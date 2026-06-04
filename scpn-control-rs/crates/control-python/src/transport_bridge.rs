// SPDX-License-Identifier: AGPL-3.0-or-later
// ──────────────────────────────────────────────────────────────────────
// SCPN Control — UDP Transport Bridge
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// ──────────────────────────────────────────────────────────────────────

use std::io::ErrorKind;
use std::mem::size_of;
use std::net::UdpSocket;
#[cfg(all(feature = "io-uring", target_os = "linux"))]
use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(all(feature = "io-uring", target_os = "linux"))]
use io_uring::{opcode, types, IoUring};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
#[cfg(all(feature = "io-uring", target_os = "linux"))]
use socket2::{Domain, Protocol, Socket, Type};

use crate::slab::{PacketSlab, SLAB_CAPACITY};

const MAGIC: u32 = 0x534e_5054; // "SNPT"
const UDP_TRANSPORT_VERSION: u16 = 1;
const DEFAULT_ENDPOINT: &str = "239.0.0.1";
const DEFAULT_PORT: u16 = 5555;
const DEFAULT_HEARTBEAT_PORT: u16 = 0;
const DEFAULT_HEARTBEAT_TIMEOUT_MS: u64 = 3;
const HEARTBEAT_CHECK_INTERVAL_MS: u64 = 1;
#[cfg(all(feature = "io-uring", target_os = "linux"))]
const IO_URING_QUEUE_DEPTH: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransportBackend {
    StdUdp,
    #[allow(dead_code)]
    IoUring,
}

impl TransportBackend {
    fn from_name(name: &str) -> PyResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "std" | "udp" => Ok(Self::StdUdp),
            "io_uring" | "io-uring" | "ioring" => Ok(Self::IoUring),
            "" => Ok(Self::StdUdp),
            other => Err(PyRuntimeError::new_err(format!(
                "unsupported transport backend '{other}' (use 'std', 'udp', or 'io-uring' when enabled)"
            ))),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::StdUdp => "std",
            Self::IoUring => "io_uring",
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TransportFrameHeader {
    pub magic: u32,
    pub version: u16,
    pub payload_bytes: u16,
    pub sequence: u64,
    pub timestamp_ns: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TransportSnapshotFrame {
    pub header: TransportFrameHeader,
    pub status: u32,
    pub reserved0: u32,
    pub r_error: f64,
    pub z_error: f64,
    pub r_command: f64,
    pub z_command: f64,
    pub acados_time_ns: u64,
    pub snn_time_ns: u64,
    pub reserve1: u64,
    pub reserve2: u64,
}

const FRAME_SIZE: usize = size_of::<TransportSnapshotFrame>();

#[pyclass]
pub struct PyUdpTransportBridge {
    endpoint: String,
    port: u16,
    ttl: u8,
    heartbeat_port: u16,
    heartbeat_timeout_ns: u64,
    max_queue: usize,
    backend: TransportBackend,
    sequence: u64,
    sender: Option<SyncSender<usize>>,
    slab: Option<Arc<PacketSlab<FRAME_SIZE, SLAB_CAPACITY>>>,
    last_heartbeat_ns: Option<Arc<AtomicU64>>,
    stop_flag: Option<Arc<AtomicBool>>,
    handle: Option<JoinHandle<()>>,
    heartbeat_handle: Option<JoinHandle<()>>,
    heartbeat_watchdog_handle: Option<JoinHandle<()>>,
}

#[pymethods]
impl PyUdpTransportBridge {
    #[new]
    #[pyo3(signature = (endpoint=DEFAULT_ENDPOINT, port=DEFAULT_PORT, ttl=1, max_queue=4, backend="std", heartbeat_port=DEFAULT_HEARTBEAT_PORT, heartbeat_timeout_ms=DEFAULT_HEARTBEAT_TIMEOUT_MS))]
    pub(crate) fn new(
        endpoint: &str,
        port: u16,
        ttl: u8,
        max_queue: usize,
        backend: &str,
        heartbeat_port: u16,
        heartbeat_timeout_ms: u64,
    ) -> PyResult<Self> {
        let backend = TransportBackend::from_name(backend)?;
        Ok(PyUdpTransportBridge {
            endpoint: endpoint.to_string(),
            port,
            ttl,
            heartbeat_port,
            heartbeat_timeout_ns: heartbeat_timeout_ms.saturating_mul(1_000_000),
            max_queue: max_queue.min(SLAB_CAPACITY).max(1),
            backend,
            sequence: 0,
            sender: None,
            slab: None,
            last_heartbeat_ns: None,
            stop_flag: None,
            handle: None,
            heartbeat_handle: None,
            heartbeat_watchdog_handle: None,
        })
    }

    fn backend(&self) -> &'static str {
        self.backend.as_str()
    }

    pub(crate) fn start(&mut self) -> PyResult<()> {
        if self.sender.is_some() {
            return Ok(());
        }

        let heartbeat_timeout_ns = self.heartbeat_timeout_ns;
        let (tx, rx) = sync_channel(self.max_queue);
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let endpoint = self.endpoint.clone();
        let port = self.port;
        let ttl = self.ttl;
        let backend = self.backend;
        let heartbeat_port = self.heartbeat_port;

        let slab = Arc::new(PacketSlab::<FRAME_SIZE, SLAB_CAPACITY>::new());
        let slab_thread = slab.clone();

        let mut heartbeat_handle = None;
        let mut heartbeat_watchdog_handle = None;
        let last_heartbeat_ns = if heartbeat_port > 0 {
            let heartbeat = Arc::new(AtomicU64::new(current_time_ns()));
            let heartbeat_thread = heartbeat.clone();
            let stop_for_heartbeat = thread_stop.clone();
            heartbeat_handle = Some(
                thread::Builder::new()
                    .name("snpt-udp-heartbeat-receiver".to_string())
                    .spawn(move || {
                        run_udp_heartbeat_monitor(
                            heartbeat_thread,
                            heartbeat_port,
                            stop_for_heartbeat,
                        );
                    })
                    .map_err(|_| {
                        PyRuntimeError::new_err("failed to start heartbeat monitor thread")
                    })?,
            );

            let stop_for_watchdog = thread_stop.clone();
            let heartbeat_watchdog = heartbeat.clone();
            heartbeat_watchdog_handle = Some(
                thread::Builder::new()
                    .name("snpt-udp-heartbeat-watchdog".to_string())
                    .spawn(move || {
                        while !stop_for_watchdog.load(Ordering::Acquire) {
                            if !heartbeat_is_alive(Some(&heartbeat_watchdog), heartbeat_timeout_ns)
                            {
                                stop_for_watchdog.store(true, Ordering::Release);
                                return;
                            }
                            thread::sleep(Duration::from_millis(HEARTBEAT_CHECK_INTERVAL_MS));
                        }
                    })
                    .map_err(|_| {
                        PyRuntimeError::new_err("failed to start heartbeat watchdog thread")
                    })?,
            );
            Some(heartbeat)
        } else {
            None
        };

        let handle = match backend {
            TransportBackend::StdUdp => {
                let heartbeat_for_thread = last_heartbeat_ns.clone();
                thread::Builder::new()
                    .name("snpt-udp-publisher-std".to_string())
                    .spawn(move || {
                        run_udp_publisher(
                            rx,
                            endpoint,
                            port,
                            ttl,
                            thread_stop,
                            slab_thread,
                            heartbeat_for_thread,
                            heartbeat_timeout_ns,
                        );
                    })
                    .map_err(|_| PyRuntimeError::new_err("failed to start transport thread"))?
            }
            TransportBackend::IoUring => {
                #[cfg(feature = "io-uring")]
                {
                    let slab_thread_uring = slab_thread.clone();
                    let heartbeat_for_thread = last_heartbeat_ns.clone();
                    thread::Builder::new()
                        .name("snpt-udp-publisher-io_uring".to_string())
                        .spawn(move || {
                            run_udp_publisher_uring(
                                rx,
                                endpoint,
                                port,
                                ttl,
                                thread_stop,
                                slab_thread_uring,
                                heartbeat_for_thread,
                                heartbeat_timeout_ns,
                            );
                        })
                        .map_err(|_| PyRuntimeError::new_err("failed to start transport thread"))?
                }

                #[cfg(not(feature = "io-uring"))]
                {
                    return Err(PyRuntimeError::new_err(
                        "io_uring transport backend requires build flag: --features io-uring",
                    ));
                }
            }
        };

        self.sender = Some(tx);
        self.stop_flag = Some(stop);
        self.slab = Some(slab);
        self.last_heartbeat_ns = last_heartbeat_ns;
        self.handle = Some(handle);
        self.heartbeat_handle = heartbeat_handle;
        self.heartbeat_watchdog_handle = heartbeat_watchdog_handle;
        Ok(())
    }

    fn payload_bytes(&self) -> usize {
        size_of::<TransportSnapshotFrame>()
    }

    fn is_running(&self) -> bool {
        self.sender.is_some() && self.handle.is_some()
    }

    fn heartbeat_timeout_ns(&self) -> u64 {
        self.heartbeat_timeout_ns
    }

    fn heartbeat_timeout_ms(&self) -> u64 {
        self.heartbeat_timeout_ns / 1_000_000
    }

    fn heartbeat_age_ns(&self) -> u64 {
        if let Some(last) = self.last_heartbeat_ns.as_ref() {
            current_time_ns().saturating_sub(last.load(Ordering::Acquire))
        } else {
            0
        }
    }

    pub(crate) fn heartbeat_expired(&self) -> bool {
        if self.heartbeat_port == 0 {
            false
        } else if let Some(last) = self.last_heartbeat_ns.as_ref() {
            !heartbeat_is_alive(Some(last.as_ref()), self.heartbeat_timeout_ns)
        } else {
            true
        }
    }

    fn stopped(&self) -> bool {
        self.stop_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire))
    }

    pub(crate) fn publish(
        &mut self,
        r_error: f64,
        z_error: f64,
        r_command: f64,
        z_command: f64,
        acados_time_ns: u64,
        snn_time_ns: u64,
        status: u32,
    ) -> PyResult<bool> {
        if !r_error.is_finite()
            || !z_error.is_finite()
            || !r_command.is_finite()
            || !z_command.is_finite()
        {
            return Err(PyRuntimeError::new_err(
                "transport payload values must be finite",
            ));
        }

        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("transport bridge is not started"))?;
        if self.stopped() {
            return Err(PyRuntimeError::new_err("transport bridge is stopped"));
        }
        let slab = self
            .slab
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("transport slab is not initialized"))?;

        if self.heartbeat_port > 0 && self.heartbeat_expired() {
            return Err(PyRuntimeError::new_err("transport heartbeat timeout"));
        }

        self.sequence = self.sequence.wrapping_add(1);
        let timestamp_ns = current_time_ns();

        let frame = TransportSnapshotFrame {
            header: TransportFrameHeader {
                magic: MAGIC,
                version: UDP_TRANSPORT_VERSION,
                payload_bytes: size_of::<TransportSnapshotFrame>() as u16,
                sequence: self.sequence,
                timestamp_ns,
            },
            status,
            reserved0: 0,
            r_error,
            z_error,
            r_command,
            z_command,
            acados_time_ns,
            snn_time_ns,
            reserve1: 0,
            reserve2: 0,
        };

        let frame_bytes = unsafe {
            std::slice::from_raw_parts(
                (&frame as *const TransportSnapshotFrame).cast::<u8>(),
                FRAME_SIZE,
            )
        };

        let Some(index) = slab.lease_and_write(frame_bytes) else {
            return Ok(false);
        };

        match sender.try_send(index) {
            Ok(()) => Ok(true),
            Err(TrySendError::Full(_)) => {
                slab.release(index);
                Ok(false)
            }
            Err(TrySendError::Disconnected(_)) => {
                slab.release(index);
                Err(PyRuntimeError::new_err("udp transport sender disconnected"))
            }
        }
    }

    pub(crate) fn stop(&mut self) -> PyResult<()> {
        if let Some(stop) = self.stop_flag.take() {
            stop.store(true, Ordering::Release);
        }

        self.sender.take();

        if let Some(heartbeat_handle) = self.heartbeat_handle.take() {
            let _ = heartbeat_handle.join();
        }

        if let Some(heartbeat_watchdog_handle) = self.heartbeat_watchdog_handle.take() {
            let _ = heartbeat_watchdog_handle.join();
        }

        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| PyRuntimeError::new_err("udp transport thread panicked"))?;
        }

        self.slab.take();
        self.last_heartbeat_ns.take();
        Ok(())
    }
}

impl Drop for PyUdpTransportBridge {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

fn run_udp_publisher(
    rx: Receiver<usize>,
    endpoint: String,
    port: u16,
    ttl: u8,
    stop: Arc<AtomicBool>,
    slab: Arc<PacketSlab<FRAME_SIZE, SLAB_CAPACITY>>,
    last_heartbeat_ns: Option<Arc<AtomicU64>>,
    heartbeat_timeout_ns: u64,
) {
    let target = format!("{endpoint}:{port}");
    let socket = match UdpSocket::bind("0.0.0.0:0") {
        Ok(socket) => socket,
        Err(_) => return,
    };

    let _ = socket.set_multicast_ttl_v4(ttl as u32);
    let _ = socket.connect(&target);

    while !stop.load(Ordering::Acquire) {
        match rx.recv_timeout(Duration::from_millis(2)) {
            Ok(index) => {
                if !heartbeat_is_alive(last_heartbeat_ns.as_deref(), heartbeat_timeout_ns) {
                    slab.release(index);
                    stop.store(true, Ordering::Release);
                    return;
                }

                let Some(frame_ptr) = slab.get_ptr_for_io(index) else {
                    slab.release(index);
                    continue;
                };
                let frame_bytes = unsafe { std::slice::from_raw_parts(frame_ptr, FRAME_SIZE) };

                if socket.send(frame_bytes).is_err() {
                    slab.release(index);
                    stop.store(true, Ordering::Release);
                    return;
                }

                slab.release(index);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if let Some(last_heartbeat_ns) = last_heartbeat_ns.as_deref() {
                    if !heartbeat_is_alive(Some(last_heartbeat_ns), heartbeat_timeout_ns) {
                        stop.store(true, Ordering::Release);
                        return;
                    }
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }

    while let Ok(index) = rx.try_recv() {
        if let Some(frame_ptr) = slab.get_ptr_for_io(index) {
            let frame_bytes = unsafe { std::slice::from_raw_parts(frame_ptr, FRAME_SIZE) };
            let _ = socket.send(frame_bytes);
        }
        slab.release(index);
    }
}

#[cfg(all(feature = "io-uring", target_os = "linux"))]
fn run_udp_publisher_uring(
    rx: Receiver<usize>,
    endpoint: String,
    port: u16,
    ttl: u8,
    stop: Arc<AtomicBool>,
    slab: Arc<PacketSlab<FRAME_SIZE, SLAB_CAPACITY>>,
    last_heartbeat_ns: Option<Arc<AtomicU64>>,
    heartbeat_timeout_ns: u64,
) {
    use std::net::SocketAddr;

    let target: SocketAddr = match format!("{endpoint}:{port}").parse() {
        Ok(addr) => addr,
        Err(_) => {
            run_udp_publisher(
                rx,
                endpoint,
                port,
                ttl,
                stop,
                slab,
                last_heartbeat_ns,
                heartbeat_timeout_ns,
            );
            return;
        }
    };

    let socket = match build_udp_socket(target, ttl) {
        Ok(socket) => socket,
        Err(_) => {
            run_udp_publisher(
                rx,
                endpoint,
                port,
                ttl,
                stop,
                slab,
                last_heartbeat_ns,
                heartbeat_timeout_ns,
            );
            return;
        }
    };

    let mut ring = match IoUring::new(IO_URING_QUEUE_DEPTH) {
        Ok(ring) => ring,
        Err(_) => {
            run_udp_publisher(
                rx,
                endpoint,
                port,
                ttl,
                stop,
                slab,
                last_heartbeat_ns,
                heartbeat_timeout_ns,
            );
            return;
        }
    };

    let fd = socket.as_raw_fd();
    let frame_size = FRAME_SIZE as u32;

    while !stop.load(Ordering::Acquire) {
        match rx.recv_timeout(Duration::from_millis(2)) {
            Ok(index) => {
                if !heartbeat_is_alive(last_heartbeat_ns.as_deref(), heartbeat_timeout_ns) {
                    slab.release(index);
                    stop.store(true, Ordering::Release);
                    return;
                }

                let Some(frame_ptr) = slab.get_ptr_for_io(index) else {
                    slab.release(index);
                    continue;
                };

                let mut sqe = opcode::Send::new(types::Fd(fd), frame_ptr, frame_size).build();
                sqe.set_user_data(index as u64);

                if let Err(err) = submit_or_fallback_to_std(
                    &mut ring,
                    &socket,
                    &sqe,
                    frame_ptr,
                    &slab,
                    &stop,
                    &last_heartbeat_ns,
                    heartbeat_timeout_ns,
                    index,
                ) {
                    let _ = err;
                    return;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if let Some(last_heartbeat_ns) = last_heartbeat_ns.as_deref() {
                    if !heartbeat_is_alive(Some(last_heartbeat_ns), heartbeat_timeout_ns) {
                        stop.store(true, Ordering::Release);
                        return;
                    }
                }

                let _ = drain_io_uring_completion(&mut ring, &slab);
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }

    while let Ok(index) = rx.try_recv() {
        let Some(frame_ptr) = slab.get_ptr_for_io(index) else {
            slab.release(index);
            continue;
        };

        let mut sqe = opcode::Send::new(types::Fd(fd), frame_ptr, frame_size).build();
        sqe.set_user_data(index as u64);

        if submit_or_fallback_to_std(
            &mut ring,
            &socket,
            &sqe,
            frame_ptr,
            &slab,
            &stop,
            &last_heartbeat_ns,
            heartbeat_timeout_ns,
            index,
        )
        .is_err()
        {
            break;
        }
    }

    // Drain completions at shutdown and drop any remaining leased slots.
    let _ = drain_io_uring_completion(&mut ring, &slab);
    while let Ok(index) = rx.try_recv() {
        slab.release(index);
    }
}

#[cfg(all(not(feature = "io-uring"), target_os = "linux"))]
#[allow(dead_code)]
fn run_udp_publisher_uring(
    rx: Receiver<usize>,
    endpoint: String,
    port: u16,
    ttl: u8,
    stop: Arc<AtomicBool>,
    slab: Arc<PacketSlab<FRAME_SIZE, SLAB_CAPACITY>>,
    last_heartbeat_ns: Option<Arc<AtomicU64>>,
    heartbeat_timeout_ns: u64,
) {
    run_udp_publisher(
        rx,
        endpoint,
        port,
        ttl,
        stop,
        slab,
        last_heartbeat_ns,
        heartbeat_timeout_ns,
    );
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn run_udp_publisher_uring(
    rx: Receiver<usize>,
    endpoint: String,
    port: u16,
    ttl: u8,
    stop: Arc<AtomicBool>,
    slab: Arc<PacketSlab<FRAME_SIZE, SLAB_CAPACITY>>,
    last_heartbeat_ns: Option<Arc<AtomicU64>>,
    heartbeat_timeout_ns: u64,
) {
    run_udp_publisher(
        rx,
        endpoint,
        port,
        ttl,
        stop,
        slab,
        last_heartbeat_ns,
        heartbeat_timeout_ns,
    );
}

#[cfg(all(feature = "io-uring", target_os = "linux"))]
fn submit_or_fallback_to_std(
    ring: &mut IoUring,
    socket: &UdpSocket,
    sqe: &io_uring::squeue::Entry,
    frame_ptr: *const u8,
    slab: &Arc<PacketSlab<FRAME_SIZE, SLAB_CAPACITY>>,
    stop: &AtomicBool,
    last_heartbeat_ns: &Option<Arc<AtomicU64>>,
    heartbeat_timeout_ns: u64,
    index: usize,
) -> Result<(), &'static str> {
    let pushed = unsafe { ring.submission().push(sqe).is_ok() };

    if !pushed {
        if let Some(false) = drain_or_submit(ring, slab)? {
            let frame_bytes = unsafe { std::slice::from_raw_parts(frame_ptr, FRAME_SIZE) };
            let _ = socket.send(frame_bytes);
            slab.release(index);
            stop.store(true, Ordering::Release);
            return Ok(());
        }

        if !unsafe { ring.submission().push(sqe).is_ok() } {
            let frame_bytes = unsafe { std::slice::from_raw_parts(frame_ptr, FRAME_SIZE) };
            let _ = socket.send(frame_bytes);
            slab.release(index);
            stop.store(true, Ordering::Release);
            return Err("io_uring submission queue full after drain");
        }
    }

    if ring.submit_and_wait(1).is_err() {
        if !heartbeat_is_alive(last_heartbeat_ns.as_deref(), heartbeat_timeout_ns) {
            slab.release(index);
            stop.store(true, Ordering::Release);
            return Err("io_uring submit failed");
        }

        let frame_bytes = unsafe { std::slice::from_raw_parts(frame_ptr, FRAME_SIZE) };
        let _ = socket.send(frame_bytes);
        slab.release(index);
        stop.store(true, Ordering::Release);
        return Err("io_uring submit failed");
    }

    if let Some(false) = drain_io_uring_completion(ring, slab) {
        let frame_bytes = unsafe { std::slice::from_raw_parts(frame_ptr, FRAME_SIZE) };
        let _ = socket.send(frame_bytes);
        slab.release(index);
        stop.store(true, Ordering::Release);
        return Err("io_uring completion failed");
    }

    Ok(())
}

#[cfg(all(feature = "io-uring", target_os = "linux"))]
fn drain_or_submit(
    ring: &mut IoUring,
    slab: &PacketSlab<FRAME_SIZE, SLAB_CAPACITY>,
) -> Result<Option<bool>, &'static str> {
    if ring.submit_and_wait(1).is_err() {
        return Err("io_uring submit failed");
    }

    Ok(drain_io_uring_completion(ring, slab))
}

#[cfg(all(feature = "io-uring", target_os = "linux"))]
fn drain_io_uring_completion(
    ring: &mut IoUring,
    slab: &PacketSlab<FRAME_SIZE, SLAB_CAPACITY>,
) -> Option<bool> {
    let mut ok = true;
    let mut had = false;

    while let Some(cqe) = ring.completion().next() {
        had = true;

        if cqe.result() < 0 {
            ok = false;
        }

        let index = cqe.user_data() as usize;
        if index < SLAB_CAPACITY {
            slab.release(index);
        }
    }

    if had {
        Some(ok)
    } else {
        None
    }
}

#[allow(dead_code)]
fn build_udp_socket(target_addr: std::net::SocketAddr, ttl: u8) -> std::io::Result<UdpSocket> {
    #[cfg(all(feature = "io-uring", target_os = "linux"))]
    {
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        let _ = socket.set_reuse_address(true);
        let _ = socket.set_broadcast(true);
        let _ = socket.set_multicast_ttl_v4(ttl as u32);
        let bind_addr: std::net::SocketAddr = "0.0.0.0:0".parse().map_err(std::io::Error::other)?;
        socket.bind(&bind_addr.into())?;
        socket.connect(&target_addr.into())?;
        let std_socket: UdpSocket = socket.into();
        return Ok(std_socket);
    }

    #[cfg(not(all(feature = "io-uring", target_os = "linux")))]
    {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        let _ = socket.set_multicast_ttl_v4(ttl as u32);
        socket.connect(target_addr)?;
        return Ok(socket);
    }
}

fn run_udp_heartbeat_monitor(
    last_heartbeat_ns: Arc<AtomicU64>,
    heartbeat_port: u16,
    stop: Arc<AtomicBool>,
) {
    let socket = match UdpSocket::bind(("0.0.0.0", heartbeat_port)) {
        Ok(socket) => socket,
        Err(_) => {
            last_heartbeat_ns.store(0, Ordering::Release);
            stop.store(true, Ordering::Release);
            return;
        }
    };

    let mut buf = [0u8; 64];
    let _ = socket.set_read_timeout(Some(Duration::from_millis(HEARTBEAT_CHECK_INTERVAL_MS)));

    while !stop.load(Ordering::Acquire) {
        match socket.recv_from(&mut buf) {
            Ok((size, _)) if size > 0 => {
                last_heartbeat_ns.store(current_time_ns(), Ordering::Release);
            }
            Ok(_) => {}
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(_) => {}
        }
    }
}

fn heartbeat_is_alive(last_heartbeat_ns: Option<&AtomicU64>, timeout_ns: u64) -> bool {
    match last_heartbeat_ns {
        Some(last_heartbeat_ns) => {
            let last_ns = last_heartbeat_ns.load(Ordering::Acquire);
            if last_ns == 0 {
                return false;
            }
            let now_ns = current_time_ns();
            now_ns.saturating_sub(last_ns) <= timeout_ns
        }
        None => true,
    }
}

#[inline]
fn current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_slab_roundtrip_and_release() {
        let slab = PacketSlab::<FRAME_SIZE, SLAB_CAPACITY>::new();
        let payload = [0x5a_u8; FRAME_SIZE];

        let index = slab
            .lease_and_write(&payload)
            .expect("initial lease should succeed");
        let ptr = slab
            .get_ptr_for_io(index)
            .expect("leased slot should provide a valid pointer");

        let frame = unsafe { std::slice::from_raw_parts(ptr, FRAME_SIZE) };
        assert_eq!(frame, &payload);
        assert_eq!(slab.in_use_count(), 1);

        slab.release(index);
        assert_eq!(slab.in_use_count(), 0);
    }

    #[test]
    fn packet_slab_capacity_exhaustion_and_recycle() {
        let slab = PacketSlab::<FRAME_SIZE, SLAB_CAPACITY>::new();
        let payload = [0xA5_u8; FRAME_SIZE];

        let mut indices = Vec::with_capacity(SLAB_CAPACITY);
        for _ in 0..SLAB_CAPACITY {
            indices.push(slab.lease_and_write(&payload).expect("slot should exist"));
        }

        assert_eq!(slab.lease_and_write(&payload), None);

        for index in indices {
            slab.release(index);
        }

        assert_eq!(slab.in_use_count(), 0);
        assert!(slab.lease_and_write(&payload).is_some());
    }

    #[test]
    fn heartbeat_expiry_uses_zero_as_stale_signal() {
        let heartbeat = AtomicU64::new(0);
        assert!(!heartbeat_is_alive(Some(&heartbeat), 1_000_000));

        heartbeat.store(current_time_ns(), Ordering::Release);
        assert!(heartbeat_is_alive(Some(&heartbeat), 1_000_000));

        heartbeat.store(
            current_time_ns().saturating_sub(2_000_000),
            Ordering::Release,
        );
        assert!(!heartbeat_is_alive(Some(&heartbeat), 1_000));
    }

    #[test]
    fn heartbeat_monitor_bind_failure_marks_timestamp_stale() {
        use std::net::UdpSocket;

        let occupied =
            UdpSocket::bind(("127.0.0.1", 0)).expect("reserved heartbeat socket should bind");
        let heartbeat_port = occupied
            .local_addr()
            .expect("reserved heartbeat socket should report local addr")
            .port();

        let heartbeat = Arc::new(AtomicU64::new(current_time_ns()));
        let stop = Arc::new(AtomicBool::new(false));
        let monitor_heartbeat = heartbeat.clone();
        let monitor_stop = stop.clone();

        let monitor = std::thread::spawn(move || {
            run_udp_heartbeat_monitor(monitor_heartbeat, heartbeat_port, monitor_stop);
        });

        std::thread::sleep(Duration::from_millis(4));
        stop.store(true, Ordering::Release);
        let _ = monitor.join();

        let last = heartbeat.load(Ordering::Acquire);
        assert_eq!(
            last, 0,
            "heartbeat monitor should force stale value when bind to port is denied"
        );
    }

    #[test]
    fn heartbeat_timeout_stops_publisher() {
        use std::net::UdpSocket;

        let payload_listener = UdpSocket::bind(("127.0.0.1", 0)).expect("payload sink should bind");
        let payload_port = payload_listener
            .local_addr()
            .expect("payload sink port should be discoverable")
            .port();
        let heartbeat_port = UdpSocket::bind(("127.0.0.1", 0))
            .expect("heartbeat sink should bind")
            .local_addr()
            .expect("heartbeat sink port should be discoverable")
            .port();

        let mut bridge =
            PyUdpTransportBridge::new("127.0.0.1", payload_port, 1, 16, "std", heartbeat_port, 3)
                .expect("bridge should initialize");

        bridge.start().expect("bridge thread should start");

        // Give the watchdog a chance to sample heartbeat freshness at least once.
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(
            !bridge.stopped(),
            "bridge should be active before heartbeat timeout"
        );

        // With no heartbeat traffic and 3ms timeout, expect stop to be asserted within 5ms.
        let stop_wait_start = std::time::Instant::now();
        let stop_deadline = stop_wait_start + std::time::Duration::from_millis(5);
        while !bridge.stopped() && std::time::Instant::now() < stop_deadline {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }

        let stopped_before_teardown = bridge.stopped();
        bridge.stop().expect("bridge should stop");

        let detection_time = stop_wait_start.elapsed();

        assert!(
            stopped_before_teardown,
            "heartbeat timeout should drive stop flag after timeout window"
        );
        assert!(
            detection_time < std::time::Duration::from_millis(5),
            "heartbeat timeout should drive stop inside deterministic window, got {detection_time:?}"
        );
    }
}
