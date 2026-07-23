// SPDX-License-Identifier: AGPL-3.0-or-later
// ──────────────────────────────────────────────────────────────────────
// SCPN Control — UDP Transport Bridge
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// ──────────────────────────────────────────────────────────────────────

use std::env;
use std::fs;
use std::io::{ErrorKind, Read};
use std::mem::size_of;
use std::net::{IpAddr, UdpSocket};
#[cfg(all(feature = "io-uring", target_os = "linux"))]
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, KeyInit, Mac};
#[cfg(all(feature = "io-uring", target_os = "linux"))]
use io_uring::{opcode, types, IoUring};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use sha2::Sha256;
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
const HEARTBEAT_MAGIC: &[u8; 8] = b"SCPNHB01";
const HEARTBEAT_SIGNED_BYTES: usize = 16;
const HEARTBEAT_TAG_BYTES: usize = 32;
const HEARTBEAT_FRAME_BYTES: usize = HEARTBEAT_SIGNED_BYTES + HEARTBEAT_TAG_BYTES;
const HEARTBEAT_MIN_KEY_BYTES: usize = 32;
const HEARTBEAT_MAX_KEY_BYTES: usize = 64;
const HEARTBEAT_KEY_FILE_ENV: &str = "SCPN_CONTROL_HEARTBEAT_KEY_FILE";
const HEARTBEAT_ALLOWED_SOURCE_ENV: &str = "SCPN_CONTROL_HEARTBEAT_ALLOWED_SOURCE";
const HEARTBEAT_BIND_HOST_ENV: &str = "SCPN_CONTROL_HEARTBEAT_BIND_HOST";
const DEFAULT_HEARTBEAT_BIND_HOST: &str = "127.0.0.1";
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

type HmacSha256 = Hmac<Sha256>;

#[derive(Clone)]
struct HeartbeatAuth {
    key: Arc<[u8]>,
    allowed_source: IpAddr,
    bind_host: IpAddr,
}

fn heartbeat_auth_from_path(
    key_path: &Path,
    allowed_source: &str,
    bind_host: &str,
) -> Result<HeartbeatAuth, String> {
    let metadata = fs::symlink_metadata(key_path)
        .map_err(|_| "heartbeat HMAC key file is not readable".to_string())?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("heartbeat HMAC key path must be a regular non-symlink file".to_string());
    }
    let key_len = usize::try_from(metadata.len())
        .map_err(|_| "heartbeat HMAC key file length is unsupported".to_string())?;
    if !(HEARTBEAT_MIN_KEY_BYTES..=HEARTBEAT_MAX_KEY_BYTES).contains(&key_len) {
        return Err(format!(
            "heartbeat HMAC key file must contain {HEARTBEAT_MIN_KEY_BYTES}..={HEARTBEAT_MAX_KEY_BYTES} bytes"
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(
                "heartbeat HMAC key file must not grant group or other permissions".to_string(),
            );
        }
    }
    let key_file = fs::File::open(key_path)
        .map_err(|_| "heartbeat HMAC key file is not readable".to_string())?;
    let opened_metadata = key_file
        .metadata()
        .map_err(|_| "heartbeat HMAC key file metadata is not readable".to_string())?;
    if !opened_metadata.is_file() || opened_metadata.len() != metadata.len() {
        return Err("heartbeat HMAC key file changed while being opened".to_string());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        if opened_metadata.dev() != metadata.dev() || opened_metadata.ino() != metadata.ino() {
            return Err("heartbeat HMAC key file changed while being opened".to_string());
        }
        if opened_metadata.permissions().mode() & 0o077 != 0 {
            return Err(
                "heartbeat HMAC key file must not grant group or other permissions".to_string(),
            );
        }
    }
    let mut key = Vec::with_capacity(key_len);
    key_file
        .take((HEARTBEAT_MAX_KEY_BYTES + 1) as u64)
        .read_to_end(&mut key)
        .map_err(|_| "heartbeat HMAC key file is not readable".to_string())?;
    if key.len() != key_len {
        return Err("heartbeat HMAC key file changed while being read".to_string());
    }
    let allowed_source = allowed_source
        .parse::<IpAddr>()
        .map_err(|_| "heartbeat allowed source must be one IP address".to_string())?;
    let bind_host = bind_host
        .parse::<IpAddr>()
        .map_err(|_| "heartbeat bind host must be one IP address".to_string())?;
    Ok(HeartbeatAuth {
        key: Arc::from(key),
        allowed_source,
        bind_host,
    })
}

fn heartbeat_auth_from_env() -> Result<HeartbeatAuth, String> {
    let key_path = env::var(HEARTBEAT_KEY_FILE_ENV).map_err(|_| {
        format!("{HEARTBEAT_KEY_FILE_ENV} is required when heartbeat monitoring is enabled")
    })?;
    let allowed_source = env::var(HEARTBEAT_ALLOWED_SOURCE_ENV).map_err(|_| {
        format!("{HEARTBEAT_ALLOWED_SOURCE_ENV} is required when heartbeat monitoring is enabled")
    })?;
    let bind_host = env::var(HEARTBEAT_BIND_HOST_ENV)
        .unwrap_or_else(|_| DEFAULT_HEARTBEAT_BIND_HOST.to_string());
    heartbeat_auth_from_path(Path::new(&key_path), &allowed_source, &bind_host)
}

fn verify_heartbeat_frame(
    frame: &[u8],
    source: IpAddr,
    auth: &HeartbeatAuth,
    last_counter: &mut u64,
) -> bool {
    if source != auth.allowed_source
        || frame.len() != HEARTBEAT_FRAME_BYTES
        || &frame[..HEARTBEAT_MAGIC.len()] != HEARTBEAT_MAGIC
    {
        return false;
    }
    let counter_bytes: [u8; 8] =
        match frame[HEARTBEAT_MAGIC.len()..HEARTBEAT_SIGNED_BYTES].try_into() {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };
    let counter = u64::from_be_bytes(counter_bytes);
    if counter == 0 {
        return false;
    }
    let Ok(mut mac) = HmacSha256::new_from_slice(auth.key.as_ref()) else {
        return false;
    };
    mac.update(&frame[..HEARTBEAT_SIGNED_BYTES]);
    if mac.verify_slice(&frame[HEARTBEAT_SIGNED_BYTES..]).is_err() {
        return false;
    }
    if counter <= *last_counter {
        return false;
    }
    *last_counter = counter;
    true
}

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
    heartbeat_auth: Option<HeartbeatAuth>,
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
            max_queue: max_queue.clamp(1, SLAB_CAPACITY),
            backend,
            sequence: 0,
            sender: None,
            slab: None,
            last_heartbeat_ns: None,
            stop_flag: None,
            handle: None,
            heartbeat_handle: None,
            heartbeat_watchdog_handle: None,
            heartbeat_auth: None,
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
        let heartbeat_auth = if heartbeat_port > 0 {
            Some(
                self.heartbeat_auth
                    .clone()
                    .map_or_else(heartbeat_auth_from_env, Ok)
                    .map_err(PyRuntimeError::new_err)?,
            )
        } else {
            None
        };

        let slab = Arc::new(PacketSlab::<FRAME_SIZE, SLAB_CAPACITY>::new());
        let slab_thread = slab.clone();

        let mut heartbeat_handle = None;
        let mut heartbeat_watchdog_handle = None;
        let last_heartbeat_ns = if heartbeat_port > 0 {
            let heartbeat = Arc::new(AtomicU64::new(monotonic_time_ns()));
            let heartbeat_thread = heartbeat.clone();
            let stop_for_heartbeat = thread_stop.clone();
            let heartbeat_auth = heartbeat_auth.ok_or_else(|| {
                PyRuntimeError::new_err("heartbeat authentication configuration is missing")
            })?;
            heartbeat_handle = Some(
                thread::Builder::new()
                    .name("snpt-udp-heartbeat-receiver".to_string())
                    .spawn(move || {
                        run_udp_heartbeat_monitor(
                            heartbeat_thread,
                            heartbeat_port,
                            stop_for_heartbeat,
                            heartbeat_auth,
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
            monotonic_time_ns().saturating_sub(last.load(Ordering::Acquire))
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

    #[expect(
        clippy::too_many_arguments,
        reason = "transport frame fields remain explicit across the PyO3 boundary"
    )]
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

        // SAFETY: `frame` is a live, fully-initialised `#[repr(C)]` value with no
        // padding, and `FRAME_SIZE == size_of::<TransportSnapshotFrame>()`. The
        // byte slice borrows `frame` for this call only and is read (copied into
        // the slab) before `frame` goes out of scope.
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

#[expect(
    clippy::too_many_arguments,
    reason = "thread entrypoint receives immutable launch settings"
)]
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
                // SAFETY: `index` was leased by the producer and handed over the
                // channel, transferring ownership to this thread; `get_ptr_for_io`
                // returned a pointer to that slot's FRAME_SIZE initialised bytes,
                // which stay valid until the `slab.release(index)` below.
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
            // SAFETY: as above — `index` owns a leased slot whose FRAME_SIZE bytes
            // are initialised and stay valid until the `slab.release(index)` below.
            let frame_bytes = unsafe { std::slice::from_raw_parts(frame_ptr, FRAME_SIZE) };
            let _ = socket.send(frame_bytes);
        }
        slab.release(index);
    }
}

#[cfg(all(feature = "io-uring", target_os = "linux"))]
#[expect(
    clippy::too_many_arguments,
    reason = "io_uring thread entrypoint receives immutable launch settings"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "io_uring thread entrypoint receives immutable launch settings"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "io_uring thread entrypoint receives immutable launch settings"
)]
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
#[expect(
    clippy::too_many_arguments,
    reason = "hot-path fallback keeps borrowed ring/socket/slab state explicit"
)]
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
    // SAFETY: the SQE references `frame_ptr`, which addresses slot `index`'s leased
    // bytes; that slot is not released until its completion is reaped in
    // `drain_io_uring_completion`, so the buffer outlives the kernel's async read.
    let pushed = unsafe { ring.submission().push(sqe).is_ok() };

    if !pushed {
        if let Some(false) = drain_or_submit(ring, slab)? {
            // SAFETY: `frame_ptr` addresses slot `index`'s initialised FRAME_SIZE
            // bytes, still leased here; read once for the std fallback send.
            let frame_bytes = unsafe { std::slice::from_raw_parts(frame_ptr, FRAME_SIZE) };
            let _ = socket.send(frame_bytes);
            slab.release(index);
            stop.store(true, Ordering::Release);
            return Ok(());
        }

        // SAFETY: same invariant as the first push — the SQE's `frame_ptr` slot
        // stays leased until its completion is drained.
        if !unsafe { ring.submission().push(sqe).is_ok() } {
            // SAFETY: `frame_ptr` addresses slot `index`'s initialised FRAME_SIZE
            // bytes, still leased here; read once for the std fallback send.
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

        // SAFETY: `frame_ptr` addresses slot `index`'s initialised FRAME_SIZE
        // bytes, still leased here; read once for the std fallback send.
        let frame_bytes = unsafe { std::slice::from_raw_parts(frame_ptr, FRAME_SIZE) };
        let _ = socket.send(frame_bytes);
        slab.release(index);
        stop.store(true, Ordering::Release);
        return Err("io_uring submit failed");
    }

    if let Some(false) = drain_io_uring_completion(ring, slab) {
        // SAFETY: `frame_ptr` addresses slot `index`'s initialised FRAME_SIZE
        // bytes, still leased here; read once for the std fallback send.
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
        Ok(std_socket)
    }

    #[cfg(not(all(feature = "io-uring", target_os = "linux")))]
    {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        let _ = socket.set_multicast_ttl_v4(ttl as u32);
        socket.connect(target_addr)?;
        Ok(socket)
    }
}

fn run_udp_heartbeat_monitor(
    last_heartbeat_ns: Arc<AtomicU64>,
    heartbeat_port: u16,
    stop: Arc<AtomicBool>,
    auth: HeartbeatAuth,
) {
    let socket = match UdpSocket::bind((auth.bind_host, heartbeat_port)) {
        Ok(socket) => socket,
        Err(_) => {
            last_heartbeat_ns.store(0, Ordering::Release);
            stop.store(true, Ordering::Release);
            return;
        }
    };

    let mut buf = [0u8; 64];
    let mut last_counter = 0_u64;
    let _ = socket.set_read_timeout(Some(Duration::from_millis(HEARTBEAT_CHECK_INTERVAL_MS)));

    while !stop.load(Ordering::Acquire) {
        match socket.recv_from(&mut buf) {
            Ok((size, source))
                if verify_heartbeat_frame(&buf[..size], source.ip(), &auth, &mut last_counter) =>
            {
                last_heartbeat_ns.store(monotonic_time_ns(), Ordering::Release);
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
            let now_ns = monotonic_time_ns();
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

#[inline]
fn monotonic_time_ns() -> u64 {
    static PROCESS_EPOCH: OnceLock<Instant> = OnceLock::new();
    let elapsed = PROCESS_EPOCH.get_or_init(Instant::now).elapsed().as_nanos();
    elapsed.min(u128::from(u64::MAX - 1)) as u64 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_HEARTBEAT_KEY: [u8; 32] = [0x42; 32];

    fn test_heartbeat_auth() -> HeartbeatAuth {
        HeartbeatAuth {
            key: Arc::from(TEST_HEARTBEAT_KEY),
            allowed_source: "127.0.0.1".parse().expect("loopback source must parse"),
            bind_host: "127.0.0.1".parse().expect("loopback bind host must parse"),
        }
    }

    fn signed_heartbeat_frame(counter: u64, key: &[u8]) -> [u8; HEARTBEAT_FRAME_BYTES] {
        let mut frame = [0_u8; HEARTBEAT_FRAME_BYTES];
        frame[..HEARTBEAT_MAGIC.len()].copy_from_slice(HEARTBEAT_MAGIC);
        frame[HEARTBEAT_MAGIC.len()..HEARTBEAT_SIGNED_BYTES]
            .copy_from_slice(&counter.to_be_bytes());
        let mut mac = HmacSha256::new_from_slice(key).expect("test HMAC key must be accepted");
        mac.update(&frame[..HEARTBEAT_SIGNED_BYTES]);
        frame[HEARTBEAT_SIGNED_BYTES..].copy_from_slice(&mac.finalize().into_bytes());
        frame
    }

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

        // SAFETY: `ptr` came from `get_ptr_for_io` for the slot just leased and
        // written above; it addresses FRAME_SIZE initialised bytes that stay valid
        // until the slot is released below.
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

        heartbeat.store(monotonic_time_ns(), Ordering::Release);
        assert!(heartbeat_is_alive(Some(&heartbeat), 1_000_000));

        heartbeat.store(
            monotonic_time_ns().saturating_sub(2_000_000),
            Ordering::Release,
        );
        assert!(!heartbeat_is_alive(Some(&heartbeat), 1_000));
    }

    #[test]
    fn heartbeat_frame_requires_source_magic_hmac_and_monotonic_counter() {
        let auth = test_heartbeat_auth();
        let mut last_counter = 0_u64;
        let source = auth.allowed_source;
        let first = signed_heartbeat_frame(1, auth.key.as_ref());
        let interoperable_tag = [
            0xea, 0x68, 0x62, 0xd2, 0xeb, 0x66, 0xc2, 0x46, 0xfb, 0x80, 0xc9, 0x8e, 0x90, 0x01,
            0xd8, 0xe1, 0x61, 0x36, 0xe8, 0x71, 0x4f, 0xa6, 0x4c, 0xb6, 0xe1, 0x4e, 0x45, 0x3b,
            0x0c, 0xc9, 0xa7, 0xf8,
        ];
        assert_eq!(&first[HEARTBEAT_SIGNED_BYTES..], interoperable_tag);

        assert!(verify_heartbeat_frame(
            &first,
            source,
            &auth,
            &mut last_counter
        ));
        assert_eq!(last_counter, 1);
        assert!(!verify_heartbeat_frame(
            &first,
            source,
            &auth,
            &mut last_counter
        ));

        let zero = signed_heartbeat_frame(0, auth.key.as_ref());
        assert!(!verify_heartbeat_frame(
            &zero,
            source,
            &auth,
            &mut last_counter
        ));
        let mut wrong_magic = signed_heartbeat_frame(2, auth.key.as_ref());
        wrong_magic[0] ^= 0xff;
        assert!(!verify_heartbeat_frame(
            &wrong_magic,
            source,
            &auth,
            &mut last_counter
        ));

        let mut tampered = signed_heartbeat_frame(2, auth.key.as_ref());
        tampered[HEARTBEAT_SIGNED_BYTES] ^= 0x01;
        assert!(!verify_heartbeat_frame(
            &tampered,
            source,
            &auth,
            &mut last_counter
        ));
        assert!(!verify_heartbeat_frame(
            &signed_heartbeat_frame(2, auth.key.as_ref()),
            "127.0.0.2".parse().expect("alternate loopback must parse"),
            &auth,
            &mut last_counter
        ));
        assert!(!verify_heartbeat_frame(
            &first[..HEARTBEAT_FRAME_BYTES - 1],
            source,
            &auth,
            &mut last_counter
        ));
        assert_eq!(last_counter, 1);
        assert!(verify_heartbeat_frame(
            &signed_heartbeat_frame(2, auth.key.as_ref()),
            source,
            &auth,
            &mut last_counter
        ));
        assert_eq!(last_counter, 2);
    }

    #[test]
    fn heartbeat_auth_loads_private_regular_key_and_exact_addresses() {
        let key_path = std::env::temp_dir().join(format!(
            "scpn-control-heartbeat-key-{}-{}",
            std::process::id(),
            current_time_ns()
        ));
        fs::write(&key_path, TEST_HEARTBEAT_KEY).expect("temporary key should be writable");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600))
                .expect("temporary key permissions should be settable");
        }

        let auth = heartbeat_auth_from_path(&key_path, "127.0.0.2", "127.0.0.1")
            .expect("private regular key and exact IP addresses should load");
        assert_eq!(auth.key.as_ref(), TEST_HEARTBEAT_KEY);
        assert_eq!(auth.allowed_source.to_string(), "127.0.0.2");
        assert_eq!(auth.bind_host.to_string(), "127.0.0.1");
        assert!(heartbeat_auth_from_path(&key_path, "not-an-ip", "127.0.0.1").is_err());
        assert!(heartbeat_auth_from_path(&key_path, "127.0.0.1", "not-an-ip").is_err());

        fs::write(&key_path, [0x42; HEARTBEAT_MIN_KEY_BYTES - 1])
            .expect("short test key should be writable");
        assert!(heartbeat_auth_from_path(&key_path, "127.0.0.1", "127.0.0.1").is_err());

        fs::remove_file(key_path).expect("temporary key should be removable");
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

        let heartbeat = Arc::new(AtomicU64::new(monotonic_time_ns()));
        let stop = Arc::new(AtomicBool::new(false));
        let monitor_heartbeat = heartbeat.clone();
        let monitor_stop = stop.clone();

        let monitor = std::thread::spawn(move || {
            run_udp_heartbeat_monitor(
                monitor_heartbeat,
                heartbeat_port,
                monitor_stop,
                test_heartbeat_auth(),
            );
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
    fn heartbeat_monitor_ignores_invalid_packet_and_accepts_authenticated_counter() {
        let port_reservation = UdpSocket::bind(("127.0.0.1", 0))
            .expect("temporary heartbeat port should be reservable");
        let heartbeat_port = port_reservation
            .local_addr()
            .expect("temporary heartbeat address should be available")
            .port();
        drop(port_reservation);

        let auth = test_heartbeat_auth();
        let heartbeat = Arc::new(AtomicU64::new(1));
        let stop = Arc::new(AtomicBool::new(false));
        let monitor_heartbeat = heartbeat.clone();
        let monitor_stop = stop.clone();
        let monitor = std::thread::spawn(move || {
            run_udp_heartbeat_monitor(monitor_heartbeat, heartbeat_port, monitor_stop, auth);
        });
        std::thread::sleep(Duration::from_millis(2));

        let sender = UdpSocket::bind(("127.0.0.1", 0)).expect("heartbeat sender should bind");
        sender
            .send_to(b"unauthenticated", ("127.0.0.1", heartbeat_port))
            .expect("invalid heartbeat should be deliverable");
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(heartbeat.load(Ordering::Acquire), 1);

        sender
            .send_to(
                &signed_heartbeat_frame(1, &TEST_HEARTBEAT_KEY),
                ("127.0.0.1", heartbeat_port),
            )
            .expect("authenticated heartbeat should be deliverable");
        let deadline = std::time::Instant::now() + Duration::from_millis(20);
        while heartbeat.load(Ordering::Acquire) == 1 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_micros(200));
        }
        assert_ne!(heartbeat.load(Ordering::Acquire), 1);

        stop.store(true, Ordering::Release);
        monitor
            .join()
            .expect("heartbeat monitor should stop cleanly");
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
            PyUdpTransportBridge::new("127.0.0.1", payload_port, 1, 16, "std", heartbeat_port, 20)
                .expect("bridge should initialize");
        bridge.heartbeat_auth = Some(test_heartbeat_auth());

        bridge.start().expect("bridge thread should start");

        // Give the watchdog a chance to sample heartbeat freshness at least once.
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(
            !bridge.stopped(),
            "bridge should be active before heartbeat timeout"
        );

        // With no heartbeat traffic and 20 ms timeout, expect stop to be asserted
        // within a bounded host-scheduler tolerance.
        let stop_wait_start = std::time::Instant::now();
        let stop_deadline = stop_wait_start + std::time::Duration::from_millis(60);
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
            detection_time < std::time::Duration::from_millis(60),
            "heartbeat timeout should drive stop inside deterministic window, got {detection_time:?}"
        );
    }
}
