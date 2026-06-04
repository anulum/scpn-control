// SPDX-License-Identifier: AGPL-3.0-or-later
// ──────────────────────────────────────────────────────────────────────
// SCPN Control — Transport benchmark
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// ──────────────────────────────────────────────────────────────────────

use std::io;
#[cfg(all(feature = "io-uring", target_os = "linux"))]
use std::net::SocketAddr;
use std::net::UdpSocket;
use std::time::{Duration, Instant};

#[cfg(all(feature = "io-uring", target_os = "linux"))]
use io_uring::{opcode, types, IoUring};

#[cfg(all(feature = "io-uring", target_os = "linux"))]
const IO_URING_QUEUE_DEPTH: u32 = 128;

#[repr(C)]
struct TransportFrameHeader {
    magic: u32,
    version: u16,
    payload_bytes: u16,
    sequence: u64,
    timestamp_ns: u64,
}

#[repr(C)]
struct TransportSnapshotFrame {
    header: TransportFrameHeader,
    status: u32,
    reserved0: u32,
    r_error: f64,
    z_error: f64,
    r_command: f64,
    z_command: f64,
    acados_time_ns: u64,
    snn_time_ns: u64,
    reserve1: u64,
    reserve2: u64,
}

const DEFAULT_ENDPOINT: &str = "127.0.0.1:5555";
const DEFAULT_BURST: usize = 10_000;
const PAYLOAD_SIZE_BYTES: usize = std::mem::size_of::<TransportSnapshotFrame>();

fn usage() -> String {
    [
        "transport_bench [backend] [endpoint] [burst] [payload_count]",
        "  backend: std | io-uring (default: std)",
        "  endpoint: host:port target (default: 127.0.0.1:5555)",
        "  burst: number of payloads to transmit (default: 10000)",
    ]
    .join("\n")
}

fn parse_args() -> Option<(String, String, usize)> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        println!("{}", usage());
        return None;
    }

    let mut index = 0;
    let backend = args
        .get(index)
        .map(|it| it.to_ascii_lowercase())
        .filter(|it| !it.is_empty())
        .unwrap_or_else(|| "std".to_string());
    if args.get(index).is_some() {
        index += 1;
    }

    let endpoint = args
        .get(index)
        .cloned()
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    if args.get(index).is_some() {
        index += 1;
    }

    let burst = args
        .get(index)
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BURST);

    Some((backend, endpoint, burst))
}

fn build_frame() -> [u8; PAYLOAD_SIZE_BYTES] {
    let frame = TransportSnapshotFrame {
        header: TransportFrameHeader {
            magic: 0x534e_5054,
            version: 1,
            payload_bytes: PAYLOAD_SIZE_BYTES as u16,
            sequence: 0,
            timestamp_ns: 0,
        },
        status: 0,
        reserved0: 0,
        r_error: 1.0,
        z_error: 2.0,
        r_command: 3.0,
        z_command: 4.0,
        acados_time_ns: 123,
        snn_time_ns: 456,
        reserve1: 0,
        reserve2: 0,
    };

    let mut bytes = [0u8; PAYLOAD_SIZE_BYTES];
    let source = unsafe {
        std::slice::from_raw_parts((frame_addr(&frame)).cast::<u8>(), PAYLOAD_SIZE_BYTES)
    };
    bytes.copy_from_slice(source);
    bytes
}

fn frame_addr<T>(frame: &T) -> *const T {
    frame as *const T
}

fn run_std_udp(endpoint: &str, burst: usize) -> io::Result<Duration> {
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    socket.connect(endpoint)?;
    let frame = build_frame();

    let start = Instant::now();
    for _ in 0..burst {
        socket.send(&frame)?;
    }
    Ok(start.elapsed())
}

#[cfg(all(feature = "io-uring", target_os = "linux"))]
fn run_uring_udp(endpoint: &str, burst: usize) -> io::Result<Duration> {
    use std::os::unix::io::AsRawFd;

    let target: SocketAddr = endpoint.parse().map_err(io::Error::other)?;
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    socket.connect(target)?;

    let fd = socket.as_raw_fd();
    let frame = build_frame();
    let frame_ptr = frame.as_ptr();
    let frame_len = u32::try_from(PAYLOAD_SIZE_BYTES)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "frame too large"))?;

    let mut ring = IoUring::new(IO_URING_QUEUE_DEPTH)?;
    let mut sent = 0usize;
    let start = Instant::now();

    while sent < burst {
        let batch = (IO_URING_QUEUE_DEPTH as usize).min(burst - sent);
        for idx in 0..batch {
            let mut sqe = opcode::Send::new(types::Fd(fd), frame_ptr, frame_len).build();
            sqe.set_user_data((sent + idx + 1) as u64);
            unsafe {
                if ring.submission().push(&sqe).is_err() {
                    return Err(io::Error::other("io_uring submission queue full"));
                }
            }
        }

        sent += batch;
        ring.submit_and_wait(1)?;
        while let Some(cqe) = ring.completion().next() {
            if cqe.result() < 0 {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("io_uring completion error: {}", cqe.result()),
                ));
            }
        }
    }

    Ok(start.elapsed())
}

#[cfg(not(all(feature = "io-uring", target_os = "linux")))]
fn run_uring_udp(_: &str, _: usize) -> io::Result<Duration> {
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "io_uring backend unavailable in this build",
    ))
}

fn main() {
    let (backend, endpoint, burst) = match parse_args() {
        Some(parts) => parts,
        None => return,
    };

    match backend.as_str() {
        "std" | "udp" => match run_std_udp(&endpoint, burst) {
            Ok(elapsed) => {
                let mean_per = elapsed / burst as u32;
                println!(
                    "backend=std burst={burst} payload={endpoint:?} duration_us={} mean_us={}",
                    elapsed.as_micros(),
                    mean_per.as_micros(),
                );
            }
            Err(error) => {
                eprintln!("std benchmark failed: {error}");
            }
        },
        "io-uring" | "io_uring" | "ioring" => match run_uring_udp(&endpoint, burst) {
            Ok(elapsed) => {
                let mean_per = elapsed / burst as u32;
                println!(
                    "backend=io-uring burst={burst} payload={endpoint:?} duration_us={} mean_us={}",
                    elapsed.as_micros(),
                    mean_per.as_micros(),
                );
            }
            Err(error) => {
                eprintln!("io-uring benchmark failed: {error}");
            }
        },
        _ => {
            eprintln!("unknown backend '{}'; valid: std, udp, io-uring", backend);
        }
    }
}
