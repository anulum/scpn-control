# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fault-Tolerance Benchmark

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path
from typing import Dict

from scpn_control.core._rust_compat import RustUdpTransportBridge


def _now_ns() -> int:
    return time.perf_counter_ns()


def _run_heartbeat_sender(stop_flag: threading.Event, host: str, port: int, interval_s: float) -> None:
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = b"\x01\x02\x03\x04"
    try:
        while not stop_flag.is_set():
            try:
                tx.sendto(payload, (host, port))
            except OSError:
                pass
            stop_flag.wait(interval_s)
    finally:
        tx.close()


def _safe_publish_once(bridge: RustUdpTransportBridge) -> bool:
    try:
        return bool(
            bridge.publish(
                r_error=0.1,
                z_error=0.2,
                r_command=0.3,
                z_command=0.4,
                acados_time_ns=1,
                snn_time_ns=2,
                status=0,
            )
        )
    except Exception:
        return False


def _measure_fault_reaction(
    heartbeat_port: int,
    heartbeat_timeout_ms: int,
    payload_port: int,
    run_seconds: float,
    monitor_interval_ms: int,
    backend: str,
) -> Dict[str, object]:
    bridge = RustUdpTransportBridge(
        endpoint="127.0.0.1",
        port=payload_port,
        max_queue=128,
        backend=backend,
        heartbeat_port=heartbeat_port,
        heartbeat_timeout_ms=heartbeat_timeout_ms,
    )

    bridge.start()

    stop_sender = threading.Event()
    sender = threading.Thread(
        target=_run_heartbeat_sender,
        args=(stop_sender, "127.0.0.1", heartbeat_port, 0.001),
        daemon=True,
    )
    sender.start()

    healthy_until_ns = 0
    end_healthy_ns = _now_ns() + int(run_seconds * 1_000_000_000)

    while _now_ns() < end_healthy_ns:
        healthy_until_ns = _now_ns()
        _safe_publish_once(bridge)
        time.sleep(monitor_interval_ms / 1_000.0)

    # Fault injection point: stop heartbeat stream.
    fault_injected_ns = _now_ns()
    stop_sender.set()
    sender.join(timeout=1.0)

    timeout_ns = heartbeat_timeout_ms * 1_000_000
    deadline_ns = fault_injected_ns + max(timeout_ns * 4, 10_000_000)

    detected_ns = None
    attempts = 0
    while _now_ns() < deadline_ns:
        attempts += 1
        if bridge.heartbeat_expired():
            detected_ns = _now_ns()
            break
        _safe_publish_once(bridge)
        time.sleep(monitor_interval_ms / 1_000.0)

    bridge.stop()

    if detected_ns is None:
        latency_ns = None
    else:
        latency_ns = detected_ns - fault_injected_ns

    return {
        "backend": backend,
        "heartbeat_port": heartbeat_port,
        "heartbeat_timeout_ms": heartbeat_timeout_ms,
        "payload_port": payload_port,
        "healthy_duration_s": run_seconds,
        "heartbeats_sent": int(run_seconds / 0.001),
        "publish_attempts_after_loss": attempts,
        "fault_injected_ns": fault_injected_ns,
        "detected_ns": detected_ns,
        "healthy_until_ns": healthy_until_ns,
        "fault_to_detection_ns": latency_ns,
        "fault_to_detection_ms": None if latency_ns is None else latency_ns / 1_000_000,
        "passed": latency_ns is not None and latency_ns <= max(timeout_ns * 2, 5_000_000),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Heartbeat severance fault injection benchmark",
    )
    parser.add_argument(
        "--backend",
        default="std",
        choices=["std", "io-uring", "udp", "io_uring"],
    )
    parser.add_argument("--heartbeat-port", type=int, default=55887)
    parser.add_argument("--payload-port", type=int, default=55888)
    parser.add_argument("--heartbeat-timeout-ms", type=int, default=3)
    parser.add_argument("--run-seconds", type=float, default=0.5)
    parser.add_argument("--monitor-interval-ms", type=int, default=1)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("validation/reports/udp_fault_tolerance_benchmark.jsonl"),
    )

    args = parser.parse_args()

    backend = args.backend
    if backend == "udp":
        backend = "std"
    elif backend in {"io_uring", "io-uring"}:
        backend = "io-uring"

    result = _measure_fault_reaction(
        heartbeat_port=args.heartbeat_port,
        heartbeat_timeout_ms=args.heartbeat_timeout_ms,
        payload_port=args.payload_port,
        run_seconds=args.run_seconds,
        monitor_interval_ms=max(args.monitor_interval_ms, 1),
        backend=backend,
    )

    result["command"] = {
        "backend": backend,
        "heartbeat_timeout_ms": args.heartbeat_timeout_ms,
        "run_seconds": args.run_seconds,
        "monitor_interval_ms": args.monitor_interval_ms,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({"result": result}, separators=(",", ":")))
        fp.write("\n")

    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
