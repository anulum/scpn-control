# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native Handoff Benchmark

"""Compare Python-orchestrated and Rust-native control-loop execution."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scpn_control.core.rust_engine import NeuroCyberneticEngine  # noqa: E402


class _UdpSink:
    """Consume local UDP frames so connected loopback sends do not fail closed."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = int(port)
        self._stop = threading.Event()
        self._packets = 0
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    @property
    def packets(self) -> int:
        return int(self._packets)

    def __enter__(self) -> "_UdpSink":
        bind_host = "0.0.0.0" if self._host.startswith("239.") else self._host
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, self._port))
        sock.settimeout(0.05)
        self._socket = sock
        self._thread = threading.Thread(target=self._run, name=f"native-handoff-udp-sink-{self._port}")
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        sock = self._socket
        if sock is None:
            return
        while not self._stop.is_set():
            try:
                sock.recvfrom(4096)
                self._packets += 1
            except TimeoutError:
                continue
            except OSError:
                return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python-orchestrated and Rust-native control execution.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--tick-interval-s", type=float, default=0.0001)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--transport-backend", default="std", choices=("std", "io-uring"))
    parser.add_argument("--transport-endpoint", default="127.0.0.1")
    parser.add_argument("--transport-port-base", type=int, default=55900)
    parser.add_argument("--transport-max-queue", type=int, default=128)
    parser.add_argument(
        "--json-out",
        default=str(REPO_ROOT / "validation" / "reports" / "native_handoff_comparison.json"),
    )
    parser.add_argument(
        "--markdown-out",
        default=str(REPO_ROOT / "validation" / "reports" / "native_handoff_comparison.md"),
    )
    return parser.parse_args()


def _run_campaign(args: argparse.Namespace, *, backend: str, port: int) -> dict[str, Any]:
    engine = NeuroCyberneticEngine(n_neurons=args.neurons, seed=args.seed)
    engine.configure_acados_targets(
        {
            "target_r": 6.2,
            "target_z": 0.0,
            "beta_n_limit": 2.5,
            "c_gB": 0.0424,
        }
    )
    engine.configure_transport(
        endpoint=args.transport_endpoint,
        port=port,
        ttl=1,
        max_queue=args.transport_max_queue,
        backend=args.transport_backend,
        heartbeat_port=0,
        heartbeat_timeout_ms=3,
    )

    started = time.perf_counter()
    with _UdpSink(args.transport_endpoint, port) as sink:
        summary = engine.execute_hardware_loop(
            steps=args.steps,
            tick_interval_s=args.tick_interval_s,
            max_publish_failures=args.transport_max_queue,
            execution_backend=backend,
        )
    wall_s = time.perf_counter() - started
    summary["wall_s"] = wall_s
    summary["requested_execution_backend"] = backend
    summary["effective_step_us"] = wall_s * 1_000_000.0 / max(1, int(summary.get("steps", 0)))
    summary["udp_sink_packets"] = sink.packets
    return summary


def _metric(summary: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    value: Any = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "status": summary.get("status"),
        "mode": _metric(summary, ("execution", "mode")),
        "steps": summary.get("steps"),
        "wall_s": summary.get("wall_s"),
        "effective_step_us": summary.get("effective_step_us"),
        "avg_cycle_us": summary.get("avg_cycle_us"),
        "dropped": summary.get("dropped"),
        "publish_failures": summary.get("publish_failures"),
        "transport_backend": _metric(summary, ("transport", "backend")),
        "native_last_cycle_ns": _metric(summary, ("native", "last_cycle_ns")),
        "native_total_cycle_ns": _metric(summary, ("native", "total_cycle_ns")),
        "native_last_acados_time_ns": _metric(summary, ("native", "last_acados_time_ns")),
        "native_last_snn_time_ns": _metric(summary, ("native", "last_snn_time_ns")),
    }


def _speedup(python_summary: dict[str, Any], native_summary: dict[str, Any]) -> float:
    python_step_us = float(python_summary.get("effective_step_us", 0.0))
    native_step_us = float(native_summary.get("effective_step_us", 0.0))
    if native_step_us <= 0.0:
        return 0.0
    return python_step_us / native_step_us


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# Native handoff comparison",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "| Backend | Status | Mode | Steps | Effective step us | Avg cycle us | Drops | Publish failures |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {status} | {mode} | {steps} | {effective_step_us:.3f} | "
            "{avg_cycle_us:.3f} | {dropped} | {publish_failures} |".format(
                name=row["name"],
                status=row["status"],
                mode=row["mode"],
                steps=row["steps"],
                effective_step_us=float(row["effective_step_us"]),
                avg_cycle_us=float(row["avg_cycle_us"]),
                dropped=row["dropped"],
                publish_failures=row["publish_failures"],
            )
        )
    lines.extend(
        [
            "",
            f"Native handoff wall-time speedup: {payload['comparison']['wall_time_speedup']:.3f}x.",
            "",
            "The Python row forces the Python orchestration path. The native row forces the PyO3 fused Rust loop.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.tick_interval_s < 0.0:
        raise SystemExit("--tick-interval-s must be >= 0")

    python_summary = _run_campaign(args, backend="python", port=args.transport_port_base)
    native_summary = _run_campaign(args, backend="native", port=args.transport_port_base + 1)

    payload = {
        "schema": "scpn-control.native_handoff_comparison.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "parameters": {
            "steps": args.steps,
            "tick_interval_s": args.tick_interval_s,
            "transport_backend": args.transport_backend,
            "transport_endpoint": args.transport_endpoint,
            "transport_port_base": args.transport_port_base,
            "transport_max_queue": args.transport_max_queue,
        },
        "rows": [
            _row("python", python_summary),
            _row("native", native_summary),
        ],
        "summaries": {
            "python": python_summary,
            "native": native_summary,
        },
        "comparison": {
            "wall_time_speedup": _speedup(python_summary, native_summary),
            "native_drops_zero": int(native_summary.get("dropped", -1)) == 0,
            "native_publish_failures_zero": int(native_summary.get("publish_failures", -1)) == 0,
            "python_udp_sink_packets": int(python_summary.get("udp_sink_packets", 0)),
            "native_udp_sink_packets": int(native_summary.get("udp_sink_packets", 0)),
        },
    }

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(Path(args.markdown_out), payload)
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))

    acceptable_status = {"completed", "normal"}
    if python_summary.get("status") not in acceptable_status:
        return 1
    if native_summary.get("status") not in acceptable_status:
        return 1
    if int(native_summary.get("dropped", 0)) != 0:
        return 1
    if int(native_summary.get("publish_failures", 0)) != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
