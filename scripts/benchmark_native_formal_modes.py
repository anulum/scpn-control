# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native Formal Mode Benchmark

"""Benchmark native formal-verification modes for the fused Rust loop."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scpn_control.core.rust_engine import NeuroCyberneticEngine  # noqa: E402

LOOP_DEADLINE_US = 100.0


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
        self._thread = threading.Thread(target=self._run, name=f"native-formal-udp-sink-{self._port}")
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


def _parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _parse_csv_strings(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark native formal verification modes.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tick-interval-s", type=float, default=0.0)
    parser.add_argument("--strides", type=_parse_csv_ints, default=[1, 5, 20, 30])
    parser.add_argument(
        "--formal-modes",
        type=_parse_csv_strings,
        default=["disabled", "async_drop", "sync_stride", "aot_certificate"],
    )
    parser.add_argument("--pacing-modes", type=_parse_csv_strings, default=["sleep"])
    parser.add_argument("--transports", default="std,io-uring")
    parser.add_argument("--transport-endpoint", default="127.0.0.1")
    parser.add_argument("--transport-port-base", type=int, default=57300)
    parser.add_argument("--transport-max-queue", type=int, default=128)
    parser.add_argument("--max-publish-failures", type=int, default=1_000_000)
    parser.add_argument("--core-snn", type=int, default=1)
    parser.add_argument("--core-z3", type=int, default=2)
    parser.add_argument("--core-net", type=int, default=3)
    parser.add_argument("--core-hb", type=int, default=3)
    parser.add_argument(
        "--json-out",
        default=str(REPO_ROOT / "validation" / "reports" / "native_formal_modes.json"),
    )
    parser.add_argument(
        "--markdown-out",
        default=str(REPO_ROOT / "validation" / "reports" / "native_formal_modes.md"),
    )
    return parser.parse_args()


def _shell(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _git_dirty() -> bool:
    unstaged = subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode != 0
    return bool(unstaged or staged)


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _formal(summary: dict[str, Any]) -> dict[str, Any]:
    native = summary.get("native")
    if isinstance(native, dict):
        formal = native.get("formal_verification")
        if isinstance(formal, dict):
            return formal
    return {}


def _run_case(
    args: argparse.Namespace,
    *,
    transport: str,
    formal_mode: str,
    pacing_mode: str,
    stride: int,
    repeat: int,
    port: int,
) -> dict[str, Any]:
    engine = NeuroCyberneticEngine(n_neurons=64, seed=7)
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
        backend=transport,
        heartbeat_port=0,
        heartbeat_timeout_ms=3,
    )
    engine.configure_native_formal_verification(
        enabled=formal_mode != "disabled",
        mode="async_drop" if formal_mode == "disabled" else formal_mode,
        max_marking=100,
        max_depth=4,
        dispatch_interval_steps=stride,
        channel_capacity=2,
    )

    started = time.perf_counter()
    with _UdpSink(args.transport_endpoint, port) as sink:
        summary = engine.execute_hardware_loop(
            steps=args.steps,
            tick_interval_s=args.tick_interval_s,
            pacing_mode=pacing_mode,
            max_publish_failures=args.max_publish_failures,
            execution_backend="native",
            core_snn=args.core_snn,
            core_z3=args.core_z3,
            core_net=args.core_net,
            core_hb=args.core_hb,
        )
    wall_s = time.perf_counter() - started
    summary["wall_s"] = wall_s
    summary["effective_step_us"] = wall_s * 1_000_000.0 / max(1, int(summary.get("steps", 0)))
    summary["udp_sink_packets"] = sink.packets
    summary["formal_mode_requested"] = formal_mode
    summary["pacing_mode_requested"] = pacing_mode
    summary["formal_stride_requested"] = stride
    summary["transport_requested"] = transport
    summary["repeat"] = repeat
    return summary


def _quantile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100.0)
    return ordered[int(index)]


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    def stats(key: str) -> dict[str, float]:
        data = values(key)
        return {
            "min": min(data),
            "p50": _quantile(data, 50),
            "p95": _quantile(data, 95),
            "p99": _quantile(data, 99),
            "max": max(data),
            "mean": statistics.fmean(data),
        }

    cycle = stats("avg_cycle_us")
    return {
        "runs": len(rows),
        "avg_cycle_us": cycle,
        "effective_step_us": stats("effective_step_us"),
        "formal_generated_total": sum(int(_formal(row).get("generated", 0)) for row in rows),
        "formal_submitted_total": sum(int(_formal(row).get("submitted", 0)) for row in rows),
        "formal_checked_total": sum(int(_formal(row).get("checked", 0)) for row in rows),
        "formal_dropped_total": sum(int(_formal(row).get("dropped", 0)) for row in rows),
        "formal_failures_total": sum(int(_formal(row).get("failures", 0)) for row in rows),
        "sync_wait_count_total": sum(int(_formal(row).get("sync_wait_count", 0)) for row in rows),
        "sync_wait_p99_ns_max": max((int(_formal(row).get("sync_wait_p99_ns", 0)) for row in rows), default=0),
        "drops_total": sum(int(row.get("dropped", 0)) for row in rows),
        "publish_failures_total": sum(int(row.get("publish_failures", 0)) for row in rows),
        "udp_sink_packets_total": sum(int(row.get("udp_sink_packets", 0)) for row in rows),
        "safety_headroom_pct_p99_cycle": max(0.0, 100.0 * (LOOP_DEADLINE_US - cycle["p99"]) / LOOP_DEADLINE_US),
    }


def _case_names(
    transports: Iterable[str],
    strides: Iterable[int],
    formal_modes: Iterable[str],
    pacing_modes: Iterable[str],
) -> list[tuple[str, str, str, int]]:
    cases: list[tuple[str, str, str, int]] = []
    for transport in transports:
        for pacing_mode in pacing_modes:
            if "disabled" in formal_modes:
                cases.append((transport, pacing_mode, "disabled", 30))
        for pacing_mode in pacing_modes:
            for mode in formal_modes:
                if mode == "disabled":
                    continue
                for stride in strides:
                    cases.append((transport, pacing_mode, mode, int(stride)))
    return cases


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Native formal verification mode benchmark",
        "",
        f"Generated: {payload['generated_at']}",
        f"Commit: `{payload['commit_short']}`",
        f"Workspace dirty: `{payload['workspace_dirty']}`",
        "",
        payload["classification"],
        "",
        "| Case | Runs | p50 cycle us | p99 cycle us | p99 headroom % | Generated | Submitted | Checked | Dropped | Failures | Sync waits | Max sync p99 ns |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in payload["summaries"].items():
        lines.append(
            "| {name} | {runs} | {p50:.6f} | {p99:.6f} | {headroom:.3f} | "
            "{generated} | {submitted} | {checked} | {dropped} | {failures} | {waits} | {wait_p99} |".format(
                name=name,
                runs=summary["runs"],
                p50=summary["avg_cycle_us"]["p50"],
                p99=summary["avg_cycle_us"]["p99"],
                headroom=summary["safety_headroom_pct_p99_cycle"],
                generated=summary["formal_generated_total"],
                submitted=summary["formal_submitted_total"],
                checked=summary["formal_checked_total"],
                dropped=summary["formal_dropped_total"],
                failures=summary["formal_failures_total"],
                waits=summary["sync_wait_count_total"],
                wait_p99=summary["sync_wait_p99_ns_max"],
            )
        )
    lines.extend(["", "Limitations:"])
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.tick_interval_s < 0.0:
        raise SystemExit("--tick-interval-s must be >= 0")

    transports = [item.strip() for item in args.transports.split(",") if item.strip()]
    formal_modes = [item.strip().lower().replace("-", "_") for item in args.formal_modes]
    pacing_modes = [item.strip().lower().replace("-", "_") for item in args.pacing_modes]
    invalid_formal_modes = sorted(set(formal_modes) - {"disabled", "async_drop", "sync_stride", "aot_certificate"})
    invalid_pacing_modes = sorted(set(pacing_modes) - {"sleep", "spin"})
    if invalid_formal_modes:
        raise SystemExit(f"unsupported formal modes: {', '.join(invalid_formal_modes)}")
    if invalid_pacing_modes:
        raise SystemExit(f"unsupported pacing modes: {', '.join(invalid_pacing_modes)}")
    runs: list[dict[str, Any]] = []
    port = args.transport_port_base
    for transport, pacing_mode, mode, stride in _case_names(transports, args.strides, formal_modes, pacing_modes):
        for repeat in range(args.repeats):
            runs.append(
                _run_case(
                    args,
                    transport=transport,
                    formal_mode=mode,
                    pacing_mode=pacing_mode,
                    stride=stride,
                    repeat=repeat,
                    port=port,
                )
            )
            port += 1

    summaries = {
        f"{transport}:{pacing_mode}:{mode}:stride_{stride}": _summarise(
            [
                row
                for row in runs
                if row["transport_requested"] == transport
                and row["pacing_mode_requested"] == pacing_mode
                and row["formal_mode_requested"] == mode
                and int(row["formal_stride_requested"]) == stride
            ]
        )
        for transport, pacing_mode, mode, stride in _case_names(transports, args.strides, formal_modes, pacing_modes)
    }
    payload = {
        "schema": "scpn-control.native_formal_modes.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "classification": "native formal mode benchmark; isolation depends on caller taskset/governor setup",
        "commit": _shell(["git", "rev-parse", "HEAD"]),
        "commit_short": _shell(["git", "rev-parse", "--short", "HEAD"]),
        "workspace_dirty": _git_dirty(),
        "execution": {
            "steps": args.steps,
            "repeats": args.repeats,
            "tick_interval_s": args.tick_interval_s,
            "strides": list(args.strides),
            "formal_modes": formal_modes,
            "pacing_modes": pacing_modes,
            "transports": transports,
            "core_snn": args.core_snn,
            "core_z3": args.core_z3,
            "core_net": args.core_net,
            "core_hb": args.core_hb,
            "max_publish_failures": args.max_publish_failures,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "loadavg": _read_text("/proc/loadavg"),
            "isolated_cpus": _read_text("/sys/devices/system/cpu/isolated"),
            "nohz_full": _read_text("/sys/devices/system/cpu/nohz_full"),
        },
        "limitations": [
            "p50/p95/p99 are across repeated campaign summaries, not per-tick histograms.",
            "workspace_dirty means the benchmark includes uncommitted local changes on top of the reported commit.",
            "async_drop deliberately drops saturated snapshots and is not strict proof coverage.",
            "sync_stride blocks on designated stride steps and exposes sync wait telemetry.",
            "aot_certificate is a compiled sufficient certificate monitor; it is not a live SMT solver.",
            "spin pacing busy-waits on a native core and should only be used for short timing experiments.",
        ],
        "summaries": summaries,
        "runs": runs,
    }

    json_path = Path(args.json_out)
    markdown_path = Path(args.markdown_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
