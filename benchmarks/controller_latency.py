# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Per-controller step-latency benchmark

"""Measure isolated per-step latency for each control law.

This is the reproducible source for the control-loop latency table cited across
the documentation and the preprint (PID, SNN NumPy, SNN Rust, nonlinear MPC,
H-infinity). Each controller is advanced for ``--iterations`` real steps after a
warm-up; per-step wall time is sampled with ``perf_counter_ns`` and reduced to
P50/P95/P99/throughput. Unlike a single wall-clock average, the per-step
percentiles are stable and the JSON artefact records full provenance so the
numbers can be regenerated on a single CI environment.

Usage::

    python benchmarks/controller_latency.py
    python benchmarks/controller_latency.py --iterations 5000 --warmup 500 --json-out out.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scpn_control.control.h_infinity_controller import HInfinityController  # noqa: E402
from scpn_control.control.neuro_cybernetic_controller import SpikingControllerPool  # noqa: E402
from scpn_control.control.nmpc_controller import NMPCConfig, NonlinearMPC  # noqa: E402
from scpn_control.core._rust_compat import RustPIDController  # noqa: E402


def _tokamak_plant(x: NDArray[np.floating[Any]], u: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Six-state tokamak dynamics used by the nonlinear MPC.

    State ``x = [Ip, beta_N, q95, li, T_axis, n_bar]``; control
    ``u = [P_aux, I_p_ref, n_gas_puff]``. This is the discrete model the MPC is
    formulated against in the unit tests, reused here so the measured MPC latency
    reflects a representative linearisation cost.
    """
    x_next = x.copy()
    dt = 0.1
    x_next[0] += dt * (u[1] - x[0]) * 0.5
    x_next[1] += dt * (u[0] - x[1]) * 0.5
    x_next[2] = 15.0 / x_next[0] if x_next[0] > 0.1 else 150.0
    x_next[3] += dt * (1.0 - x[3]) * 0.1
    if x[5] > 0.1:
        x_next[4] += dt * (2.0 * u[0] / x[5] - x[4]) * 0.5
    x_next[5] += dt * (u[2] - 0.5 * x[5])
    return x_next.astype(np.float64, copy=False)


def _cpu_model() -> str:
    """Best-effort CPU model string for benchmark provenance."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _percentile(ordered: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 1]) of a pre-sorted list."""
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[idx])


def _measure(step: Callable[[int], object], *, iterations: int, warmup: int) -> dict[str, Any]:
    """Time ``step(i)`` over ``iterations`` samples after ``warmup`` discards."""
    for i in range(warmup):
        step(i)
    samples_us: list[float] = []
    for i in range(iterations):
        start = time.perf_counter_ns()
        step(warmup + i)
        samples_us.append((time.perf_counter_ns() - start) / 1000.0)
    ordered = sorted(samples_us)
    p50 = _percentile(ordered, 0.50)
    return {
        "n": len(ordered),
        "p50_us": p50,
        "p95_us": _percentile(ordered, 0.95),
        "p99_us": _percentile(ordered, 0.99),
        "mean_us": sum(ordered) / len(ordered) if ordered else 0.0,
        "min_us": ordered[0] if ordered else 0.0,
        "max_us": ordered[-1] if ordered else 0.0,
        "throughput_khz": (1.0e3 / p50) if p50 > 0.0 else 0.0,
    }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _rust_symbol_available(symbol: str) -> bool:
    try:
        import scpn_control_rs
    except ImportError:
        return False
    return hasattr(scpn_control_rs, symbol)


def _entry(name: str, backend: str, stats: dict[str, Any] | None, status: str, note: str = "") -> dict[str, Any]:
    return {"name": name, "backend": backend, "stats": stats, "status": status, "note": note}


def _pid_entries(iterations: int, warmup: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    pid_np = RustPIDController._PurePythonPID(1.0, 0.1, 0.05)
    entries.append(
        _entry(
            "PID",
            "numpy",
            _measure(lambda i: pid_np.step(math.sin(i * 0.01)), iterations=iterations, warmup=warmup),
            "measured",
        )
    )
    if _rust_symbol_available("PyPIDController"):
        pid_rs = RustPIDController(kp=1.0, ki=0.1, kd=0.05)
        entries.append(
            _entry(
                "PID",
                "rust",
                _measure(lambda i: pid_rs.step(math.sin(i * 0.01)), iterations=iterations, warmup=warmup),
                "measured",
            )
        )
    else:
        entries.append(_entry("PID", "rust", None, "unavailable: PyPIDController not built"))
    return entries


def _snn_entries(iterations: int, warmup: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    pool = SpikingControllerPool(
        n_neurons=64,
        gain=1.0,
        seed=7,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
    )
    entries.append(
        _entry(
            "SNN",
            "numpy",
            _measure(lambda i: pool.step(math.sin(i * 0.01)), iterations=iterations, warmup=warmup),
            "measured",
        )
    )
    if _rust_symbol_available("PySnnController"):
        from scpn_control.core._rust_compat import RustSnnController

        snn = RustSnnController(target_r=6.2, target_z=0.0)
        entries.append(
            _entry(
                "SNN",
                "rust",
                _measure(
                    lambda i: snn.step(6.2 + 0.01 * math.sin(i * 0.01), 0.01 * math.cos(i * 0.01)),
                    iterations=iterations,
                    warmup=warmup,
                ),
                "measured",
            )
        )
    else:
        entries.append(_entry("SNN", "rust", None, "unavailable: PySnnController not built"))
    return entries


def _mpc_entries(iterations: int, warmup: int) -> list[dict[str, Any]]:
    # The internal pure-Python QP is ~20 ms/tick; cap the sample count so the
    # millisecond-scale rows do not dominate wall time. nothing is left out: every
    # QP backend is measured where its dependency is installed and explicitly
    # marked unavailable otherwise.
    iterations = min(iterations, 500)
    warmup = min(warmup, 50)
    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 20.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = np.array([10.0, 1.0, 1.0])
    backend_dep = {"internal": None, "scipy": "scipy", "osqp": "osqp", "casadi": "casadi", "acados": "acados_template"}
    entries: list[dict[str, Any]] = []
    for qp, dep in backend_dep.items():
        if dep is not None and not _module_available(dep):
            entries.append(_entry("MPC (Np=10)", qp, None, f"unavailable: {dep} not installed"))
            continue
        mpc = NonlinearMPC(_tokamak_plant, NMPCConfig(horizon=10, max_sqp_iter=3, qp_backend=qp))

        def _step(i: int, _mpc: NonlinearMPC = mpc) -> object:
            # Real-time iteration (one linearisation + one QP per tick) is the
            # scheme that runs in the kHz control loop, not the full multi-SQP solve.
            return _mpc.step_rti(x0, x_ref + 0.01 * math.sin(i * 0.01), u_prev)

        entries.append(_entry("MPC (Np=10)", qp, _measure(_step, iterations=iterations, warmup=warmup), "measured"))
    return entries


def _h_infinity_entries(iterations: int, warmup: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    n = 3
    ctrl = HInfinityController(A=-0.5 * np.eye(n), B1=np.eye(n), B2=np.eye(n), C1=np.eye(n), C2=np.eye(n))
    entries.append(
        _entry(
            "H-infinity",
            "numpy",
            _measure(lambda i: ctrl.step(math.sin(i * 0.01), 1.0e-3), iterations=iterations, warmup=warmup),
            "measured",
            "general state-space",
        )
    )
    if _rust_symbol_available("PyHInfController"):
        from scpn_control.core._rust_compat import RustHInfController

        rust_ctrl = RustHInfController()
        entries.append(
            _entry(
                "H-infinity",
                "rust",
                _measure(lambda i: rust_ctrl.step(math.sin(i * 0.01), 1.0e-3), iterations=iterations, warmup=warmup),
                "measured",
                "2-state VDE",
            )
        )
    else:
        entries.append(_entry("H-infinity", "rust", None, "unavailable: PyHInfController not built"))
    return entries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-controller step-latency benchmark.")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument(
        "--json-out",
        default=str(REPO_ROOT / "validation" / "reports" / "controller_latency.json"),
    )
    parser.add_argument(
        "--markdown-out",
        default=str(REPO_ROOT / "validation" / "reports" / "controller_latency.md"),
    )
    return parser.parse_args()


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Control-loop latency (per controller)",
        "",
        f"Generated: {payload['generated_at']}",
        f"Iterations: {payload['parameters']['iterations']} (warm-up "
        f"{payload['parameters']['warmup']}), platform: {payload['platform']['platform']}",
        "",
        "| Controller | Backend | P50 (us) | P95 (us) | P99 (us) | Throughput (kHz) | Note |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for entry in payload["controllers"]:
        stats = entry["stats"]
        note = entry.get("note") or entry.get("status", "")
        if stats is None:
            lines.append(
                f"| {entry['name']} | {entry['backend']} | n/a | n/a | n/a | n/a | {entry.get('status', '')} |"
            )
            continue
        lines.append(
            f"| {entry['name']} | {entry['backend']} | {stats['p50_us']:.1f} | "
            f"{stats['p95_us']:.1f} | {stats['p99_us']:.1f} | {stats['throughput_khz']:.1f} | {note} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    controllers: list[dict[str, Any]] = [
        *_pid_entries(args.iterations, args.warmup),
        *_snn_entries(args.iterations, args.warmup),
        *_mpc_entries(args.iterations, args.warmup),
        *_h_infinity_entries(args.iterations, args.warmup),
    ]

    payload = {
        "schema": "scpn-control.controller_latency.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
        },
        "parameters": {"iterations": args.iterations, "warmup": args.warmup},
        "controllers": controllers,
    }

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(Path(args.markdown_out), payload)
    for entry in controllers:
        stats = entry["stats"]
        if stats is None:
            print(f"{entry['name']:14s} {entry['backend']:6s} unavailable")
        else:
            print(
                f"{entry['name']:14s} {entry['backend']:6s} "
                f"p50={stats['p50_us']:.2f}us p99={stats['p99_us']:.2f}us "
                f"throughput={stats['throughput_khz']:.1f}kHz"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
