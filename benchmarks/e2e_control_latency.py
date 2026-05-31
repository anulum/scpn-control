# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — E2E Control Latency
"""
Measures the full control cycle latency, not just a bare kernel step.

Pipeline per iteration:
  1. Construct plasma sensor state (simulated)
  2. Equilibrium: single SOR Picard step on GS equation
  3. Transport: update_transport_model + evolve_profiles (Crank-Nicolson)
  4. H-infinity controller .step()
  5. Actuator clamp + slew-rate limit

Usage:
    python benchmarks/e2e_control_latency.py
    python benchmarks/e2e_control_latency.py --json
    python benchmarks/e2e_control_latency.py --iterations 5000
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.validate_e2e_latency_evidence import build_e2e_latency_evidence_payload


def _write_minimal_config(path: Path) -> Path:
    """Write a small-grid GS config for benchmarking."""
    cfg = {
        "reactor_name": "Bench-16x16",
        "grid_resolution": [16, 16],
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0},
        ],
        "solver": {
            "max_iterations": 50,
            "convergence_threshold": 1e-4,
            "relaxation_factor": 0.15,
            "solver_method": "sor",
        },
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _percentile(sorted_arr: np.ndarray, p: float) -> float:
    idx = int(p / 100.0 * (len(sorted_arr) - 1))
    return float(sorted_arr[idx])


def bench_kernel_only(kernel: Any, n_warmup: int, n_iter: int) -> np.ndarray:
    """Measure a single SOR step (kernel-only, no transport/control)."""
    source = -kernel.cfg["physics"]["vacuum_permeability"] * kernel.RR * kernel.J_phi
    for _ in range(n_warmup):
        kernel._sor_step(kernel.Psi, source)

    times_ns = np.empty(n_iter, dtype=np.int64)
    for i in range(n_iter):
        t0 = time.perf_counter_ns()
        kernel._sor_step(kernel.Psi, source)
        times_ns[i] = time.perf_counter_ns() - t0
    return times_ns


def bench_e2e(transport: Any, hinf: Any, n_warmup: int, n_iter: int) -> np.ndarray:
    """Measure the full control cycle: sensor→equilibrium→transport→control→actuator."""
    rng = np.random.default_rng(42)
    dt_transport = 0.001  # 1 ms transport step
    dt_control = 1e-4  # 100 µs control sampling
    P_aux = 10.0  # MW
    u_max = 1e6  # A, actuator saturation
    slew_max = 1e8  # A/s
    u_prev = 0.0

    source = -transport.cfg["physics"]["vacuum_permeability"] * transport.RR * transport.J_phi

    for _ in range(n_warmup):
        _e2e_iteration(transport, hinf, rng, source, dt_transport, dt_control, P_aux, u_max, slew_max, u_prev)

    times_ns = np.empty(n_iter, dtype=np.int64)
    for i in range(n_iter):
        t0 = time.perf_counter_ns()
        u_prev = _e2e_iteration(transport, hinf, rng, source, dt_transport, dt_control, P_aux, u_max, slew_max, u_prev)
        times_ns[i] = time.perf_counter_ns() - t0
    return times_ns


def _e2e_iteration(
    transport: Any,
    hinf: Any,
    rng: np.random.Generator,
    source: np.ndarray,
    dt_transport: float,
    dt_control: float,
    P_aux: float,
    u_max: float,
    slew_max: float,
    u_prev: float,
) -> float:
    """Single end-to-end control iteration."""
    # 1. Sensor read (simulated plasma state perturbation)
    z_displacement = 0.01 * rng.standard_normal()

    # 2. Equilibrium: one SOR step on current GS source
    transport._sor_step(transport.Psi, source)

    # 3. Transport: update coefficients + advance profiles
    transport.update_transport_model(P_aux)
    transport.evolve_profiles(dt_transport, P_aux)

    # 4. Controller: H-inf step on vertical displacement error
    u_raw = hinf.step(z_displacement, dt_control)

    # 5. Actuator limits: absolute clamp + slew rate
    delta_max = slew_max * dt_control
    u_clamped = np.clip(u_raw, u_prev - delta_max, u_prev + delta_max)
    u_clamped = np.clip(u_clamped, -u_max, u_max)
    return float(u_clamped)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E control latency benchmark")
    parser.add_argument("--iterations", type=int, default=1000, help="Measurement iterations (default: 1000)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations (default: 50)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    parser.add_argument("--output-json", type=Path, default=None, help="Persist benchmark JSON evidence")
    parser.add_argument(
        "--target-hardware-id",
        default="local-host-unqualified",
        help="Operator-supplied hardware identifier for published evidence",
    )
    parser.add_argument(
        "--target-hardware-class",
        default="unspecified-local",
        help="Hardware class, for example raspberry-pi, jetson, or industrial-pc",
    )
    parser.add_argument("--rt-kernel", default="unknown", help="Real-time kernel or scheduler evidence label")
    args = parser.parse_args()

    # Lazy imports so --help is fast
    from scpn_control.control.h_infinity_controller import get_radial_robust_controller
    from scpn_control.core.integrated_transport_solver import TransportSolver

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = _write_minimal_config(Path(tmpdir) / "bench.json")
        transport = TransportSolver(cfg_path)
        transport.set_neoclassical(R0=4.0, a=2.0, B0=5.3)

    hinf = get_radial_robust_controller(gamma_growth=100.0, damping=10.0)

    # Kernel-only benchmark
    kernel_ns = bench_kernel_only(transport, args.warmup, args.iterations)
    kernel_ns.sort()
    kernel_us = kernel_ns / 1000.0

    # E2E benchmark
    hinf.reset()
    e2e_ns = bench_e2e(transport, hinf, args.warmup, args.iterations)
    e2e_ns.sort()
    e2e_us = e2e_ns / 1000.0

    results = build_e2e_latency_evidence_payload(
        {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "grid": "16x16",
            "target_hardware": {
                "id": args.target_hardware_id,
                "class": args.target_hardware_class,
                "machine": platform.machine(),
                "processor": platform.processor(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rt_kernel": args.rt_kernel,
            },
            "kernel_only_us": {
                "p50": round(_percentile(kernel_us, 50), 1),
                "p95": round(_percentile(kernel_us, 95), 1),
                "p99": round(_percentile(kernel_us, 99), 1),
            },
            "e2e_us": {
                "p50": round(_percentile(e2e_us, 50), 1),
                "p95": round(_percentile(e2e_us, 95), 1),
                "p99": round(_percentile(e2e_us, 99), 1),
            },
            "e2e_overhead_factor": round(_percentile(e2e_us, 50) / max(_percentile(kernel_us, 50), 0.1), 1),
        }
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print()
    print("SCPN Control — End-to-End Control Latency Benchmark")
    print("=" * 55)
    print(f"  Iterations: {args.iterations}   Warmup: {args.warmup}   Grid: 16x16")
    print()
    print(f"  {'Stage':<30} {'P50 µs':>8} {'P95 µs':>8} {'P99 µs':>8}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8}")
    for label, key in [("Kernel only (1 SOR step)", "kernel_only_us"), ("Full E2E control cycle", "e2e_us")]:
        d = results[key]
        print(f"  {label:<30} {d['p50']:>8.1f} {d['p95']:>8.1f} {d['p99']:>8.1f}")
    print()
    print(f"  E2E / Kernel overhead: {results['e2e_overhead_factor']}x")
    print()
    print("  E2E includes: sensor read + SOR step + transport model")
    print("  update + Crank-Nicolson profile advance + H-inf controller")
    print("  step + actuator clamp/slew.")
    print()


if __name__ == "__main__":
    main()
