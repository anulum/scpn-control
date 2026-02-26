#!/usr/bin/env python3
"""Full pipeline benchmark: GS equilibrium → transport → controller.

Measures end-to-end latency for the complete SCPN control cycle.
Reports per-stage timing and total throughput.

Usage:
    python examples/full_pipeline_benchmark.py [--n-iter 1000]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from scpn_control.core.fusion_kernel import FusionKernel
from scpn_control.core.scaling_laws import ipb98y2_tau_e, load_ipb98y2_coefficients
from scpn_control.core.neural_transport import TransportInputs, critical_gradient_model
from scpn_control.control.h_infinity_controller import get_radial_robust_controller


def benchmark_scaling_law(n_iter: int, coefficients: dict) -> float:
    """Time n_iter evaluations of IPB98(y,2)."""
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ipb98y2_tau_e(
            Ip=rng.uniform(5, 20),
            BT=rng.uniform(2, 8),
            ne19=rng.uniform(3, 15),
            Ploss=rng.uniform(10, 150),
            R=rng.uniform(1, 8),
            kappa=rng.uniform(1.2, 2.0),
            epsilon=rng.uniform(0.2, 0.4),
            coefficients=coefficients,
        )
    return (time.perf_counter() - t0) / n_iter


def benchmark_transport(n_iter: int) -> float:
    """Time n_iter evaluations of the critical-gradient model."""
    rng = np.random.default_rng(1)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        inp = TransportInputs(
            grad_ti=rng.uniform(2, 12),
            grad_te=rng.uniform(2, 12),
            q=rng.uniform(1, 4),
        )
        critical_gradient_model(inp)
    return (time.perf_counter() - t0) / n_iter


def benchmark_hinf_step(n_iter: int) -> float:
    """Time n_iter H-infinity controller steps."""
    ctrl = get_radial_robust_controller(gamma_growth=100.0, damping=10.0)
    ctrl.reset()
    rng = np.random.default_rng(2)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        ctrl.step(rng.uniform(-0.1, 0.1), dt=0.001)
    return (time.perf_counter() - t0) / n_iter


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline benchmark")
    parser.add_argument("--n-iter", type=int, default=5000)
    args = parser.parse_args()
    n = args.n_iter

    coefficients = load_ipb98y2_coefficients()

    scaling_us = benchmark_scaling_law(n, coefficients) * 1e6
    transport_us = benchmark_transport(n) * 1e6
    hinf_us = benchmark_hinf_step(n) * 1e6
    total_us = scaling_us + transport_us + hinf_us

    results = {
        "n_iter": n,
        "ipb98y2_us_per_call": round(scaling_us, 2),
        "transport_us_per_call": round(transport_us, 2),
        "h_infinity_us_per_step": round(hinf_us, 2),
        "total_pipeline_us": round(total_us, 2),
        "throughput_khz": round(1e3 / total_us, 1) if total_us > 0 else 0,
    }

    print(json.dumps(results, indent=2))
    print(f"\nPipeline: {total_us:.1f} us/cycle = {results['throughput_khz']:.1f} kHz")


if __name__ == "__main__":
    main()
