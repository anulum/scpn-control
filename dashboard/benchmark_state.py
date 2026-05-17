# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard benchmark state
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Streamlit-independent controller benchmark helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np


Clock = Callable[[], float]


@dataclass(frozen=True)
class BenchmarkTiming:
    """Per-step controller timing metrics for dashboard display."""

    iterations: int
    pid_us_per_step: float
    snn_us_per_step: float
    ratio: float


def _require_iterations(iterations: int) -> int:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 2:
        raise ValueError("iterations must be an integer >= 2.")
    return iterations


def benchmark_controller_latency(
    *,
    iterations: int,
    seed: int = 42,
    clock: Clock = time.perf_counter,
) -> BenchmarkTiming:
    """Measure dashboard PID and SNN proxy controller loop latency."""
    checked_iterations = _require_iterations(iterations)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    rng = np.random.default_rng(seed)

    start_s = clock()
    kp, ki, kd = 1.0, 0.1, 0.01
    integral, prev_error = 0.0, 0.0
    for _ in range(checked_iterations):
        error = float(rng.standard_normal())
        integral += error
        derivative = error - prev_error
        _ = kp * error + ki * integral + kd * derivative
        prev_error = error
    pid_us = (clock() - start_s) / checked_iterations * 1.0e6

    start_s = clock()
    v = np.zeros(50, dtype=np.float64)
    for _ in range(checked_iterations):
        error = float(rng.standard_normal())
        v = v * 0.9 + error * rng.standard_normal(50) * 0.1
        spikes = v > 1.0
        v[spikes] = 0.0
    snn_us = (clock() - start_s) / checked_iterations * 1.0e6

    safe_snn_us = max(float(snn_us), 1.0e-12)
    return BenchmarkTiming(
        iterations=checked_iterations,
        pid_us_per_step=float(pid_us),
        snn_us_per_step=float(snn_us),
        ratio=float(pid_us) / safe_snn_us,
    )


__all__ = ["BenchmarkTiming", "Clock", "benchmark_controller_latency"]
