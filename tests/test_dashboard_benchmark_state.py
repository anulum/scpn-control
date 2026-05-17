# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard benchmark state tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Tests for Streamlit-independent dashboard benchmark helpers."""

from __future__ import annotations

import pytest

from dashboard.benchmark_state import BenchmarkTiming, benchmark_controller_latency


class StepClock:
    def __init__(self, step_s: float = 1.0e-4) -> None:
        self._now = 0.0
        self._step_s = step_s

    def __call__(self) -> float:
        current = self._now
        self._now += self._step_s
        return current


def test_benchmark_controller_latency_returns_positive_microsecond_metrics() -> None:
    timing = benchmark_controller_latency(iterations=64, seed=19, clock=StepClock())

    assert isinstance(timing, BenchmarkTiming)
    assert timing.iterations == 64
    assert timing.pid_us_per_step > 0.0
    assert timing.snn_us_per_step > 0.0
    assert timing.ratio > 0.0


def test_benchmark_controller_latency_is_deterministic_with_injected_clock() -> None:
    first = benchmark_controller_latency(iterations=32, seed=5, clock=StepClock())
    second = benchmark_controller_latency(iterations=32, seed=5, clock=StepClock())

    assert first == second


@pytest.mark.parametrize("iterations", [0, 1, True])
def test_benchmark_controller_latency_rejects_invalid_iterations(iterations: int) -> None:
    with pytest.raises(ValueError, match="iterations"):
        benchmark_controller_latency(iterations=iterations)


def test_benchmark_controller_latency_rejects_invalid_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        benchmark_controller_latency(iterations=32, seed=True)
