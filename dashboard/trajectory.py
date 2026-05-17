# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard trajectory state
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Streamlit-independent trajectory preparation for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


SCENARIOS = ("PID", "SNN", "Combined")


@dataclass(frozen=True)
class TrajectorySeries:
    """Closed-loop trajectory arrays prepared for dashboard display."""

    scenario: str
    target: float
    states: NDArray[np.float64]
    errors: NDArray[np.float64]


def _scenario_gains(scenario: str) -> tuple[float, float]:
    if scenario == "PID":
        return 0.10, 0.010
    if scenario == "SNN":
        return 0.085, 0.007
    if scenario == "Combined":
        return 0.13, 0.006
    raise ValueError(f"scenario must be one of {SCENARIOS}.")


def simulate_closed_loop_trajectory(
    *,
    steps: int,
    scenario: str,
    seed: int = 42,
    target: float = 1.0,
    initial_state: float = 0.5,
) -> TrajectorySeries:
    """Simulate a bounded one-state dashboard control trajectory."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer >= 2.")
    if not np.isfinite(target) or not np.isfinite(initial_state):
        raise ValueError("target and initial_state must be finite.")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    proportional_gain, noise_scale = _scenario_gains(scenario)
    rng = np.random.default_rng(seed)
    state = float(np.clip(initial_state, 0.0, 2.0))
    states = np.empty(steps, dtype=np.float64)
    errors = np.empty(steps, dtype=np.float64)

    for index in range(steps):
        error = target - state
        action = proportional_gain * error + noise_scale * rng.standard_normal()
        state = float(np.clip(state + action, 0.0, 2.0))
        states[index] = state
        errors[index] = error

    return TrajectorySeries(
        scenario=scenario,
        target=float(target),
        states=states,
        errors=errors,
    )


__all__ = ["SCENARIOS", "TrajectorySeries", "simulate_closed_loop_trajectory"]
