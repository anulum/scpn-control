# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard trajectory tests
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Tests for Streamlit-independent dashboard trajectory preparation."""

from __future__ import annotations

import numpy as np
import pytest

from dashboard.trajectory import simulate_closed_loop_trajectory


def test_simulate_closed_loop_trajectory_returns_bounded_finite_series() -> None:
    trajectory = simulate_closed_loop_trajectory(steps=64, scenario="PID", seed=17)

    assert trajectory.states.shape == (64,)
    assert trajectory.errors.shape == (64,)
    assert np.all(np.isfinite(trajectory.states))
    assert np.all(np.isfinite(trajectory.errors))
    assert np.all((trajectory.states >= 0.0) & (trajectory.states <= 2.0))
    assert trajectory.target == pytest.approx(1.0)
    assert abs(trajectory.errors[-1]) < abs(trajectory.errors[0])


def test_simulate_closed_loop_trajectory_is_seed_deterministic() -> None:
    first = simulate_closed_loop_trajectory(steps=48, scenario="SNN", seed=3)
    second = simulate_closed_loop_trajectory(steps=48, scenario="SNN", seed=3)

    np.testing.assert_allclose(first.states, second.states)
    np.testing.assert_allclose(first.errors, second.errors)


@pytest.mark.parametrize("scenario", ["PID", "SNN", "Combined"])
def test_simulate_closed_loop_trajectory_supports_dashboard_scenarios(scenario: str) -> None:
    trajectory = simulate_closed_loop_trajectory(steps=32, scenario=scenario, seed=5)

    assert trajectory.scenario == scenario
    assert trajectory.states.shape == trajectory.errors.shape == (32,)


@pytest.mark.parametrize("steps", [0, 1, True])
def test_simulate_closed_loop_trajectory_rejects_invalid_steps(steps: int) -> None:
    with pytest.raises(ValueError, match="steps"):
        simulate_closed_loop_trajectory(steps=steps, scenario="PID")


def test_simulate_closed_loop_trajectory_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="scenario"):
        simulate_closed_loop_trajectory(steps=32, scenario="unknown")
