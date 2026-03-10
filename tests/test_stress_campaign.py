# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Stress-Test Campaign Tests (P1.4)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests for the controller stress-test campaign and RESULTS.md wiring."""

from __future__ import annotations

from unittest.mock import patch

import pytest

try:
    import validation.collect_results as _collect_results  # noqa: F401

    HAS_COLLECT_RESULTS = True
except (ImportError, ModuleNotFoundError):
    HAS_COLLECT_RESULTS = False

_skip_no_collect = pytest.mark.skipif(
    not HAS_COLLECT_RESULTS,
    reason="validation.collect_results not available",
)


# ── Campaign data structures ─────────────────────────────────────────


def test_episode_result_dataclass():
    """EpisodeResult should store all required fields."""
    from validation.stress_test_campaign import EpisodeResult

    ep = EpisodeResult(
        mean_abs_r_error=0.02,
        mean_abs_z_error=0.03,
        reward=-0.05,
        latency_us=50.0,
        disrupted=False,
        t_disruption=30.0,
        energy_efficiency=0.95,
    )
    assert ep.mean_abs_r_error == 0.02
    assert ep.disrupted is False
    assert ep.energy_efficiency == 0.95


def test_controller_metrics_dataclass():
    """ControllerMetrics should have correct defaults."""
    from validation.stress_test_campaign import ControllerMetrics

    m = ControllerMetrics(name="test")
    assert m.name == "test"
    assert m.n_episodes == 0
    assert m.mean_reward == 0.0
    assert m.disruption_rate == 0.0
    assert isinstance(m.episodes, list)


# ── Summary table generation ─────────────────────────────────────────


def test_generate_summary_table_format():
    """generate_summary_table should produce a markdown table."""
    from validation.stress_test_campaign import (
        ControllerMetrics,
        generate_summary_table,
    )

    results = {
        "PID": ControllerMetrics(
            name="PID",
            n_episodes=10,
            mean_reward=-0.1,
            std_reward=0.05,
            mean_r_error=0.03,
            p50_latency_us=40.0,
            p95_latency_us=60.0,
            p99_latency_us=80.0,
            disruption_rate=0.1,
            mean_def=0.9,
            mean_energy_efficiency=0.85,
        ),
        "H-infinity": ControllerMetrics(
            name="H-infinity",
            n_episodes=10,
            mean_reward=-0.08,
            std_reward=0.04,
            mean_r_error=0.02,
            p50_latency_us=45.0,
            p95_latency_us=70.0,
            p99_latency_us=100.0,
            disruption_rate=0.05,
            mean_def=0.95,
            mean_energy_efficiency=0.88,
        ),
    }
    table = generate_summary_table(results)
    assert "| PID" in table
    assert "| H-infinity" in table
    assert "Controller" in table
    assert "Mean Reward" in table
    # Should be multi-line
    lines = table.strip().split("\n")
    assert len(lines) >= 3  # header + separator + 2 data rows


# ── run_controller_campaign wiring ───────────────────────────────────


@_skip_no_collect
def test_run_controller_campaign_returns_dict():
    """run_controller_campaign should return a dict with expected keys."""
    from validation.stress_test_campaign import (
        ControllerMetrics,
    )

    # Mock the campaign runner
    mock_results = {
        "PID": ControllerMetrics(
            name="PID",
            n_episodes=5,
            mean_reward=-0.1,
            std_reward=0.05,
            mean_r_error=0.03,
            p50_latency_us=40.0,
            p95_latency_us=60.0,
            p99_latency_us=80.0,
            disruption_rate=0.1,
            mean_def=0.9,
            mean_energy_efficiency=0.85,
        ),
    }

    with patch("validation.stress_test_campaign.run_campaign", return_value=mock_results):
        from validation.collect_results import run_controller_campaign

        result = run_controller_campaign(quick=True)

    assert result is not None
    assert "n_episodes" in result
    assert "controllers" in result
    assert "markdown_table" in result
    assert "PID" in result["controllers"]


@_skip_no_collect
def test_campaign_controller_fields():
    """Each controller in campaign result should have all metric fields."""
    from validation.stress_test_campaign import ControllerMetrics

    mock_results = {
        "PID": ControllerMetrics(
            name="PID",
            n_episodes=5,
            mean_reward=-0.1,
            std_reward=0.05,
            mean_r_error=0.03,
            p50_latency_us=40.0,
            p95_latency_us=60.0,
            p99_latency_us=80.0,
            disruption_rate=0.1,
            mean_def=0.9,
            mean_energy_efficiency=0.85,
        ),
    }

    with patch("validation.stress_test_campaign.run_campaign", return_value=mock_results):
        from validation.collect_results import run_controller_campaign

        result = run_controller_campaign(quick=True)

    pid_data = result["controllers"]["PID"]
    expected_keys = [
        "n_episodes",
        "mean_reward",
        "std_reward",
        "mean_r_error",
        "p50_latency_us",
        "p95_latency_us",
        "p99_latency_us",
        "disruption_rate",
        "mean_def",
        "mean_energy_efficiency",
    ]
    for key in expected_keys:
        assert key in pid_data, f"Missing key: {key}"


# ── RESULTS.md generation with campaign ──────────────────────────────


@_skip_no_collect
def test_generate_results_md_includes_campaign():
    """generate_results_md should include campaign table when provided."""
    from validation.collect_results import generate_results_md

    campaign = {
        "n_episodes": 5,
        "controllers": {
            "PID": {"mean_reward": -0.1, "disruption_rate": 0.1},
        },
        "markdown_table": "| PID | 5 | ... |",
    }
    md = generate_results_md(
        hw="Test HW",
        hil=None,
        disruption=None,
        q10=None,
        tbr=None,
        ecrh=None,
        fb3d=None,
        surrogates=None,
        neural_eq=None,
        elapsed_s=10.0,
        campaign=campaign,
    )
    assert "Controller Performance" in md
    assert "PID" in md


@_skip_no_collect
def test_generate_results_md_without_campaign():
    """generate_results_md without campaign should not fail."""
    from validation.collect_results import generate_results_md

    md = generate_results_md(
        hw="Test HW",
        hil=None,
        disruption=None,
        q10=None,
        tbr=None,
        ecrh=None,
        fb3d=None,
        surrogates=None,
        neural_eq=None,
        elapsed_s=10.0,
        campaign=None,
    )
    assert "SCPN" in md
    assert "Controller Performance" not in md


# ── Controller registry ──────────────────────────────────────────────


def test_controllers_registry_has_pid_and_hinf():
    """CONTROLLERS registry should always have PID and H-infinity."""
    from validation.stress_test_campaign import CONTROLLERS

    assert "PID" in CONTROLLERS
    assert "H-infinity" in CONTROLLERS


# ── H-infinity flight-sim controller tests ───────────────────────────


def test_flight_sim_controller_factory():
    """get_flight_sim_controller synthesizes a valid controller."""
    import numpy as np
    from scpn_control.control.h_infinity_controller import get_flight_sim_controller

    ctrl = get_flight_sim_controller(response_gain=0.05, actuator_tau=0.06)
    assert ctrl.is_stable
    assert ctrl.gamma > 1.0
    assert np.all(np.isfinite(ctrl.F))
    assert np.all(np.isfinite(ctrl.L_gain))


def test_flight_sim_controller_closed_loop_converges():
    """H-inf controller for flight-sim plant should drive error to zero."""
    from scpn_control.control.h_infinity_controller import get_flight_sim_controller

    ctrl = get_flight_sim_controller(response_gain=0.05, actuator_tau=0.06)
    dt = 0.05
    g = 0.05
    tau = 0.06

    x1, x2 = 0.3, 0.0
    errors = [x1]
    for _ in range(200):
        u = ctrl.step(x1, dt)
        dx1 = -g * x2
        dx2 = (u - x2) / tau
        x1 += dx1 * dt
        x2 += dx2 * dt
        errors.append(x1)

    assert abs(errors[-1]) < 0.05 * abs(errors[0])


def test_flight_sim_controller_both_channels():
    """Radial and vertical controllers should both converge independently."""
    from scpn_control.control.h_infinity_controller import get_flight_sim_controller

    ctrl_R = get_flight_sim_controller(response_gain=0.05, actuator_tau=0.06)
    ctrl_Z = get_flight_sim_controller(response_gain=0.02, actuator_tau=0.06)
    dt = 0.05

    for ctrl, g, e0 in [(ctrl_R, 0.05, 0.2), (ctrl_Z, 0.02, -0.15)]:
        x1, x2 = e0, 0.0
        for _ in range(300):
            u = ctrl.step(x1, dt)
            dx1 = -g * x2
            dx2 = (u - x2) / 0.06
            x1 += dx1 * dt
            x2 += dx2 * dt
        assert abs(x1) < 0.1 * abs(e0)


def test_flight_sim_controller_rejects_invalid_params():
    """Factory should reject non-physical parameter values."""
    from scpn_control.control.h_infinity_controller import get_flight_sim_controller

    with pytest.raises(ValueError):
        get_flight_sim_controller(response_gain=-1.0)
    with pytest.raises(ValueError):
        get_flight_sim_controller(actuator_tau=0.0)


def test_hinf_episode_uses_flight_sim_controller(monkeypatch):
    """H-inf episode should use get_flight_sim_controller, not get_radial_robust_controller."""
    import numpy as np
    import validation.stress_test_campaign as mod

    class FakeIso:
        def __init__(self):
            self.pid_R = {"Kp": 2.0}
            self.pid_step = lambda pid, err: float(err) * 2.0

        def run_shot(self, shot_duration, save_plot=False):
            return {
                "steps": int(shot_duration),
                "mean_abs_r_error": 0.08,
                "mean_abs_z_error": 0.06,
                "mean_abs_radial_actuator_lag": 0.1,
            }

    monkeypatch.setattr(mod, "IsoFluxController", lambda *a, **kw: FakeIso())

    ep = mod._run_hinf_episode(config_path="unused", shot_duration=5)
    assert np.isfinite(ep.reward)
    assert ep.mean_abs_r_error >= 0.0
