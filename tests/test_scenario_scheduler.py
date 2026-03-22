# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Scenario Scheduler Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.control.scenario_scheduler import (
    FeedforwardController,
    ScenarioOptimizer,
    ScenarioSchedule,
    ScenarioWaveform,
    iter_15ma_baseline,
    nstx_u_1ma_standard,
)


def test_scenario_waveform_interpolation():
    times = np.array([0.0, 10.0, 20.0])
    values = np.array([0.0, 100.0, 50.0])
    wf = ScenarioWaveform("test", times, values)

    # Exact breakpoints
    assert np.isclose(wf(0.0), 0.0)
    assert np.isclose(wf(10.0), 100.0)
    assert np.isclose(wf(20.0), 50.0)

    # Midpoints
    assert np.isclose(wf(5.0), 50.0)
    assert np.isclose(wf(15.0), 75.0)

    # Extrapolation (flat)
    assert np.isclose(wf(-5.0), 0.0)
    assert np.isclose(wf(25.0), 50.0)


def test_scenario_validation():
    times = np.array([0.0, 10.0, 5.0])  # Non-monotonic
    values = np.array([0.0, 10.0, 20.0])
    wf = ScenarioWaveform("Ip", times, values)
    sched = ScenarioSchedule({"Ip": wf})

    errors = sched.validate()
    assert len(errors) > 0
    assert "monotonic" in errors[0]

    times2 = np.array([0.0, 10.0, 20.0])
    values2 = np.array([0.0, -10.0, 20.0])  # Negative Ip
    wf2 = ScenarioWaveform("Ip", times2, values2)
    sched2 = ScenarioSchedule({"Ip": wf2})

    errors2 = sched2.validate()
    assert len(errors2) > 0
    assert "negative" in errors2[0]


def test_feedforward_controller():
    sched = iter_15ma_baseline()

    def dummy_feedback(x, x_ref, t, dt):
        return np.array([1.0, 2.0, 3.0])  # Dummy fb trim

    ctrl = FeedforwardController(sched, dummy_feedback)

    x = np.zeros(1)
    u = ctrl.step(x, 60.0, 0.1)

    # At t=60, P_aux = 50, Ip = 15
    # Expect u = ff + fb = [50, 15, 0] + [1, 2, 3] = [51, 17, 3]
    assert np.isclose(u[0], 51.0)
    assert np.isclose(u[1], 17.0)


def test_scenario_optimizer():
    # Simple linear plant: x_next = x + u * dt
    def plant(x, u, dt):
        return x + u * dt

    target = np.array([10.0, 15.0])

    opt = ScenarioOptimizer(plant, target, T_total=10.0, dt=1.0)

    sched = opt.optimize(n_iter=50)
    assert sched is not None
    assert sched.duration() == 10.0


def test_factory_scenarios():
    iter_sched = iter_15ma_baseline()
    assert len(iter_sched.validate()) == 0
    assert iter_sched.duration() == 480.0

    nstx_sched = nstx_u_1ma_standard()
    assert len(nstx_sched.validate()) == 0
    assert nstx_sched.duration() == 2.0


def test_scenario_empty_duration():
    """Cover scenario_scheduler.py line 36: empty waveforms returns 0."""
    sched = ScenarioSchedule({})
    assert sched.duration() == 0.0


def test_scenario_negative_power_validation():
    """Cover scenario_scheduler.py line 50: negative heating power flagged."""
    wf = ScenarioWaveform("P_NBI", np.array([0, 10]), np.array([5.0, -1.0]))
    sched = ScenarioSchedule({"P_NBI": wf})
    errors = sched.validate()
    assert any("negative heating" in e for e in errors)


def test_scenario_non_positive_density_validation():
    """Cover scenario_scheduler.py line 53: non-positive density flagged."""
    wf = ScenarioWaveform("n_e", np.array([0, 10]), np.array([1.0, 0.0]))
    sched = ScenarioSchedule({"n_e": wf})
    errors = sched.validate()
    assert any("non-positive density" in e for e in errors)
