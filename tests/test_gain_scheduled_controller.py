# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
from __future__ import annotations

import numpy as np

from scpn_control.control.gain_scheduled_controller import (
    GainScheduledController,
    OperatingRegime,
    RegimeController,
    RegimeDetector,
    ScenarioWaveform,
    iter_baseline_schedule,
)


def test_regime_detection():
    detector = RegimeDetector()
    state = np.zeros(2)

    # Ramp up
    dstate = np.array([0.2, 0.0])  # dIp/dt = 0.2 > 0.1
    reg = detector.detect(state, dstate, tau_E=1.0, p_disrupt=0.0)
    for _ in range(5):
        reg = detector.detect(state, dstate, tau_E=1.0, p_disrupt=0.0)
    assert reg == OperatingRegime.RAMP_UP

    # L-mode flat
    dstate = np.array([0.01, 0.0])
    for _ in range(6):
        reg = detector.detect(state, dstate, tau_E=1.0, p_disrupt=0.0)
    assert reg == OperatingRegime.L_MODE_FLAT

    # H-mode flat
    for _ in range(6):
        reg = detector.detect(state, dstate, tau_E=2.0, p_disrupt=0.0)
    assert reg == OperatingRegime.H_MODE_FLAT

    # Disruption
    for _ in range(6):
        reg = detector.detect(state, dstate, tau_E=2.0, p_disrupt=0.9)
    assert reg == OperatingRegime.DISRUPTION_MITIGATION


def test_bumpless_transfer():
    controllers = {
        OperatingRegime.RAMP_UP: RegimeController(
            OperatingRegime.RAMP_UP,
            Kp=np.ones(1) * 2.0,
            Ki=np.zeros(1),
            Kd=np.zeros(1),
            x_ref=np.ones(1),
            constraints={},
        ),
        OperatingRegime.L_MODE_FLAT: RegimeController(
            OperatingRegime.L_MODE_FLAT,
            Kp=np.ones(1) * 1.0,
            Ki=np.zeros(1),
            Kd=np.zeros(1),
            x_ref=np.ones(1),
            constraints={},
        ),
    }

    gsc = GainScheduledController(controllers)
    gsc.current_regime = OperatingRegime.RAMP_UP

    x = np.zeros(1)

    # Switch exactly at t=1.0
    u0 = gsc.step(x, 0.9, 0.1, OperatingRegime.RAMP_UP)
    assert u0[0] == 2.0

    u1 = gsc.step(x, 1.0, 0.1, OperatingRegime.L_MODE_FLAT)
    # alpha = 0.0
    assert u1[0] == 2.0

    u2 = gsc.step(x, 1.25, 0.25, OperatingRegime.L_MODE_FLAT)
    # alpha = 0.5 -> Kp = 1.5
    assert u2[0] == 1.5

    u3 = gsc.step(x, 1.5, 0.25, OperatingRegime.L_MODE_FLAT)
    # alpha = 1.0 -> Kp = 1.0
    assert u3[0] == 1.0


def test_scenario_waveforms():
    times = np.array([0, 10, 20])
    values = np.array([0.0, 10.0, 10.0])
    wf = ScenarioWaveform("test", times, values)

    assert wf(0.0) == 0.0
    assert wf(5.0) == 5.0
    assert wf(10.0) == 10.0
    assert wf(15.0) == 10.0


def test_iter_baseline_schedule():
    sched = iter_baseline_schedule()
    errors = sched.validate()
    assert len(errors) == 0
    assert sched.duration() == 480.0

    val_60 = sched.evaluate(60.0)
    assert val_60["Ip"] == 15.0
    assert val_60["P_NBI"] == 33.0

    val_455 = sched.evaluate(455.0)
    # Midpoint between 430 and 480
    assert val_455["Ip"] == 6.0


def test_gain_schedule_interpolates():
    """Gains vary smoothly between operating points during bumpless transfer.

    Rugh & Shamma 2000, Automatica 36, 1401, §3: gain-scheduled controllers
    must not produce step transients at regime boundaries.
    Linear interpolation: K(α) = (1-α) K_old + α K_new, α ∈ [0,1].
    """
    Kp_ramp = np.ones(1) * 4.0
    Kp_flat = np.ones(1) * 1.0

    controllers = {
        OperatingRegime.RAMP_UP: RegimeController(
            OperatingRegime.RAMP_UP,
            Kp=Kp_ramp,
            Ki=np.zeros(1),
            Kd=np.zeros(1),
            x_ref=np.ones(1),
            constraints={},
        ),
        OperatingRegime.L_MODE_FLAT: RegimeController(
            OperatingRegime.L_MODE_FLAT,
            Kp=Kp_flat,
            Ki=np.zeros(1),
            Kd=np.zeros(1),
            x_ref=np.ones(1),
            constraints={},
        ),
    }

    gsc = GainScheduledController(controllers)
    x = np.zeros(1)

    # Step to L_MODE_FLAT at t=0.0; tau_switch=0.5 s
    tau = gsc.tau_switch
    n_steps = 20
    kp_samples = []
    for i in range(n_steps + 1):
        t = i * tau / n_steps
        dt = tau / n_steps
        gsc.step(x, t, dt, OperatingRegime.L_MODE_FLAT)
        kp_samples.append(float(gsc.Kp[0]))

    # Gains must be monotonically decreasing from Kp_ramp toward Kp_flat
    diffs = np.diff(kp_samples)
    assert np.all(diffs <= 1e-9), "gains not monotonically decreasing during bumpless transfer"
    # After tau_switch, gain must equal Kp_flat
    assert np.isclose(kp_samples[-1], float(Kp_flat[0]), atol=1e-9)


def test_ramp_down_detection():
    """Cover gain_scheduled_controller.py line 102: dIp_dt < -ramp_rate -> RAMP_DOWN."""
    detector = RegimeDetector()
    state = np.zeros(2)
    dstate = np.array([-0.2, 0.0])  # dIp/dt = -0.2 < -0.1
    for _ in range(6):
        reg = detector.detect(state, dstate, tau_E=1.0, p_disrupt=0.0)
    assert reg == OperatingRegime.RAMP_DOWN


def test_disruption_resets_integral():
    """Cover gain_scheduled_controller.py line 165: disruption zeros integral_error."""
    controllers = {
        OperatingRegime.RAMP_UP: RegimeController(
            OperatingRegime.RAMP_UP,
            Kp=np.ones(1),
            Ki=np.ones(1),
            Kd=np.zeros(1),
            x_ref=np.ones(1),
            constraints={},
        ),
        OperatingRegime.DISRUPTION_MITIGATION: RegimeController(
            OperatingRegime.DISRUPTION_MITIGATION,
            Kp=np.ones(1) * 0.1,
            Ki=np.zeros(1),
            Kd=np.zeros(1),
            x_ref=np.zeros(1),
            constraints={},
        ),
    }
    gsc = GainScheduledController(controllers)
    # Accumulate integral in RAMP_UP: error = x_ref(1) - x(0) = 1, integral += 1*dt
    gsc.step(np.zeros(1), 0.0, 0.1, OperatingRegime.RAMP_UP)
    gsc.step(np.zeros(1), 0.1, 0.1, OperatingRegime.RAMP_UP)
    integral_before = gsc.integral_error.copy()
    assert np.any(integral_before != 0.0)

    # Switch to DISRUPTION at x=x_ref so error=0 after the reset
    # During bumpless transfer alpha=0 -> x_ref=old(1), so use x=1 to make error=0
    gsc.step(np.ones(1), 0.2, 0.1, OperatingRegime.DISRUPTION_MITIGATION)
    # The integral was reset to 0, then error=(1-1)*dt=0 added -> integral=0
    assert np.all(gsc.integral_error == 0.0)


def test_empty_scenario_duration():
    """Cover gain_scheduled_controller.py line 217: empty waveforms -> duration 0."""
    from scpn_control.control.gain_scheduled_controller import ScenarioSchedule

    sched = ScenarioSchedule({})
    assert sched.duration() == 0.0


def test_scenario_validate_non_monotonic():
    """Cover gain_scheduled_controller.py line 224: non-monotonic times error."""
    from scpn_control.control.gain_scheduled_controller import ScenarioSchedule

    wf = ScenarioWaveform("bad", np.array([0.0, 5.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    sched = ScenarioSchedule({"bad": wf})
    errors = sched.validate()
    assert len(errors) > 0
    assert "non-monotonic" in errors[0]
