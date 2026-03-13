# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Gain Scheduled Controller Tests
# ──────────────────────────────────────────────────────────────────────
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
