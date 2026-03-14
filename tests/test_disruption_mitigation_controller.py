# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption Mitigation Controller Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Tests for the closed-loop disruption mitigation controller.
"""

from __future__ import annotations


from scpn_control.control.spi_mitigation import DisruptionMitigationController, MitigationState


def test_controller_initial_state():
    """Verify that the controller starts in the IDLE state."""
    ctrl = DisruptionMitigationController()
    assert ctrl.state == MitigationState.IDLE


def test_controller_arming_sequence():
    """Verify that arming requires N consecutive samples."""
    ctrl = DisruptionMitigationController(n_consecutive=3, threshold_arm=0.5)
    dt = 0.01

    # 1. Single sample above threshold
    state = ctrl.update(0.6, dt)
    assert state == MitigationState.IDLE
    assert ctrl.consecutive_count == 1

    # 2. Reset if one sample falls below
    state = ctrl.update(0.4, dt)
    assert state == MitigationState.IDLE
    assert ctrl.consecutive_count == 0

    # 3. Three consecutive samples
    ctrl.update(0.6, dt)
    ctrl.update(0.6, dt)
    state = ctrl.update(0.6, dt)
    assert state == MitigationState.ARMED


def test_controller_fire_and_cooldown():
    """Verify that firing transitions to cooldown and prevents immediate re-fire."""
    ctrl = DisruptionMitigationController(threshold_arm=0.5, threshold_fire=0.8, n_consecutive=1, tau_cooldown_s=0.1)
    dt = 0.01

    # Arm
    ctrl.update(0.6, dt)
    assert ctrl.state == MitigationState.ARMED

    # Fire
    state = ctrl.update(0.9, dt)
    # The controller should transition ARMED -> FIRED (logged) -> COOLDOWN
    assert state == MitigationState.COOLDOWN

    # Check that it stays in COOLDOWN regardless of p_disrupt
    state = ctrl.update(0.9, dt)
    assert state == MitigationState.COOLDOWN

    # Wait for cooldown to expire
    for _ in range(10):
        state = ctrl.update(0.9, dt)

    assert state == MitigationState.IDLE


def test_controller_hysteresis():
    """Verify that the controller returns to IDLE if probability drops significantly."""
    ctrl = DisruptionMitigationController(threshold_arm=0.5, n_consecutive=1)
    dt = 0.01

    # Arm
    ctrl.update(0.6, dt)
    assert ctrl.state == MitigationState.ARMED

    # Drop below 0.8 * threshold_arm (0.4)
    state = ctrl.update(0.3, dt)
    assert state == MitigationState.IDLE
