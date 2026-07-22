# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Anti-Windup PID Controller Tests
"""Unit coverage for the anti-windup PID law and its flight-sim wiring.

Mirrors the Rust ``control_control::pid`` test suite: legacy-identity, saturation,
directional anti-windup (freeze under sustained saturation, still integrate away
from the limit), slew limiting, envelope validation, and reset semantics.
"""

from __future__ import annotations

import math

import pytest

from scpn_control.control.pid_controller import PIDController
from scpn_control.control.tokamak_flight_sim import IsoFluxController


class _DummyKernel:
    """Minimal kernel stand-in exposing an empty coil configuration."""

    def __init__(self, config_file: str) -> None:
        self.cfg: dict[str, object] = {"coils": []}


def test_unconfigured_envelope_is_bit_identical_to_ideal_law() -> None:
    pid = PIDController(kp=2.0, ki=0.1, kd=0.5)
    err_sum = 0.0
    last_err = 0.0
    for error in (1.0, -0.5, 3.0, -2.0, 0.25):
        err_sum += error
        d_err = error - last_err
        last_err = error
        expected = 2.0 * error + 0.1 * err_sum + 0.5 * d_err
        assert pid.step(error) == expected


def test_output_saturation_clamps_symmetrically() -> None:
    pid = PIDController(kp=10.0, ki=0.0, kd=0.0).with_output_limits(-5.0, 5.0)
    assert pid.step(100.0) == 5.0
    assert pid.step(-100.0) == -5.0


def test_anti_windup_freezes_integrator_under_sustained_saturation() -> None:
    limit = 5.0
    guarded = PIDController(kp=1.0, ki=0.5, kd=0.0).with_output_limits(-limit, limit)
    naive = PIDController(kp=1.0, ki=0.5, kd=0.0)
    for _ in range(200):
        applied = guarded.step(10.0)
        assert applied <= limit + 1e-12
        naive.step(10.0)
    # Error reverses: the guarded controller responds immediately; the naive one
    # is still discharging a huge wound-up integral.
    guarded_recovery = guarded.step(-1.0)
    naive_recovery = naive.step(-1.0)
    assert guarded_recovery < 0.0
    assert naive_recovery > 100.0


def test_anti_windup_still_integrates_away_from_the_limit() -> None:
    pid = PIDController(kp=1.0, ki=1.0, kd=0.0).with_output_limits(-5.0, 5.0)
    for _ in range(20):
        pid.step(10.0)
    first = pid.step(-1.0)
    second = pid.step(-1.0)
    assert first < 5.0
    assert second < first


def test_slew_limit_bounds_per_step_change() -> None:
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0).with_slew_limit(2.0)
    assert pid.step(1.0) == 2.0
    assert pid.step(1.0) == 4.0


def test_reset_clears_slew_and_integral_state() -> None:
    pid = PIDController(kp=100.0, ki=0.0, kd=0.0).with_slew_limit(2.0)
    pid.step(1.0)
    pid.reset()
    assert pid.step(1.0) == 2.0


def test_rejects_non_finite_gains() -> None:
    with pytest.raises(ValueError, match="gains must be finite"):
        PIDController(kp=math.nan, ki=0.1, kd=0.2)


def test_rejects_non_finite_error() -> None:
    pid = PIDController(kp=1.0, ki=0.1, kd=0.2)
    with pytest.raises(ValueError, match="error input must be finite"):
        pid.step(math.inf)


def test_output_limit_validation() -> None:
    with pytest.raises(ValueError, match="output limits must be finite"):
        PIDController(kp=1.0, ki=0.0, kd=0.0).with_output_limits(math.nan, 1.0)
    with pytest.raises(ValueError, match="output_min must not exceed output_max"):
        PIDController(kp=1.0, ki=0.0, kd=0.0).with_output_limits(1.0, -1.0)


def test_slew_limit_validation() -> None:
    with pytest.raises(ValueError, match="slew_max must be finite and > 0"):
        PIDController(kp=1.0, ki=0.0, kd=0.0).with_slew_limit(0.0)
    with pytest.raises(ValueError, match="slew_max must be finite and > 0"):
        PIDController(kp=1.0, ki=0.0, kd=0.0).with_slew_limit(math.inf)


def test_flight_sim_wires_anti_windup_envelope_into_both_pids() -> None:
    limit = 3.0
    sim = IsoFluxController(
        config_file="dummy.json",
        kernel_factory=_DummyKernel,
        verbose=False,
        actuator_current_delta_limit=limit,
    )
    assert isinstance(sim.pid_R, PIDController)
    assert isinstance(sim.pid_Z, PIDController)
    # A sustained saturating position error must not wind either integrator up:
    # the output stays clamped and recovery on error reversal is immediate.
    for _ in range(200):
        assert abs(sim.pid_R.step(100.0)) <= limit + 1e-12
    assert sim.pid_R.step(-1.0) < 0.0
