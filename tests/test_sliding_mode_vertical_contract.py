# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Sliding-mode public numeric contract
"""Exercise public sliding-mode admission and idealized-gain scope."""

from __future__ import annotations

import math

import pytest

from scpn_control.control.sliding_mode_vertical import (
    SuperTwistingSMC,
    VerticalStabilizer,
    estimate_convergence_time,
    lyapunov_certificate,
)


@pytest.mark.parametrize(
    "parameters",
    (
        {"alpha": math.nan},
        {"beta": -1.0},
        {"c": 0.0},
        {"u_max": math.inf},
    ),
)
def test_super_twisting_rejects_invalid_configuration(parameters: dict[str, float]) -> None:
    """Reject a controller with invalid gain, surface or actuator envelope."""
    defaults = {"alpha": 10.0, "beta": 20.0, "c": 1.0, "u_max": 100.0}
    defaults.update(parameters)
    with pytest.raises(ValueError, match="finite and positive"):
        SuperTwistingSMC(**defaults)


@pytest.mark.parametrize(
    "error,derivative,dt",
    ((1e308, 1e308, 0.01), (1.0, 0.0, 1e308), (1.0, 0.0, math.nan), (1.0, 0.0, -0.01)),
)
def test_step_refuses_invalid_derived_numeric_without_state_change(error: float, derivative: float, dt: float) -> None:
    """A bad public step cannot poison or advance the integral state."""
    smc = SuperTwistingSMC(alpha=10.0, beta=20.0, c=1.0, u_max=100.0)
    with pytest.raises(ValueError, match="finite|non-negative"):
        smc.step(error, derivative, dt)
    assert smc.v == 0.0


def test_step_rejects_nonfinite_measurement_and_command_overflow() -> None:
    """Direct observation and output overflow both refuse publication."""
    smc = SuperTwistingSMC(alpha=10.0, beta=20.0, c=1.0, u_max=100.0)
    with pytest.raises(ValueError, match="error and derivative"):
        smc.step(math.nan, 0.0, 0.01)
    assert smc.v == 0.0

    strong = SuperTwistingSMC(alpha=1e308, beta=1.0, c=1.0, u_max=100.0)
    with pytest.raises(ValueError, match="control command"):
        strong.step(1e308, 0.0, 0.01)
    assert strong.v == 0.0


def test_integral_overflow_preserves_last_accepted_value() -> None:
    """A second huge finite update cannot replace the last clipped integrator."""
    smc = SuperTwistingSMC(alpha=1.0, beta=1e308, c=1.0, u_max=1e308)
    smc.step(-1.0, 0.0, 1.0)
    accepted = smc.v
    with pytest.raises(ValueError, match="integral state"):
        smc.step(-1.0, 0.0, 1.0)
    assert smc.v == accepted


def test_mutated_actuator_envelope_refuses_instead_of_inverting_clip() -> None:
    """A writable envelope cannot turn a later step into an invalid clamp."""
    smc = SuperTwistingSMC(alpha=1.0, beta=2.0, c=1.0, u_max=10.0)
    smc.u_max = -10.0
    with pytest.raises(ValueError, match="u_max"):
        smc.step(1.0, 0.0, 0.01)
    assert smc.v == 0.0


def test_vertical_stabilizer_rejects_invalid_geometry() -> None:
    """A zero major radius cannot define the documented restoring coefficient."""
    smc = SuperTwistingSMC(alpha=10.0, beta=20.0, c=1.0, u_max=100.0)
    with pytest.raises(ValueError, match="R0"):
        VerticalStabilizer(n_index=-1.0, Ip_MA=15.0, R0=0.0, m_eff=1.0, tau_wall=0.01, smc=smc)


@pytest.mark.parametrize(
    "attribute,value,reason",
    (("R0", 0.0, "R0"), ("n_index", math.nan, "n_index"), ("Ip", -1.0, "Ip_MA")),
)
def test_vertical_step_rejects_mutated_geometry_before_integrator_change(
    attribute: str, value: float, reason: str
) -> None:
    """Writable model geometry cannot make a later wrapper step appear valid."""
    smc = SuperTwistingSMC(alpha=1.0, beta=2.0, c=1.0, u_max=10.0)
    stabilizer = VerticalStabilizer(n_index=-1.0, Ip_MA=15.0, R0=6.2, m_eff=1.0, tau_wall=0.01, smc=smc)
    setattr(stabilizer, attribute, value)
    with pytest.raises(ValueError, match=reason):
        stabilizer.step(0.1, 0.0, 0.0, 0.01)
    assert smc.v == 0.0


@pytest.mark.parametrize(
    "field,value,reason",
    (
        ("n_index", math.nan, "n_index"),
        ("Ip_MA", -1.0, "Ip_MA"),
        ("Ip_MA", 1e308, "conversion"),
        ("Ip_MA", 1e154, "K_vs"),
        ("n_index", 1e308, "K_vs"),
    ),
)
def test_vertical_stabilizer_rejects_invalid_or_overflowed_coefficient(field: str, value: float, reason: str) -> None:
    """Public geometry construction refuses invalid or unrepresentable force."""
    smc = SuperTwistingSMC(alpha=10.0, beta=20.0, c=1.0, u_max=100.0)
    parameters = {"n_index": -1.0, "Ip_MA": 15.0, "R0": 6.2, "m_eff": 1.0, "tau_wall": 0.01}
    parameters[field] = value
    with pytest.raises(ValueError, match=reason):
        VerticalStabilizer(smc=smc, **parameters)


def test_idealized_gain_screen_refuses_invalid_disturbance_and_beta() -> None:
    """Invalid disturbance bounds and insufficient beta cannot pass a gain screen."""
    assert not lyapunov_certificate(alpha=10.0, beta=20.0, L_max=-1.0)
    assert not lyapunov_certificate(alpha=math.nan, beta=20.0, L_max=2.0)
    assert estimate_convergence_time(alpha=10.0, beta=1.0, L_max=2.0, s0=4.0) == math.inf
    assert estimate_convergence_time(alpha=10.0, beta=20.0, L_max=2.0, s0=math.nan) == math.inf
