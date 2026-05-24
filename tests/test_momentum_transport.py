# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Momentum Transport Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.momentum_transport import (
    PRANDTL_MOMENTUM,
    MomentumTransportSolver,
    RotationDiagnostics,
    exb_shearing_rate,
    intrinsic_rotation_torque,
    nbi_torque,
    radial_electric_field,
    rice_intrinsic_velocity,
    turbulence_suppression_factor,
)


def test_nbi_torque():
    P_nbi = np.ones(50) * 1e6
    torque_co = nbi_torque(P_nbi, R0=6.2, v_beam=1e6, theta_inj_deg=30.0)
    assert np.all(torque_co > 0.0)

    torque_counter = nbi_torque(P_nbi, R0=6.2, v_beam=1e6, theta_inj_deg=-30.0)
    assert np.all(torque_counter < 0.0)


def test_solver_viscous_damping():
    rho = np.linspace(0, 1, 50)
    solver = MomentumTransportSolver(rho, R0=6.2, a=2.0, B0=5.3)

    # Initialize with some rotation
    solver.omega_phi = np.ones(50) * 1e4

    chi_i = np.ones(50) * 1.0
    ne = np.ones(50) * 5.0
    Ti = np.ones(50) * 5.0
    T_zero = np.zeros(50)

    # Step forward
    for _ in range(10):
        solver.step(0.1, chi_i, ne, Ti, T_zero, T_zero)

    # Should damp towards zero
    assert solver.omega_phi[25] < 1e4


def test_solver_steady_state():
    rho = np.linspace(0, 1, 50)
    solver = MomentumTransportSolver(rho, R0=6.2, a=2.0, B0=5.3)

    chi_i = np.ones(50) * 1.0
    ne = np.ones(50) * 5.0
    Ti = np.ones(50) * 5.0

    T_nbi = np.ones(50) * 1.0
    T_zero = np.zeros(50)

    # Drive to steady state
    for _ in range(100):
        solver.step(0.1, chi_i, ne, Ti, T_nbi, T_zero)

    assert solver.omega_phi[0] > 0.0


def test_exb_shear_suppression():
    rho = np.linspace(0, 1, 50)
    omega = 1e5 * (1.0 - rho**2)  # Peaked rotation
    B_theta = 0.5 * rho

    rate = exb_shearing_rate(omega, B_theta, B0=5.3, R0=6.2, rho=rho, a=2.0)
    # Should be non-zero where shear is non-zero
    assert rate[25] > 0.0

    gamma_max = np.ones(50) * 1e4
    supp = turbulence_suppression_factor(rate, gamma_max)

    # Where rate > gamma, supp should be < 0.5
    if rate[25] > 1e4:
        assert supp[25] < 0.5


def test_radial_electric_field():
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 5.0
    Ti = 5.0 * (1.0 - rho**2)
    omega = 1e4 * np.ones(50)
    B_theta = 0.5 * rho

    Er = radial_electric_field(ne, Ti, omega, B_theta, B0=5.3, R0=6.2, rho=rho, a=2.0)

    # Expect Er to be composed of dp/dr (negative) and v*B (positive)
    assert len(Er) == 50


def test_diagnostics():
    omega = np.ones(50) * 1e4
    Ti = np.ones(50) * 10.0

    diag = RotationDiagnostics()
    mach = diag.mach_number(omega, Ti, R0=6.2)

    assert np.all(mach > 0.0)
    assert mach[0] < 1.0  # Usually subsonic

    stab = diag.rwm_stabilization_criterion(omega, tau_wall=0.01)
    assert stab  # 10000 * 0.01 = 100 > 0.01


# ── new physics tests ────────────────────────────────────────────────────────


def test_prandtl_number() -> None:
    # χ_φ = Pr · χ_i; Peeters et al. 2011, Nucl. Fusion 51, 083015, Fig. 5.
    rho = np.linspace(0, 1, 20)
    solver = MomentumTransportSolver(rho, R0=6.2, a=2.0, B0=5.3, prandtl=PRANDTL_MOMENTUM)
    chi_i = np.ones(20) * 2.0
    chi_phi = solver.prandtl * chi_i
    np.testing.assert_allclose(chi_phi, PRANDTL_MOMENTUM * chi_i)
    assert PRANDTL_MOMENTUM == 0.7


def test_rice_scaling() -> None:
    # Higher W_p/I_p → faster intrinsic rotation; Rice et al. 2007, Eq. 3.
    v_low = rice_intrinsic_velocity(W_p_MJ=1.0, I_p_MA=2.0)  # 0.5 MJ/MA
    v_high = rice_intrinsic_velocity(W_p_MJ=4.0, I_p_MA=2.0)  # 2.0 MJ/MA
    assert v_high > v_low
    # Linearity in W_p/I_p
    np.testing.assert_allclose(v_high / v_low, 4.0, rtol=1e-9)


def test_exb_shear_positive() -> None:
    # Positive dω/dr → positive shearing rate; Burrell 1997, Phys. Plasmas 4, 1499, Eq. 1.
    rho = np.linspace(0.0, 1.0, 50)
    omega_rising = 1e4 * rho  # monotonically increasing → positive gradient
    B_theta = 0.5 * np.ones(50)
    rate = exb_shearing_rate(omega_rising, B_theta, B0=5.3, R0=6.2, rho=rho, a=2.0)
    assert np.all(rate >= 0.0)
    assert rate[25] > 0.0


def test_nbi_torque_direction() -> None:
    # Co-injection (θ > 0) → positive torque; Stacey & Sigmar 1985, Phys. Fluids 28, 2800.
    P_nbi = np.ones(50) * 1e6
    torque_co = nbi_torque(P_nbi, R0=6.2, v_beam=1e6, theta_inj_deg=30.0)
    assert np.all(torque_co > 0.0)

    torque_ctr = nbi_torque(P_nbi, R0=6.2, v_beam=1e6, theta_inj_deg=-30.0)
    assert np.all(torque_ctr < 0.0)


def test_nbi_torque_rejects_nonphysical_inputs() -> None:
    with pytest.raises(ValueError, match="P_nbi_profile"):
        nbi_torque(np.array([-1.0]), R0=6.2, v_beam=1e6, theta_inj_deg=30.0)
    with pytest.raises(ValueError, match="R0"):
        nbi_torque(np.ones(3), R0=0.0, v_beam=1e6, theta_inj_deg=30.0)
    assert np.all(nbi_torque(np.ones(3), R0=6.2, v_beam=0.0, theta_inj_deg=30.0) == 0.0)


def test_nbi_torque_rejects_nonfinite_power_and_geometry() -> None:
    with pytest.raises(ValueError, match="P_nbi_profile"):
        nbi_torque(np.array([np.nan]), R0=6.2, v_beam=1e6, theta_inj_deg=30.0)
    with pytest.raises(ValueError, match="theta_inj_deg"):
        nbi_torque(np.ones(3), R0=6.2, v_beam=1e6, theta_inj_deg=float("nan"))
    with pytest.raises(ValueError, match="v_beam"):
        nbi_torque(np.ones(3), R0=6.2, v_beam=float("inf"), theta_inj_deg=30.0)


def test_intrinsic_rotation_torque_rejects_inconsistent_profiles() -> None:
    with pytest.raises(ValueError, match="matching shape"):
        intrinsic_rotation_torque(np.ones(3), np.ones(4), R0=6.2, a=2.0)
    with pytest.raises(ValueError, match="R0 and a"):
        intrinsic_rotation_torque(np.ones(3), np.ones(3), R0=-1.0, a=2.0)
    with pytest.raises(ValueError, match="grad_Ti"):
        intrinsic_rotation_torque(np.array([1.0, np.nan]), np.ones(2), R0=6.2, a=2.0)


def test_exb_shearing_rate_rejects_nonphysical_inputs() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    omega = np.ones(5)
    btheta = np.ones(5)

    with pytest.raises(ValueError, match="matching shape"):
        exb_shearing_rate(omega[:-1], btheta, B0=5.3, R0=6.2, rho=rho, a=2.0)
    with pytest.raises(ValueError, match="sorted"):
        exb_shearing_rate(omega, btheta, B0=5.3, R0=6.2, rho=rho[::-1], a=2.0)
    with pytest.raises(ValueError, match="B0, R0, and a"):
        exb_shearing_rate(omega, btheta, B0=0.0, R0=6.2, rho=rho, a=2.0)
    with pytest.raises(ValueError, match="omega_phi"):
        exb_shearing_rate(np.array([1.0, np.nan, 1.0, 1.0, 1.0]), btheta, B0=5.3, R0=6.2, rho=rho, a=2.0)


def test_turbulence_suppression_rejects_nonphysical_growth_rates() -> None:
    with pytest.raises(ValueError, match="matching shape"):
        turbulence_suppression_factor(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="gamma_max"):
        turbulence_suppression_factor(np.ones(3), -np.ones(3))
    with pytest.raises(ValueError, match="omega_ExB"):
        turbulence_suppression_factor(np.array([1.0, np.nan]), np.ones(2))


def test_radial_electric_field_rejects_nonphysical_profiles() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    ne = np.ones(5)
    Ti = np.ones(5)
    omega = np.ones(5)
    btheta = np.ones(5)

    with pytest.raises(ValueError, match="matching shape"):
        radial_electric_field(ne[:-1], Ti, omega, btheta, B0=5.3, R0=6.2, rho=rho, a=2.0)
    with pytest.raises(ValueError, match="sorted"):
        radial_electric_field(ne, Ti, omega, btheta, B0=5.3, R0=6.2, rho=rho[::-1], a=2.0)
    with pytest.raises(ValueError, match="ne must be positive"):
        radial_electric_field(np.zeros(5), Ti, omega, btheta, B0=5.3, R0=6.2, rho=rho, a=2.0)
    with pytest.raises(ValueError, match="B0, R0, and a"):
        radial_electric_field(ne, Ti, omega, btheta, B0=5.3, R0=0.0, rho=rho, a=2.0)
    with pytest.raises(ValueError, match="B_theta"):
        radial_electric_field(ne, Ti, omega, np.array([1.0, np.nan, 1.0, 1.0, 1.0]), B0=5.3, R0=6.2, rho=rho, a=2.0)


def test_rice_intrinsic_velocity_rejects_nonphysical_scaling_inputs() -> None:
    with pytest.raises(ValueError, match="W_p_MJ"):
        rice_intrinsic_velocity(W_p_MJ=-1.0, I_p_MA=2.0)
    with pytest.raises(ValueError, match="I_p_MA"):
        rice_intrinsic_velocity(W_p_MJ=1.0, I_p_MA=0.0)
    with pytest.raises(ValueError, match="W_p_MJ"):
        rice_intrinsic_velocity(W_p_MJ=float("nan"), I_p_MA=2.0)


def test_rotation_diagnostics_rejects_nonphysical_inputs() -> None:
    diag = RotationDiagnostics()

    with pytest.raises(ValueError, match="matching shape"):
        diag.mach_number(np.ones(3), np.ones(4), R0=6.2)
    with pytest.raises(ValueError, match="Ti_keV"):
        diag.mach_number(np.ones(3), np.zeros(3), R0=6.2)
    with pytest.raises(ValueError, match="omega_phi"):
        diag.rwm_stabilization_criterion(np.array([]), tau_wall=0.01)
    with pytest.raises(ValueError, match="tau_wall"):
        diag.rwm_stabilization_criterion(np.ones(3), tau_wall=0.0)
    with pytest.raises(ValueError, match="omega_phi"):
        diag.mach_number(np.array([np.nan]), np.ones(1), R0=6.2)
    with pytest.raises(ValueError, match="tau_wall"):
        diag.rwm_stabilization_criterion(np.ones(3), tau_wall=float("inf"))


def test_momentum_solver_rejects_nonphysical_constructor_inputs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        MomentumTransportSolver(np.array([0.0]), R0=6.2, a=2.0, B0=5.3)
    with pytest.raises(ValueError, match="strictly increasing"):
        MomentumTransportSolver(np.array([0.0, 0.0, 1.0]), R0=6.2, a=2.0, B0=5.3)
    with pytest.raises(ValueError, match="R0, a, and B0"):
        MomentumTransportSolver(np.linspace(0, 1, 5), R0=0.0, a=2.0, B0=5.3)
    with pytest.raises(ValueError, match="prandtl"):
        MomentumTransportSolver(np.linspace(0, 1, 5), R0=6.2, a=2.0, B0=5.3, prandtl=0.0)
    with pytest.raises(ValueError, match="rho"):
        MomentumTransportSolver(np.array([0.0, np.nan, 1.0]), R0=6.2, a=2.0, B0=5.3)
    with pytest.raises(ValueError, match="prandtl"):
        MomentumTransportSolver(np.linspace(0, 1, 5), R0=6.2, a=2.0, B0=5.3, prandtl=float("nan"))


def test_momentum_solver_rejects_nonphysical_step_inputs() -> None:
    rho = np.linspace(0, 1, 5)
    solver = MomentumTransportSolver(rho, R0=6.2, a=2.0, B0=5.3)
    chi_i = np.ones(5)
    ne = np.ones(5)
    Ti = np.ones(5)
    torque = np.zeros(5)

    with pytest.raises(ValueError, match="dt"):
        solver.step(0.0, chi_i, ne, Ti, torque, torque)
    with pytest.raises(ValueError, match="matching shape"):
        solver.step(0.1, chi_i[:-1], ne, Ti, torque, torque)
    with pytest.raises(ValueError, match="chi_i"):
        solver.step(0.1, -chi_i, ne, Ti, torque, torque)
    with pytest.raises(ValueError, match="ne must be positive"):
        solver.step(0.1, chi_i, np.zeros(5), Ti, torque, torque)
    with pytest.raises(ValueError, match="T_nbi"):
        solver.step(0.1, chi_i, ne, Ti, np.full(5, np.nan), torque)
    solver.omega_phi[2] = np.nan
    with pytest.raises(ValueError, match="omega_phi"):
        solver.step(0.1, chi_i, ne, Ti, torque, torque)
