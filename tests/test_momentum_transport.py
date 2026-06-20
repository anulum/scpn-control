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
    _finite_1d_grid,
    _finite_array,
    _finite_scalar,
    _nonnegative_profile_or_scalar,
    _uniform_axis_to_edge_rho_grid,
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


def test_solver_applies_collisional_damping_implicitly() -> None:
    rho = np.linspace(0.0, 1.0, 16)
    solver = MomentumTransportSolver(rho, R0=6.2, a=2.0, B0=5.3)
    initial_omega = 1.5e4 * (1.0 - 0.2 * rho)
    solver.omega_phi = initial_omega.copy()

    chi_i = np.zeros_like(rho)
    ne = np.ones_like(rho) * 5.0
    Ti = np.ones_like(rho) * 7.0
    torque = np.zeros_like(rho)
    dt = 0.25
    damping_frequency = np.ones_like(rho) * 0.8

    omega = solver.step(
        dt,
        chi_i,
        ne,
        Ti,
        torque,
        torque,
        momentum_damping_frequency_s=damping_frequency,
    )

    expected = initial_omega / (1.0 + dt * damping_frequency)
    np.testing.assert_allclose(omega[1:-1], expected[1:-1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(omega[0], omega[1], rtol=1e-12, atol=1e-12)
    assert omega[-1] == 0.0


def test_solver_rejects_nonphysical_collisional_damping_profiles() -> None:
    rho = np.linspace(0.0, 1.0, 8)
    solver = MomentumTransportSolver(rho, R0=6.2, a=2.0, B0=5.3)
    chi_i = np.ones_like(rho)
    ne = np.ones_like(rho)
    Ti = np.ones_like(rho)
    torque = np.zeros_like(rho)

    with pytest.raises(ValueError, match="momentum_damping_frequency_s"):
        solver.step(0.1, chi_i, ne, Ti, torque, torque, momentum_damping_frequency_s=-0.1)
    with pytest.raises(ValueError, match="matching shape"):
        solver.step(0.1, chi_i, ne, Ti, torque, torque, momentum_damping_frequency_s=np.ones(7))
    with pytest.raises(ValueError, match="momentum_damping_frequency_s"):
        solver.step(0.1, chi_i, ne, Ti, torque, torque, momentum_damping_frequency_s=np.full(8, np.nan))


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


def test_exb_shearing_rate_uses_nonuniform_rho_coordinates() -> None:
    uniform = np.linspace(0.0, 1.0, 96)
    rho = uniform**1.6
    a = 2.0
    R0 = 6.2
    B0 = 5.3
    omega = 2.5e4 * (rho * a) ** 2
    B_theta = np.ones_like(rho) * 0.45

    rate = exb_shearing_rate(omega, B_theta, B0=B0, R0=R0, rho=rho, a=a)
    B_tot = np.sqrt(B0**2 + B_theta**2)
    expected = np.abs((R0 * B_theta / B_tot) * (5.0e4 * rho * a))

    np.testing.assert_allclose(rate[4:-4], expected[4:-4], rtol=3.0e-2, atol=1.0e-8)


def test_radial_electric_field_uses_nonuniform_rho_coordinates() -> None:
    uniform = np.linspace(0.0, 1.0, 96)
    rho = uniform**1.6
    a = 2.0
    ne = np.ones_like(rho) * 5.0
    Ti = 8.0 - 0.5 * (rho * a) ** 2
    omega = np.zeros_like(rho)
    btheta = np.zeros_like(rho)

    Er = radial_electric_field(ne, Ti, omega, btheta, B0=5.3, R0=6.2, rho=rho, a=a)
    expected = -1.0 * rho * a * 1.0e3

    np.testing.assert_allclose(Er[4:-4], expected[4:-4], rtol=3.0e-2, atol=1.0e-8)


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
    with pytest.raises(ValueError, match="strictly increasing"):
        exb_shearing_rate(omega, btheta, B0=5.3, R0=6.2, rho=np.array([0.0, 0.25, 0.25, 0.75, 1.0]), a=2.0)
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
    with pytest.raises(ValueError, match="strictly increasing"):
        radial_electric_field(
            ne,
            Ti,
            omega,
            btheta,
            B0=5.3,
            R0=6.2,
            rho=np.array([0.0, 0.25, 0.25, 0.75, 1.0]),
            a=2.0,
        )
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
    with pytest.raises(ValueError, match="uniform"):
        MomentumTransportSolver(np.array([0.0, 0.1, 0.4, 1.0]), R0=6.2, a=2.0, B0=5.3)


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


def test_finite_scalar_rejects_negative_and_nonpositive() -> None:
    with pytest.raises(ValueError, match="x must be non-negative"):
        _finite_scalar("x", -1.0, nonnegative=True)
    with pytest.raises(ValueError, match="x must be positive"):
        _finite_scalar("x", 0.0, positive=True)


def test_finite_array_rejects_nonpositive_element() -> None:
    with pytest.raises(ValueError, match="x must be positive"):
        _finite_array("x", np.array([1.0, -1.0]), positive=True)


def test_finite_1d_grid_reports_required_minimum_size() -> None:
    with pytest.raises(ValueError, match="at least 3 points"):
        _finite_1d_grid("grid", np.array([1.0]), minimum_size=3)


def test_uniform_grid_requires_axis_and_edge_anchors() -> None:
    with pytest.raises(ValueError, match="rho must start at the magnetic axis"):
        _uniform_axis_to_edge_rho_grid(np.array([0.1, 0.5, 1.0]))
    with pytest.raises(ValueError, match="rho must end at the plasma edge"):
        _uniform_axis_to_edge_rho_grid(np.array([0.0, 0.45, 0.9]))


def test_intrinsic_rotation_torque_returns_gradient_scaled_profile() -> None:
    grad_Ti = np.array([1.0, 2.0, 3.0])
    torque = intrinsic_rotation_torque(grad_Ti, np.ones(3), R0=6.2, a=2.0)
    np.testing.assert_allclose(torque, -1e-3 * grad_Ti)


def test_nonnegative_profile_or_scalar_broadcasts_scalar() -> None:
    out = _nonnegative_profile_or_scalar("damping", 2.0, (4,))
    np.testing.assert_array_equal(out, np.full(4, 2.0))


def test_mach_number_rejects_nonpositive_major_radius() -> None:
    with pytest.raises(ValueError, match="R0 must be positive"):
        RotationDiagnostics.mach_number(np.ones(5), np.ones(5), R0=0.0)


def test_step_rejects_chi_profile_mismatched_with_grid() -> None:
    solver = MomentumTransportSolver(np.linspace(0.0, 1.0, 50), R0=6.2, a=2.0, B0=5.3)
    solver.omega_phi = np.zeros(49)  # match the misshaped inputs so the grid guard fires
    short = np.ones(49)
    with pytest.raises(ValueError, match="transport profiles must match the solver rho grid"):
        solver.step(0.1, short, short, short, np.zeros(49), np.zeros(49))


def test_step_rejects_corrupted_rho_grid() -> None:
    solver = MomentumTransportSolver(np.linspace(0.0, 1.0, 50), R0=6.2, a=2.0, B0=5.3)
    corrupted = solver.rho.copy()
    corrupted[25] = corrupted[24]  # break strict monotonicity
    solver.rho = corrupted
    chi_i = np.ones(50)
    with pytest.raises(ValueError, match="rho grid must remain finite and strictly increasing"):
        solver.step(0.1, chi_i, np.ones(50) * 5.0, np.ones(50) * 5.0, np.zeros(50), np.zeros(50))
