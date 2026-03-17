# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
from __future__ import annotations

import numpy as np

from scpn_control.core.momentum_transport import (
    PRANDTL_MOMENTUM,
    MomentumTransportSolver,
    RotationDiagnostics,
    exb_shearing_rate,
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
