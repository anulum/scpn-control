# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Current Diffusion Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.current_diffusion import (
    MU_0,
    CurrentDiffusionSolver,
    coulomb_log,
    neoclassical_resistivity,
    psi_from_q,
    q_from_psi,
    resistive_diffusion_time,
)


def test_neoclassical_resistivity():
    eta = neoclassical_resistivity(Te_keV=1.0, ne_19=1.0, Z_eff=1.5, epsilon=0.1)
    assert eta > 0.0
    # Spitzer resistivity ~ 1.65e-9 * 1.5 * 17 = 4.2e-8
    assert 1e-8 < eta < 1e-7


def test_q_from_psi():
    rho = np.linspace(0, 1, 50)
    R0 = 2.0
    a = 0.5
    B0 = 1.0
    # Parabolic current implies q ~ 1 + rho^2 ...
    # Let's set a simple psi: psi(rho) = - a^2 B0 / R0 * rho^2 / 2
    # Then dpsi/drho = - a^2 B0 / R0 * rho
    # q(rho) = -rho * a^2 * B0 / (R0 * (-a^2 B0 / R0 * rho)) = 1.0
    psi = -(a**2) * B0 / R0 * rho**2 / 2.0

    q = q_from_psi(rho, psi, R0, a, B0)
    assert np.allclose(q, 1.0, atol=0.1)


def test_resistive_diffusion_time():
    tau = resistive_diffusion_time(a=2.0, eta=1e-8)
    assert tau > 100.0  # Should be ~ 500s


def test_pure_ohmic_relaxation():
    rho = np.linspace(0, 1, 50)
    solver = CurrentDiffusionSolver(rho, R0=2.0, a=0.5, B0=1.0)

    Te = np.ones(50)
    ne = np.ones(50)
    j_bs = np.zeros(50)
    j_cd = np.zeros(50)

    psi_initial = solver.psi.copy()

    # Step by a large fraction of tau_R
    dt = 10.0
    for _ in range(10):
        solver.step(dt, Te, ne, 1.5, j_bs, j_cd)

    # The profile should have evolved
    assert not np.allclose(solver.psi, psi_initial)
    # Since Te is flat, resistivity is uniform, it should relax to a parabolic profile
    # which means q is relatively flat.
    q = q_from_psi(rho, solver.psi, 2.0, 0.5, 1.0)
    assert np.all(q > 0)


def test_conservation_and_steady_state():
    rho = np.linspace(0, 1, 50)
    solver = CurrentDiffusionSolver(rho, R0=2.0, a=0.5, B0=1.0)

    Te = np.ones(50)
    ne = np.ones(50)
    j_bs = np.zeros(50)
    j_cd = np.ones(50) * 1e5  # 100 kA/m^2 constant drive

    # Evolve to steady state
    dt = 1.0
    for _ in range(100):
        solver.step(dt, Te, ne, 1.5, j_bs, j_cd)

    # Check that it stopped evolving
    psi_new = solver.psi.copy()
    solver.step(dt, Te, ne, 1.5, j_bs, j_cd)
    assert np.allclose(psi_new, solver.psi, rtol=1e-3, atol=1e-3)


def test_bootstrap_steepens_q():
    rho = np.linspace(0, 1, 50)
    solver = CurrentDiffusionSolver(rho, R0=2.0, a=0.5, B0=1.0)

    Te = np.ones(50)
    ne = np.ones(50)
    j_bs = np.exp(-(((rho - 0.5) / 0.1) ** 2)) * 1e6  # Off-axis bootstrap
    j_cd = np.zeros(50)

    # Relax
    for _ in range(5):
        solver.step(10.0, Te, ne, 1.5, j_bs, j_cd)

    q = q_from_psi(rho, solver.psi, 2.0, 0.5, 1.0)
    # Due to off-axis current, q-profile should be hollow or steepened
    assert q[0] > 0


def test_coulomb_log_varies_with_temperature():
    """ln_Λ(1 keV) < ln_Λ(10 keV) — Wesson 2011 Eq. 2.12.4."""
    ne_19 = 1.0  # 10¹⁹ m⁻³, fixed
    ln_lam_1kev = coulomb_log(Te_keV=1.0, ne_19=ne_19)
    ln_lam_10kev = coulomb_log(Te_keV=10.0, ne_19=ne_19)
    assert ln_lam_1kev < ln_lam_10kev
    # Both must be physically reasonable: 10 < ln_Λ < 30
    assert 10.0 < ln_lam_1kev < 30.0
    assert 10.0 < ln_lam_10kev < 30.0


def test_current_diffusion_conserves_ip():
    """Total Ip approximately conserved over short time with no source.

    Ip ∝ ∫ j_∥ dA = ∫ (1/μ₀) ∇²ψ dA; with Dirichlet edge BC the integral
    can drift, but over short times (dt << τ_R) the change should be small.
    Reference: Jardin (2010), Ch. 8, Eq. 8.5.
    """
    rho = np.linspace(0, 1, 50)
    R0, a, B0 = 2.0, 0.5, 1.0
    solver = CurrentDiffusionSolver(rho, R0=R0, a=a, B0=B0)

    Te = np.ones(50) * 1.0  # 1 keV
    ne = np.ones(50)  # 10¹⁹ m⁻³
    j_bs = np.zeros(50)
    j_cd = np.zeros(50)

    # Ip proxy: ∑ (−∂ψ/∂ρ / (μ₀ a)) Δρ  (sign from toroidal current definition)
    drho = rho[1] - rho[0]

    def ip_proxy() -> float:
        dpsi = np.gradient(solver.psi, drho)
        return float(np.sum(-dpsi / (MU_0 * a)) * drho)

    ip0 = ip_proxy()
    tau_R = resistive_diffusion_time(a, neoclassical_resistivity(1.0, 1.0, 1.5, 0.1))
    # Step for 0.1 % of τ_R — negligible diffusion expected
    dt = 1e-3 * tau_R
    for _ in range(5):
        solver.step(dt, Te, ne, 1.5, j_bs, j_cd)

    ip1 = ip_proxy()
    # Relative change < 5 %
    assert abs(ip1 - ip0) / (abs(ip0) + 1e-30) < 0.05


def test_q_from_psi_singular_denom():
    """Singular flux gradients fall back to the neighbouring finite q value."""

    rho = np.linspace(0, 1, 50)
    # Flat psi near center means dpsi/drho ~ 0 at ρ=0
    psi = np.zeros(50)
    psi[0] = 0.0
    psi[1] = 1e-15
    psi[2] = 3e-15
    for i in range(3, 50):
        psi[i] = -float(i) * 0.01
    q = q_from_psi(rho, psi, R0=2.0, a=0.5, B0=1.0)
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0)


def test_psi_from_q_roundtrip():
    """psi_from_q inverts q_from_psi for a monotone positive q profile."""
    rho = np.linspace(0, 1, 50)
    q_input = 1.0 + 2.0 * rho**2
    psi_recon = psi_from_q(rho, q_input, R0=2.0, a=0.5, B0=1.0)
    assert psi_recon[-1] == 0.0
    q_back = q_from_psi(rho, psi_recon, R0=2.0, a=0.5, B0=1.0)
    assert np.allclose(q_back[5:], q_input[5:], rtol=0.1)


def test_psi_q_roundtrip_on_nonuniform_flux_grid():
    """q↔psi transforms must honour actual rho spacing, not assume uniform grids."""
    uniform = np.linspace(0.0, 1.0, 96)
    rho = uniform**1.6
    q_input = 1.0 + 1.5 * rho + 0.4 * rho**2

    psi_recon = psi_from_q(rho, q_input, R0=2.0, a=0.5, B0=1.0)
    q_back = q_from_psi(rho, psi_recon, R0=2.0, a=0.5, B0=1.0)

    assert psi_recon[-1] == pytest.approx(0.0, abs=1e-14)
    assert np.all(np.isfinite(q_back))
    assert np.all(q_back > 0.0)
    np.testing.assert_allclose(q_back[4:-4], q_input[4:-4], rtol=2.5e-2, atol=2.5e-2)


def test_nonuniform_q_from_psi_matches_analytic_constant_q():
    uniform = np.linspace(0.0, 1.0, 96)
    rho = uniform**1.7
    R0, a, B0 = 2.0, 0.5, 1.0
    psi = -(a**2) * B0 * rho**2 / (2.0 * R0)

    q = q_from_psi(rho, psi, R0, a, B0)

    np.testing.assert_allclose(q[4:-4], np.ones_like(q[4:-4]), rtol=2.5e-2, atol=2.5e-2)


def test_nonuniform_psi_from_q_matches_analytic_constant_q():
    uniform = np.linspace(0.0, 1.0, 96)
    rho = uniform**1.7
    R0, a, B0 = 2.0, 0.5, 1.0

    psi = psi_from_q(rho, np.ones_like(rho), R0, a, B0)
    expected = -(a**2) * B0 * (rho**2 - 1.0) / (2.0 * R0)

    np.testing.assert_allclose(psi, expected, rtol=2.5e-3, atol=2.5e-3)


def test_q_from_psi_rejects_invalid_flux_grid_domains():
    rho = np.linspace(0, 1, 5)
    psi = -(0.5**2) * rho**2 / (2.0 * 2.0)

    with pytest.raises(ValueError, match="rho"):
        q_from_psi(np.array([0.0, 0.5]), psi[:2], R0=2.0, a=0.5, B0=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        q_from_psi(rho[::-1], psi, R0=2.0, a=0.5, B0=1.0)
    with pytest.raises(ValueError, match="psi"):
        q_from_psi(rho, np.array([0.0, np.nan, 0.0, 0.0, 0.0]), R0=2.0, a=0.5, B0=1.0)
    with pytest.raises(ValueError, match="B0"):
        q_from_psi(rho, psi, R0=2.0, a=0.5, B0=0.0)


def test_psi_from_q_rejects_invalid_safety_factor_domains():
    rho = np.linspace(0, 1, 5)
    q = 1.0 + rho**2

    with pytest.raises(ValueError, match="q"):
        psi_from_q(rho, np.array([1.0, 1.2, 0.0, 1.6, 2.0]), R0=2.0, a=0.5, B0=1.0)
    with pytest.raises(ValueError, match="matching shape"):
        psi_from_q(rho, q[:-1], R0=2.0, a=0.5, B0=1.0)
    with pytest.raises(ValueError, match="rho"):
        psi_from_q(np.array([0.0, np.nan, 1.0, 1.5, 2.0]), q, R0=2.0, a=0.5, B0=1.0)


def test_neoclassical_resistivity_rejects_nonphysical_domain_values():
    with pytest.raises(ValueError, match="Te_keV must be finite and > 0"):
        neoclassical_resistivity(Te_keV=0.0, ne_19=1.0, Z_eff=1.5, epsilon=0.1)
    with pytest.raises(ValueError, match="ne_19 must be finite and > 0"):
        neoclassical_resistivity(Te_keV=1.0, ne_19=-1.0, Z_eff=1.5, epsilon=0.1)
    with pytest.raises(ValueError, match="epsilon must be finite and within"):
        neoclassical_resistivity(Te_keV=1.0, ne_19=1.0, Z_eff=1.5, epsilon=1.0)


def test_resistive_diffusion_time_rejects_nonphysical_inputs():
    with pytest.raises(ValueError, match="minor radius"):
        resistive_diffusion_time(a=0.0, eta=1e-8)
    with pytest.raises(ValueError, match="resistivity"):
        resistive_diffusion_time(a=2.0, eta=-1e-8)


def test_current_diffusion_step_rejects_invalid_time_and_profiles():
    rho = np.linspace(0, 1, 16)
    solver = CurrentDiffusionSolver(rho, R0=2.0, a=0.5, B0=1.0)
    Te = np.ones(16)
    ne = np.ones(16)
    j = np.zeros(16)

    with pytest.raises(ValueError, match="dt must be finite and > 0"):
        solver.step(0.0, Te, ne, 1.5, j, j)

    bad_te = Te.copy()
    bad_te[3] = -0.1
    with pytest.raises(ValueError, match="Te must contain finite positive values"):
        solver.step(1.0, bad_te, ne, 1.5, j, j)

    with pytest.raises(ValueError, match="j_cd must have shape"):
        solver.step(1.0, Te, ne, 1.5, j, np.zeros(15))


def test_current_diffusion_solver_rejects_nonuniform_or_non_normalised_rho():
    with pytest.raises(ValueError, match="uniform"):
        CurrentDiffusionSolver(np.array([0.0, 0.1, 0.4, 1.0]), R0=2.0, a=0.5, B0=1.0)

    with pytest.raises(ValueError, match="start at 0"):
        CurrentDiffusionSolver(np.linspace(0.1, 1.0, 8), R0=2.0, a=0.5, B0=1.0)

    with pytest.raises(ValueError, match="end at 1"):
        CurrentDiffusionSolver(np.linspace(0.0, 0.9, 8), R0=2.0, a=0.5, B0=1.0)
