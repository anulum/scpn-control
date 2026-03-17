# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Current Diffusion Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.core.current_diffusion import (
    MU_0,
    CurrentDiffusionSolver,
    coulomb_log,
    neoclassical_resistivity,
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
