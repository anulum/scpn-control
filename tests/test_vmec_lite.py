# ──────────────────────────────────────────────────────────────────────
# SCPN Control — VMEC-lite Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.core.vmec_lite import (
    AxisymmetricTokamakBoundary,
    StellaratorBoundary,
    VMECLiteSolver,
)


def test_axisymmetric_tokamak_convergence():
    solver = VMECLiteSolver(n_s=11, m_pol=2, n_tor=0, n_fp=1)

    b_R, b_Z = AxisymmetricTokamakBoundary.from_parameters(R0=6.2, a=2.0, kappa=1.7, delta=0.33)
    solver.set_boundary(b_R, b_Z)

    p_prof = np.linspace(1e5, 0.0, 11)
    iota_prof = np.linspace(1.0, 0.3, 11)  # q = 1 to 3.3

    solver.set_profiles(p_prof, iota_prof)

    res = solver.solve(max_iter=50, tol=1e-3)

    assert res.iterations > 0
    # Just verify output structure
    assert res.R_mn.shape == (11, solver.basis.n_modes)


def test_stellarator_convergence():
    solver = VMECLiteSolver(n_s=11, m_pol=2, n_tor=1, n_fp=5)

    b_R, b_Z = StellaratorBoundary.w7x_standard()
    solver.set_boundary(b_R, b_Z)

    p_prof = np.linspace(5e4, 0.0, 11)
    iota_prof = np.ones(11) * 0.9  # W7-X is roughly shearless

    solver.set_profiles(p_prof, iota_prof)

    res = solver.solve(max_iter=50, tol=1e-3)

    assert res.iterations > 0
    assert res.Z_mn.shape == (11, solver.basis.n_modes)


def test_basis_evaluation():
    solver = VMECLiteSolver(m_pol=1, n_tor=0, n_fp=1)
    # Modes: (0,0), (1,0)
    coeffs = np.array([6.2, 2.0])

    theta = np.array([0.0, np.pi / 2, np.pi])
    zeta = np.zeros(3)

    R_val = solver.basis.evaluate(coeffs, theta, zeta, is_sin=False)

    assert np.isclose(R_val[0], 8.2)  # 6.2 + 2.0 * cos(0)
    assert np.isclose(R_val[1], 6.2)  # 6.2 + 2.0 * cos(pi/2)
    assert np.isclose(R_val[2], 4.2)  # 6.2 + 2.0 * cos(pi)


# ── New citation-driven tests ─────────────────────────────────────────


def test_vmec_spectral_reconstruction():
    """Hirshman & Whitson 1983: R and Z finite for all s, θ, ζ.

    R(s,θ,ζ) = Σ R_mn cos(mθ − nζ),  Z = Σ Z_mn sin(mθ − nζ)  [Eqs. 1–2].
    """
    solver = VMECLiteSolver(n_s=11, m_pol=2, n_tor=1, n_fp=5)
    b_R, b_Z = StellaratorBoundary.w7x_standard()
    solver.set_boundary(b_R, b_Z)
    solver.set_profiles(np.linspace(5e4, 0.0, 11), np.ones(11) * 0.9)
    res = solver.solve(max_iter=100, tol=1e-3)

    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    TH, ZE = np.meshgrid(theta, zeta, indexing="ij")

    for s_idx in range(solver.n_s):
        R_surf = solver.basis.evaluate(res.R_mn[s_idx], TH.ravel(), ZE.ravel(), is_sin=False)
        Z_surf = solver.basis.evaluate(res.Z_mn[s_idx], TH.ravel(), ZE.ravel(), is_sin=True)
        assert np.all(np.isfinite(R_surf)), f"R not finite at s_idx={s_idx}"
        assert np.all(np.isfinite(Z_surf)), f"Z not finite at s_idx={s_idx}"


def test_vmec_force_balance_residual():
    """Force balance residual |∇p − J×B| proxy stays bounded.

    Freidberg 2014, Ideal MHD, Ch. 3: ∇p = J × B in equilibrium.
    The solver minimises this residual; after convergence it must be small.
    """
    solver = VMECLiteSolver(n_s=11, m_pol=2, n_tor=0, n_fp=1)
    b_R, b_Z = AxisymmetricTokamakBoundary.from_parameters(R0=6.2, a=2.0, kappa=1.7, delta=0.33)
    solver.set_boundary(b_R, b_Z)
    solver.set_profiles(np.linspace(1e5, 0.0, 11), np.linspace(1.0, 0.3, 11))
    res = solver.solve(max_iter=200, tol=1e-4)
    # Residual must be finite and not worse than the starting point (inf)
    assert np.isfinite(res.force_residual)
    # Convergence to within a reasonable bound (not asking for full physics)
    assert res.force_residual < 1.0


def test_vmec_converges_with_loose_tolerance():
    solver = VMECLiteSolver(n_s=11, m_pol=1, n_tor=0, n_fp=1)
    b_R, b_Z = AxisymmetricTokamakBoundary.from_parameters(R0=6.2, a=2.0, kappa=1.7, delta=0.33)
    solver.set_boundary(b_R, b_Z)
    solver.set_profiles(np.zeros(11), np.linspace(1.0, 0.3, 11))
    res = solver.solve(max_iter=500, tol=1e10)
    assert res.converged is True
    assert res.force_residual < 1e10
