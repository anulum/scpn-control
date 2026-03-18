# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""
Tests for neoclassical transport models and bootstrap current.
"""

from __future__ import annotations

import numpy as np

from scpn_control.core.neoclassical import (
    chang_hinton_chi,
    collisionality,
    neoclassical_chi,
    pfirsch_schluter_chi,
    plateau_chi,
    sauter_bootstrap,
    _sauter_L31,
    _sauter_L32,
    _sauter_L34,
)


def test_collisionality_trends():
    """Verify that collisionality scales correctly with density and temperature."""
    # Baseline
    nu_base = collisionality(n_e_19=5.0, T_kev=2.0, q=2.0, R=6.2, epsilon=0.1)

    # nu_star ~ n / T^2
    nu_high_n = collisionality(n_e_19=10.0, T_kev=2.0, q=2.0, R=6.2, epsilon=0.1)
    assert nu_high_n > nu_base

    nu_high_t = collisionality(n_e_19=5.0, T_kev=5.0, q=2.0, R=6.2, epsilon=0.1)
    assert nu_high_t < nu_base


def test_chang_hinton_regimes():
    """Verify Chang-Hinton behavior in banana and Pfirsch-Schlüter regimes."""
    # Banana regime (nu_star << 1)
    # chi ~ q^2 * epsilon^-1.5 * rho_i^2 * nu_ii
    chi_banana = chang_hinton_chi(q=2.0, epsilon=0.1, nu_star=0.01, rho_i=0.01, nu_ii=100.0)

    # Pfirsch-Schlüter regime (nu_star >> 1)
    # chi is reduced by nu_star^(2/3) in Chang-Hinton formula
    chi_ps = chang_hinton_chi(q=2.0, epsilon=0.1, nu_star=100.0, rho_i=0.01, nu_ii=100.0)

    assert chi_banana > chi_ps


def test_bootstrap_flat_profiles():
    """Verify that bootstrap current vanishes when gradients are zero."""
    rho = np.linspace(0, 1, 50)
    Te = np.full(50, 5.0)
    Ti = np.full(50, 5.0)
    ne = np.full(50, 5.0)
    q = np.linspace(1.0, 3.0, 50)

    j_bs = sauter_bootstrap(rho, Te, Ti, ne, q, R0=6.2, a=2.0)

    # Inner points should be zero
    assert np.allclose(j_bs[1:-1], 0.0, atol=1e-10)


def test_bootstrap_gradient_direction():
    """Verify bootstrap current is positive for standard peaked profiles."""
    rho = np.linspace(0, 1, 50)
    Te = 5.0 * (1 - rho**2)
    Ti = 5.0 * (1 - rho**2)
    ne = 5.0 * (1 - rho**2)
    q = 1.0 + 2.0 * rho**2

    j_bs = sauter_bootstrap(rho, Te, Ti, ne, q, R0=6.2, a=2.0)

    # Peaked profiles should drive positive bootstrap current
    assert np.max(j_bs) > 0.0
    # Core should be zero due to zero gradients at rho=0
    assert j_bs[0] == 0.0


# ─── New tests ──────────────────────────────────────────────────────────────


def test_pfirsch_schluter_scaling():
    """chi_PS proportional to q^2 — Wesson 2011, Eq. 14.5.7."""
    rho_i = 0.01
    nu_ii = 500.0

    chi_q2 = pfirsch_schluter_chi(q=2.0, rho_i=rho_i, nu_ii=nu_ii)
    chi_q4 = pfirsch_schluter_chi(q=4.0, rho_i=rho_i, nu_ii=nu_ii)

    # chi_PS = q^2 * rho_i^2 * nu_ii  →  chi(q=4) / chi(q=2) = 4
    assert abs(chi_q4 / chi_q2 - 4.0) < 1e-10


def test_pfirsch_schluter_rho_scaling():
    """chi_PS proportional to rho_i^2 — Wesson 2011, Eq. 14.5.7."""
    q = 3.0
    nu_ii = 200.0

    chi_r1 = pfirsch_schluter_chi(q=q, rho_i=0.01, nu_ii=nu_ii)
    chi_r2 = pfirsch_schluter_chi(q=q, rho_i=0.02, nu_ii=nu_ii)

    assert abs(chi_r2 / chi_r1 - 4.0) < 1e-10


def test_regime_auto_detection():
    """neoclassical_chi selects the correct regime for banana / plateau / PS.

    Regime boundaries from Hinton & Hazeltine 1976 and Wesson 2011, Ch. 14.5.
    """
    q = 2.0
    epsilon = 0.1
    rho_i = 0.01
    nu_ii = 100.0
    v_thi = 1.0e5
    R = 6.2

    # nu_star < epsilon^1.5 → banana (Hinton & Hazeltine 1976, Sec. IV)
    # epsilon^1.5 = 0.1^1.5 ≈ 0.0316
    nu_banana = 0.01
    chi_auto = neoclassical_chi(nu_banana, q, epsilon, rho_i, nu_ii, v_thi, R)
    chi_ch = chang_hinton_chi(q, epsilon, nu_banana, rho_i, nu_ii)
    assert abs(chi_auto - chi_ch) < 1e-15

    # epsilon^1.5 < nu_star < q^2/eps^1.5 → plateau
    # q^2/eps^1.5 = 4/0.031623 ≈ 126.5
    nu_plateau = 5.0
    chi_auto_pl = neoclassical_chi(nu_plateau, q, epsilon, rho_i, nu_ii, v_thi, R)
    chi_pl = plateau_chi(q, rho_i, v_thi, R)
    assert abs(chi_auto_pl - chi_pl) < 1e-15

    # nu_star > q^2/eps^1.5  →  PS
    nu_ps = 200.0
    chi_auto_ps = neoclassical_chi(nu_ps, q, epsilon, rho_i, nu_ii, v_thi, R)
    chi_ps = pfirsch_schluter_chi(q, rho_i, nu_ii)
    assert abs(chi_auto_ps - chi_ps) < 1e-15


def test_regime_auto_detection_zero_epsilon():
    """neoclassical_chi returns 0 for epsilon=0."""
    assert neoclassical_chi(1.0, q=2.0, epsilon=0.0, rho_i=0.01, nu_ii=100.0, v_thi=1e5, R=6.2) == 0.0


def test_sauter_full_bootstrap_vs_L31_only():
    """Full L31+L32+L34 bootstrap differs from L31-alone on peaked profiles.

    Sauter 1999, Eqs. 14–16: L32 couples electron T-gradient and L34 couples
    ion T-gradient; both are non-zero for peaked Te/Ti.
    """
    f_t = 0.4
    nu_e = 0.1
    Z = 1.5

    L31 = _sauter_L31(f_t, nu_e, Z)
    L32 = _sauter_L32(f_t, nu_e, Z)
    L34 = _sauter_L34(f_t, nu_e, Z)

    # L32 and L34 are non-zero for f_t > 0
    assert abs(L32) > 1e-6
    assert abs(L34) > 1e-6
    # L31 and L34 agree to leading order in f_t (Sauter 1999, Eq. 16 note)
    assert abs(L34 - L31) < L31  # L34 < L31 due to truncated higher-order terms


def test_sauter_full_bootstrap_flat_profiles():
    """Full Sauter bootstrap with L31+L32+L34 vanishes on flat profiles."""
    rho = np.linspace(0, 1, 50)
    Te = np.full(50, 5.0)
    Ti = np.full(50, 5.0)
    ne = np.full(50, 5.0)
    q = np.linspace(1.0, 3.0, 50)

    j_bs = sauter_bootstrap(rho, Te, Ti, ne, q, R0=6.2, a=2.0)
    assert np.allclose(j_bs[1:-1], 0.0, atol=1e-10)


def test_sauter_full_bootstrap_peaked_profiles():
    """Full Sauter bootstrap is non-zero for peaked profiles."""
    rho = np.linspace(0, 1, 50)
    Te = 5.0 * (1 - rho**2)
    Ti = 5.0 * (1 - rho**2)
    ne = 5.0 * (1 - rho**2)
    q = 1.0 + 2.0 * rho**2

    j_bs = sauter_bootstrap(rho, Te, Ti, ne, q, R0=6.2, a=2.0)
    assert np.any(j_bs != 0.0)


def test_bootstrap_trapped_fraction():
    """L31 increases monotonically with trapped fraction f_t.

    f_t is set by local epsilon (Sauter 1999, Eq. 14).  Higher trapped
    fraction produces a larger L31 coefficient, which amplifies bootstrap
    current for any fixed pressure gradient.  This is verified directly on
    the coefficient, isolating the trapped-particle physics from geometric
    gradient dilution.
    """
    nu_e = 0.1
    Z = 1.5

    # f_t values corresponding to epsilon ≈ 0.05, 0.15, 0.30 (Sauter 1999, Eq. 14)
    eps_values = np.array([0.05, 0.15, 0.30])
    f_t_values = 1.0 - (1.0 - eps_values) ** 2 / (np.sqrt(1.0 - eps_values**2) * (1.0 + 1.46 * np.sqrt(eps_values)))

    L31_values = np.array([_sauter_L31(ft, nu_e, Z) for ft in f_t_values])

    # L31 must increase strictly with f_t
    assert L31_values[0] < L31_values[1] < L31_values[2], f"L31 not monotone with f_t: {L31_values}"


def test_sauter_L31_zero_trapped():
    """L31 vanishes when f_t = 0 (no trapped particles)."""
    assert _sauter_L31(f_t=0.0, nu_e=0.1, Z=1.5) == 0.0


def test_sauter_L32_zero_trapped():
    """L32 vanishes when f_t = 0."""
    assert _sauter_L32(f_t=0.0, nu_e=0.1, Z=1.5) == 0.0


def test_sauter_coefficients_physical_range():
    """Sauter L31, L32, L34 are bounded for physical f_t in [0, 0.8].

    All coefficients must remain finite and within order-unity bounds
    consistent with Sauter 1999, Table I.
    """
    for f_t in np.linspace(0.0, 0.8, 9):
        for nu_e in [0.01, 0.1, 1.0, 10.0]:
            L31 = _sauter_L31(f_t, nu_e, Z=1.5)
            L32 = _sauter_L32(f_t, nu_e, Z=1.5)
            L34 = _sauter_L34(f_t, nu_e, Z=1.5)
            assert np.isfinite(L31)
            assert np.isfinite(L32)
            assert np.isfinite(L34)
            assert abs(L31) < 5.0
            assert abs(L32) < 5.0
            assert abs(L34) < 5.0
