# ──────────────────────────────────────────────────────────────────────
# SCPN Control — NTM Dynamics Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.core.ntm_dynamics import (
    NTMIslandDynamics,
    _ggj_delta_prime,
    bootstrap_from_local,
    find_rational_surfaces,
)


def test_find_rational_surfaces():
    rho = np.linspace(0, 1, 100)
    q = np.linspace(1.0, 3.5, 100)
    a = 1.0

    surfaces = find_rational_surfaces(q, rho, a, m_max=5, n_max=3)
    # Expected q values between 1.0 and 3.5:
    # 1/1=1, 3/2=1.5, 2/1=2, 5/2=2.5, 3/1=3
    # plus maybe others like 4/3=1.33, 5/3=1.66, 7/3(m_max=5 no), 4/2=2(skip duplicate), 5/4(n_max=3 no)
    q_vals = [s.q for s in surfaces]
    assert 1.5 in q_vals  # 3/2
    assert 2.0 in q_vals  # 2/1
    assert 3.0 in q_vals  # 3/1


def test_classical_stable_decay():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)
    # No drive, no ECCD -> classical Delta' < 0
    t, w = ntm.evolve(
        w0=0.05,
        t_span=(0.0, 1.0),
        dt=0.01,
        j_bs=0.0,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
    )
    assert w[-1] < w[0]


def test_bootstrap_drive_growth():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)
    # Large bootstrap current -> island grows
    t, w = ntm.evolve(
        w0=0.01,
        t_span=(0.0, 1.0),
        dt=0.01,
        j_bs=1e5,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
    )
    assert w[-1] > w[0]


def test_island_saturation():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)
    t, w = ntm.evolve(
        w0=0.01,
        t_span=(0.0, 50.0),
        dt=0.1,
        j_bs=5e4,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
    )
    dw_dt = ntm.dw_dt(w[-1], j_bs=5e4, j_phi=1e6, j_cd=0.0, eta=1e-7)
    assert abs(dw_dt) < 0.5
    assert w[-1] > 0.01


def test_eccd_stabilization():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)

    w0 = 0.05
    j_bs = 1e5
    j_phi = 1e6
    eta = 1e-7

    # Without ECCD, it should grow
    dw_dt_no_cd = ntm.dw_dt(w0, j_bs, j_phi, 0.0, eta)
    assert dw_dt_no_cd > 0

    # With strong ECCD, it should shrink
    dw_dt_with_cd = ntm.dw_dt(w0, j_bs, j_phi, 2e5, eta, d_cd=0.05)
    assert dw_dt_with_cd < 0


def test_eccd_misalignment():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)

    w0 = 0.05
    j_bs = 1e5
    j_phi = 1e6
    eta = 1e-7
    j_cd = 2e5

    # Perfect alignment (d_cd ~ w) -> strong stabilization
    dw_dt_aligned = ntm.dw_dt(w0, j_bs, j_phi, j_cd, eta, d_cd=0.05)

    # Misaligned / broad deposition (d_cd >> w) -> weaker stabilization
    dw_dt_broad = ntm.dw_dt(w0, j_bs, j_phi, j_cd, eta, d_cd=0.5)

    # Negative is more stable, so aligned should be more negative than broad
    assert dw_dt_aligned < dw_dt_broad


def test_polarization_threshold():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)

    # w_d determines the critical width for bootstrap drive to take over
    # If w0 is very small, polarization term dominates and keeps it stable
    w0_small = 1e-4
    j_bs = 1e5
    j_phi = 1e6
    eta = 1e-7

    dw_dt_small = ntm.dw_dt(w0_small, j_bs, j_phi, 0.0, eta, w_d=1e-3, w_pol=5e-4)
    assert dw_dt_small <= 0.0


def test_diamagnetic_stabilizes_small_islands():
    """w < w_d → diamagnetic term dominates; dw/dt must be negative.

    Sauter et al. 1997, Phys. Plasmas 4, 1654, Eq. 20:
      term_dia = -a4 * (w_d/w)² * (s_hat/s_hat_ref)
    For w < w_d this term drives dw/dt strongly negative.
    """
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0, s_hat=1.0)

    w_d = 5e-3  # banana width [m]
    w = 1e-3  # island width < w_d → diamagnetic dominates

    # Provide enough bootstrap to normally grow the island at larger w,
    # but the w < w_d condition makes term_dia >> term_bs.
    dw_dt_val = ntm.dw_dt(
        w,
        j_bs=1e5,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
        w_d=w_d,
        w_pol=1e-6,  # disable polarization to isolate diamagnetic effect
    )
    assert dw_dt_val < 0.0, f"Expected negative dw/dt for w < w_d; got {dw_dt_val}"


def test_ggj_delta_prime():
    """Unfavorable pressure gradient (D_R > 0) makes Δ'_GGJ negative → more stable.

    Glasser, Greene & Johnson 1975, Phys. Fluids 18, 875, Eq. 42:
      D_R = -2 μ_0 q² R₀² p' / (s² B_pol²)
    For p' < 0 (peaked profile) D_R > 0 → Δ'_GGJ = -m*D_R/(r_s*s) < 0.
    """
    m, r_s, s_hat = 2, 0.5, 1.5
    q, R0, B_pol = 2.0, 3.0, 0.3
    pressure_gradient = -5e4  # Pa/m (peaked, inward-directed gradient)

    delta_ggj = _ggj_delta_prime(m, r_s, s_hat, pressure_gradient, B_pol, q, R0)
    # D_R > 0 for p' < 0 → Δ'_GGJ < 0 (stabilizing)
    assert delta_ggj < 0.0, f"Expected Δ'_GGJ < 0 for unfavorable curvature; got {delta_ggj}"

    # Flat-gradient baseline → Δ'_GGJ = 0
    delta_flat = _ggj_delta_prime(m, r_s, s_hat, 0.0, B_pol, q, R0)
    assert delta_flat == 0.0


def test_sauter_bootstrap_vs_direct():
    """bootstrap_from_local and direct j_bs agree within 20%.

    For a typical ITER-like surface: ε=0.3, p'=-5e4 Pa/m, B_pol=0.3 T, L31≈0.3.
    Both paths use Sauter 1999, Eq. 14.
    """
    pressure_gradient = -5e4  # Pa/m
    epsilon = 0.3
    B_pol = 0.3  # T
    L31 = 0.3  # Sauter 1999, Eq. 14 coefficient (typical banana regime)

    j_bs_computed = bootstrap_from_local(pressure_gradient, epsilon, B_pol, L31)

    # Direct formula: j_bs = -ε^0.5 * p' * L31 / B_pol
    j_bs_direct = -np.sqrt(epsilon) * pressure_gradient * L31 / B_pol

    assert abs(j_bs_computed) > 0.0
    assert abs(j_bs_computed - j_bs_direct) / max(abs(j_bs_direct), 1.0) < 0.20


def test_seed_island_threshold():
    """Below the combined polarization + diamagnetic floor, islands don't grow.

    For w < max(w_pol, w_d), the stabilizing terms (term_pol + term_dia)
    overwhelm bootstrap drive, so dw/dt <= 0 even with active bootstrap.
    Sauter 1997, Eqs. 19–20.
    """
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0, s_hat=1.0)

    w_pol = 3e-3
    w_d = 2e-3
    # Island well below both thresholds
    w_seed = 5e-4

    dw_dt_val = ntm.dw_dt(
        w_seed,
        j_bs=1e5,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
        w_d=w_d,
        w_pol=w_pol,
    )
    assert dw_dt_val <= 0.0, f"Seed island should not grow below polarization/diamagnetic floor; dw/dt={dw_dt_val}"
