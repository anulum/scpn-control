# ──────────────────────────────────────────────────────────────────────
# SCPN Control — NTM Dynamics Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.core.ntm_dynamics import (
    NTMIslandDynamics,
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
    # Evolve for a long time to reach saturation
    t, w = ntm.evolve(
        w0=0.01,
        t_span=(0.0, 10.0),
        dt=0.1,
        j_bs=5e4,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
    )
    # Saturation means derivative goes to ~0
    dw_dt = ntm.dw_dt(w[-1], j_bs=5e4, j_phi=1e6, j_cd=0.0, eta=1e-7)
    assert abs(dw_dt) < 1e-2
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
