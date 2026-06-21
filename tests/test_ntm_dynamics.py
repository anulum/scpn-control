# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NTM Dynamics Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.ntm_dynamics import (
    NTMController,
    NTMIslandDynamics,
    _ggj_delta_prime,
    bootstrap_from_local,
    eccd_stabilization_factor,
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

    # With sufficient ECCD, it should shrink
    dw_dt_with_cd = ntm.dw_dt(w0, j_bs, j_phi, 2e5, eta, d_cd=0.05)
    assert dw_dt_with_cd < 0


def test_eccd_misalignment():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)

    w0 = 0.05
    j_bs = 1e5
    j_phi = 1e6
    eta = 1e-7
    j_cd = 2e5

    # Perfect alignment (d_cd ~ w) -> larger stabilization
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
    For w < w_d this term drives dw/dt negative.
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
    """Local full-Sauter bootstrap agrees with profile Sauter evaluation."""
    from scpn_control.core.neoclassical import sauter_bootstrap

    rho = np.linspace(0.2, 0.8, 7)
    q = 1.0 + 2.0 * rho**2
    Te = 8.0 * (1.0 - 0.6 * rho**2)
    Ti = 7.0 * (1.0 - 0.5 * rho**2)
    ne = 9.0 * (1.0 - 0.4 * rho**2)

    i = 3
    dr = rho[1] - rho[0]
    j_local = bootstrap_from_local(
        ne_19=float(ne[i]),
        Te_keV=float(Te[i]),
        Ti_keV=float(Ti[i]),
        q=float(q[i]),
        rho=float(rho[i]),
        R0=6.2,
        a=2.0,
        B0=5.3,
        z_eff=1.5,
        dne_dr=float(np.gradient(ne, dr)[i] / 2.0),
        dTe_dr=float(np.gradient(Te, dr)[i] / 2.0),
        dTi_dr=float(np.gradient(Ti, dr)[i] / 2.0),
    )

    j_profile = sauter_bootstrap(rho, Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3, z_eff=1.5)[i]
    assert np.isfinite(j_local)
    assert np.isfinite(j_profile)
    assert abs(j_profile) > 0.0
    assert abs(j_local - j_profile) / max(abs(j_profile), 1.0) < 0.25


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


def test_eccd_stabilization_factor_edge():
    """Non-positive island width or deposition width gives zero ECCD coupling."""
    assert eccd_stabilization_factor(0.05, 0.0) == 0.0
    assert eccd_stabilization_factor(0.0, 0.05) == 0.0
    assert eccd_stabilization_factor(-0.1, 0.05) == 0.0
    assert eccd_stabilization_factor(0.05, -0.1) == 0.0
    with pytest.raises(ValueError, match="finite"):
        eccd_stabilization_factor(float("nan"), 0.05)


def test_find_rational_surfaces_equal_q():
    """Flat q profiles do not create interpolated rational-surface crossings."""
    rho = np.linspace(0, 1, 10)
    q = np.ones(10) * 2.0  # flat q = 2/1
    surfaces = find_rational_surfaces(q, rho, a=1.0)
    assert len(surfaces) == 0


def test_ggj_delta_prime_zero_shear():
    """Degenerate shear or poloidal field gives no finite GGJ correction."""
    assert _ggj_delta_prime(2, 0.5, 1e-7, -5e4, 0.3, 2.0, 3.0) == 0.0
    assert _ggj_delta_prime(2, 0.5, 1.0, -5e4, 1e-11, 2.0, 3.0) == 0.0


def test_bootstrap_from_local_edge():
    """bootstrap_from_local fails closed for invalid state and returns zero on-axis."""
    with pytest.raises(ValueError, match="B0"):
        bootstrap_from_local(
            ne_19=8.0,
            Te_keV=5.0,
            Ti_keV=5.0,
            q=2.0,
            rho=0.5,
            R0=6.2,
            a=2.0,
            B0=0.0,
            z_eff=1.5,
            dne_dr=-1.0,
            dTe_dr=-1.0,
            dTi_dr=-1.0,
        )
    assert (
        bootstrap_from_local(
            ne_19=8.0,
            Te_keV=5.0,
            Ti_keV=5.0,
            q=2.0,
            rho=0.0,
            R0=6.2,
            a=2.0,
            B0=5.3,
            z_eff=1.5,
            dne_dr=-1.0,
            dTe_dr=-1.0,
            dTi_dr=-1.0,
        )
        == 0.0
    )
    assert (
        bootstrap_from_local(
            ne_19=8.0,
            Te_keV=5.0,
            Ti_keV=5.0,
            q=2.0,
            rho=-0.1,
            R0=6.2,
            a=2.0,
            B0=5.3,
            z_eff=1.5,
            dne_dr=-1.0,
            dTe_dr=-1.0,
            dTi_dr=-1.0,
        )
        == 0.0
    )


def test_dw_dt_with_rho_theta_i_override():
    """dw_dt uses rho_theta_i * sqrt(2*beta_pol) as the banana-width override."""
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)
    dw1 = ntm.dw_dt(0.02, j_bs=1e5, j_phi=1e6, j_cd=0.0, eta=1e-7, w_d=1e-3)
    dw2 = ntm.dw_dt(
        0.02,
        j_bs=1e5,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
        rho_theta_i=0.01,
        beta_pol=0.5,
    )
    # w_d override should produce a different result
    assert dw1 != dw2


def test_find_rational_surfaces_rejects_multidimensional_profiles() -> None:
    """Rational-surface search requires one-dimensional q and rho profiles."""
    two_d = np.ones((2, 5))
    with pytest.raises(ValueError, match="one-dimensional"):
        find_rational_surfaces(two_d, two_d, a=1.0)


def test_bootstrap_from_local_returns_zero_for_vanishing_poloidal_field() -> None:
    """A vanishingly small toroidal field collapses B_pol below the floor, giving zero drive."""
    j_bs = bootstrap_from_local(
        ne_19=8.0,
        Te_keV=5.0,
        Ti_keV=5.0,
        q=2.0,
        rho=0.5,
        R0=6.2,
        a=2.0,
        B0=1e-12,
        z_eff=1.5,
        dne_dr=-1.0,
        dTe_dr=-1.0,
        dTi_dr=-1.0,
    )
    assert j_bs == 0.0


def test_dw_dt_returns_zero_below_seed_floor() -> None:
    """An island narrower than the 1e-6 m numerical floor has no growth rate."""
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)
    assert ntm.dw_dt(1e-7, j_bs=1e5, j_phi=1e6, j_cd=0.0, eta=1e-7) == 0.0


def test_dw_dt_applies_ggj_correction_with_pressure_gradient() -> None:
    """A finite pressure gradient with B_pol adds the GGJ resistive-interchange term to Δ'."""
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0, s_hat=1.5, q_s=2.0)
    without_ggj = ntm.dw_dt(0.02, j_bs=1e5, j_phi=1e6, j_cd=0.0, eta=1e-7)
    with_ggj = ntm.dw_dt(
        0.02,
        j_bs=1e5,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
        pressure_gradient=-5e4,
        B_pol=0.3,
    )
    # The GGJ term shifts Δ', so the growth rate must change.
    assert with_ggj != without_ggj


def test_ntm_controller_deactivation():
    """NTMController deactivates when the island width drops below target."""
    ctrl = NTMController(w_onset=0.02, w_target=0.005)

    # Activate
    power = ctrl.step(w=0.03, rho_rs=0.5, max_power=20.0)
    assert ctrl.active is True
    assert power == 20.0

    # Still active — w above target
    power = ctrl.step(w=0.01, rho_rs=0.5, max_power=20.0)
    assert ctrl.active is True

    # Drop below target — deactivate
    power = ctrl.step(w=0.004, rho_rs=0.5, max_power=20.0)
    assert ctrl.active is False
    assert power == 0.0


def test_find_rational_surfaces_rejects_nonphysical_inputs() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    q = np.linspace(1.0, 2.0, 5)

    with pytest.raises(ValueError, match="equal length"):
        find_rational_surfaces(q[:-1], rho, a=1.0)
    with pytest.raises(ValueError, match="q values"):
        find_rational_surfaces(np.zeros(5), rho, a=1.0)
    with pytest.raises(ValueError, match="finite"):
        find_rational_surfaces(np.array([1.0, np.nan, 2.0, 2.5, 3.0]), rho, a=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        find_rational_surfaces(q, rho[::-1], a=1.0)
    with pytest.raises(ValueError, match="a must be positive"):
        find_rational_surfaces(q, rho, a=0.0)
    with pytest.raises(ValueError, match="m_max and n_max"):
        find_rational_surfaces(q, rho, a=1.0, m_max=0)


def test_ggj_delta_prime_rejects_nonphysical_geometry() -> None:
    with pytest.raises(ValueError, match="m"):
        _ggj_delta_prime(0, 0.5, 1.0, -5e4, 0.3, 2.0, 3.0)
    with pytest.raises(ValueError, match="r_s"):
        _ggj_delta_prime(2, 0.0, 1.0, -5e4, 0.3, 2.0, 3.0)
    with pytest.raises(ValueError, match="finite"):
        _ggj_delta_prime(2, 0.5, float("nan"), -5e4, 0.3, 2.0, 3.0)


def test_bootstrap_from_local_rejects_nonphysical_state() -> None:
    with pytest.raises(ValueError, match="ne_19"):
        bootstrap_from_local(
            ne_19=0.0,
            Te_keV=5.0,
            Ti_keV=5.0,
            q=2.0,
            rho=0.5,
            R0=6.2,
            a=2.0,
            B0=5.3,
            z_eff=1.5,
            dne_dr=-1.0,
            dTe_dr=-1.0,
            dTi_dr=-1.0,
        )
    with pytest.raises(ValueError, match="z_eff"):
        bootstrap_from_local(
            ne_19=8.0,
            Te_keV=5.0,
            Ti_keV=5.0,
            q=2.0,
            rho=0.5,
            R0=6.2,
            a=2.0,
            B0=5.3,
            z_eff=float("nan"),
            dne_dr=-1.0,
            dTe_dr=-1.0,
            dTi_dr=-1.0,
        )


def test_ntm_island_dynamics_rejects_nonphysical_constructor_inputs() -> None:
    with pytest.raises(ValueError, match="r_s"):
        NTMIslandDynamics(r_s=0.0, m=2, n=1, a=1.0, R0=3.0, B0=2.0)
    with pytest.raises(ValueError, match="r_s"):
        NTMIslandDynamics(r_s=1.1, m=2, n=1, a=1.0, R0=3.0, B0=2.0)
    with pytest.raises(ValueError, match="a must be smaller"):
        NTMIslandDynamics(r_s=0.5, m=2, n=1, a=3.0, R0=3.0, B0=2.0)
    with pytest.raises(ValueError, match="m and n"):
        NTMIslandDynamics(r_s=0.5, m=0, n=1, a=1.0, R0=3.0, B0=2.0)
    with pytest.raises(ValueError, match="q_s"):
        NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0, q_s=0.0)
    with pytest.raises(ValueError, match="Delta_prime_0"):
        NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0, Delta_prime_0=float("nan"))


def test_ntm_dw_dt_rejects_nonphysical_inputs() -> None:
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)

    with pytest.raises(ValueError, match="j_phi"):
        ntm.dw_dt(0.02, j_bs=1e5, j_phi=0.0, j_cd=0.0, eta=1e-7)
    with pytest.raises(ValueError, match="eta"):
        ntm.dw_dt(0.02, j_bs=1e5, j_phi=1e6, j_cd=0.0, eta=0.0)
    with pytest.raises(ValueError, match="w_d"):
        ntm.dw_dt(0.02, j_bs=1e5, j_phi=1e6, j_cd=0.0, eta=1e-7, w_d=-1.0)
    with pytest.raises(ValueError, match="rho_theta_i"):
        ntm.dw_dt(0.02, j_bs=1e5, j_phi=1e6, j_cd=0.0, eta=1e-7, rho_theta_i=-1.0)
    with pytest.raises(ValueError, match="supplied together"):
        ntm.dw_dt(0.02, j_bs=1e5, j_phi=1e6, j_cd=0.0, eta=1e-7, rho_theta_i=1e-3)
    with pytest.raises(ValueError, match="finite"):
        ntm.dw_dt(float("nan"), j_bs=1e5, j_phi=1e6, j_cd=0.0, eta=1e-7)


def test_ntm_evolve_rejects_nonphysical_time_inputs() -> None:
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)

    with pytest.raises(ValueError, match="t_span"):
        ntm.evolve(0.01, (1.0, 0.0), 0.01, j_bs=0.0, j_phi=1e6, j_cd=0.0, eta=1e-7)
    with pytest.raises(ValueError, match="dt"):
        ntm.evolve(0.01, (0.0, 1.0), 0.0, j_bs=0.0, j_phi=1e6, j_cd=0.0, eta=1e-7)
    with pytest.raises(ValueError, match="w0"):
        ntm.evolve(0.0, (0.0, 1.0), 0.01, j_bs=0.0, j_phi=1e6, j_cd=0.0, eta=1e-7)
    with pytest.raises(ValueError, match="t_end"):
        ntm.evolve(0.01, (0.0, float("nan")), 0.01, j_bs=0.0, j_phi=1e6, j_cd=0.0, eta=1e-7)


def test_ntm_controller_rejects_nonphysical_inputs() -> None:
    with pytest.raises(ValueError, match="w_target"):
        NTMController(w_onset=0.01, w_target=0.02)

    ctrl = NTMController(w_onset=0.02, w_target=0.005)
    with pytest.raises(ValueError, match="w must be non-negative"):
        ctrl.step(w=-1.0, rho_rs=0.5)
    with pytest.raises(ValueError, match="rho_rs"):
        ctrl.step(w=0.01, rho_rs=1.5)
    with pytest.raises(ValueError, match="max_power"):
        ctrl.step(w=0.01, rho_rs=0.5, max_power=-1.0)
    with pytest.raises(ValueError, match="finite"):
        ctrl.step(w=float("nan"), rho_rs=0.5)
