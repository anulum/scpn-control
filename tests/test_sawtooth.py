# ──────────────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from scpn_control.core.sawtooth import (
    PorcelliParams,
    SawtoothCycler,
    SawtoothMonitor,
    _alfven_time,
    _bussac_dW_mhd,
    _ion_diamagnetic_freq,
    _poloidal_beta_q1,
    _resistive_crit_dW,
    _resistive_time,
    kadomtsev_crash,
    porcelli_trigger,
)


def test_no_crash_q_above_1():
    rho = np.linspace(0, 1, 50)
    q = np.linspace(1.1, 3.0, 50)
    shear = np.linspace(0.1, 1.0, 50)

    monitor = SawtoothMonitor(rho)
    assert monitor.find_q1_radius(q) is None
    assert not monitor.check_trigger(q, shear)


def test_trigger_q_below_1():
    rho = np.linspace(0, 1, 50)
    q = np.linspace(0.8, 3.0, 50)
    # q=1 is at rho ~ 0.09

    # Low shear -> no trigger
    shear_low = np.full(50, 0.05)
    monitor = SawtoothMonitor(rho, s_crit=0.1)
    assert not monitor.check_trigger(q, shear_low)

    # High shear -> trigger
    shear_high = np.full(50, 0.2)
    assert monitor.check_trigger(q, shear_high)


def test_kadomtsev_crash():
    rho = np.linspace(0, 1, 100)
    q = 0.8 + 2.0 * rho**2  # q=1 at rho=sqrt(0.1) ~ 0.316
    T = 5.0 * (1 - rho**2) ** 2
    n = 1.0 * (1 - rho**2)

    T_new, n_new, q_new, rho_1, rho_mix = kadomtsev_crash(rho, T, n, q, R0=2.0, a=0.5)

    assert rho_1 > 0.0
    assert rho_mix > rho_1

    # T inside q=1 should be flattened
    idx_1 = int(rho_1 * 100)
    assert np.allclose(T_new[0], T_new[idx_1 - 1])
    assert T_new[0] < T[0]  # Core T drops

    # q should be reset to 1.01 inside mixing radius
    idx_mix = int(rho_mix * 100)
    assert np.allclose(q_new[0], 1.01)
    assert np.allclose(q_new[idx_mix - 1], 1.01)


def test_density_conservation():
    rho = np.linspace(0, 1, 100)
    q = 0.8 + 2.0 * rho**2
    T = 5.0 * (1 - rho**2)
    n = 2.0 * (1 - rho**2)

    T_new, n_new, q_new, rho_1, rho_mix = kadomtsev_crash(rho, T, n, q, R0=2.0, a=0.5)

    def total_particles(n_prof):
        return trapezoid(n_prof * rho, rho)

    N_before = total_particles(n)
    N_after = total_particles(n_new)

    assert np.isclose(N_before, N_after, rtol=1e-2)


def test_sawtooth_cycler():
    rho = np.linspace(0, 1, 50)
    q = 0.8 + 2.0 * rho**2
    shear = np.full(50, 0.2)
    T = np.linspace(5.0, 0.1, 50)
    n = np.linspace(1.0, 0.1, 50)

    cycler = SawtoothCycler(rho, R0=2.0, a=0.5, s_crit=0.1)
    event = cycler.step(0.1, q, shear, T, n)

    assert event is not None
    assert event.T_drop > 0
    assert event.seed_energy > 0

    # After crash, q is reset to 1.01, so trigger should be false
    event2 = cycler.step(0.1, q, shear, T, n)
    assert event2 is None


# ---------------------------------------------------------------------------
# Porcelli 1996 trigger tests
# ---------------------------------------------------------------------------


def _base_profile(N: int = 100):
    """Shared analytic profile: q=1 at rho~0.316, parabolic T, flat-ish n."""
    rho = np.linspace(0, 1, N)
    q = 0.8 + 2.0 * rho**2  # q=1 at rho=sqrt(0.1)≈0.316
    shear = 2.0 * rho / np.maximum(q, 0.1)  # s = (rho/q) dq/drho
    return rho, q, shear


def test_porcelli_condition1_triggers():
    """High β_p1, low shear → Condition 1 fires.

    High pressure inside q=1 makes β_p1 large → δW_MHD << 0.
    Low shear keeps δW_crit small → -δW_crit is only mildly negative.
    Porcelli et al. 1996, PPCF 38, 2163, Condition 1.
    """
    rho, q, shear = _base_profile()
    # Peaked T profile → high β_p1
    T = 20.0 * (1 - rho**2) ** 2
    n = 5.0 * np.ones_like(rho)  # flat density, 5×10¹⁹ m⁻³

    # Low-resistivity, low-field params → δW_crit is small
    params = PorcelliParams(B_T=1.0, B_pol=0.1, T_i_keV=2.0, eta=1e-8, v_A=5e6)
    result = porcelli_trigger(rho, T, n, q, shear, R0=3.0, a=1.0, params=params)
    assert result, "High β_p1 profile should trigger Condition 1"


def test_porcelli_stable():
    """Low β_p1, high shear → Porcelli trigger stays false.

    Low-pressure flat profile → δW_MHD ~ 0.
    High shear → s1² dominates Bussac formula → δW_MHD > 0 (stabilised).
    Bussac et al. 1975, Phys. Rev. Lett. 35, 1638, Eq. (3).
    """
    rho = np.linspace(0, 1, 100)
    # Build q so q=1 is crossed with large shear (steep q slope at crossing)
    q = 0.5 + 3.0 * rho**2  # q=1 at rho≈0.408, dq/drho large
    shear = 6.0 * rho / np.maximum(q, 0.1)

    # Very low temperature → negligible pressure → β_p1 ≈ 0
    T = 0.01 * np.ones_like(rho)
    n = 0.1 * np.ones_like(rho)

    params = PorcelliParams(B_T=5.0, B_pol=1.0, T_i_keV=0.01, eta=1e-6, v_A=1e7)
    result = porcelli_trigger(rho, T, n, q, shear, R0=3.0, a=1.0, params=params)
    assert not result, "Low-pressure, high-shear profile should not trigger"


def test_porcelli_vs_shear_consistency():
    """Both trigger models fire for a strongly unstable profile.

    A strongly peaked, low-shear profile should be flagged by both
    the shear threshold (Kadomtsev proxy) and the Porcelli conditions.
    """
    rho = np.linspace(0, 1, 100)
    q = 0.8 + 2.0 * rho**2
    shear_val = np.full(100, 0.5)  # above s_crit=0.1
    T = 15.0 * (1 - rho**2) ** 2
    n = 3.0 * np.ones_like(rho)

    monitor = SawtoothMonitor(rho, s_crit=0.1)

    shear_fires = monitor.check_trigger(q, shear_val, trigger_model="shear")
    porcelli_fires = monitor.check_trigger(
        q,
        shear_val,
        trigger_model="porcelli",
        T=T,
        n=n,
        R0=3.0,
        a=1.0,
        porcelli_params=PorcelliParams(B_T=2.0, B_pol=0.2, T_i_keV=2.0, eta=1e-7, v_A=5e6),
    )

    assert shear_fires, "Shear model should fire for this profile"
    assert porcelli_fires, "Porcelli model should also fire for this profile"


def test_crash_energy_conservation():
    """Total thermal energy is conserved within 5% across a Kadomtsev crash.

    Kadomtsev reconnection redistributes but does not destroy energy.
    Kadomtsev 1975, Sov. J. Plasma Phys. 1, 389.
    """
    rho = np.linspace(0, 1, 200)
    q = 0.8 + 2.0 * rho**2
    T = 5.0 * (1 - rho**2) ** 2
    n = 2.0 * (1 - rho**2)

    R0, a = 3.0, 1.0

    def _total_energy(Te, ne):
        # W = 3/2 ∫ n T dV,  dV = 4π² R₀ a² ρ dρ
        energy_dens = 1.5 * (ne * 1e19) * (Te * 1e3 * 1.602e-19)
        vol_element = 4.0 * np.pi**2 * R0 * a**2 * rho
        return float(trapezoid(energy_dens * vol_element, rho))

    W_before = _total_energy(T, n)
    T_new, n_new, _, _, _ = kadomtsev_crash(rho, T, n, q, R0, a)
    W_after = _total_energy(T_new, n_new)

    assert W_before > 0.0
    assert abs(W_after - W_before) / W_before < 0.05, (
        f"Energy not conserved: before={W_before:.4g} J, after={W_after:.4g} J, "
        f"rel_err={abs(W_after - W_before) / W_before:.3%}"
    )


# ---------------------------------------------------------------------------
# Coverage gap closers
# ---------------------------------------------------------------------------


def test_poloidal_beta_q1_small_idx():
    """_poloidal_beta_q1 returns 0 when the q=1 surface is at rho < 2 grid pts (line 64)."""
    rho = np.linspace(0, 1, 100)
    T = 5.0 * np.ones_like(rho)
    n = 5.0 * np.ones_like(rho)
    # rho_1 = 0.005 → searchsorted gives idx < 2
    result = _poloidal_beta_q1(rho, T, n, B_pol=0.3, rho_1=0.005)
    assert result == 0.0


def test_porcelli_no_q1_surface():
    """porcelli_trigger returns False when q > 1 everywhere (line 160)."""
    rho = np.linspace(0, 1, 50)
    q = np.linspace(1.1, 3.0, 50)
    shear = np.full(50, 0.5)
    T = 10.0 * (1 - rho**2)
    n = 5.0 * np.ones_like(rho)
    assert not porcelli_trigger(rho, T, n, q, shear, R0=3.0, a=1.0)


def test_porcelli_condition2_ideal_kink():
    """Condition 2 (ideal kink) fires with high beta_p1 and very low resistivity.

    Porcelli 1996, Eq. 14 — ideal-kink trigger with very low eta so
    Condition 1 (resistive) does not fire first but Condition 2 does.
    """
    rho = np.linspace(0, 1, 100)
    q = 0.8 + 2.0 * rho**2
    shear = 2.0 * rho / np.maximum(q, 0.1)
    T = 30.0 * (1 - rho**2) ** 2
    n = 8.0 * np.ones_like(rho)

    params = PorcelliParams(B_T=1.0, B_pol=0.05, T_i_keV=5.0, eta=1e-10, v_A=5e6)
    result = porcelli_trigger(rho, T, n, q, shear, R0=3.0, a=1.0, params=params)
    assert result, "High beta_p1 + low eta should fire Condition 2 (ideal kink)"


def test_resistive_crit_dW_formula():
    """_resistive_crit_dW matches Porcelli 1996, Eq. 18 hand computation."""
    eps_1 = 0.1
    omega_star_i = 1e4
    tau_R = 1e3
    s1 = 0.5
    expected = (np.pi**2 * eps_1**4 * omega_star_i**2 * tau_R) / (12.0 * s1)
    assert _resistive_crit_dW(eps_1, omega_star_i, tau_R, s1) == pytest.approx(expected)


def test_alfven_time_formula():
    """_alfven_time matches tau_A = R / (v_A * s1)."""
    assert _alfven_time(R0=3.0, v_A=1e7, s1=0.5) == pytest.approx(3.0 / (1e7 * 0.5))


def test_alfven_time_near_zero_shear():
    """_alfven_time clamps s1 to 1e-6 to avoid division by zero (line 80)."""
    tau = _alfven_time(R0=3.0, v_A=1e7, s1=0.0)
    assert np.isfinite(tau)
    assert tau > 0


def test_resistive_time_formula():
    """_resistive_time matches tau_R = mu_0 * r1^2 / (1.22 * eta)."""
    from scpn_control.core.sawtooth import MU_0

    r1 = 0.3
    eta = 1e-7
    expected = MU_0 * r1**2 / (1.22 * eta)
    assert _resistive_time(r1, eta) == pytest.approx(expected)


def test_ion_diamagnetic_freq_formula():
    """_ion_diamagnetic_freq matches omega_*i = k_theta * T_i / (e * B * r1)."""
    from scpn_control.core.sawtooth import E_CHARGE

    Ti_keV = 2.0
    Ti_J = Ti_keV * 1e3 * E_CHARGE
    k_theta = 1.0 / 0.3
    B_T = 2.0
    r1 = 0.3
    expected = k_theta * Ti_J / (E_CHARGE * B_T * r1)
    assert _ion_diamagnetic_freq(k_theta, Ti_keV, B_T, r1) == pytest.approx(expected)


def test_bussac_dW_mhd_formula():
    """_bussac_dW_mhd = -0.5*(beta_p1 - s1^2)."""
    assert _bussac_dW_mhd(beta_p1=1.0, s1=0.5) == pytest.approx(-0.5 * (1.0 - 0.25))
    # Stable case: s1^2 > beta_p1 → dW > 0
    assert _bussac_dW_mhd(beta_p1=0.1, s1=1.0) > 0


def test_kadomtsev_crash_no_q1():
    """kadomtsev_crash returns copies unchanged when q > 1 everywhere (line 283)."""
    rho = np.linspace(0, 1, 50)
    q = np.linspace(1.5, 3.0, 50)
    T = 5.0 * np.ones_like(rho)
    n = 2.0 * np.ones_like(rho)
    T_new, n_new, q_new, rho_1, rho_mix = kadomtsev_crash(rho, T, n, q, R0=2.0, a=0.5)
    np.testing.assert_array_equal(T_new, T)
    np.testing.assert_array_equal(n_new, n)
    assert rho_1 == 0.0
    assert rho_mix == 0.0


def test_kadomtsev_crash_idx_mix_zero():
    """kadomtsev_crash handles degenerate case where mixing radius is at edge (line 303)."""
    rho = np.linspace(0, 1, 50)
    # q slightly below 1 at the very core only, mixing radius at rho=0 effectively
    q = 0.99 + 3.0 * rho**2
    T = 5.0 * (1 - rho**2)
    n = 2.0 * np.ones_like(rho)
    T_new, n_new, q_new, rho_1, rho_mix = kadomtsev_crash(rho, T, n, q, R0=2.0, a=0.5)
    assert np.all(np.isfinite(T_new))
    assert np.all(np.isfinite(n_new))


def test_monitor_check_trigger_porcelli_missing_profiles():
    """SawtoothMonitor.check_trigger raises ValueError when porcelli model lacks T/n."""
    rho = np.linspace(0, 1, 50)
    q = 0.8 + 2.0 * rho**2
    shear = np.full(50, 0.5)
    monitor = SawtoothMonitor(rho)
    with pytest.raises(ValueError, match="porcelli trigger requires"):
        monitor.check_trigger(q, shear, trigger_model="porcelli")


def test_monitor_find_q1_equal_values():
    """find_q1_radius handles q1 == q2 at crossing (line 211)."""
    rho = np.linspace(0, 1, 50)
    q = 0.8 + 2.0 * rho**2
    # Force q values exactly equal at the crossing
    idx = np.where(np.diff(np.sign(q - 1.0)))[0][0]
    q[idx] = 1.0
    q[idx + 1] = 1.0
    monitor = SawtoothMonitor(rho)
    rho_1 = monitor.find_q1_radius(q)
    assert rho_1 is not None
    assert rho_1 == pytest.approx(rho[idx])


def test_monitor_check_trigger_shear_edge_cases():
    """check_trigger handles idx==0 and idx>=len(rho) shear interpolation (lines 246-248)."""
    rho = np.linspace(0, 1, 50)
    # q=1 right at the first grid point
    q = np.ones_like(rho)
    q[0] = 0.99
    q[1:] = 1.01 + np.arange(49) * 0.01
    shear = np.full(50, 0.2)
    monitor = SawtoothMonitor(rho, s_crit=0.1)
    result = monitor.check_trigger(q, shear)
    # Should not crash regardless of the interpolation edge case
    assert isinstance(result, bool)
