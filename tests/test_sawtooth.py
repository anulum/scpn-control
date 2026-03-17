# ──────────────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from scpn_control.core.sawtooth import (
    PorcelliParams,
    SawtoothCycler,
    SawtoothMonitor,
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
    q = 0.8 + 2.0 * rho**2       # q=1 at rho=sqrt(0.1)≈0.316
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
    n = 5.0 * np.ones_like(rho)   # flat density, 5×10¹⁹ m⁻³

    # Low-resistivity, low-field params → δW_crit is small
    params = PorcelliParams(
        B_T=1.0, B_pol=0.1, T_i_keV=2.0, eta=1e-8, v_A=5e6
    )
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
    q = 0.5 + 3.0 * rho**2         # q=1 at rho≈0.408, dq/drho large
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
        porcelli_params=PorcelliParams(
            B_T=2.0, B_pol=0.2, T_i_keV=2.0, eta=1e-7, v_A=5e6
        ),
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
        f"rel_err={abs(W_after-W_before)/W_before:.3%}"
    )
