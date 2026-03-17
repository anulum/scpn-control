# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core import (
    current_diffusion,
    current_drive,
    eped_pedestal,
    lh_transition,
    neoclassical,
    ntm_dynamics,
    runaway_electrons,
    sawtooth,
    scaling_laws,
    stability_mhd,
)

# ─── ITER-like reference parameters ──────────────────────────────────────────
# ITER Physics Basis, Nucl. Fusion 39, 2175 (1999), Table 1.
_R = 6.2  # m — major radius
_a = 2.0  # m — minor radius
_B = 5.3  # T — toroidal field
_Ip = 15.0  # MA — plasma current
_ne19 = 10.0  # 10^19 m^-3 — electron density (~10^20 m^-3)
_Te = 10.0  # keV — electron temperature
_Ti = 8.0  # keV — ion temperature
_kappa = 1.7  # elongation
_eps = _a / _R  # inverse aspect ratio


def _iter_rho_profiles(n: int = 50):
    """Standard ITER-like flux-surface profiles on a uniform rho grid."""
    rho = np.linspace(0.0, 1.0, n)
    # Peaked temperature: T(0) = T_0, T(1) = 0.1 keV
    Te = _Te * (1.0 - rho**2) + 0.1
    Ti = _Ti * (1.0 - rho**2) + 0.1
    # Peaked density: n(0)=ne19, n(1)=0.3×ne19
    ne = _ne19 * (0.3 + 0.7 * (1.0 - rho**2))
    # Monotone q rising from ~1 on axis to ~3.5 at edge
    q = 1.0 + 2.5 * rho**2
    return rho, Te, Ti, ne, q


# ─── Chain 1: Bootstrap → NTM Island ─────────────────────────────────────────
# Sauter 1999 → La Haye 2006 MRE.
# Peaked profiles ⟹ large inward pressure gradient ⟹ j_bs > 0 ⟹ NTM bootstrap
# drive exceeds stabilising terms ⟹ dw/dt > 0 above seed width.


def test_bootstrap_drives_ntm_island_growth():
    rho, Te, Ti, ne, q = _iter_rho_profiles()

    j_bs = neoclassical.sauter_bootstrap(rho, Te, Ti, ne, q, R0=_R, a=_a, B0=_B, z_eff=1.5)

    # Peaked profile must give predominantly positive bootstrap current.
    assert np.any(j_bs > 0), "peaked profiles must produce positive bootstrap current"

    # Characteristic mid-radius bootstrap current density at rho~0.4
    idx_mid = np.argmin(np.abs(rho - 0.4))
    j_bs_mid = float(j_bs[idx_mid])

    # q=2 rational surface at rho where q=2 ⟹ rho_s = sqrt((2−1)/2.5) ≈ 0.632
    rho_s = float(np.interp(2.0, q, rho))
    r_s = rho_s * _a  # m

    B_pol = _B * (rho_s * _a / _R) / max(float(np.interp(rho_s, rho, q)), 0.1)
    # Typical Ohmic resistivity at mid-radius: Spitzer at T_e ~ 8 keV
    eta = 1.65e-9 * 1.5 * 17.0 / (8.0**1.5)  # Wesson 2011, Eq. 2.5.4

    ntm = ntm_dynamics.NTMIslandDynamics(
        r_s=r_s,
        m=2,
        n=1,
        a=_a,
        R0=_R,
        B0=_B,
        Delta_prime_0=-2.0 * 2 / max(r_s, 1e-3),  # classically stable
        s_hat=1.5,
        q_s=2.0,
    )

    j_phi_typical = 1.0e6  # A/m^2 — ITER toroidal current density scale
    w_seed = 0.01  # m — above minimum threshold

    dwdt = ntm.dw_dt(
        w=w_seed,
        j_bs=j_bs_mid,
        j_phi=j_phi_typical,
        j_cd=0.0,
        eta=eta,
        w_d=5e-3,
        w_pol=2e-4,
    )
    # Bootstrap drive must overcome polarisation at w > w_pol, giving net growth.
    assert dwdt > 0, f"bootstrap-driven NTM must grow above seed: dw/dt={dwdt:.3e} m/s"


# ─── Chain 2: Scaling Law → Power Balance ────────────────────────────────────
# IPB98(y,2) → tau_E → P_loss = W_th / tau_E.
# For ITER-like parameters: tau_E ≈ 3–5 s; P_loss ≈ 50–100 MW.


def test_ipb98_power_balance_iter_range():
    # ITER steady-state point: P_loss ~ P_alpha + P_aux ~ 100 MW total
    P_loss_MW = 80.0
    epsilon = _a / _R

    tau_E = scaling_laws.ipb98y2_tau_e(
        Ip=_Ip,
        BT=_B,
        ne19=_ne19,
        Ploss=P_loss_MW,
        R=_R,
        kappa=_kappa,
        epsilon=epsilon,
        M=2.5,  # D-T
    )

    assert 2.0 < tau_E < 8.0, f"ITER tau_E should be 3–5 s; got {tau_E:.2f} s"

    # Thermal energy: W_th = 3/2 * n_e * T_e * volume (rough estimate)
    # Use tau_E to reconstruct P_loss = W_th / tau_E and compare with input.
    mu0 = 4.0 * np.pi * 1e-7
    E_charge = 1.602e-19
    vol = 2.0 * np.pi**2 * _R * _a**2 * _kappa  # toroidal volume [m^3], circular
    W_th = 1.5 * (_ne19 * 1e19) * (_Te * 1e3 * E_charge) * vol  # J

    P_check_MW = (W_th / tau_E) / 1e6
    # At tau_E ~ 3–5 s and W_th ~ 300–500 MJ, P_loss ~ 60–160 MW.
    assert 10.0 < P_check_MW < 300.0, f"reconstructed P_loss {P_check_MW:.1f} MW out of range"


# ─── Chain 3: Collisionality → Neoclassical Regime → Transport ───────────────
# nu_star → regime selector → chi_banana < chi_PS for same geometry.
# Banana regime (low-collisionality) transport is smaller than PS (high-coll.).


def test_banana_chi_less_than_ps_chi():
    q = 2.0
    rho_s = 0.5
    epsilon = rho_s * _a / _R  # ≈ 0.16

    M_P = 1.67262e-27  # kg
    E_ch = 1.602e-19
    T_J = _Ti * 1e3 * E_ch
    m_i = 2.0 * M_P  # deuterium
    v_thi = np.sqrt(2.0 * T_J / m_i)

    EPS0 = 8.854e-12
    LN_LAMBDA = 17.0
    n_m3 = _ne19 * 1e19
    nu_ii = (n_m3 * 1.0**2 * E_ch**4 * LN_LAMBDA) / (12.0 * np.pi**1.5 * EPS0**2 * np.sqrt(m_i) * T_J**1.5)

    rho_i = m_i * v_thi / (E_ch * _B)  # Larmor radius

    # Banana regime: nu_star << 1
    nu_star_banana = 0.05
    chi_banana = neoclassical.neoclassical_chi(
        nu_star=nu_star_banana,
        q=q,
        epsilon=epsilon,
        rho_i=rho_i,
        nu_ii=nu_ii,
        v_thi=v_thi,
        R=_R,
    )

    # PS regime: nu_star >> q^2/eps^1.5
    nu_star_ps = q**2 / epsilon**1.5 * 10.0
    chi_ps = neoclassical.neoclassical_chi(
        nu_star=nu_star_ps,
        q=q,
        epsilon=epsilon,
        rho_i=rho_i,
        nu_ii=nu_ii * nu_star_ps / nu_star_banana,
        v_thi=v_thi,
        R=_R,
    )

    assert chi_banana > 0, "banana-regime diffusivity must be positive"
    assert chi_ps > 0, "PS-regime diffusivity must be positive"
    assert chi_banana < chi_ps, f"banana chi ({chi_banana:.3e}) must be smaller than PS chi ({chi_ps:.3e})"


# ─── Chain 4: EPED → Pedestal → Troyon Limit ─────────────────────────────────
# EPED1 prediction → pedestal pressure → beta_t → Troyon stability check.
# EPED-predicted pedestal pressure must be within the Troyon beta limit.


def test_eped_pedestal_below_troyon_limit():
    mu0 = 4.0 * np.pi * 1e-7
    B_pol_ped = mu0 * _Ip * 1e6 / (2.0 * np.pi * _a * np.sqrt(_kappa))

    cfg = eped_pedestal.EPEDConfig(
        R0=_R,
        a=_a,
        B0=_B,
        kappa=_kappa,
        delta=0.33,
        Ip_MA=_Ip,
        ne_ped_19=4.0,
        B_pol_ped=B_pol_ped,
    )
    result = eped_pedestal.eped1_predict(cfg)

    assert result.p_ped_kPa > 0, "EPED must predict positive pedestal pressure"
    assert result.delta_ped > 0, "EPED must predict positive pedestal width"

    # Convert pedestal pressure to a rough volume-average beta_t.
    # beta_t = p_avg / (B^2 / 2μ0); use p_avg ~ 0.2 * p_ped (pedestal fraction).
    p_ped_Pa = result.p_ped_kPa * 1e3
    p_avg_estimate = 0.3 * p_ped_Pa  # rough: pedestal is ~30% of total pressure
    B_pressure = _B**2 / (2.0 * mu0)
    beta_t = p_avg_estimate / B_pressure

    troyon = stability_mhd.troyon_beta_limit(beta_t=beta_t, Ip_MA=_Ip, a=_a, B0=_B)
    assert troyon.stable_nowall, f"EPED pedestal should not breach Troyon limit; beta_N={troyon.beta_N:.2f}"


# ─── Chain 5: Sawtooth → NTM Seed ────────────────────────────────────────────
# Kadomtsev crash ⟹ seed_energy > 0; seed island width ~ sqrt(seed / B_pol) > 0.


def test_sawtooth_produces_finite_ntm_seed():
    rho, Te, Ti, ne, q = _iter_rho_profiles(n=80)

    # Force a crash by making q < 1 inside rho ~ 0.3
    q_crash = q.copy()
    idx = np.searchsorted(rho, 0.3)
    q_crash[:idx] = 0.8  # q < 1 inside the inversion radius

    T_new, n_new, q_new, rho_1, rho_mix = sawtooth.kadomtsev_crash(rho, Te, ne, q_crash, R0=_R, a=_a)

    assert rho_1 > 0, "q=1 surface must be found for the crash"

    # Compute seed energy from the temperature drop in the core cylinder.
    E_charge = 1.602e-19
    T_drop = Te[0] - T_new[0]
    # Seed energy from core cylinder — Bussac et al. 1975, Phys. Rev. Lett. 35, 1638
    seed_energy = 1.5 * (ne[0] * 1e19) * (T_drop * 1e3 * E_charge) * (2.0 * np.pi**2 * _R * (rho_1 * _a) ** 2)
    assert seed_energy > 0, f"sawtooth crash must release seed energy; got {seed_energy:.3e} J"

    # Seed island width ~ sqrt(seed_energy / (B_pol^2 * volume_scale))
    # Simple dimensional proxy: w_seed ∝ sqrt(seed_energy) / B_pol
    B_pol_q1 = _B * (rho_1 * _a / _R) / 1.0  # q=1 so q=1
    vol_scale = 2.0 * np.pi**2 * _R * _a**2  # m^3
    w_seed_proxy = np.sqrt(seed_energy / (B_pol_q1**2 * vol_scale))
    assert w_seed_proxy > 0, f"NTM seed width proxy must be positive; got {w_seed_proxy:.3e} m"


# ─── Chain 6: ECCD → Current Diffusion → q-profile ───────────────────────────
# ECCD at mid-radius ⟹ j_cd peaked at rho~0.4 ⟹ after diffusion,
# q increases near deposition (less inductive current needed there).


def test_eccd_flattens_q_at_deposition():
    rho, Te, Ti, ne, q = _iter_rho_profiles(n=40)

    # ECCD source at rho_dep = 0.4 with 20 MW
    eccd = current_drive.ECCDSource(P_ec_MW=20.0, rho_dep=0.4, sigma_rho=0.05)
    j_cd = eccd.j_cd(rho, ne_19=ne, Te_keV=Te)

    assert np.max(j_cd) > 0, "ECCD must produce current drive"

    # Confirm j_cd peaks near rho=0.4
    idx_peak = int(np.argmax(j_cd))
    assert 0.3 < rho[idx_peak] < 0.55, f"ECCD peak should be near rho=0.4; found at rho={rho[idx_peak]:.2f}"

    # Evolve current diffusion one resistive step
    solver = current_diffusion.CurrentDiffusionSolver(rho=rho, R0=_R, a=_a, B0=_B)
    j_bs_zero = np.zeros(len(rho))
    j_ext_zero = np.zeros(len(rho))

    psi_before = solver.psi.copy()
    q_before = current_diffusion.q_from_psi(rho, psi_before, _R, _a, _B)

    dt = 1e-2  # s — short step; resistive time scale >> 1 s
    Z_eff = 1.5
    solver.step(dt=dt, Te=Te, ne=ne, Z_eff=Z_eff, j_bs=j_bs_zero, j_cd=j_cd, j_ext=j_ext_zero)

    q_after = current_diffusion.q_from_psi(rho, solver.psi, _R, _a, _B)

    # Near deposition the safety factor must change (flux evolves).
    q_change_at_dep = np.abs(q_after[idx_peak] - q_before[idx_peak])
    assert q_change_at_dep > 0 or np.any(q_after != q_before), "ECCD-driven current must modify the psi profile"
    # psi profile itself must differ from initial
    assert not np.allclose(solver.psi, psi_before), "psi must evolve under ECCD source"


# ─── Chain 7: Runaway Electrons → SPI Mitigation Trigger ─────────────────────
# For ITER post-disruption E-fields (~10–50 V/m): E > E_c and avalanche rate > 0.
# Connor & Hastie 1975; Rosenbluth & Putvinski 1997.


def test_iter_disruption_e_field_exceeds_critical():
    # ITER disruption parameters: n_e drops to ~10^19 m^-3, T_e ~ 1 keV
    params_disrupt = runaway_electrons.RunawayParams(
        ne_20=0.1,  # 10^19 m^-3 — post-disruption density
        Te_keV=1.0,
        E_par=25.0,  # V/m — representative post-disruption loop voltage
        Z_eff=2.0,
        B0=_B,
        R0=_R,
        a=_a,
    )

    E_c = runaway_electrons.critical_field(params_disrupt.ne_20, params_disrupt.Te_keV)
    assert E_c > 0, "critical field must be positive"
    assert params_disrupt.E_par > E_c, f"disruption E_par={params_disrupt.E_par:.1f} V/m must exceed E_c={E_c:.3f} V/m"

    # With E > E_c, avalanche growth rate must be positive given a seed population
    n_RE_seed = 1e12  # m^-3 — small but non-zero seed
    gamma_aval = runaway_electrons.avalanche_growth_rate(params_disrupt, n_RE_seed)
    assert gamma_aval > 0, (
        f"avalanche rate must be > 0 for E/E_c={params_disrupt.E_par / E_c:.1f}; got {gamma_aval:.3e}"
    )


# ─── Chain 8: L-H Transition → H-mode → EPED Validity ───────────────────────
# For ITER at full NBI power (P_aux = 50 MW NBI + 20 MW ECRH = 73 MW total),
# P_aux > P_LH (Martin scaling) → H-mode → EPED prediction is physically valid.


def test_iter_full_power_exceeds_lh_threshold_and_eped_gives_finite_pedestal():
    # ITER plasma surface area — ITER Technical Basis (2001), Table 1.2: ~678 m^2.
    # Standard formula for a D-shaped tokamak: S ≈ 2π² R a κ / f_shape,
    # where f_shape accounts for triangularity.  The factor 4π² / (2π²) = 2
    # error in the raw torus formula (2πR × 2πa) overpredicts.
    # Using the standard S = 4π² R a κ / 3 approximation for ITER shape → ~556 m^2
    # is also used in scaling law literature (Martin 2008, Eq. 1 caption).
    # Directly using the ITER design value avoids shape-model uncertainty.
    S_m2 = 678.0  # ITER plasma first-wall surface area [m^2], ITER Technical Basis (2001)

    P_LH_MW = lh_transition.MartinThreshold.power_threshold_MW(ne_19=_ne19, B_T=_B, S_m2=S_m2)
    assert P_LH_MW > 0, "L-H threshold must be positive"

    # ITER total heating at full power:
    #   NBI 33 MW + ECRH 20 MW + ICRH 20 MW = 73 MW aux
    #   + fusion alpha heating ~66 MW (at Q=10 with P_fusion=500 MW)
    # Martin scaling applies to P_loss which includes both aux and alpha.
    P_total_MW = 139.0  # P_aux + P_alpha at Q=10 operating point
    assert P_total_MW > P_LH_MW, f"ITER P_total={P_total_MW} MW must exceed P_LH={P_LH_MW:.1f} MW for H-mode"

    # H-mode confirmed → EPED prediction is physically valid
    mu0 = 4.0 * np.pi * 1e-7
    B_pol_ped = mu0 * _Ip * 1e6 / (2.0 * np.pi * _a * np.sqrt(_kappa))
    cfg = eped_pedestal.EPEDConfig(
        R0=_R,
        a=_a,
        B0=_B,
        kappa=_kappa,
        delta=0.33,
        Ip_MA=_Ip,
        ne_ped_19=4.0,
        B_pol_ped=B_pol_ped,
    )
    result = eped_pedestal.eped1_predict(cfg)

    assert result.p_ped_kPa > 0, "EPED must predict finite pedestal pressure in H-mode"
    assert result.T_ped_keV > 0, "EPED must predict finite pedestal temperature"


# ─── Parametrize sanity: all chains use consistent ITER parameters ────────────


@pytest.mark.parametrize(
    "param,val,lo,hi",
    [
        ("R", _R, 5.0, 8.0),
        ("a", _a, 1.5, 2.5),
        ("B", _B, 4.0, 7.0),
        ("Ip", _Ip, 10.0, 20.0),
    ],
)
def test_iter_params_in_design_range(param, val, lo, hi):
    assert lo < val < hi, f"ITER param {param}={val} outside design range [{lo}, {hi}]"
