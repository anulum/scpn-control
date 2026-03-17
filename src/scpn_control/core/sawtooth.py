# ──────────────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

# Physical constants
MU_0: float = 4.0 * np.pi * 1e-7  # H/m
E_CHARGE: float = 1.602176634e-19  # C  — CODATA 2018

# Porcelli 1996 geometric coefficient for ideal-kink term
# Porcelli et al. 1996, PPCF 38, 2163, Eq. (14) — c_rho ~ 0.1 (Table 1)
_C_RHO: float = 0.1


@dataclass
class SawtoothEvent:
    crash_time: float
    rho_1: float
    rho_mix: float
    T_drop: float
    seed_energy: float


# ---------------------------------------------------------------------------
# Bussac / Porcelli MHD energy integral
# ---------------------------------------------------------------------------


def _bussac_dW_mhd(beta_p1: float, s1: float) -> float:
    """Bussac normalised MHD energy integral for the (1,1) mode.

    δW_MHD = -½(β_p1 - s1²)

    Bussac et al. 1975, Phys. Rev. Lett. 35, 1638, Eq. (3).
    Sign convention: δW_MHD < 0 → unstable.
    s1² term is the magnetic-shear stabilisation.
    """
    return -0.5 * (beta_p1 - s1**2)


def _poloidal_beta_q1(
    rho: np.ndarray,
    T: np.ndarray,
    n: np.ndarray,
    B_pol: float,
    rho_1: float,
) -> float:
    """Volume-averaged β_p inside the q=1 surface.

    β_p1 = 2μ_0 <p(ρ<ρ_1)> / B_pol²
    Porcelli et al. 1996, PPCF 38, 2163, Eq. (5).
    """
    idx = np.searchsorted(rho, rho_1)
    if idx < 2:
        return 0.0
    rho_in = rho[:idx]
    # p in Pa: n in 10¹⁹ m⁻³, T in keV
    p_in = (n[:idx] * 1e19) * (T[:idx] * 1e3 * E_CHARGE)
    vol_int = trapezoid(p_in * rho_in, rho_in)
    vol_tot = trapezoid(rho_in, rho_in)
    p_avg = vol_int / vol_tot if vol_tot > 0.0 else p_in[0]
    return float(2.0 * MU_0 * p_avg / B_pol**2)


def _alfven_time(R0: float, v_A: float, s1: float) -> float:
    """Alfvén transit time at q=1 surface.

    τ_A = R / (v_A · s1)
    Porcelli et al. 1996, PPCF 38, 2163, Eq. (10).
    """
    return R0 / (v_A * max(s1, 1e-6))


def _resistive_crit_dW(eps_1: float, omega_star_i: float, tau_R: float, s1: float) -> float:
    """Critical δW for the resistive internal-kink trigger (Condition 1).

    δW_crit = π² ε_1⁴ ω_*i² τ_R / (12 s_1)
    Porcelli et al. 1996, PPCF 38, 2163, Eq. (18).
    """
    return (np.pi**2 * eps_1**4 * omega_star_i**2 * tau_R) / (12.0 * max(s1, 1e-6))


def _resistive_time(r1_m: float, eta: float) -> float:
    """Resistive diffusion time at q=1 surface.

    τ_R = μ_0 r_1² / (1.22 η)
    Porcelli et al. 1996, PPCF 38, 2163, Eq. (16).
    1.22 is the Spitzer resistivity numerical prefactor.
    """
    return MU_0 * r1_m**2 / (1.22 * max(eta, 1e-12))


def _ion_diamagnetic_freq(k_theta: float, Ti_keV: float, B_T: float, r1_m: float) -> float:
    """Ion diamagnetic frequency ω_*i at q=1 surface.

    ω_*i = k_θ T_i / (e B r_1)
    Porcelli et al. 1996, PPCF 38, 2163, Eq. (11).
    k_theta is the poloidal wavenumber (≈ 1/r1 for m=1).
    """
    Ti_J = Ti_keV * 1e3 * E_CHARGE
    return k_theta * Ti_J / (E_CHARGE * B_T * r1_m)


# ---------------------------------------------------------------------------
# Porcelli trigger
# ---------------------------------------------------------------------------


@dataclass
class PorcelliParams:
    """Parameters for the Porcelli 1996 trigger criterion.

    All SI except T_i_keV (keV) and n (10¹⁹ m⁻³).
    """

    B_T: float = 2.0  # Tesla — toroidal field on axis
    B_pol: float = 0.3  # Tesla — poloidal field at q=1
    T_i_keV: float = 1.0  # keV  — ion temperature at q=1
    eta: float = 1e-7  # Ω·m  — plasma resistivity at q=1
    v_A: float = 1e7  # m/s  — Alfvén speed (B/√(μ₀ρ_mass))


def porcelli_trigger(
    rho: np.ndarray,
    T: np.ndarray,
    n: np.ndarray,
    q: np.ndarray,
    shear: np.ndarray,
    R0: float,
    a: float,
    params: PorcelliParams | None = None,
) -> bool:
    """Porcelli 1996 sawtooth trigger: fires when any of three conditions holds.

    Condition 1 — resistive internal kink:
        δW_MHD < -δW_crit   (Porcelli 1996, Eq. 18)
    Condition 2 — ideal internal kink:
        δW_MHD < -½ c_ρ² ε_1 ω_*i (ω_*i τ_A)  (Porcelli 1996, Eq. 14)
    Condition 3 — fast-ion fishbone:
        inactive (no fast-ion β input)

    Bussac 1975, Phys. Rev. Lett. 35, 1638 — δW_MHD formula.
    Porcelli et al. 1996, PPCF 38, 2163 — trigger conditions.
    """
    if params is None:
        params = PorcelliParams()

    monitor = SawtoothMonitor(rho)
    rho_1 = monitor.find_q1_radius(q)
    if rho_1 is None or rho_1 < 1e-6:
        return False

    # Shear at q=1 surface
    s1 = float(np.interp(rho_1, rho, shear))

    # β_p1 from volume average inside q=1
    beta_p1 = _poloidal_beta_q1(rho, T, n, params.B_pol, rho_1)

    dW = _bussac_dW_mhd(beta_p1, s1)

    r1_m = rho_1 * a  # q=1 radius in metres
    eps_1 = r1_m / R0  # inverse aspect ratio at q=1

    k_theta = 1.0 / max(r1_m, 1e-4)  # m=1 poloidal wavenumber
    omega_star = _ion_diamagnetic_freq(k_theta, params.T_i_keV, params.B_T, r1_m)
    tau_R = _resistive_time(r1_m, params.eta)
    tau_A = _alfven_time(R0, params.v_A, s1)

    # Condition 1 — resistive kink (Porcelli 1996, Eq. 18)
    dW_crit_res = _resistive_crit_dW(eps_1, omega_star, tau_R, s1)
    if dW < -dW_crit_res:
        return True

    # Condition 2 — ideal kink (Porcelli 1996, Eq. 14)
    dW_ideal_thresh = -0.5 * _C_RHO**2 * eps_1 * omega_star * (omega_star * tau_A)
    # Condition 3 — fast-ion fishbone: inactive without fast-ion β
    return dW < dW_ideal_thresh


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class SawtoothMonitor:
    def __init__(self, rho: np.ndarray, s_crit: float = 0.1):
        self.rho = rho
        self.s_crit = s_crit

    def find_q1_radius(self, q: np.ndarray) -> float | None:
        """Return normalised radius where q=1 by linear interpolation."""
        if q[0] >= 1.0 or np.min(q) >= 1.0:
            return None
        q_diff = q - 1.0
        crossings = np.where(np.diff(np.sign(q_diff)))[0]
        if len(crossings) == 0:
            return None
        idx = crossings[0]
        r1, r2 = self.rho[idx], self.rho[idx + 1]
        q1, q2 = q[idx], q[idx + 1]
        if q1 == q2:
            return float(r1)
        frac = (1.0 - q1) / (q2 - q1)
        return float(r1 + frac * (r2 - r1))

    def check_trigger(
        self,
        q: np.ndarray,
        shear: np.ndarray,
        trigger_model: str = "shear",
        T: np.ndarray | None = None,
        n: np.ndarray | None = None,
        R0: float = 3.0,
        a: float = 1.0,
        porcelli_params: PorcelliParams | None = None,
    ) -> bool:
        """Return True when the sawtooth trigger fires.

        trigger_model="shear" (default) — shear threshold (Kadomtsev-like).
        trigger_model="porcelli"        — Porcelli 1996 three-condition test.

        Kadomtsev 1975, Sov. J. Plasma Phys. 1, 389.
        Porcelli et al. 1996, PPCF 38, 2163.
        """
        if trigger_model == "porcelli":
            if T is None or n is None:
                raise ValueError("porcelli trigger requires T and n arrays")
            return porcelli_trigger(self.rho, T, n, q, shear, R0, a, porcelli_params)

        # Default: shear threshold at q=1
        rho_1 = self.find_q1_radius(q)
        if rho_1 is None:
            return False
        idx = np.searchsorted(self.rho, rho_1)
        if idx == 0:
            s1 = shear[0]
        elif idx >= len(self.rho):
            s1 = shear[-1]
        else:
            r1, r2 = self.rho[idx - 1], self.rho[idx]
            s_1, s_2 = shear[idx - 1], shear[idx]
            if r1 == r2:
                s1 = s_1
            else:
                frac = (rho_1 - r1) / (r2 - r1)
                s1 = s_1 + frac * (s_2 - s_1)
        return bool(s1 > self.s_crit)


# ---------------------------------------------------------------------------
# Kadomtsev crash
# ---------------------------------------------------------------------------


def kadomtsev_crash(
    rho: np.ndarray, T: np.ndarray, n: np.ndarray, q: np.ndarray, R0: float, a: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Apply Kadomtsev full-reconnection crash.

    Returns (T_new, n_new, q_new, rho_1, rho_mix).

    Helical flux proxy: dψ*/dρ = ρ(1/q − 1), integrated from axis.
    rho_mix is where ψ* returns to zero outside the q=1 surface.
    Inside rho_mix, T, n are volume-averaged and q is reset to 1.01.

    Kadomtsev 1975, Sov. J. Plasma Phys. 1, 389.
    Porcelli et al. 1996, PPCF 38, 2163, §2.
    """
    monitor = SawtoothMonitor(rho)
    rho_1 = monitor.find_q1_radius(q)

    if rho_1 is None:
        return T.copy(), n.copy(), q.copy(), 0.0, 0.0

    integrand = rho * (1.0 / np.maximum(q, 1e-6) - 1.0)
    psi_star = np.zeros_like(rho)
    for i in range(1, len(rho)):
        psi_star[i] = psi_star[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * (rho[i] - rho[i - 1])

    idx_1 = np.searchsorted(rho, rho_1)
    rho_mix = rho[-1]

    for i in range(idx_1, len(rho)):
        if psi_star[i] <= 0.0:
            if i > 0 and psi_star[i - 1] > 0.0:
                frac = psi_star[i - 1] / (psi_star[i - 1] - psi_star[i])
                rho_mix = rho[i - 1] + frac * (rho[i] - rho[i - 1])
            else:
                rho_mix = rho[i]
            break

    idx_mix = np.searchsorted(rho, rho_mix)
    if idx_mix == 0:
        return T.copy(), n.copy(), q.copy(), rho_1, rho_mix

    rho_inner = rho[:idx_mix]

    def _volume_average(prof: np.ndarray) -> float:
        # dV ∝ ρ dρ in circular cross-section
        vol_int = trapezoid(prof[:idx_mix] * rho_inner, rho_inner)
        vol_tot = trapezoid(rho_inner, rho_inner)
        return float(vol_int / vol_tot) if vol_tot > 0.0 else float(prof[0])

    T_mix = _volume_average(T)
    n_mix = _volume_average(n)

    T_new = T.copy()
    n_new = n.copy()
    q_new = q.copy()

    T_new[:idx_mix] = T_mix
    n_new[:idx_mix] = n_mix
    q_new[:idx_mix] = 1.01  # Kadomtsev post-crash q profile flattened to q=1

    return T_new, n_new, q_new, rho_1, rho_mix


# ---------------------------------------------------------------------------
# Cycler
# ---------------------------------------------------------------------------


class SawtoothCycler:
    """Tracks and triggers sawtooth crashes."""

    def __init__(
        self,
        rho: np.ndarray,
        R0: float,
        a: float,
        s_crit: float = 0.1,
        trigger_model: str = "shear",
        porcelli_params: PorcelliParams | None = None,
    ):
        self.rho = rho
        self.R0 = R0
        self.a = a
        self.trigger_model = trigger_model
        self.porcelli_params = porcelli_params
        self.monitor = SawtoothMonitor(rho, s_crit)
        self.time = 0.0

    def step(
        self,
        dt: float,
        q: np.ndarray,
        shear: np.ndarray,
        T: np.ndarray,
        n: np.ndarray,
    ) -> SawtoothEvent | None:
        self.time += dt

        fired = self.monitor.check_trigger(
            q,
            shear,
            trigger_model=self.trigger_model,
            T=T,
            n=n,
            R0=self.R0,
            a=self.a,
            porcelli_params=self.porcelli_params,
        )

        if not fired:
            return None

        T_core_old = T[0]

        def _plasma_energy(Te: np.ndarray, ne: np.ndarray) -> float:
            # W = 3/2 ∫ n T dV,  dV = 4π² R₀ a² ρ dρ  (toroidal volume element)
            energy_dens = 1.5 * (ne * 1e19) * (Te * 1e3 * E_CHARGE)
            vol_element = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.rho
            return float(trapezoid(energy_dens * vol_element, self.rho))

        W_before = _plasma_energy(T, n)
        T_new, n_new, q_new, rho_1, rho_mix = kadomtsev_crash(self.rho, T, n, q, self.R0, self.a)
        W_after = _plasma_energy(T_new, n_new)

        np.copyto(T, T_new)
        np.copyto(n, n_new)
        np.copyto(q, q_new)

        T_drop = T_core_old - T[0]
        # Seed energy for NTM seeding: pressure drop in the q=1 cylinder
        # Bussac et al. 1975, Phys. Rev. Lett. 35, 1638
        core_energy_drop = (
            1.5 * (n[0] * 1e19) * (T_drop * 1e3 * E_CHARGE) * (2.0 * np.pi**2 * self.R0 * (rho_1 * self.a) ** 2)
        )
        _ = W_before - W_after  # retained for caller diagnostics if needed

        return SawtoothEvent(
            crash_time=self.time,
            rho_1=rho_1,
            rho_mix=rho_mix,
            T_drop=T_drop,
            seed_energy=float(core_energy_drop),
        )
