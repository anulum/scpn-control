# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# Contact: protoscience@anulum.li  ORCID: 0009-0009-3560-0851
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import erfc


# SI fundamental constants
E_CHARGE = 1.602176634e-19  # C
M_E = 9.1093837015e-31  # kg  — CODATA 2018
C_LIGHT = 2.99792458e8  # m/s
EPS_0 = 8.8541878128e-12  # F/m
ALPHA_FINE = 7.2973525693e-3  # fine-structure constant — CODATA 2018

# Kinetic energy at rest [J]
M_E_C2_J = M_E * C_LIGHT**2
# [MeV]
M_E_C2_MeV = M_E_C2_J / (E_CHARGE * 1e6)  # 0.51100 MeV


@dataclass
class RunawayParams:
    ne_20: float  # electron density [10^20 m^-3]
    Te_keV: float  # electron temperature [keV]
    E_par: float  # parallel electric field [V/m]
    Z_eff: float  # effective charge
    B0: float  # magnetic field [T]
    R0: float  # major radius [m]
    a: float = 2.0  # minor radius [m]


# ---------------------------------------------------------------------------
# Coulomb logarithm
# ---------------------------------------------------------------------------


def coulomb_log(ne_20: float, Te_keV: float) -> float:
    """Coulomb logarithm for a thermal plasma.

    For T_e > 10 eV:
        ln Λ = 14.9 - 0.5 ln(n_e / 10^20) + ln(T_e / 10^3)
    where n_e is in m^-3 and T_e in eV.

    Wesson 2011, Tokamaks 4th ed., Eq. 2.12.4.
    """
    Te_eV = Te_keV * 1e3
    # Clamp to the regime where the formula is valid (T_e > 10 eV)
    Te_eV = max(Te_eV, 10.0)
    ln_lambda = 14.9 - 0.5 * np.log(ne_20) + np.log(Te_eV / 1e3)
    # Physical lower bound (Wesson Ch. 14 context: typically 10–20 in tokamaks)
    return float(max(ln_lambda, 2.0))


# ---------------------------------------------------------------------------
# Critical and Dreicer fields
# ---------------------------------------------------------------------------


def dreicer_field(ne_20: float, Te_keV: float) -> float:
    """Dreicer field E_D = n_e e^3 ln Λ / (4π ε₀² T_e).

    Connor & Hastie, Nucl. Fusion 15, 415 (1975).
    """
    ln_lambda = coulomb_log(ne_20, Te_keV)
    n_e = ne_20 * 1e20
    Te_J = Te_keV * 1e3 * E_CHARGE
    return float(n_e * E_CHARGE**3 * ln_lambda / (4.0 * np.pi * EPS_0**2 * Te_J))


def critical_field(ne_20: float, Te_keV: float = 1.0) -> float:
    """Critical electric field E_c = n_e e^3 ln Λ / (4π ε₀² m_e c²).

    Rosenbluth & Putvinski, Nucl. Fusion 37, 1355 (1997).
    Note: E_D / E_c = m_e c² / T_e ≈ 51 at T_e = 10 keV.
    """
    ln_lambda = coulomb_log(ne_20, Te_keV)
    n_e = ne_20 * 1e20
    return float(n_e * E_CHARGE**3 * ln_lambda / (4.0 * np.pi * EPS_0**2 * M_E * C_LIGHT**2))


# ---------------------------------------------------------------------------
# Generation rates
# ---------------------------------------------------------------------------


def dreicer_generation_rate(params: RunawayParams) -> float:
    """Primary (Dreicer) seed runaway generation rate [m^-3 s^-1].

    Connor & Hastie, Nucl. Fusion 15, 415 (1975).
    Smith et al., Phys. Plasmas 15, 072502 (2008) — hot-tail context uses
    same Dreicer base rate.
    """
    if params.E_par <= 0.0 or params.Te_keV <= 0.0 or params.ne_20 <= 0.0:
        return 0.0

    ln_lambda = coulomb_log(params.ne_20, params.Te_keV)
    n_e = params.ne_20 * 1e20
    Te_J = params.Te_keV * 1e3 * E_CHARGE

    v_te = np.sqrt(2.0 * Te_J / M_E)

    E_D = n_e * E_CHARGE**3 * ln_lambda / (4.0 * np.pi * EPS_0**2 * Te_J)

    if params.E_par / E_D < 1e-4:
        return 0.0

    nu_ee = n_e * E_CHARGE**4 * ln_lambda / (4.0 * np.pi * EPS_0**2 * M_E**2 * v_te**3)

    Z = params.Z_eff
    E_ratio = params.E_par / E_D

    h_z = (Z + 1.0) / 16.0 * (Z + 1.0 + 2.0 * np.sqrt(1.0 + 1.0 / Z))

    # Prefactor C_D = 0.35 — Connor & Hastie (1975), Eq. 63
    C_D = 0.35

    exponent = -E_D / (4.0 * params.E_par) - np.sqrt((1.0 + Z) * E_D / params.E_par)

    if exponent < -500:
        return 0.0

    rate = C_D * n_e * nu_ee * E_ratio ** (-h_z) * np.exp(exponent)
    return float(max(0.0, rate))


def avalanche_growth_rate(params: RunawayParams, n_RE: float) -> float:
    """Avalanche multiplication rate [m^-3 s^-1].

    Full pitch-angle-scattering formula:
        dn_RE/dt = n_RE (E/E_c - 1) / (τ_s ln Λ) × F(E/E_c, Z_eff)^{-1/2}
    where
        F = 1 - E_c/E + 4π(Z+1)² / [3(Z+1)(E/E_c)²]

    Rosenbluth & Putvinski, Nucl. Fusion 37, 1355 (1997), Eq. 15.
    """
    if n_RE <= 0.0 or params.E_par <= 0.0:
        return 0.0

    ln_lambda = coulomb_log(params.ne_20, params.Te_keV)
    E_c = critical_field(params.ne_20, params.Te_keV)
    if params.E_par <= E_c:
        return 0.0

    n_e = params.ne_20 * 1e20
    # Relativistic slowing-down time τ_s — Rosenbluth & Putvinski (1997), Eq. 4
    tau_s = 4.0 * np.pi * EPS_0**2 * M_E**2 * C_LIGHT**3 / (n_e * E_CHARGE**4 * ln_lambda)

    E_over_Ec = params.E_par / E_c
    Z = params.Z_eff

    # Pitch-angle scattering correction factor F — R&P (1997), Eq. 15
    F = 1.0 - 1.0 / E_over_Ec + 4.0 * np.pi * (Z + 1.0) ** 2 / (3.0 * (Z + 1.0) * E_over_Ec**2)

    # Guard: F <= 0 means the formula is outside its validity domain
    if F <= 0.0:
        return 0.0

    rate = n_RE * (E_over_Ec - 1.0) / (tau_s * ln_lambda * np.sqrt(F))
    return float(max(0.0, rate))


def hot_tail_seed(Te_pre_keV: float, Te_post_keV: float, ne_20: float, quench_time_ms: float) -> float:
    """Seed RE density from thermal-quench hot-tail mechanism [m^-3].

    Smith et al., Phys. Plasmas 15, 072502 (2008).
    Parametric fit to Smith (2008) Fig. 3: v_c/v_te scales as τ_q^0.2
    at the V_C_V_TE_REF = 4.0 reference (τ_q = 1 ms).
    """
    if Te_post_keV >= Te_pre_keV or Te_post_keV <= 0:
        return 0.0

    ratio = Te_pre_keV / Te_post_keV
    V_C_V_TE_REF = 4.0  # Smith (2008) Fig. 3, τ_q = 1 ms reference point
    v_c_v_te = V_C_V_TE_REF * (quench_time_ms / 1.0) ** 0.2

    n_e = ne_20 * 1e20

    if v_c_v_te > 30.0:
        return 0.0

    n_seed = n_e * erfc(v_c_v_te) * ratio**1.5
    return float(max(0.0, n_seed))


# ---------------------------------------------------------------------------
# Synchrotron energy limit
# ---------------------------------------------------------------------------


def synchrotron_energy_limit(E_par: float, E_c: float) -> float:
    """Maximum RE energy [MeV] limited by synchrotron radiation.

    E_max = m_e c² (E/E_c)^{1/3} (3 / (4 α_f))^{1/3}

    Martin-Solis et al., Phys. Plasmas 13, 062509 (2006), Eq. 12.
    α_f = 7.297e-3 is the fine-structure constant (CODATA 2018).
    """
    if E_c <= 0.0 or E_par <= E_c:
        return 0.0

    E_ratio = E_par / E_c
    # Martin-Solis (2006), Eq. 12
    gamma_max = E_ratio ** (1.0 / 3.0) * (3.0 / (4.0 * ALPHA_FINE)) ** (1.0 / 3.0)
    E_max_MeV = gamma_max * M_E_C2_MeV
    return float(E_max_MeV)


# ---------------------------------------------------------------------------
# Evolution integrator
# ---------------------------------------------------------------------------


class RunawayEvolution:
    def __init__(self, params: RunawayParams) -> None:
        self.params = params

    def step(self, dt: float, n_RE: float, E_par: float) -> float:
        self.params.E_par = E_par

        rate_D = dreicer_generation_rate(self.params)
        rate_A = avalanche_growth_rate(self.params, n_RE)

        return float(n_RE + (rate_D + rate_A) * dt)

    def evolve(
        self,
        n_RE_0: float,
        E_par_profile: Callable[[float], float],
        t_span: tuple[float, float],
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_start, t_end = t_span
        n_steps = int(np.ceil((t_end - t_start) / dt))

        t_arr = np.linspace(t_start, t_end, n_steps + 1)
        n_RE_arr = np.zeros(n_steps + 1)
        n_RE_arr[0] = n_RE_0

        for i in range(n_steps):
            n_RE_arr[i + 1] = self.step(dt, n_RE_arr[i], E_par_profile(t_arr[i]))

        return t_arr, n_RE_arr

    def current_fraction(self, n_RE: float, I_p_MA: float) -> float:
        """Fraction of total plasma current carried by REs (v ≈ c)."""
        if I_p_MA <= 0.0:
            return 0.0
        j_RE = E_CHARGE * n_RE * C_LIGHT
        I_RE = j_RE * np.pi * self.params.a**2
        return float(min(1.0, I_RE / (I_p_MA * 1e6)))


# ---------------------------------------------------------------------------
# Mitigation assessment
# ---------------------------------------------------------------------------


class RunawayMitigationAssessment:
    @staticmethod
    def required_density_for_suppression(E_par: float, Z_eff: float, Te_keV: float = 1.0) -> float:
        """Electron density [10^20 m^-3] required to raise E_c above E_par.

        Derived by inverting critical_field: n_e = E_par (4π ε₀² m_e c²) / (e³ ln Λ).
        Uses the temperature-dependent Coulomb log (Wesson 2011, Eq. 2.12.4).
        """
        if E_par <= 0.0:
            return 0.0

        # Iterative solve: ln Λ depends on ne_20 we are solving for.
        # One Newton step from a flat-log starting guess converges in <3 iterations.
        ne_guess = 1.0
        for _ in range(5):
            ln_lambda = coulomb_log(ne_guess, Te_keV)
            ne_guess = E_par * (4.0 * np.pi * EPS_0**2 * M_E * C_LIGHT**2) / (E_CHARGE**3 * ln_lambda * 1e20)
        return float(ne_guess)

    @staticmethod
    def maximum_re_energy(B0: float, R0: float, E_par: float = 10.0, ne_20: float = 10.0, Te_keV: float = 1.0) -> float:
        """Maximum RE energy [MeV] from synchrotron balance.

        Martin-Solis et al., Phys. Plasmas 13, 062509 (2006), Eq. 12.
        Defaults represent a disruption scenario: E_par=10 V/m, post-TQ
        ne_20=10, Te_keV=1.
        """
        E_c = critical_field(ne_20, Te_keV)
        E_max = synchrotron_energy_limit(E_par, E_c)
        # Orbit-loss upper bound: e B0 R0 c (Wesson 2011, Ch. 14)
        E_orbit_MeV = E_CHARGE * B0 * R0 * C_LIGHT / (E_CHARGE * 1e6)
        return float(min(E_max, E_orbit_MeV)) if E_max > 0.0 else float(E_orbit_MeV)

    @staticmethod
    def wall_heat_load(n_RE: float, E_max_MeV: float, A_wet: float, volume: float = 800.0) -> float:
        """Deposited energy density [MJ/m^2] assuming instantaneous loss.

        Assumes mean RE energy ≈ E_max / 2 (flat distribution upper bound).
        """
        E_avg_J = (E_max_MeV / 2.0) * 1e6 * E_CHARGE
        W_total_J = n_RE * volume * E_avg_J

        if A_wet <= 0.0:
            return float("inf")

        return float((W_total_J / 1e6) / A_wet)
