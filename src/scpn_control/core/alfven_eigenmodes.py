# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851 | Contact: protoscience@anulum.li
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Physical constants (SI)
_M_P = 1.67262192e-27  # kg, proton mass
_M_E = 9.10938e-31  # kg, electron mass
_E_C = 1.60218e-19  # C, elementary charge
_MU_0 = 4.0 * np.pi * 1e-7  # H/m


@dataclass
class AlfvenGap:
    rho_location: float
    omega_lower: float
    omega_upper: float
    m_coupling: int


class AlfvenContinuum:
    """
    Shear Alfvén continuum and TAE gap structure.

    Cheng & Chance 1986, Phys. Fluids 29, 3695 — gap condition q = (2m+1)/(2n).
    """

    def __init__(
        self,
        rho: np.ndarray,
        q: np.ndarray,
        ne: np.ndarray,
        B0: float,
        R0: float,
        m_i_amu: float = 2.5,
        a: float = 2.0,
    ):
        self.rho = rho
        self.q = q
        self.ne = ne
        self.B0 = B0
        self.R0 = R0
        self.a = a

        m_i = m_i_amu * _M_P
        self.rho_mass = ne * 1e19 * (m_i + _M_E)
        self.v_A = B0 / np.sqrt(_MU_0 * np.maximum(self.rho_mass, 1e-12))

    def alfven_speed(self, rho_eval: float) -> float:
        return float(np.interp(rho_eval, self.rho, self.v_A))

    def continuum(self, m: int, n: int) -> np.ndarray:
        """
        ω_A(ρ) = |n q − m| / (q R₀) · v_A
        Cheng & Chance 1986, Eq. 2.
        """
        k_par = np.abs(n * self.q - m) / np.maximum(self.q * self.R0, 1e-6)
        return np.asarray(k_par * self.v_A)

    def find_gaps(self, n: int) -> list[AlfvenGap]:
        """
        TAE gap midpoint at q = (2m+1)/(2n); width ≃ ε·ω₀.
        Cheng & Chance 1986, Eq. 3.6.
        """
        gaps: list[AlfvenGap] = []
        for m in range(1, 10):
            q_gap = (2.0 * m + 1.0) / (2.0 * n)
            q_diff = self.q - q_gap
            crossings = np.where(np.diff(np.sign(q_diff)))[0]

            for idx in crossings:
                r1, r2 = self.rho[idx], self.rho[idx + 1]
                q1, q2 = self.q[idx], self.q[idx + 1]
                if q1 == q2:
                    continue

                frac = (q_gap - q1) / (q2 - q1)
                rho_gap = r1 + frac * (r2 - r1)

                v_A_gap = self.alfven_speed(rho_gap)
                omega_0 = v_A_gap / (2.0 * q_gap * self.R0)

                # Gap width ≃ ε ω₀, ε = r/R — Cheng & Chance 1986, §3
                eps = rho_gap * self.a / self.R0
                gaps.append(
                    AlfvenGap(
                        rho_location=float(rho_gap),
                        omega_lower=float(omega_0 * (1.0 - eps)),
                        omega_upper=float(omega_0 * (1.0 + eps)),
                        m_coupling=m,
                    )
                )

        return gaps


class TAEMode:
    """
    Toroidal Alfvén eigenmode frequency and damping.

    Cheng & Chance 1986, Phys. Fluids 29, 3695.
    """

    def __init__(
        self,
        n: int,
        q_rational: float,
        v_A: float,
        R0: float,
        T_e_keV: float | None = None,
        m_coupling: int = 1,
    ):
        self.n = n
        self.q = q_rational
        self.v_A = v_A
        self.R0 = R0
        self.T_e_keV = T_e_keV
        self.m = m_coupling

    def frequency(self) -> float:
        """ω_TAE = v_A / (2 q R₀) — Cheng & Chance 1986, Eq. 3.5."""
        return self.v_A / (2.0 * self.q * self.R0)

    def frequency_kHz(self) -> float:
        return self.frequency() / (2.0 * np.pi * 1e3)

    def electron_landau_damping(self) -> float:
        """
        Electron Landau damping rate for a TAE.

        γ_e = −(π/4)·(ω / |k_∥ v_the|) · ω · exp(−(ω / k_∥ v_the)²)

        where
            v_the = sqrt(2 T_e / m_e),  T_e in SI
            k_∥   = (n − m/q) / (q R₀)   (parallel wave vector)

        Returns |γ_e| (positive damping magnitude).

        Rosenbluth & Rutherford 1975, Phys. Rev. Lett. 34, 1428, Eq. 9.
        Fallback: γ_damp = 0.01 ω when T_e_keV is None (no physics input).
        """
        omega = self.frequency()
        if self.T_e_keV is None:
            return 0.01 * omega  # Rosenbluth & Rutherford 1975 order-of-magnitude

        T_e_J = self.T_e_keV * 1e3 * _E_C  # keV → J
        v_the = np.sqrt(2.0 * T_e_J / _M_E)  # thermal electron speed, m/s

        # k_∥ = (n − m/q) / (q R₀)  — Cheng & Chance 1986, Eq. 2
        k_par = (self.n - self.m / self.q) / (self.q * self.R0)
        k_par_abs = abs(k_par)

        if k_par_abs < 1e-12 or v_the < 1.0:
            return 0.01 * omega

        xi = omega / (k_par_abs * v_the)
        # Rosenbluth & Rutherford 1975, Eq. 9
        gamma_e = (np.pi / 4.0) * xi * omega * np.exp(-(xi**2))
        return float(gamma_e)


class FastParticleDrive:
    """
    Fast-particle (alpha/beam-ion) drive for Alfvén eigenmodes.

    Fu & Van Dam 1989, Phys. Fluids B 1, 1949.
    Heidbrink 2008, Phys. Plasmas 15, 055501 — review.
    """

    def __init__(self, E_fast_keV: float, n_fast_frac: float, m_fast_amu: float = 4.0):
        self.E_fast_keV = E_fast_keV
        self.n_fast_frac = n_fast_frac
        self.m_fast = m_fast_amu * _M_P

        E_J = E_fast_keV * 1e3 * _E_C
        self.v_fast = np.sqrt(2.0 * E_J / self.m_fast)

    def beta_fast(self, ne_20: float, B0: float) -> float:
        n_e = ne_20 * 1e20
        n_fast = n_e * self.n_fast_frac
        E_J = self.E_fast_keV * 1e3 * _E_C
        p_fast = (2.0 / 3.0) * n_fast * E_J
        p_mag = B0**2 / (2.0 * _MU_0)
        return float(p_fast / p_mag)

    def resonance_function(self, v_f: float, v_A: float) -> float:
        """
        Analytic fast-ion resonance function.

        Primary resonance (v_f ≃ v_A):
            F_main = (v_f/v_A)³ · exp(−(v_f/v_A − 1)² / (2·0.2²))

        Sideband resonance (v_f ≃ v_A/3):
            F_sb = 0.15·(v_f/v_A)² · exp(−(v_f/v_A − 1/3)² / (2·0.1²))

        Fu & Van Dam 1989, Phys. Fluids B 1, 1949, Eq. 28.
        σ_main = 0.2, σ_side = 0.1, A_side = 0.15 — ibid., Table 1.
        """
        x = v_f / v_A
        # σ_main = 0.2 — Fu & Van Dam 1989, Table 1
        f_main = x**3 * np.exp(-((x - 1.0) ** 2) / (2.0 * 0.2**2))
        # A_side = 0.15, σ_side = 0.1 — Fu & Van Dam 1989, Table 1
        f_side = 0.15 * x**2 * np.exp(-((x - 1.0 / 3.0) ** 2) / (2.0 * 0.1**2))
        return float(f_main + f_side)

    def growth_rate(self, tae: TAEMode, beta_fast: float) -> float:
        """
        γ_fast / ω ≃ β_f · q² · F(v_f/v_A)
        Fu & Van Dam 1989, Phys. Fluids B 1, 1949, Eq. 15.
        Heidbrink 2008, Phys. Plasmas 15, 055501, Eq. 10.
        """
        omega = tae.frequency()
        F = self.resonance_function(self.v_fast, tae.v_A)
        return float(omega * beta_fast * tae.q**2 * F)


@dataclass
class TAEStabilityResult:
    n: int
    m: int
    frequency_kHz: float
    gamma_drive: float
    gamma_damp: float
    gamma_net: float
    unstable: bool


class AlfvenStabilityAnalysis:
    """
    Combined TAE stability: electron Landau damping vs. fast-particle drive.

    Rosenbluth & Rutherford 1975, Phys. Rev. Lett. 34, 1428.
    Fu & Van Dam 1989, Phys. Fluids B 1, 1949.
    Heidbrink 2008, Phys. Plasmas 15, 055501.
    """

    def __init__(
        self,
        continuum: AlfvenContinuum,
        fast_params: FastParticleDrive,
        T_e_keV: float | None = None,
    ):
        self.continuum = continuum
        self.fast_params = fast_params
        self.T_e_keV = T_e_keV

    def tae_stability(self, n_range: range = range(1, 6)) -> list[TAEStabilityResult]:
        results: list[TAEStabilityResult] = []
        for n in n_range:
            gaps = self.continuum.find_gaps(n)
            for gap in gaps:
                v_A = self.continuum.alfven_speed(gap.rho_location)
                q_gap = (2.0 * gap.m_coupling + 1.0) / (2.0 * n)
                tae = TAEMode(
                    n,
                    q_gap,
                    v_A,
                    self.continuum.R0,
                    T_e_keV=self.T_e_keV,
                    m_coupling=gap.m_coupling,
                )

                freq_kHz = tae.frequency_kHz()

                idx = int(np.searchsorted(self.continuum.rho, gap.rho_location))
                idx = min(idx, len(self.continuum.ne) - 1)
                ne_20 = self.continuum.ne[idx] / 10.0  # ne in 10^19 → 10^20

                b_fast = self.fast_params.beta_fast(max(ne_20, 0.1), self.continuum.B0)
                gamma_drive = self.fast_params.growth_rate(tae, b_fast)

                # Rosenbluth & Rutherford 1975, Eq. 9 (or 0.01ω fallback)
                gamma_damp = tae.electron_landau_damping()

                gamma_net = gamma_drive - gamma_damp
                results.append(
                    TAEStabilityResult(
                        n=n,
                        m=gap.m_coupling,
                        frequency_kHz=freq_kHz,
                        gamma_drive=gamma_drive,
                        gamma_damp=gamma_damp,
                        gamma_net=gamma_net,
                        unstable=gamma_net > 0.0,
                    )
                )

        return results

    def critical_beta_fast(self, n: int) -> float:
        """
        β_f,crit where γ_net = 0 for the most resonant mode.
        γ_drive = C·β_f  →  β_crit = γ_damp / C = γ_damp / (γ_drive / β_current)
        """
        res = self.tae_stability(range(n, n + 1))
        if not res:
            return float("inf")

        best = max(res, key=lambda r: r.gamma_drive)
        if best.gamma_drive == 0:
            return float("inf")

        b_current = self.fast_params.beta_fast(1.0, self.continuum.B0)
        if b_current == 0:
            return float("inf")

        return float(b_current * (best.gamma_damp / best.gamma_drive))

    def alpha_particle_loss_estimate(self, gamma_net: float, tau_sd: float = 0.5) -> float:
        """Fraction of alpha power lost; loss ∝ γ·τ_sd (Heidbrink 2008, §V)."""
        if gamma_net <= 0:
            return 0.0
        return float(min(1.0, 1e-4 * gamma_net * tau_sd))


def rsae_frequency(
    q_min: float,
    n: int,
    m: int,
    v_A: float,
    R0: float,
    T_e_keV: float = 10.0,
    T_i_keV: float = 10.0,
    m_i_amu: float = 2.5,
) -> float:
    """
    Reversed-Shear Alfvén Eigenmode (RSAE) frequency.

    ω²_RSAE = ω²_BAE + (v_A / (2 q_min R₀))² · (n − m/q_min)²

    where the beta-induced acoustic gap (BAE) frequency is
        ω_BAE = v_thi · sqrt(7/4 + T_e/T_i) / R₀,
        v_thi = sqrt(T_i / m_i).

    Sharapov et al. 2001, Phys. Lett. A 289, 127, Eq. 5.
    Geodesic acoustic coupling: Zonca & Chen 1996, Plasma Phys. 38, 2011.
    """
    T_i_J = T_i_keV * 1e3 * _E_C
    m_i = m_i_amu * _M_P
    v_thi = np.sqrt(T_i_J / m_i)

    # ω_BAE — Sharapov et al. 2001, Eq. 4; factor (7/4 + T_e/T_i) from geodesic coupling
    omega_BAE = v_thi * np.sqrt(7.0 / 4.0 + T_e_keV / T_i_keV) / R0

    # Alfvénic term — Sharapov et al. 2001, Eq. 5
    delta_q = n - m / q_min  # dimensionless frequency detuning
    omega_alf_sq = (v_A / (2.0 * q_min * R0)) ** 2 * delta_q**2

    return float(np.sqrt(omega_BAE**2 + omega_alf_sq))
