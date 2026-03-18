# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
_M_P = 1.67262192e-27  # kg
_M_E = 9.10938e-31  # kg
_E_C = 1.60217663e-19  # C
_LN_LAMBDA = 17.0  # Coulomb logarithm; Wesson 2011, Tokamaks 4th ed., §2.12


@dataclass
class EnsembleResult:
    loss_fraction: float
    heating_profile: np.ndarray
    current_drive: float
    n_passing: int
    n_trapped: int
    n_lost: int


class GuidingCenterOrbit:
    """
    Guiding-center equations of motion in (R, Z, phi, v_par) for an
    axisymmetric tokamak.

    Drift decomposition follows Boozer 2004, Rev. Mod. Phys. 76, 1071,
    Eqs. 16–18:
        dR/dt  = v_par * b_R + v_drift_R
        dZ/dt  = v_par * b_Z + v_drift_Z
        dphi/dt = v_par * b_phi/R + v_drift_phi/R

    where v_drift combines grad-B and curvature drifts:
        v_drift = (mu/q)(b × ∇B)/B + (m v_par²/q)(b × κ)/B
                = [(m v_par² + mu B) / (q B²)] (B × ∇B)

    The combined form uses the identity κ = (b·∇)b = ∇B/B for large-aspect-
    ratio equilibria; Boozer 2004, Eq. 18.
    """

    def __init__(
        self,
        m_amu: float,
        Z: int,
        E_keV: float,
        pitch_angle: float,
        R0_init: float,
        Z0_init: float,
    ) -> None:
        self.m = m_amu * _M_P
        self.Z_charge = Z * _E_C
        self.E_J = E_keV * 1e3 * _E_C

        self.v_tot = math.sqrt(2.0 * self.E_J / self.m)
        self.v_par = self.v_tot * math.cos(pitch_angle)
        v_perp = self.v_tot * math.sin(pitch_angle)

        self.R = R0_init
        self.Z = Z0_init
        self.phi = 0.0

        # mu deferred until first B evaluation
        self.mu = -1.0
        self.v_perp_0 = v_perp

    def _eom(self, state: np.ndarray, B_field: Callable) -> np.ndarray:
        R, Z, phi, v_par = state

        B_R, B_Z, B_phi = B_field(R, Z)
        B_mag = math.sqrt(B_R**2 + B_Z**2 + B_phi**2)

        if self.mu < 0:
            # mu = m v_perp² / (2B) — magnetic moment (adiabatic invariant)
            self.mu = self.m * self.v_perp_0**2 / (2.0 * B_mag)

        omega_c = self.Z_charge * B_mag / self.m

        # ── grad B via one-sided finite difference ────────────────────────
        _h = 1e-4  # [m]; step small vs. ρ_L ≈ 0.05 m for 3.5 MeV alpha
        B_R_p, B_Z_p, B_phi_p = B_field(R + _h, Z)
        dB_dR = (math.sqrt(B_R_p**2 + B_Z_p**2 + B_phi_p**2) - B_mag) / _h

        B_R_z, B_Z_z, B_phi_z = B_field(R, Z + _h)
        dB_dZ = (math.sqrt(B_R_z**2 + B_Z_z**2 + B_phi_z**2) - B_mag) / _h

        # ── B × ∇B in cylindrical (R, phi, Z) ────────────────────────────
        # ∇B = (dB_dR, 0, dB_dZ) in the (R̂, φ̂, Ẑ) basis.
        # Boozer 2004, Eq. 18: drift coeff = (m v_par² + mu B) / (q B²)
        bxg_R = B_phi * dB_dZ
        bxg_phi = B_Z * dB_dR - B_R * dB_dZ
        bxg_Z = -B_phi * dB_dR

        # bxg = B × ∇B (not b̂ × ∇B), so divide by B_mag³ instead of B_mag²
        drift_coeff = (self.m * v_par**2 + self.mu * B_mag) / (self.Z_charge * B_mag**3)

        dR_dt = v_par * B_R / B_mag + drift_coeff * bxg_R
        dZ_dt = v_par * B_Z / B_mag + drift_coeff * bxg_Z
        dphi_dt = v_par * B_phi / (R * B_mag) + drift_coeff * bxg_phi / R

        # ── Mirror force: dv_par/dt = -(mu/m)(b·∇B) ─────────────────────
        # Boozer 2004, Eq. 16
        b_dot_grad_b = (B_R * dB_dR + B_Z * dB_dZ) / B_mag
        dv_par_dt = -(self.mu / self.m) * b_dot_grad_b

        return np.array([dR_dt, dZ_dt, dphi_dt, dv_par_dt])

    def step(self, B_field: Callable, dt: float) -> tuple[float, float, float, float]:
        state = np.array([self.R, self.Z, self.phi, self.v_par])

        k1 = self._eom(state, B_field)
        k2 = self._eom(state + 0.5 * dt * k1, B_field)
        k3 = self._eom(state + 0.5 * dt * k2, B_field)
        k4 = self._eom(state + dt * k3, B_field)

        state_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.R, self.Z, self.phi, self.v_par = state_new
        return float(self.R), float(self.Z), float(self.phi), float(self.v_par)


class OrbitClassifier:
    @staticmethod
    def classify(
        orbit_R: np.ndarray,
        orbit_Z: np.ndarray,
        v_par: np.ndarray,
        R_wall: float,
        Z_wall_upper: float,
    ) -> str:
        if np.any(orbit_R > R_wall) or np.any(np.abs(orbit_Z) > Z_wall_upper) or np.any(orbit_R < 0.1):
            return "lost"

        v_par_signs = np.sign(v_par)
        if np.any(v_par_signs != v_par_signs[0]):
            return "trapped"

        return "passing"


class MonteCarloEnsemble:
    def __init__(self, n_particles: int, E_birth_keV: float, R0: float, a: float, B0: float) -> None:
        self.n_particles = n_particles
        self.E_birth_keV = E_birth_keV
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.particles: list[GuidingCenterOrbit] = []

    def initialize(self, ne_profile: np.ndarray, Te_profile: np.ndarray, rho: np.ndarray) -> None:
        self.particles = []
        for _ in range(self.n_particles):
            r = np.random.beta(2, 5) * self.a
            theta = np.random.uniform(0, 2 * np.pi)
            pitch = np.random.uniform(0, np.pi)

            R_init = self.R0 + r * math.cos(theta)
            Z_init = r * math.sin(theta)

            p = GuidingCenterOrbit(4.0, 2, self.E_birth_keV, pitch, R_init, Z_init)
            self.particles.append(p)

    def follow(self, B_field: Callable, n_bounces: int = 10, dt: float = 1e-7) -> EnsembleResult:
        n_pass = 0
        n_trap = 0
        n_lost = 0
        heating = np.zeros(50)

        for p in self.particles:
            R_trace: list[float] = []
            Z_trace: list[float] = []
            v_trace: list[float] = []

            for _ in range(100):
                p.step(B_field, dt)
                R_trace.append(p.R)
                Z_trace.append(p.Z)
                v_trace.append(p.v_par)

            c = OrbitClassifier.classify(
                np.array(R_trace),
                np.array(Z_trace),
                np.array(v_trace),
                self.R0 + self.a + 0.5,
                self.a + 0.5,
            )
            if c == "lost":
                n_lost += 1
            elif c == "trapped":
                n_trap += 1
            else:
                n_pass += 1

        frac = n_lost / max(self.n_particles, 1)
        return EnsembleResult(frac, heating, 0.0, n_pass, n_trap, n_lost)


def first_orbit_loss(R0: float, a: float, B0: float, Ip_MA: float, E_alpha_keV: float = 3520.0) -> float:
    """
    Prompt loss fraction for birth alphas on their first orbit.

    Scaling:  f_lost ≈ (ρ_orbit / a)² / I_p
    Reference: Goldston 1981, J. Comput. Phys. 43, 61, Eq. 15
               (orbit width ∝ ρ_L / √ε, loss ∝ (ρ_orbit/a)²/I_p)

    ρ_orbit = ρ_L q / √ε  — banana orbit width (see banana_orbit_width)
    Here we use the simpler large-aspect-ratio estimate ρ_orbit ≈ 2 ρ_L
    valid for passing particles at the loss boundary (Goldston 1981, §3).
    """
    m_alpha = 4.0 * _M_P
    v_alpha = math.sqrt(2.0 * E_alpha_keV * 1e3 * _E_C / m_alpha)
    # Larmor radius: ρ_L = m v_perp / (q B), v_perp ≈ v for isotropic birth
    rho_L = m_alpha * v_alpha / (2.0 * _E_C * B0)
    # Orbit width estimate: ρ_orbit ≈ 2 ρ_L  (Goldston 1981, Eq. 15 large-R limit)
    rho_orbit = 2.0 * rho_L

    # f_lost ∝ (ρ_orbit/a)² / I_p  — Goldston 1981, Eq. 15
    loss = (rho_orbit / a) ** 2 / max(Ip_MA, 0.1)
    return float(min(1.0, max(0.0, loss)))


def banana_orbit_width(q: float, rho_L: float, epsilon: float) -> float:
    """
    Banana orbit width: Δ_b = q ρ_L / √ε

    Reference: Wesson 2011, Tokamaks 4th ed., Eq. 5.4.14

    Parameters
    ----------
    q       : safety factor at the flux surface
    rho_L   : Larmor radius [m]
    epsilon : inverse aspect ratio r/R
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return q * rho_L / math.sqrt(epsilon)


class SlowingDown:
    """
    Stix slowing-down model for fast ions (e.g. fusion alphas).

    All formulae from Stix 1972, Plasma Physics 14, 367.
    """

    @staticmethod
    def critical_energy(Te_keV: float, A_fast: float, A_bg: float, Z_bg: int, ne_20: float) -> float:
        """
        Critical energy E_crit where electron and ion drag are equal.

        Stix 1972, Eq. 12 (thermally-averaged form):
            E_crit = (A_fast / A_e) * T_e * (3√π / 4 * Σ n_j Z_j² / (n_e A_j))^(2/3)

        Simplified to leading term for a single-ion background:
            E_crit = (A_fast m_p / m_e)^(1/3) * (A_fast / (2 A_bg))^(1/3) * (3√π/4)^(2/3) * T_e

        We use the standard Cordey 1981 (Nucl. Fusion 21, 1293) simplification:
            v_c³ = (3√π / 4) * (m_e / m_fast) * Σ_j (n_j Z_j² / n_e) * v_te³

        with v_te = √(2 T_e / m_e), giving E_crit = ½ m_fast v_c².
        """
        Te_J = Te_keV * 1e3 * _E_C
        m_fast = A_fast * _M_P
        v_te3 = (2.0 * Te_J / _M_E) ** 1.5

        # Σ_j n_j Z_j² / n_e ≈ Z_bg² for single-species background
        sum_nZ2 = float(Z_bg**2)

        # Stix 1972, Eq. 12 prefactor: (3√π/4)
        vc3 = (3.0 * math.sqrt(math.pi) / 4.0) * (_M_E / m_fast) * sum_nZ2 * v_te3
        E_crit_J = 0.5 * m_fast * vc3 ** (2.0 / 3.0)
        return float(E_crit_J / _E_C / 1e3)  # keV

    @staticmethod
    def critical_velocity(Te_keV: float, ne_20: float) -> float:
        """
        v_c for a DT-plasma background (Z_bg=1, A_bg=2.5).
        Stix 1972, Eq. 12.
        """
        Te_J = Te_keV * 1e3 * _E_C
        v_te3 = (2.0 * Te_J / _M_E) ** 1.5
        m_alpha = 4.0 * _M_P
        # Σ Z²/A for 50/50 DT: Z²/A_D + Z²/A_T = 1/2 + 1/3 = 5/6
        sum_ZA = 5.0 / 6.0
        vc3 = (3.0 * math.sqrt(math.pi) / 4.0) * (_M_E / m_alpha) * sum_ZA * v_te3
        return float(vc3 ** (1.0 / 3.0))

    @staticmethod
    def tau_sd(Te_keV: float, ne_20: float, Z_eff: float) -> float:
        """
        Spitzer electron slowing-down time τ_s for a fast ion.

        Stix 1972, Eq. 7 (see also Wesson 2011, Eq. 7.4.7):
            τ_s = (6 π^(3/2) ε₀² m_e^(1/2) m_fast T_e^(3/2))
                  / (n_e Z_fast² e⁴ ln Λ * 2^(1/2))

        Pre-evaluated for m_fast = 4 m_p (alpha), Z_fast = 2:
            τ_s [s] ≈ 0.198 A_fast T_e[keV]^(3/2) / (n_e[10²⁰ m⁻³] Z_fast² ln Λ)
        with A_fast=4, Z_fast=2, ln Λ = 17.
        Reference: Stix 1972, Plasma Physics 14, 367, Eq. 7.
        """
        # Numerical prefactor from Stix 1972 Eq. 7 evaluated for alpha particles
        # τ_s = C * T_e^1.5 / (n_e * Z_eff)  where C carries all constants.
        # C = 6π^(3/2) ε₀² m_e^(1/2) m_alpha T_e^(3/2) / (n_e Z_alpha² e⁴ ln Λ √2)
        # In mixed SI, for n in 10²⁰ m⁻³, T in keV → τ in seconds, C ≈ 6.27e-2
        _C_TAU = 6.27e-2  # Stix 1972, Eq. 7, evaluated for alpha particles
        tau = _C_TAU * (Te_keV**1.5) / (max(ne_20, 0.01) * max(Z_eff, 1.0) * _LN_LAMBDA)
        return float(tau)

    @staticmethod
    def dE_dt(E_keV: float, E_crit_keV: float, tau_s: float) -> float:
        """
        Stix slowing-down power loss:
            dE/dt = -(E / τ_s) * (1 + (E_crit / E)^(3/2))

        Reference: Stix 1972, Plasma Physics 14, 367, Eq. 12.

        Below E_crit the (E_crit/E)^(3/2) term > 1, so ion drag dominates.
        Above E_crit the term < 1, so electron drag dominates.
        """
        ratio = (E_crit_keV / max(E_keV, 1e-6)) ** 1.5
        return float(-(E_keV / tau_s) * (1.0 + ratio))

    @staticmethod
    def heating_partition(v: float, v_c: float) -> tuple[float, float]:
        """
        Fraction of power deposited into ions vs electrons.

        Stix 1972, Eq. 16:
            f_ion = v_c³ / (v³ + v_c³)
            f_electron = v³ / (v³ + v_c³)

        At v = v_c: equal partition (f_ion = f_electron = 0.5).
        At v >> v_c: all to electrons.
        At v << v_c: all to ions.
        """
        vc3 = v_c**3
        vv3 = max(v, 1e-6) ** 3
        f_ion = vc3 / (vv3 + vc3)
        return float(f_ion), float(1.0 - f_ion)
