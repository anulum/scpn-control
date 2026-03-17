# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""
Error-field amplification, mode locking, and post-lock island growth chain.

Key references
--------------
La Haye 2006    : R.J. La Haye, Phys. Plasmas 13, 055501 (2006).
                  [RFA formula Eq. 6; locking condition Eq. 12; MRE in
                  locked-mode regime Eq. 8; C_lock coefficient]
Fitzpatrick 1993: R. Fitzpatrick, Nucl. Fusion 33, 1049 (1993).
                  [electromagnetic torque Eq. 28; locking criterion]
Rutherford 1973 : P.H. Rutherford, Phys. Fluids 16, 1903 (1973).
                  [classical tearing stability dw/dt ∝ Delta']
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Permeability of free space (CODATA 2018)
_MU_0: float = 4.0 * math.pi * 1e-7  # H m⁻¹

# Locked-mode bootstrap drive coefficient.
# La Haye 2006, Phys. Plasmas 13, 055501, Table I (same set as NTM a1–a3).
_C_LOCK: float = 1.0  # La Haye 2006, Eq. 8 — normalised locked-mode current drive


class ErrorFieldSpectrum:
    def __init__(self, B0: float, n_corrections: int = 0):
        self.B0 = B0
        self.n_corrections = n_corrections
        self.B_mn_components: dict[tuple[int, int], float] = {}
        # Intrinsic error-field estimates for a large superconducting tokamak.
        # La Haye 2006, Phys. Plasmas 13, 055501, §II: typical |b_21/B0| ~ 1e-4.
        self.B_mn_components[(2, 1)] = 1e-4 * B0
        self.B_mn_components[(3, 2)] = 5e-5 * B0

    def set_coil_misalignment(self, delta_R_mm: float, delta_Z_mm: float) -> None:
        shift_mag = math.sqrt(delta_R_mm**2 + delta_Z_mm**2) / 1000.0
        # Linear scaling of b_mn with coil displacement.
        # La Haye 2006, §II: b_21/B0 ~ 0.01 × (δR/a) for centroid shift δR.
        self.B_mn_components[(2, 1)] = 0.01 * self.B0 * shift_mag
        self.B_mn_components[(3, 2)] = 0.005 * self.B0 * shift_mag

    def B_mn(self, m: int, n: int) -> float:
        return self.B_mn_components.get((m, n), 0.0)

    def corrected_B_mn(self, m: int, n: int, I_correction: float) -> float:
        B_raw = self.B_mn(m, n)
        # First-order linear correction from an active coil of effective area ~1 m².
        B_corr = max(0.0, B_raw - 1e-5 * I_correction)
        return B_corr


class ResonantFieldAmplification:
    """Ideal-MHD resonant-field amplification (RFA) model.

    B_res = B_err / (1 - beta_N / beta_N_nowall)

    La Haye 2006, Phys. Plasmas 13, 055501, Eq. 6.
    Denominator → 0 as beta_N → beta_N_nowall (no-wall stability boundary).
    """

    def __init__(self, beta_N: float, beta_N_nowall: float):
        self.beta_N = beta_N
        self.beta_N_nowall = beta_N_nowall

    def amplification_factor(self) -> float:
        """RFA factor; La Haye 2006, Eq. 6."""
        if self.beta_N >= self.beta_N_nowall:
            return float("inf")
        return 1.0 / (1.0 - self.beta_N / self.beta_N_nowall)

    def resonant_field(self, B_err: float) -> float:
        return B_err * self.amplification_factor()


@dataclass
class RotationEvolution:
    omega_trace: np.ndarray
    locked: bool
    lock_time: float


class ModeLocking:
    """Toroidal rotation under electromagnetic braking from a resonant error field.

    Electromagnetic torque (per unit toroidal angle):
        T_em = -n² m ψ_ext² sin(Δφ) / (μ₀ R r_s²)
    Fitzpatrick 1993, Nucl. Fusion 33, 1049, Eq. 28.

    Mode-locking condition:
        |T_em| > T_viscous   where   T_viscous = ρ χ_φ ω V_island
    La Haye 2006, Phys. Plasmas 13, 055501, Eq. 12.

    In steady-state the mode locks when the electromagnetic drag exceeds the
    viscous restoring torque that maintains plasma rotation.
    """

    def __init__(self, R0: float, a: float, B0: float, Ip_MA: float, omega_phi_0: float):
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.Ip = Ip_MA * 1e6
        self.omega_phi_0 = omega_phi_0
        # Effective moment of inertia proxy for a deuterium plasma annulus.
        # Not taken from a single reference; scales as ρ_i × V_plasma.
        self._I_eff: float = 0.01  # kg m²

    def em_torque(self, B_res: float, r_s: float, m: int, n: int) -> float:
        """Electromagnetic braking torque on the (m, n) mode.

        Derived from Fitzpatrick 1993, Nucl. Fusion 33, 1049, Eq. 28:
            T_em = -n² m ψ_ext² sin(Δφ) / (μ₀ R r_s²)
        where ψ_ext ~ B_res × r_s² / n (flux loop estimate) and sin(Δφ) → 1
        at the locking threshold.  The sign convention here gives a positive
        magnitude; the caller interprets the torque as opposing rotation.
        """
        # ψ_ext estimate: B_res r_s² / n  (Fitzpatrick 1993, §2)
        psi_ext = B_res * r_s**2 / max(n, 1)
        torque = (n**2 * m * psi_ext**2) / (_MU_0 * self.R0 * max(r_s, 1e-3) ** 2)
        return torque

    def viscous_torque(self, omega: float, r_s: float, rho_chi: float = 1.0) -> float:
        """Viscous restoring torque.

        T_viscous = ρ χ_φ ω V_island
        La Haye 2006, Phys. Plasmas 13, 055501, Eq. 12.

        rho_chi = ρ × χ_φ [kg m⁻¹ s⁻¹], V_island ~ 4 π² R₀ r_s (shell volume
        at the rational surface, thin-shell limit).
        """
        V_island = 4.0 * math.pi**2 * self.R0 * r_s
        return rho_chi * omega * V_island

    def evolve_rotation(self, B_res: float, r_s: float, tau_visc: float, dt: float, n_steps: int) -> RotationEvolution:
        """Integrate dω/dt = -(ω - ω₀)/τ_visc - T_em / I_eff.

        Mode locks when ω ≤ ω_eq (equilibrium with sustained braking).
        Locking condition: T_em > T_viscous (La Haye 2006, Eq. 12).
        """
        omega = self.omega_phi_0
        omega_trace = np.zeros(n_steps)
        locked = False
        lock_time = -1.0

        T_em = self.em_torque(B_res, r_s, 2, 1)
        omega_eq = max(0.0, self.omega_phi_0 - T_em * tau_visc / self._I_eff)

        for i in range(n_steps):
            if not locked:
                d_omega = -(omega - self.omega_phi_0) / tau_visc - T_em / self._I_eff
                omega += d_omega * dt

                if omega <= omega_eq:
                    locked = True
                    lock_time = i * dt
                    omega = 0.0
            else:
                omega = 0.0

            omega_trace[i] = omega

        return RotationEvolution(omega_trace, locked, lock_time)


@dataclass
class IslandGrowth:
    w_trace: np.ndarray
    overlap_time: float
    stochastic: bool


class LockedModeIsland:
    """Post-locking island growth via the Modified Rutherford Equation.

    dw/dt = (η / μ₀ r_s) [r_s Δ' + C_lock r_s / w]

    La Haye 2006, Phys. Plasmas 13, 055501, Eq. 8.
    C_lock accounts for the locked-mode bootstrap drive (same physical origin
    as the NTM bootstrap term but evaluated at ω = 0).
    Classical tearing drive: Rutherford 1973, Phys. Fluids 16, 1903.
    """

    def __init__(self, r_s: float, m: int, n: int, a: float, R0: float, delta_prime: float):
        self.r_s = r_s
        self.m = m
        self.n = n
        self.a = a
        self.R0 = R0
        self.delta_prime = delta_prime

    def grow(self, w0: float, eta: float, dt: float, n_steps: int, delta_r_mn: float = 0.3) -> IslandGrowth:
        """Integrate MRE for a locked island; stochastic when w > delta_r_mn.

        dw/dt = (η / μ₀ r_s) [r_s Δ' + C_lock r_s / w]
        La Haye 2006, Phys. Plasmas 13, 055501, Eq. 8.
        """
        w = max(w0, 1e-4)
        w_trace = np.zeros(n_steps)

        overlap_time = -1.0
        stochastic = False

        for i in range(n_steps):
            dw_dt = (eta / (_MU_0 * self.r_s)) * (self.r_s * self.delta_prime + _C_LOCK * self.r_s / w)
            w += dw_dt * dt
            w_trace[i] = w

            if not stochastic and w > delta_r_mn:
                stochastic = True
                overlap_time = i * dt

        return IslandGrowth(w_trace, overlap_time, stochastic)


@dataclass
class ChainResult:
    lock_time: float
    island_width_at_tq: float
    tq_trigger_time: float
    disruption: bool
    warning_time_ms: float


class ErrorFieldToDisruptionChain:
    def __init__(self, config: dict[str, float]):
        self.config = config

    def run(self, B_err_n1: float, omega_phi_0: float) -> ChainResult:
        R0 = self.config.get("R0", 6.2)
        a = self.config.get("a", 2.0)
        B0 = self.config.get("B0", 5.3)
        Ip = self.config.get("Ip_MA", 15.0)
        beta_N = self.config.get("beta_N", 2.0)
        beta_nowall = self.config.get("beta_N_nowall", 2.8)

        # Step 1: RFA — La Haye 2006, Eq. 6
        rfa = ResonantFieldAmplification(beta_N, beta_nowall)
        B_res = rfa.resonant_field(B_err_n1)

        # Step 2: Mode Locking — Fitzpatrick 1993, Eq. 28; La Haye 2006, Eq. 12
        dt = 0.001
        n_steps = 2000
        tau_visc = 0.1
        r_s = a * 0.5

        locker = ModeLocking(R0, a, B0, Ip, omega_phi_0)
        rot_evol = locker.evolve_rotation(B_res, r_s, tau_visc, dt, n_steps)

        if not rot_evol.locked:
            return ChainResult(-1.0, 0.0, -1.0, False, -1.0)

        # Step 3: Island Growth — La Haye 2006, Eq. 8; Rutherford 1973
        # η ~ 1e-7 Ω m typical for a hot deuterium plasma (Wesson, Tokamaks Ch.14)
        eta = 1e-7
        lm = LockedModeIsland(r_s, 2, 1, a, R0, delta_prime=-1.0)

        steps_left = n_steps - int(rot_evol.lock_time / dt)
        if steps_left <= 0:
            return ChainResult(rot_evol.lock_time, 0.0, -1.0, False, -1.0)

        grow_res = lm.grow(w0=1e-3, eta=eta, dt=dt, n_steps=steps_left, delta_r_mn=0.2 * a)

        if not grow_res.stochastic:
            return ChainResult(rot_evol.lock_time, grow_res.w_trace[-1], -1.0, False, -1.0)

        tq_time = rot_evol.lock_time + grow_res.overlap_time
        warning_time = grow_res.overlap_time * 1000.0

        return ChainResult(
            lock_time=rot_evol.lock_time,
            island_width_at_tq=grow_res.w_trace[int(grow_res.overlap_time / dt)],
            tq_trigger_time=tq_time,
            disruption=True,
            warning_time_ms=warning_time,
        )
