# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851 — Contact: protoscience@anulum.li
from __future__ import annotations

import math

import numpy as np


# ── named constants ─────────────────────────────────────────────────────────
# Sentinel for the ideal-kink (β > β_wall) regime; returned wherever a finite
# growth rate is physically undefined.
_IDEAL_KINK_RATE: float = 1e6  # s⁻¹ (representative large value, not physical)


class RWMPhysics:
    """
    Resistive Wall Mode growth rate with rotation stabilization.

    References
    ----------
    Bondeson & Ward 1994, Phys. Rev. Lett. 72, 2709  (γ_wall formula, Ω_crit)
    Fitzpatrick 2001, Phys. Plasmas 8, 4489           (rotation stabilization γ_rot)
    Strait et al. 2003, Nucl. Fusion 43, 430          (wall dissipation, τ_eff)
    Garofalo et al. 2002, Phys. Plasmas 9, 1997       (experimental validation)
    """

    def __init__(
        self,
        beta_n: float,
        beta_n_nowall: float,
        beta_n_wall: float,
        tau_wall: float,
        omega_phi: float = 0.0,
        wall_radius: float | None = None,
        plasma_radius: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        beta_n         : normalised beta (dimensionless)
        beta_n_nowall  : no-wall stability limit (dimensionless)
        beta_n_wall    : ideal-wall stability limit (dimensionless)
        tau_wall       : resistive wall L/R time, s
        omega_phi      : toroidal rotation frequency, rad s⁻¹ (default 0)
        wall_radius    : conducting-wall minor radius b, m (optional)
        plasma_radius  : plasma-edge minor radius d, m (optional)
        """
        self.beta_n = beta_n
        self.beta_n_nowall = beta_n_nowall
        self.beta_n_wall = beta_n_wall
        self.tau_wall = tau_wall
        self.omega_phi = omega_phi
        self.wall_radius = wall_radius
        self.plasma_radius = plasma_radius

    # ── effective wall time ──────────────────────────────────────────────────

    def tau_eff(self) -> float:
        """
        Effective wall time corrected for wall–plasma gap.

        τ_eff = τ_wall × (b/d)²

        Strait et al. 2003, Nucl. Fusion 43, 430, Eq. (3).
        When b == d the expression recovers τ_wall.
        """
        if self.wall_radius is None or self.plasma_radius is None:
            return self.tau_wall
        if self.plasma_radius <= 0.0:
            return self.tau_wall
        return self.tau_wall * (self.wall_radius / self.plasma_radius) ** 2

    # ── stability flag ───────────────────────────────────────────────────────

    def is_unstable(self) -> bool:
        """True when β_N lies between the no-wall and ideal-wall limits."""
        return self.beta_n_nowall < self.beta_n < self.beta_n_wall

    # ── growth rate components ───────────────────────────────────────────────

    def _gamma_wall(self, tau: float) -> float:
        """
        RWM growth rate in the absence of rotation.

        γ_wall = (1/τ_eff) × (β_N − β_N,nowall) / (β_N,wall − β_N)

        Bondeson & Ward 1994, Phys. Rev. Lett. 72, 2709, Eq. (4).
        """
        if tau <= 0.0:
            return _IDEAL_KINK_RATE if self.beta_n > self.beta_n_nowall else 0.0
        if self.beta_n >= self.beta_n_wall:
            return _IDEAL_KINK_RATE
        if self.beta_n <= self.beta_n_nowall:
            return 0.0
        return (1.0 / tau) * (self.beta_n - self.beta_n_nowall) / (self.beta_n_wall - self.beta_n)

    def _gamma_rot(self, tau: float) -> float:
        """
        Rotation-stabilization contribution to the growth rate.

        γ_rot = −Ω_φ² τ_eff / (1 + (Ω_φ τ_eff)²)

        Fitzpatrick 2001, Phys. Plasmas 8, 4489, Eq. (14).
        γ_rot ≤ 0 always (stabilizing); vanishes when Ω_φ = 0.
        """
        if self.omega_phi == 0.0:
            return 0.0
        x = self.omega_phi * tau  # dimensionless: Ω_φ τ_eff
        return -(x**2 / tau) / (1.0 + x**2)

    def growth_rate(self) -> float:
        """
        Total RWM growth rate including rotation stabilization.

        γ_total = γ_wall + γ_rot

        Bondeson & Ward 1994, Eq. (4); Fitzpatrick 2001, Eq. (14).
        γ_total < 0 indicates rotation-stabilized regime.
        """
        tau = self.tau_eff()
        gw = self._gamma_wall(tau)
        if gw >= _IDEAL_KINK_RATE:
            return gw  # ideal kink; rotation cannot stabilize
        return gw + self._gamma_rot(tau)

    # ── critical rotation ────────────────────────────────────────────────────

    def critical_rotation(self) -> float:
        """
        Minimum toroidal rotation for passive RWM stabilization (no coil feedback).

        Derived by setting γ_wall + γ_rot = 0 with the Fitzpatrick (2001) γ_rot:

            γ_wall = A/τ_eff,  γ_rot = −(Ωτ)²/(τ(1+(Ωτ)²))
            A = (β_N − β_N,nowall)/(β_N,wall − β_N)

        Solving:  (Ω_crit τ_eff)² = A/(1−A)
            → Ω_crit = (1/τ_eff) × √[A / (1−A)]

        Fitzpatrick 2001, Phys. Plasmas 8, 4489, Eq. (14) (γ_rot source).
        Bondeson & Ward 1994, Phys. Rev. Lett. 72, 2709 (rotation-stabilization concept).

        Returns 0 when β_N ≤ β_N,nowall (already stable).
        Returns ∞ when β_N ≥ β_N,wall (ideal kink; rotation cannot stabilize).
        """
        tau = self.tau_eff()
        if tau <= 0.0:
            return math.inf
        if self.beta_n <= self.beta_n_nowall:
            return 0.0
        if self.beta_n >= self.beta_n_wall:
            return math.inf
        # A = normalised distance into the unstable window
        a = (self.beta_n - self.beta_n_nowall) / (self.beta_n_wall - self.beta_n)
        if a >= 1.0:
            # γ_wall ≥ 1/τ; γ_rot saturates at −1/τ; rotation cannot fully cancel
            return math.inf
        return math.sqrt(a / (1.0 - a)) / tau


class RWMFeedbackController:
    """
    Sensor-coil PD feedback for RWM stabilization.

    The closed-loop effective growth rate combines wall physics (including
    rotation) with proportional feedback suppression:

        γ_eff = γ_total − G_p M_coil γ_total / (1 + γ_total τ_ctrl)

    Garofalo et al. 2002, Phys. Plasmas 9, 1997, Sec. III.
    """

    def __init__(
        self,
        n_sensors: int,
        n_coils: int,
        G_p: float,
        G_d: float,
        tau_controller: float = 1e-4,
        M_coil: float = 1.0,
    ) -> None:
        self.n_sensors = n_sensors
        self.n_coils = n_coils
        self.G_p = G_p
        self.G_d = G_d
        self.tau_controller = tau_controller
        self.M_coil = M_coil
        self.prev_B_r = np.zeros(n_sensors)

    def step(self, B_r_sensors: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute feedback coil currents from sensor measurements.

        I = G_p B_r + G_d dB_r/dt
        """
        dB_dt = np.zeros_like(B_r_sensors) if dt <= 0.0 else (B_r_sensors - self.prev_B_r) / dt
        self.prev_B_r = B_r_sensors.copy()

        signal = self.G_p * B_r_sensors + self.G_d * dB_dt
        if self.n_sensors == self.n_coils:
            return signal
        return np.full(self.n_coils, float(np.mean(signal)))

    def effective_growth_rate(self, rwm: RWMPhysics) -> float:
        """
        Closed-loop growth rate.

        γ_eff = γ_total − G_p M_coil γ_total / (1 + γ_total τ_ctrl)

        Garofalo et al. 2002, Phys. Plasmas 9, 1997, Eq. (5).
        Uses γ_total from RWMPhysics (includes rotation if Ω_φ ≠ 0).
        """
        gamma = rwm.growth_rate()
        if gamma == 0.0:
            return 0.0
        if gamma >= _IDEAL_KINK_RATE:
            return gamma
        stabilization = self.G_p * self.M_coil * gamma / (1.0 + gamma * self.tau_controller)
        return gamma - stabilization

    def is_stabilized(self, rwm: RWMPhysics) -> bool:
        """True when the closed-loop growth rate is negative."""
        return self.effective_growth_rate(rwm) < 0.0


class RWMStabilityAnalysis:
    """Utility methods for stability-window and gain analysis."""

    @staticmethod
    def required_feedback_gain(
        beta_n: float,
        beta_n_nowall: float,
        beta_n_wall: float,
        tau_wall: float,
        tau_controller: float,
        M_coil: float = 1.0,
        omega_phi: float = 0.0,
        wall_radius: float | None = None,
        plasma_radius: float | None = None,
    ) -> float:
        """
        Minimum G_p such that γ_eff < 0.

        From γ_eff = γ_total (1 − G_p M_coil / (1 + γ_total τ_ctrl)) < 0 :
            G_p > (1 + γ_total τ_ctrl) / M_coil

        Garofalo et al. 2002, Phys. Plasmas 9, 1997, Sec. III.
        Returns 0 when already stable; ∞ for ideal kink.
        """
        rwm = RWMPhysics(
            beta_n,
            beta_n_nowall,
            beta_n_wall,
            tau_wall,
            omega_phi=omega_phi,
            wall_radius=wall_radius,
            plasma_radius=plasma_radius,
        )
        gamma = rwm.growth_rate()
        if gamma <= 0.0:
            return 0.0
        if gamma >= _IDEAL_KINK_RATE:
            return float("inf")
        return (1.0 + gamma * tau_controller) / M_coil
