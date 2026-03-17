# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
from __future__ import annotations

import math

import numpy as np

# Boundary-layer thickness for chattering suppression.
# Replaces sign(s) with s/(|s|+δ).
# Slotine & Li 1991, "Applied Nonlinear Control", Prentice Hall, Ch. 7, §7.2.
_BOUNDARY_LAYER_DELTA: float = 0.01  # dimensionless

# Minimum denominator guard to prevent division by zero.
_EPS: float = 1e-12


def _sat(s: float, delta: float = _BOUNDARY_LAYER_DELTA) -> float:
    """Boundary-layer saturation function s/(|s|+δ).

    Slotine & Li 1991, Ch. 7, Eq. (7.15): replaces sign(s) to eliminate
    chattering while preserving the sliding behaviour inside |s| > δ.
    """
    return s / (abs(s) + delta)


class SuperTwistingSMC:
    """Second-order sliding mode controller: super-twisting algorithm.

    Convergence in finite time to s = 0 with continuous control output.

    Algorithm (Levant 1993, Int. J. Control 58, 1247, Eq. (1)):
        u(t)  = -α |s|^{1/2} sat(s)  +  v(t)
        v̇(t) = -β sat(s)

    Gain conditions for finite-time convergence under bounded disturbance L:
        α > √(2L),  β > L
    (Moreno & Osorio 2012, IEEE TAC 57, 1049, Theorem 1)

    Sliding surface follows Utkin 1992, "Sliding Modes in Control and
    Optimization", Springer, Ch. 2: s = e + c ė.
    """

    def __init__(self, alpha: float, beta: float, c: float, u_max: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.u_max = u_max
        self.v = 0.0

    def sliding_surface(self, e: float, de_dt: float) -> float:
        """s = e + c ė  (Utkin 1992, Ch. 2)."""
        return e + self.c * de_dt

    def step(self, e: float, de_dt: float, dt: float) -> float:
        """Advance one super-twisting step.

        Levant 1993, Eq. (1):
            u  = -α |s|^{1/2} sat(s) + v
            v̇  = -β sat(s)
        """
        s = self.sliding_surface(e, de_dt)

        if dt > 0:
            self.v -= self.beta * _sat(s) * dt

        self.v = np.clip(self.v, -self.u_max, self.u_max)

        u = -self.alpha * math.sqrt(abs(s)) * _sat(s) + self.v
        return float(np.clip(u, -self.u_max, self.u_max))


class VerticalStabilizer:
    """Vertical-position (VS) controller for an elongated tokamak.

    Vertical instability growth rate for elongated plasmas:
        γ ≈ (κ - 1) / τ_wall
    where κ is elongation, τ_wall is the resistive wall time.
    Lazarus et al. 1990, Nucl. Fusion 30, 111, §2.

    Real-time VS implementation at DIII-D:
    Humphreys et al. 2009, Nucl. Fusion 49, 115003.

    Restoring force on the plasma column (rigid-body model):
        F = -n μ₀ I_p² / (4π R₀) · Z  ≡  -K_vs · Z
    where n is the field-index (n < 0 for unstable equilibria),
    μ₀ = 4π×10⁻⁷ H/m (SI),  I_p in A,  R₀ in m.
    (Wesson 2004, "Tokamaks", Oxford, §3.7)
    """

    # Permeability of free space [H/m] (SI)
    MU0: float = 4.0 * math.pi * 1e-7

    def __init__(
        self,
        n_index: float,
        Ip_MA: float,
        R0: float,
        m_eff: float,
        tau_wall: float,
        smc: SuperTwistingSMC,
    ) -> None:
        self.n_index = n_index
        self.Ip = Ip_MA * 1e6  # convert MA → A
        self.R0 = R0
        self.m_eff = m_eff
        self.tau_wall = tau_wall
        self.smc = smc

    @property
    def K_vs(self) -> float:
        """Vertical restoring force coefficient [N/m].

        K_vs = -n μ₀ I_p² / (4π R₀)
        Wesson 2004, §3.7, Eq. (3.7.4).
        Negative n_index (n < 0) makes K_vs > 0 (unstable).
        """
        return -self.n_index * self.MU0 * self.Ip**2 / (4.0 * math.pi * self.R0)

    def step(self, Z_meas: float, Z_ref: float, dZ_dt_meas: float, dt: float) -> float:
        """Compute coil correction voltage.

        Plant: m_eff Z̈ = -K_vs Z + F_coil  (linearised, Humphreys 2009).
        e = Z_meas - Z_ref; SMC drives e → 0.
        """
        e = Z_meas - Z_ref
        return self.smc.step(e, dZ_dt_meas, dt)


def lyapunov_certificate(alpha: float, beta: float, L_max: float) -> bool:
    """Check gain conditions for finite-time convergence.

    Moreno & Osorio 2012, IEEE TAC 57, 1049, Theorem 1:
        α > √(2 L_max),  β > L_max
    """
    cond1 = alpha > math.sqrt(2.0 * max(L_max, _EPS))
    cond2 = beta > max(L_max, _EPS)
    return cond1 and cond2


def estimate_convergence_time(alpha: float, beta: float, L_max: float, s0: float) -> float:
    """Upper bound on time to reach s = 0.

    T ≤ 2 |s₀|^{1/2} / (α - √(2 L_max))
    Moreno & Osorio 2012, Eq. (18).
    """
    if L_max < 0 or alpha <= math.sqrt(2.0 * L_max):
        return float("inf")

    denom = alpha - math.sqrt(2.0 * L_max)
    if denom <= 0:
        return float("inf")

    return 2.0 * math.sqrt(abs(s0)) / denom
