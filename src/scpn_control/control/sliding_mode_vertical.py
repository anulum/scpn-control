# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Sliding mode vertical.

"""Sliding-mode vertical-position controller with bounded actuator commands."""

from __future__ import annotations

import math

import numpy as np

# Boundary-layer thickness for chattering suppression.
# Replaces sign(s) with s/(|s|+δ).
# Slotine & Li 1991, "Applied Nonlinear Control", Prentice Hall, Ch. 7, §7.2.
_BOUNDARY_LAYER_DELTA: float = 0.01  # same units as the sliding surface


def _sat(s: float, delta: float = _BOUNDARY_LAYER_DELTA) -> float:
    """Boundary-layer saturation function s/(|s|+δ).

    Slotine & Li 1991, Ch. 7, Eq. (7.15): replaces sign(s) to eliminate
    chattering while preserving the sliding behaviour inside |s| > δ.
    """
    return s / (abs(s) + delta)


class SuperTwistingSMC:
    """Second-order sliding mode controller: super-twisting algorithm.

    Discrete, boundary-smoothed and actuator-limited approximation.

    Algorithm (Levant 1993, Int. J. Control 58, 1247, Eq. (1)):
        u(t)  = -α |s|^{1/2} sat(s)  +  v(t)
        v̇(t) = -β sat(s)

    The ideal continuous-time sign-law analysis does not certify finite-time
    convergence for this smoothed and saturated implementation.

    Sliding surface follows Utkin 1992, "Sliding Modes in Control and
    Optimisation", Springer, Ch. 2: s = e + c ė. When e is metres, c is
    seconds and the fixed boundary-layer delta is metres. The emitted
    command has caller-defined units; no coil-voltage conversion is supplied.
    """

    def __init__(self, alpha: float, beta: float, c: float, u_max: float) -> None:
        for name, value in (("alpha", alpha), ("beta", beta), ("c", c), ("u_max", u_max)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.u_max = u_max
        self.v = 0.0

    def sliding_surface(self, e: float, de_dt: float) -> float:
        """Return finite ``s = e + c ė`` for finite error observations."""
        if not math.isfinite(e) or not math.isfinite(de_dt):
            raise ValueError("error and derivative must be finite")
        s = e + self.c * de_dt
        if not math.isfinite(s):
            raise ValueError("sliding surface must be finite")
        return s

    def step(self, e: float, de_dt: float, dt: float) -> float:
        """Advance one super-twisting step.

        Levant 1993, Eq. (1):
            u  = -α |s|^{1/2} sat(s) + v
            v̇  = -β sat(s)
        """
        for name, value in (("alpha", self.alpha), ("beta", self.beta), ("c", self.c), ("u_max", self.u_max)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")
        s = self.sliding_surface(e, de_dt)
        sat_s = _sat(s)
        integral_change = self.beta * sat_s * dt
        if not math.isfinite(integral_change):
            raise ValueError("integral update must be finite")
        candidate_v = self.v - integral_change
        if not math.isfinite(candidate_v):
            raise ValueError("integral state must be finite")
        candidate_v = float(np.clip(candidate_v, -self.u_max, self.u_max))
        raw_u = -self.alpha * math.sqrt(abs(s)) * sat_s + candidate_v
        if not math.isfinite(raw_u):
            raise ValueError("control command must be finite")
        self.v = candidate_v
        return float(np.clip(raw_u, -self.u_max, self.u_max))


class VerticalStabilizer:
    """Vertical-position (VS) controller for an elongated tokamak.

    Vertical instability growth rate for elongated plasmas:
        γ ≈ (κ - 1) / τ_wall
    where κ is elongation, τ_wall is the resistive wall time.
    Lazarus et al. 1990, Nucl. Fusion 30, 111, §2.

    Real-time VS implementation at DIII-D:
    Humphreys et al. 2009, Nucl. Fusion 49, 115003.

    Model force on the plasma column (rigid-body approximation):
        F = n μ₀ I_p² / (4π R₀) · Z  ≡  -K_vs · Z
    where n is a signed model index. This is not a calibrated facility
    equilibrium or stability certificate;
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
        if not math.isfinite(n_index):
            raise ValueError("n_index must be finite")
        if not math.isfinite(Ip_MA) or Ip_MA < 0.0:
            raise ValueError("Ip_MA must be finite and non-negative")
        for name, value in (("R0", R0), ("m_eff", m_eff), ("tau_wall", tau_wall)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.n_index = n_index
        self.Ip = Ip_MA * 1e6  # convert MA → A
        if not math.isfinite(self.Ip):
            raise ValueError("Ip_MA conversion must be finite")
        self.R0 = R0
        self.m_eff = m_eff
        self.tau_wall = tau_wall
        self.smc = smc
        _ = self.K_vs

    @property
    def K_vs(self) -> float:
        """Vertical restoring force coefficient [N/m].

        K_vs = -n μ₀ I_p² / (4π R₀)
        Wesson 2004, §3.7, Eq. (3.7.4).
        In the declared ``-K_vs Z`` force convention, positive ``K_vs`` is
        restoring and negative ``K_vs`` is destabilizing.
        """
        if not math.isfinite(self.n_index):
            raise ValueError("n_index must be finite")
        if not math.isfinite(self.Ip) or self.Ip < 0.0:
            raise ValueError("Ip_MA converted current must be finite and non-negative")
        for name, value in (("R0", self.R0), ("m_eff", self.m_eff), ("tau_wall", self.tau_wall)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        try:
            coefficient = -self.n_index * self.MU0 * self.Ip**2 / (4.0 * math.pi * self.R0)
        except OverflowError as exc:
            raise ValueError("K_vs must be finite") from exc
        if not math.isfinite(coefficient):
            raise ValueError("K_vs must be finite")
        return coefficient

    def step(self, Z_meas: float, Z_ref: float, dZ_dt_meas: float, dt: float) -> float:
        """Compute a bounded, uncalibrated vertical correction command.

        Plant: m_eff Z̈ = -K_vs Z + F_coil  (linearised, Humphreys 2009).
        e = Z_meas - Z_ref; SMC drives e → 0.
        """
        _ = self.K_vs
        e = Z_meas - Z_ref
        return self.smc.step(e, dZ_dt_meas, dt)


def lyapunov_certificate(alpha: float, beta: float, L_max: float) -> bool:
    """Screen ideal sign-law gains, without certifying the discrete controller.

    This compatibility function checks ``α > √(2 L_max), β > L_max`` for
    finite nonnegative ``L_max``. Those inequalities alone do not establish
    a Lyapunov proof for the boundary-smoothed, saturated, sampled controller.
    """
    if not all(math.isfinite(value) for value in (alpha, beta, L_max)):
        return False
    if alpha <= 0.0 or beta <= 0.0 or L_max < 0.0:
        return False
    return alpha > math.sqrt(2.0) * math.sqrt(L_max) and beta > L_max


def estimate_convergence_time(alpha: float, beta: float, L_max: float, s0: float) -> float:
    """Return an idealized sign-law expression, never a runtime guarantee.

    Compute ``2 √|s₀| / (α - √(2 L_max))`` only for admitted idealized
    gains and finite inputs. The result is not an upper bound for this runtime.
    """
    if not all(math.isfinite(value) for value in (alpha, beta, L_max, s0)):
        return float("inf")
    if not lyapunov_certificate(alpha, beta, L_max):
        return float("inf")

    denom = alpha - math.sqrt(2.0) * math.sqrt(L_max)
    return 2.0 * math.sqrt(abs(s0)) / denom
