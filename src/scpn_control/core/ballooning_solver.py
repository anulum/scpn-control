# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Ballooning Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Full Ballooning Equation Solver
# ──────────────────────────────────────────────────────────────────────
"""Ballooning-equation eigenvalue solver and marginal-stability search routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scipy.integrate import solve_ivp

from scpn_control.core.stability_mhd import QProfile


def _require_finite_scalar(name: str, value: float) -> float:
    """Return a finite scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _require_nonnegative_scalar(name: str, value: float) -> float:
    """Return a finite non-negative scalar or fail closed."""
    scalar = _require_finite_scalar(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return scalar


def _require_positive_scalar(name: str, value: float) -> float:
    """Return a finite positive scalar or fail closed."""
    scalar = _require_finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return scalar


@dataclass
class BallooningEigenResult:
    """Result of solving the ballooning equation for a single (s, alpha) pair."""

    theta: AnyFloatArray
    xi: AnyFloatArray
    is_stable: bool


class BallooningEquation:
    """
    Second-order ODE for ideal MHD ballooning stability in the s-alpha model.
    """

    def __init__(self, s: float, alpha: float, theta_max: float = 20 * np.pi, n_theta: int = 2001):
        self.s = _require_finite_scalar("s", s)
        self.alpha = _require_finite_scalar("alpha", alpha)
        self.theta_max = _require_positive_scalar("theta_max", theta_max)
        if not isinstance(n_theta, int) or n_theta < 3:
            raise ValueError("n_theta must be an integer >= 3")
        self.n_theta = n_theta

    def f(self, theta: float) -> float:
        """Field-line bending term coefficient."""
        return float(1.0 + (self.s * theta - self.alpha * np.sin(theta)) ** 2)

    def g(self, theta: float) -> float:
        """Curvature drive term coefficient."""
        return float(self.alpha * (np.cos(theta) + (self.s * theta - self.alpha * np.sin(theta)) * np.sin(theta)))

    def solve(self) -> BallooningEigenResult:
        """
        Solve via Newcomb shooting: ξ crossing zero signals instability.
        No zero crossing within [0, θ_max] means stable.
        """

        def eqs(t: float, y: AnyFloatArray) -> list[float]:
            u1, u2 = y
            du1 = u2 / self.f(t)
            du2 = -self.g(t) * u1
            return [du1, du2]

        class _ZeroCrossing:
            terminal = True
            direction = -1

            def __call__(self, t: float, y: AnyFloatArray) -> float:
                return float(y[0])

        t_span = (0.0, self.theta_max)
        # solve_ivp without t_eval to let it stop early smoothly, or with t_eval
        # and it will return up to the event.
        sol = solve_ivp(
            eqs,
            t_span,
            [1.0, 0.0],
            events=_ZeroCrossing(),
            rtol=1e-5,
            atol=1e-5,
        )

        # Newcomb criterion: zero crossing of ξ(θ) signals instability.
        # No zero crossing means the ballooning mode is stable.
        is_stable = len(sol.t_events[0]) == 0

        return BallooningEigenResult(
            theta=sol.t,
            xi=sol.y[0],
            is_stable=is_stable,
        )


def find_marginal_stability(s: float, alpha_min: float = 0.0, alpha_max: float = 2.0, tol: float = 1e-3) -> float:
    """
    Binary search for alpha_crit at fixed shear s.
    Returns the critical alpha (first stability boundary).
    """
    s = _require_finite_scalar("s", s)
    alpha_min = _require_nonnegative_scalar("alpha_min", alpha_min)
    alpha_max = _require_nonnegative_scalar("alpha_max", alpha_max)
    tol = _require_positive_scalar("tol", tol)
    if alpha_max <= alpha_min:
        raise ValueError("alpha_max must be greater than alpha_min")
    amin = alpha_min

    # Check lower bound
    if s <= 0.0:
        return 0.0

    eq_min = BallooningEquation(s, amin)
    if not eq_min.solve().is_stable:
        return 0.0  # Unstable even at min alpha

    # Find a valid unstable upper bound (don't jump to the 2nd stability region)
    amax = amin + 0.1
    found_unstable = False
    for _ in range(20):
        if not BallooningEquation(s, amax).solve().is_stable:
            found_unstable = True
            break
        amax += 0.2
        if amax > 3.0:
            break

    if not found_unstable:
        # Stable everywhere up to 3.0
        return alpha_max

    while (amax - amin) > tol:
        mid = (amin + amax) / 2.0
        eq = BallooningEquation(s, mid)
        if eq.solve().is_stable:
            amin = mid
        else:
            amax = mid

    return (amin + amax) / 2.0


def compute_stability_diagram(s_range: AnyFloatArray, alpha_min: float = 0.0, alpha_max: float = 2.0) -> FloatArray:
    """
    Compute alpha_crit(s) for an array of shear values.
    """
    s_values = np.asarray(s_range, dtype=float)
    if s_values.ndim != 1:
        raise ValueError("s_range must be a one-dimensional array")
    if not np.all(np.isfinite(s_values)):
        raise ValueError("s_range must contain only finite values")
    alpha_crit = np.zeros_like(s_values)
    for i, s in enumerate(s_values):
        alpha_crit[i] = find_marginal_stability(s, alpha_min, alpha_max)
    return alpha_crit


class BallooningStabilityAnalysis:
    """
    Performs ballooning stability analysis given a QProfile.
    """

    def analyze(self, q_profile: QProfile) -> FloatArray:
        """
        Extracts (s, alpha) at each radial point and returns per-radius stability margin.
        Positive margin means stable.
        margin = alpha_crit(s) - alpha_actual
        """
        margin = np.zeros_like(q_profile.rho)
        for i in range(len(q_profile.rho)):
            s = q_profile.shear[i]
            alpha = q_profile.alpha_mhd[i]
            if s <= 0.0:
                # s <= 0 is often stable to ideal ballooning, but standard s-alpha
                # breaks down. Let's say alpha_crit = 0.0 for s=0.
                alpha_crit = 0.0
            else:
                alpha_crit = find_marginal_stability(s)

            margin[i] = alpha_crit - alpha
        return margin
