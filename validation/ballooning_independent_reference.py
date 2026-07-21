#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent s-alpha ballooning marginal-stability reference
"""Independent numerical s-alpha ballooning first-stability boundary.

This module solves the ideal-MHD s-alpha ballooning equation
(Connor, Hastie & Taylor, Phys. Rev. Lett. 40 (1978) 396) numerically to find
the marginal normalised pressure gradient ``alpha_crit(s)``, used to cross-check
the analytic approximation ``alpha_crit = s(1 - s/2)`` (low shear) / ``0.6 s``
(high shear) in :func:`scpn_control.core.stability_mhd.ballooning_stability`.

Independence is structural: the reference integrates the ballooning ODE

    d/dtheta[(1 + Lambda^2) dF/dtheta] + alpha[cos theta + Lambda sin theta] F = 0,
    Lambda(theta) = s theta - alpha sin theta,

and locates the marginal ``alpha`` at which the nodeless ground-state eigenmode
first develops a node (the ideal first-stability boundary). It shares no code
with the production algebraic fit, so agreement is genuine validation rather than
a re-statement of the same formula.

Validity: the simple s-alpha first-stability boundary is well defined for moderate
magnetic shear (``s`` >~ 0.4). At low shear the s-alpha diagram opens access to a
second-stability region and the single first-node criterion no longer returns the
physical first boundary; the reference therefore reports a validity flag and the
cross-check restricts quantitative comparison to ``s`` in ``[0.5, 2.0]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]

_THETA_SPAN = 6.0 * np.pi  # integration half-domain in ballooning angle
_N_THETA = 6000
_RTOL = 1.0e-8
_ATOL = 1.0e-10
_ALPHA_MAX = 3.0
_BISECT_ITERS = 44


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def ballooning_mode_has_node(
    s: float,
    alpha: float,
    *,
    theta_span: float = _THETA_SPAN,
    n_theta: int = _N_THETA,
) -> bool:
    """Return True when the s-alpha ballooning ground state develops a node.

    Integrates the ballooning ODE from ``-theta_span`` to ``+theta_span`` with a
    decaying boundary condition and counts interior sign changes of ``F``. A node
    marks the ideal ballooning instability, so the marginal ``alpha_crit`` is the
    smallest ``alpha`` for which this returns True.
    """
    s = _finite("s", s)
    alpha = _finite("alpha", alpha, positive=False)
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if not isinstance(n_theta, int) or isinstance(n_theta, bool) or n_theta < 100:
        raise ValueError("n_theta must be an integer >= 100")
    theta_span = _finite("theta_span", theta_span, positive=True)

    theta = np.linspace(-theta_span, theta_span, n_theta)

    def rhs(t: float, y: list[float]) -> list[float]:
        f, g = y  # g = (1 + Lambda^2) F'
        lam = s * t - alpha * np.sin(t)
        p = 1.0 + lam * lam
        return [g / p, -alpha * (np.cos(t) + lam * np.sin(t)) * f]

    sol = solve_ivp(
        rhs,
        (theta[0], theta[-1]),
        [1.0e-8, 0.0],
        t_eval=theta,
        rtol=_RTOL,
        atol=_ATOL,
        method="RK45",
    )
    f = np.asarray(sol.y[0], dtype=np.float64)
    signs = np.sign(f)
    return bool(int(np.sum(signs[:-1] * signs[1:] < 0)) >= 1)


def ballooning_alpha_crit(s: float) -> float:
    """Marginal normalised pressure gradient ``alpha_crit`` at magnetic shear ``s``.

    Found by bisection on :func:`ballooning_mode_has_node` in ``[0, alpha_max]``.
    Returns ``alpha_max`` when no node is found below it (the low-shear
    second-stability regime, where this simple boundary is not physical — see the
    module docstring).
    """
    s = _finite("s", s, positive=True)
    lo, hi = 0.0, _ALPHA_MAX
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if ballooning_mode_has_node(s, mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class BallooningReferencePoint:
    """One numerically resolved first-stability point."""

    s: float
    alpha_crit_numeric: float
    first_stability_regime: bool  # False in the low-shear second-stability regime


def ballooning_reference_curve(shear_values: FloatArray) -> list[BallooningReferencePoint]:
    """Numerically resolve ``alpha_crit`` across a shear grid.

    A point is flagged out of the first-stability regime when the bisection runs
    into the ``alpha_max`` ceiling (no first node found), which happens at low
    shear where second-stability access breaks the single-boundary picture.
    """
    grid = np.asarray(shear_values, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError("shear_values must be a non-empty 1-D array")
    points: list[BallooningReferencePoint] = []
    for s in grid:
        ac = ballooning_alpha_crit(float(s))
        in_first = ac < 0.98 * _ALPHA_MAX
        points.append(BallooningReferencePoint(s=float(s), alpha_crit_numeric=ac, first_stability_regime=in_first))
    return points
