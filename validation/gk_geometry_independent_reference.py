#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — independent finite-difference Miller geometry reference
"""Independent Miller local-equilibrium geometry reference implementation.

This module is a deliberately *independent* second implementation of the Miller
(Phys. Plasmas 5 (1998) 973) local-equilibrium metric, used to cross-check the
production analytic geometry in :mod:`scpn_control.core.gk_geometry`.

Independence is structural: the metric coefficients here are obtained by
high-order central *finite differences* of the flux-surface definition

    R(r, theta) = R0(r) + r cos(theta + arcsin(delta(r)) sin theta)
    Z(r, theta) = kappa(r) r sin(theta)

with the radial variation of the shape functions supplied by the local shears

    kappa(r) = kappa0 [1 + s_kappa (r - r0) / r0]
    delta(r) = delta0 + sqrt(1 - delta0^2) s_delta (r - r0) / r0
    R0(r)    = R0_0 + (dR0/dr) r

The reference therefore differentiates the surface *definition* directly and
cannot share any error in the production analytic derivatives. Agreement between
the two is genuine validation rather than a self-consistency tautology.

The ``R0(r) = R0_0 + (dR0/dr) r`` centre convention deliberately mirrors the
production surface in :mod:`scpn_control.core.gk_geometry` (``R0`` treated as the
magnetic-axis major radius with the Shafranov shift accumulating to the surface),
so the cross-check isolates the *metric* rather than a surface-centre convention
difference. The radial derivative ``dR/dr`` is identical under either centre
convention, so the metric coefficients are validated regardless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IndependentMillerMetric:
    """Finite-difference Miller metric sampled on a caller-supplied theta grid.

    All arrays share the shape of the input ``theta`` grid.
    """

    theta: FloatArray  # ballooning angle [rad]
    R: FloatArray  # major radius R(theta) [m]
    Z: FloatArray  # vertical position Z(theta) [m]
    dR_dr: FloatArray  # dR/dr at constant theta [dimensionless]
    dZ_dr: FloatArray  # dZ/dr at constant theta [dimensionless]
    jacobian: FloatArray  # (r, theta) -> (R, Z) Jacobian [m^2]
    g_rr: FloatArray  # |grad r|^2 [dimensionless]
    g_rt: FloatArray  # grad r . grad theta [m^-1]
    g_tt: FloatArray  # |grad theta|^2 [m^-2]
    B_toroidal: FloatArray  # B_phi(theta) = B0 R0 / R [T]
    b_dot_grad_theta: FloatArray  # B . grad(theta) / |B| [m^-1]


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _surface(
    r: float,
    theta: FloatArray,
    *,
    R0_0: float,
    r0: float,
    kappa0: float,
    delta0: float,
    s_kappa: float,
    s_delta: float,
    dR0_dr: float,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(R, Z)`` on the surface labelled by minor radius ``r``."""
    kappa = kappa0 * (1.0 + s_kappa * (r - r0) / r0)
    delta = delta0 + np.sqrt(1.0 - delta0**2) * s_delta * (r - r0) / r0
    R0 = R0_0 + dR0_dr * r
    shape_angle = theta + np.arcsin(delta) * np.sin(theta)
    R = R0 + r * np.cos(shape_angle)
    Z = kappa * r * np.sin(theta)
    return R.astype(np.float64), Z.astype(np.float64)


def independent_miller_metric(
    *,
    R0: float,
    a: float,
    rho: float,
    kappa: float = 1.0,
    delta: float = 0.0,
    s_kappa: float = 0.0,
    s_delta: float = 0.0,
    q: float = 1.4,
    dR_dr: float = 0.0,
    B0: float = 5.3,
    theta: FloatArray,
    eps_r_rel: float = 1.0e-5,
    eps_theta: float = 1.0e-5,
) -> IndependentMillerMetric:
    """Compute the Miller metric by finite differences of the surface definition.

    Parameters mirror :func:`scpn_control.core.gk_geometry.miller_geometry` so the
    two can be compared on a shared ``theta`` grid. Derivatives use a fourth-order
    central stencil, which keeps the truncation error near machine precision for
    the smooth Miller surface while remaining structurally independent of the
    production analytic derivatives.

    Parameters
    ----------
    R0, a : float
        Major and minor radius [m] (both positive).
    rho : float
        Normalised flux label ``r / a`` in ``(0, 1]``.
    kappa, delta : float
        Elongation and triangularity on the reference surface.
    s_kappa, s_delta : float
        Elongation and triangularity shear ``(r/kappa) dkappa/dr`` and
        ``(r/sqrt(1-delta^2)) ddelta/dr``.
    q : float
        Safety factor (positive), used for the poloidal-field pitch relation.
    dR_dr : float
        Shafranov-shift gradient ``dR0/dr``.
    B0 : float
        Toroidal field at ``R0`` [T] (positive).
    theta : numpy.ndarray
        Ballooning-angle grid [rad] to evaluate the metric on.
    eps_r_rel, eps_theta : float
        Relative radial and absolute poloidal finite-difference steps.

    Returns
    -------
    IndependentMillerMetric
        The finite-difference metric sampled on ``theta``.
    """
    R0 = _finite("R0", R0, positive=True)
    a = _finite("a", a, positive=True)
    rho = _finite("rho", rho)
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must lie in (0, 1]")
    kappa = _finite("kappa", kappa, positive=True)
    delta = _finite("delta", delta)
    if not -1.0 < delta < 1.0:
        raise ValueError("delta must lie in (-1, 1)")
    s_kappa = _finite("s_kappa", s_kappa)
    s_delta = _finite("s_delta", s_delta)
    q = _finite("q", q, positive=True)
    dR_dr = _finite("dR_dr", dR_dr)
    B0 = _finite("B0", B0, positive=True)
    eps_r_rel = _finite("eps_r_rel", eps_r_rel, positive=True)
    eps_theta = _finite("eps_theta", eps_theta, positive=True)

    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim != 1 or theta_arr.size < 1:
        raise ValueError("theta must be a non-empty 1-D array")

    r0 = rho * a
    shape = {
        "R0_0": R0,
        "r0": r0,
        "kappa0": kappa,
        "delta0": delta,
        "s_kappa": s_kappa,
        "s_delta": s_delta,
        "dR0_dr": dR_dr,
    }
    eps_r = eps_r_rel * a

    def surf(r: float, th: FloatArray) -> tuple[FloatArray, FloatArray]:
        return _surface(r, th, **shape)

    R_s, Z_s = surf(r0, theta_arr)
    if np.any(R_s <= 0.0):
        raise ValueError("Miller surface major radius must remain positive")

    # Radial derivatives (4th-order central) at the reference surface.
    Rp2, Zp2 = surf(r0 + 2.0 * eps_r, theta_arr)
    Rp1, Zp1 = surf(r0 + eps_r, theta_arr)
    Rm1, Zm1 = surf(r0 - eps_r, theta_arr)
    Rm2, Zm2 = surf(r0 - 2.0 * eps_r, theta_arr)
    dR_dr_arr = (-Rp2 + 8.0 * Rp1 - 8.0 * Rm1 + Rm2) / (12.0 * eps_r)
    dZ_dr_arr = (-Zp2 + 8.0 * Zp1 - 8.0 * Zm1 + Zm2) / (12.0 * eps_r)

    # Poloidal derivatives (4th-order central) at the reference surface.
    Ra2, Za2 = surf(r0, theta_arr + 2.0 * eps_theta)
    Ra1, Za1 = surf(r0, theta_arr + eps_theta)
    Rb1, Zb1 = surf(r0, theta_arr - eps_theta)
    Rb2, Zb2 = surf(r0, theta_arr - 2.0 * eps_theta)
    dR_dt = (-Ra2 + 8.0 * Ra1 - 8.0 * Rb1 + Rb2) / (12.0 * eps_theta)
    dZ_dt = (-Za2 + 8.0 * Za1 - 8.0 * Zb1 + Zb2) / (12.0 * eps_theta)

    jac = dR_dr_arr * dZ_dt - dR_dt * dZ_dr_arr
    jac = np.where(np.abs(jac) < 1e-30, 1e-30, jac)
    g_rr = (dR_dt**2 + dZ_dt**2) / jac**2
    g_rt = -(dR_dr_arr * dR_dt + dZ_dr_arr * dZ_dt) / jac**2
    g_tt = (dR_dr_arr**2 + dZ_dr_arr**2) / jac**2

    B_phi = B0 * R0 / R_s
    abs_jac_over_r = np.abs(jac) / max(r0, 1e-6)
    B_p = (r0 * B_phi) / (q * R_s * abs_jac_over_r + 1e-30)
    B_mag = np.sqrt(B_phi**2 + B_p**2)
    b_dot_grad_theta = B_p * np.sqrt(g_tt) / np.maximum(B_mag, 1e-30)

    return IndependentMillerMetric(
        theta=theta_arr,
        R=R_s,
        Z=Z_s,
        dR_dr=dR_dr_arr.astype(np.float64),
        dZ_dr=dZ_dr_arr.astype(np.float64),
        jacobian=jac.astype(np.float64),
        g_rr=g_rr.astype(np.float64),
        g_rt=g_rt.astype(np.float64),
        g_tt=g_tt.astype(np.float64),
        B_toroidal=B_phi.astype(np.float64),
        b_dot_grad_theta=b_dot_grad_theta.astype(np.float64),
    )
