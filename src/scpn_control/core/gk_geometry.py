# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Miller Flux-Tube Geometry
"""
Miller parameterisation of local magnetic equilibrium geometry for flux-tube gyrokinetic calculations.

Computes metric coefficients, field-line curvature, and the Jacobian
on a ballooning-angle grid from (R0, a, kappa, delta, q, s_hat, ...).

Reference: Miller et al., Phys. Plasmas 5 (1998) 973.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass
class MillerGeometry:
    """Flux-tube geometry on a ballooning-angle grid.

    All arrays have shape ``(n_theta,)``.
    """

    theta: NDArray[np.float64]  # ballooning angle grid
    R: NDArray[np.float64]  # major radius R(theta) [m]
    Z: NDArray[np.float64]  # vertical position Z(theta) [m]
    B_mag: NDArray[np.float64]  # |B|(theta) [T]
    jacobian: NDArray[np.float64]  # flux-surface Jacobian
    g_rr: NDArray[np.float64]  # |grad r|^2
    g_rt: NDArray[np.float64]  # grad r . grad theta
    g_tt: NDArray[np.float64]  # |grad theta|^2
    metric_determinant: NDArray[np.float64]  # g^rr g^tt - (g^rtheta)^2
    kappa_n: NDArray[np.float64]  # normal curvature
    kappa_g: NDArray[np.float64]  # geodesic curvature
    b_dot_grad_theta: NDArray[np.float64]  # B . grad(theta) / B


def miller_geometry(
    R0: float,
    a: float,
    rho: float,
    kappa: float = 1.0,
    delta: float = 0.0,
    s_kappa: float = 0.0,
    s_delta: float = 0.0,
    q: float = 1.4,
    s_hat: float = 0.78,
    alpha_MHD: float = 0.0,
    dR_dr: float = 0.0,
    B0: float = 5.3,
    n_theta: int = 64,
    n_period: int = 2,
) -> MillerGeometry:
    """Compute Miller geometry on a ballooning-angle grid.

    Parameters
    ----------
    R0 : float
        Major radius [m].
    a : float
        Minor radius [m].
    rho : float
        Normalised flux coordinate (r/a).
    kappa, delta : float
        Elongation and triangularity.
    s_kappa, s_delta : float
        Shear of elongation/triangularity: (r/kappa)(dkappa/dr), etc.
    q, s_hat : float
        Safety factor and magnetic shear s = (r/q)(dq/dr).
    alpha_MHD : float
        Shafranov shift parameter alpha = -q^2 R0 (dp/dr) / (B0^2 / 2mu0).
    dR_dr : float
        Shafranov shift gradient dR_axis/dr (typically negative).
    B0 : float
        Toroidal field at R0 [T].
    n_theta : int
        Grid points per 2*pi period.
    n_period : int
        Number of poloidal periods (ballooning copies).
    """
    R0 = _finite_scalar("R0", R0, positive=True)
    a = _finite_scalar("a", a, positive=True)
    rho = _finite_scalar("rho", rho)
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must lie in (0, 1]")
    kappa = _finite_scalar("kappa", kappa, positive=True)
    delta = _finite_scalar("delta", delta)
    if not -1.0 < delta < 1.0:
        raise ValueError("delta must lie in (-1, 1)")
    s_kappa = _finite_scalar("s_kappa", s_kappa)
    s_delta = _finite_scalar("s_delta", s_delta)
    q = _finite_scalar("q", q, positive=True)
    _finite_scalar("s_hat", s_hat)
    _finite_scalar("alpha_MHD", alpha_MHD)
    dR_dr = _finite_scalar("dR_dr", dR_dr)
    B0 = _finite_scalar("B0", B0, positive=True)
    n_theta = _positive_int("n_theta", n_theta)
    n_period = _positive_int("n_period", n_period)
    if n_theta * n_period < 4:
        raise ValueError("theta grid must contain at least four samples for metric and curvature derivatives")

    r = rho * a
    theta: NDArray[np.float64] = np.linspace(
        -n_period * np.pi, n_period * np.pi, n_theta * n_period, endpoint=False
    ).astype(np.float64)

    # Miller et al. Eq. (1)-(2): flux surface shape
    delta_angle = np.arcsin(delta)
    R_s = R0 + r * np.cos(theta + delta_angle * np.sin(theta)) + dR_dr * r
    Z_s = kappa * r * np.sin(theta)
    if np.any(R_s <= 0.0):
        raise ValueError("Miller surface major radius must remain positive")

    # Derivatives w.r.t. theta
    dR_dt = -r * np.sin(theta + delta_angle * np.sin(theta)) * (1 + delta_angle * np.cos(theta))
    dZ_dt = kappa * r * np.cos(theta)

    # Derivatives w.r.t. r (at constant theta). The flux-surface shape functions
    # kappa(r) and delta(r) vary radially, so the elongation- and triangularity-
    # shear (s_kappa, s_delta) enter the radial derivatives (Miller et al. 1998,
    # Eqs. 36-37; d(arcsin delta)/dr = s_delta / r, dkappa/dr = kappa * s_kappa / r):
    #   dR/dr = dR0/dr + cos(theta_R) - s_delta * sin(theta) * sin(theta_R)
    #   dZ/dr = kappa * sin(theta) * (1 + s_kappa)
    # where theta_R = theta + arcsin(delta) * sin(theta). Dropping the s_kappa /
    # s_delta terms leaves the circular and fixed-shaping (s=0) metric exact but
    # corrupts g_rr/g_rt/g_tt for finite shaping-shear; verified against an
    # independent finite-difference reference in validation/validate_gk_geometry_independent.py.
    theta_R = theta + delta_angle * np.sin(theta)
    dR_dr_tot = np.cos(theta_R) + dR_dr - s_delta * np.sin(theta) * np.sin(theta_R)
    dZ_dr_r = kappa * np.sin(theta) * (1.0 + s_kappa)

    # Jacobian of (r, theta) → (R, Z): J = dR/dr * dZ/dtheta - dR/dtheta * dZ/dr
    jac = dR_dr_tot * dZ_dt - dR_dt * dZ_dr_r
    jac = np.where(np.abs(jac) < 1e-30, 1e-30, jac)

    # |grad r|^2 = (dR/dtheta)^2 + (dZ/dtheta)^2) / J^2
    g_rr = (dR_dt**2 + dZ_dt**2) / jac**2

    # grad r . grad theta = -(dR/dr * dR/dtheta + dZ/dr * dZ/dtheta) / J^2
    g_rt = -(dR_dr_tot * dR_dt + dZ_dr_r * dZ_dt) / jac**2

    # |grad theta|^2 = (dR/dr^2 + dZ/dr^2) / J^2
    g_tt = (dR_dr_tot**2 + dZ_dr_r**2) / jac**2
    metric_determinant = g_rr * g_tt - g_rt**2

    # Toroidal field on a local flux surface: B_phi = B0 * R0 / R.
    B_phi = B0 * R0 / R_s

    # Local poloidal field from the safety-factor pitch relation.
    abs_jac_over_r = np.abs(jac) / max(r, 1e-6)
    B_p = (r * B_phi) / (q * R_s * abs_jac_over_r + 1e-30)

    B_mag = np.sqrt(B_phi**2 + B_p**2)

    # Parallel derivative metric b·∇theta = B_p |∇theta| / |B|.
    b_dot_grad_theta = B_p * np.sqrt(g_tt) / np.maximum(B_mag, 1e-30)

    # Curvature from Miller flux-surface geometry (Miller 1998, Eqs. 18-19;
    # Beer, Cowley & Hammett 1995, Eq. 2.8).
    # Geometric curvature of the (R, Z) contour: kappa = |r' x r''| / |r'|^3
    d2R_dt2 = np.gradient(dR_dt, theta)
    d2Z_dt2 = np.gradient(dZ_dt, theta)
    dl_dt = np.sqrt(dR_dt**2 + dZ_dt**2)
    dl_dt = np.maximum(dl_dt, 1e-30)

    # Signed curvature of the poloidal cross-section contour
    curvature_cross = (dR_dt * d2Z_dt2 - dZ_dt * d2R_dt2) / dl_dt**3

    # Normal curvature: projection onto grad-psi (radial) direction,
    # plus the toroidal curvature contribution 1/R
    kappa_n = curvature_cross / np.maximum(np.sqrt(g_rr), 1e-30) + (1.0 / R_s)
    # Sign convention: negative on outboard (unfavorable)
    kappa_n = -np.abs(kappa_n) * np.sign(np.cos(theta) + 1e-30)

    # Geodesic curvature from poloidal variation of |B|
    dB_ds = np.gradient(B_mag, theta) / dl_dt
    kappa_g = -dB_ds / np.maximum(B_mag, 1e-30)

    return MillerGeometry(
        theta=np.asarray(theta, dtype=np.float64),
        R=R_s,
        Z=Z_s,
        B_mag=B_mag,
        jacobian=jac,
        g_rr=g_rr,
        g_rt=g_rt,
        g_tt=g_tt,
        metric_determinant=metric_determinant,
        kappa_n=kappa_n,
        kappa_g=kappa_g,
        b_dot_grad_theta=b_dot_grad_theta,
    )


def circular_geometry(
    R0: float = 2.78,
    a: float = 1.0,
    rho: float = 0.5,
    q: float = 1.4,
    s_hat: float = 0.78,
    B0: float = 2.0,
    n_theta: int = 64,
    n_period: int = 2,
) -> MillerGeometry:
    """Circular cross-section limit (kappa=1, delta=0).

    Useful for verification against analytic results and the
    Cyclone Base Case (Dimits et al. 2000).
    """
    return miller_geometry(
        R0=R0,
        a=a,
        rho=rho,
        kappa=1.0,
        delta=0.0,
        s_kappa=0.0,
        s_delta=0.0,
        q=q,
        s_hat=s_hat,
        alpha_MHD=0.0,
        dR_dr=0.0,
        B0=B0,
        n_theta=n_theta,
        n_period=n_period,
    )
