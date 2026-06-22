# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Equilibrium shape and profile metrics from a poloidal-flux map

"""Reusable equilibrium shape/profile metrics from a reconstructed flux map.

These pure functions extract the standard macroscopic descriptors from a
poloidal-flux map ``psi`` on a uniform ``(R, Z)`` grid and the fitted profile
coefficients, independent of any particular reconstruction class so EFIT-style
inverses, the free-boundary kernel, and kinetic EFIT can share them:

- geometry (R0, minor radius, elongation, upper/lower triangularity) from the
  last-closed-flux-surface contour,
- internal inductance ``li(3)`` from the poloidal-field volume integral,
- poloidal beta from the pressure-profile volume integral,
- the edge safety factor ``q95`` from the toroidal flux function and a
  flux-surface contour line integral.

Flux convention: ``psi`` is the per-radian poloidal flux with
``Delta* psi = -mu0 R^2 p' - FF'`` and ``B_pol = |grad psi| / R`` (so Ampere's law,
the closed line integral of ``B_pol`` around a surface equals ``mu0 Ip``, holds); profiles ``p'(psi_N)`` and ``FF'(psi_N)`` are
polynomials in the normalised flux ``psi_N`` (0 at the axis, 1 at the boundary).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray

MU0 = 4.0e-7 * np.pi

_HAS_CONTOURPY = importlib.util.find_spec("contourpy") is not None


@dataclass(frozen=True)
class EquilibriumShape:
    """Macroscopic equilibrium descriptors extracted from a flux map."""

    R0: float
    a: float
    kappa: float
    delta_upper: float
    delta_lower: float
    q95: float
    beta_pol: float
    li: float


def plasma_boundary(psi: AnyFloatArray, R: AnyFloatArray, Z: AnyFloatArray) -> AnyFloatArray:
    """Trace the last closed flux surface as angle-sorted ``(R, Z)`` points.

    The plasma is the positive-flux region (fixed-boundary convention); the
    boundary is the outer ring of plasma cells. Returns an empty ``(0, 2)`` array
    when no positive-flux plasma exists.
    """
    psi_arr = np.asarray(psi, dtype=float)
    psi_max = float(np.max(psi_arr))
    if psi_max <= 0.0:
        return np.empty((0, 2), dtype=float)

    plasma_mask = psi_arr > max(1e-12, psi_max * 1e-6)
    padded = np.pad(plasma_mask, 1, mode="constant", constant_values=False)
    inner = padded[1:-1, 1:-1]
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    boundary = inner & ~interior

    r_idx, z_idx = np.nonzero(boundary)
    if r_idx.size == 0:
        return np.empty((0, 2), dtype=float)  # pragma: no cover - non-empty plasma always rings

    points = np.column_stack((np.asarray(R)[r_idx], np.asarray(Z)[z_idx]))
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return np.asarray(points[np.argsort(angles)])


def largest_flux_contour(
    psi_N: AnyFloatArray, R: AnyFloatArray, Z: AnyFloatArray, level: float
) -> AnyFloatArray | None:
    """Longest closed-ish ``psi_N = level`` contour as ``(R, Z)`` points, or None.

    Uses contourpy for a sub-grid contour (smooth shape/triangularity); returns
    None when contourpy is unavailable or the level has no contour.
    """
    if not _HAS_CONTOURPY:
        return None  # pragma: no cover - contourpy present on CI/viz installs
    from contourpy import contour_generator

    # contourpy expects z[y, x] with x along axis 1; psi_N is psi_N[R, Z].
    generator = contour_generator(
        x=np.asarray(R, dtype=float), y=np.asarray(Z, dtype=float), z=np.asarray(psi_N, dtype=float).T
    )
    lines = generator.lines(float(level))
    if not lines:
        return None
    return np.asarray(max((np.asarray(line, dtype=float) for line in lines), key=len))


def boundary_geometry(boundary: AnyFloatArray) -> tuple[float, float, float, float, float]:
    """Geometric (R0, a, kappa, delta_upper, delta_lower) from boundary points."""
    pts = np.asarray(boundary, dtype=float)
    r = pts[:, 0]
    z = pts[:, 1]
    r_max, r_min = float(np.max(r)), float(np.min(r))
    z_max, z_min = float(np.max(z)), float(np.min(z))
    r0 = 0.5 * (r_max + r_min)
    a = max(0.5 * (r_max - r_min), 1e-12)
    kappa = (z_max - z_min) / (2.0 * a)
    r_at_zmax = float(r[int(np.argmax(z))])
    r_at_zmin = float(r[int(np.argmin(z))])
    delta_upper = (r0 - r_at_zmax) / a
    delta_lower = (r0 - r_at_zmin) / a
    return r0, a, kappa, delta_upper, delta_lower


def poloidal_field(psi: AnyFloatArray, R: AnyFloatArray, Z: AnyFloatArray) -> AnyFloatArray:
    """Poloidal-field magnitude ``|B_pol| = |grad psi| / R`` on the grid.

    Uses the per-radian flux convention consistent with the Grad-Shafranov source
    ``Delta* psi = -mu0 R^2 p' - FF'`` (so the closed
    line integral of ``B_pol`` equals ``mu0 Ip`` by Ampere's law). This is the physically-correct absolute field; note the magnetic
    *diagnostic* response (``DiagnosticResponse``) uses a 1/(2 pi) scaled sensor
    convention internally, which is self-consistent for the inverse fit.
    """
    psi_arr = np.asarray(psi, dtype=float)
    r_axis = np.asarray(R, dtype=float)
    dpsi_dr = np.gradient(psi_arr, r_axis, axis=0, edge_order=2)
    dpsi_dz = np.gradient(psi_arr, np.asarray(Z, dtype=float), axis=1, edge_order=2)
    rr = r_axis[:, np.newaxis]
    b_r = -dpsi_dz / rr
    b_z = dpsi_dr / rr
    return np.asarray(np.sqrt(b_r**2 + b_z**2))


def _volume_element(R: AnyFloatArray, Z: AnyFloatArray) -> tuple[AnyFloatArray, float]:
    rr = np.asarray(R, dtype=float)[:, np.newaxis]
    d_r = float(np.mean(np.diff(np.asarray(R, dtype=float))))
    d_z = float(np.mean(np.diff(np.asarray(Z, dtype=float))))
    cell = 2.0 * np.pi * rr * d_r * d_z  # axisymmetric volume of a grid cell
    return np.broadcast_to(cell, (len(R), len(Z))), d_r * d_z


def internal_inductance(psi: AnyFloatArray, R: AnyFloatArray, Z: AnyFloatArray, ip: float, r0: float) -> float:
    """Normalised internal inductance ``li(3) = 2 int B_pol^2 dV / (mu0^2 Ip^2 R0)``."""
    psi_arr = np.asarray(psi, dtype=float)
    b_pol = poloidal_field(psi_arr, R, Z)
    dv, _ = _volume_element(R, Z)
    mask = psi_arr > max(1e-12, float(np.max(psi_arr)) * 1e-6)
    int_b2 = float(np.sum((b_pol**2 * dv)[mask]))
    denom = MU0**2 * ip**2 * max(r0, 1e-12)
    if denom <= 0.0:
        return 0.0
    return 2.0 * int_b2 / denom


def pressure_grid(psi_N: AnyFloatArray, p_coeffs: AnyFloatArray, psi_axis: float) -> AnyFloatArray:
    """Pressure ``p(psi_N) = psi_axis * sum_k p_k (1 - psi_N^{k+1})/(k+1)`` (p=0 at edge).

    Integrates the fitted ``p'(psi_N)`` from the boundary inward using
    ``dpsi = -psi_axis dpsi_N`` (psi_N = 1 - psi/psi_axis).
    """
    psi_n = np.asarray(psi_N, dtype=float)
    pressure = np.zeros_like(psi_n)
    for k, coeff in enumerate(np.asarray(p_coeffs, dtype=float)):
        pressure = pressure + coeff * (1.0 - psi_n ** (k + 1)) / (k + 1)
    return np.asarray(psi_axis * pressure)


def poloidal_beta(
    psi: AnyFloatArray,
    psi_N: AnyFloatArray,
    R: AnyFloatArray,
    Z: AnyFloatArray,
    p_coeffs: AnyFloatArray,
    psi_axis: float,
    a: float,
    ip: float,
) -> float:
    """Poloidal beta ``8 pi^2 a^2 <p> / (mu0 Ip^2)`` from the fitted pressure profile.

    ``<p>`` is the plasma-volume-averaged pressure; negative pressure (an
    unphysical magnetic-only profile fit) is clipped to zero for the average.
    """
    psi_arr = np.asarray(psi, dtype=float)
    pressure = np.clip(pressure_grid(psi_N, p_coeffs, psi_axis), 0.0, None)
    dv, _ = _volume_element(R, Z)
    mask = psi_arr > max(1e-12, float(np.max(psi_arr)) * 1e-6)
    volume = float(np.sum(dv[mask]))
    if volume <= 0.0 or ip == 0.0:
        return 0.0
    mean_p = float(np.sum((pressure * dv)[mask]) / volume)
    return 8.0 * np.pi**2 * a**2 * mean_p / (MU0 * ip**2)


def _toroidal_flux_function_sq(psi_N_value: float, ff_coeffs: AnyFloatArray, psi_axis: float, f_edge: float) -> float:
    """F^2(psi_N) = F_edge^2 + 2 psi_axis sum_k ff_k (1 - psi_N^{k+1})/(k+1)."""
    integral = 0.0
    for k, coeff in enumerate(np.asarray(ff_coeffs, dtype=float)):
        integral += coeff * (1.0 - psi_N_value ** (k + 1)) / (k + 1)
    return float(f_edge**2 + 2.0 * psi_axis * integral)


def safety_factor_q95(
    psi: AnyFloatArray,
    psi_N: AnyFloatArray,
    R: AnyFloatArray,
    Z: AnyFloatArray,
    ff_coeffs: AnyFloatArray,
    psi_axis: float,
    vacuum_rb_phi: float,
    *,
    surface: float = 0.95,
) -> float:
    """Edge safety factor ``q95 = (F/2pi) * contour-integral dl/(R^2 B_pol)``.

    Uses the toroidal flux function from the vacuum field plus the FF' integral,
    and a flux-surface contour line integral at ``psi_N = surface``. Returns NaN
    when the contour engine is unavailable or the surface has no closed contour.
    """
    r_axis = np.asarray(R, dtype=float)
    z_axis = np.asarray(Z, dtype=float)
    b_pol = poloidal_field(psi, R, Z)

    contour = largest_flux_contour(np.asarray(psi_N, dtype=float), R, Z, surface)
    if contour is None or contour.shape[0] < 3:
        return float("nan")

    from scipy.interpolate import RegularGridInterpolator

    b_interp = RegularGridInterpolator((r_axis, z_axis), b_pol, bounds_error=False, fill_value=None)
    seg = np.diff(contour, axis=0)
    dl = np.sqrt(np.sum(seg**2, axis=1))
    mid = 0.5 * (contour[:-1] + contour[1:])
    r_mid = mid[:, 0]
    b_mid = np.asarray(b_interp(mid), dtype=float)
    valid = (b_mid > 1e-12) & (r_mid > 1e-12)
    if not np.any(valid):
        return float("nan")  # pragma: no cover - defensive: real contours have valid B_pol/R points
    integrand = np.zeros_like(dl)
    integrand[valid] = dl[valid] / (r_mid[valid] ** 2 * b_mid[valid])
    line_integral = float(np.sum(integrand))

    f_surface = np.sqrt(max(_toroidal_flux_function_sq(float(surface), ff_coeffs, psi_axis, vacuum_rb_phi), 0.0))
    return float(f_surface / (2.0 * np.pi) * line_integral)


def cylindrical_q95(a: float, kappa: float, r0: float, ip: float, vacuum_rb_phi: float) -> float:
    """Elongation-corrected cylindrical edge safety factor (contour-free estimate).

    ``q ~ 2 pi a^2 F (1 + kappa^2) / (2 mu0 R0^2 |Ip|)`` with ``F = R0 B_phi0``;
    used as the q95 fallback when the flux-surface contour engine is unavailable.
    """
    if abs(ip) < 1e-30 or r0 <= 0.0:
        return float("nan")
    return float(2.0 * np.pi * a**2 * abs(vacuum_rb_phi) * (1.0 + kappa**2) / (2.0 * MU0 * r0**2 * abs(ip)))


def compute_equilibrium_shape(
    psi: AnyFloatArray,
    R: AnyFloatArray,
    Z: AnyFloatArray,
    p_coeffs: AnyFloatArray,
    ff_coeffs: AnyFloatArray,
    ip: float,
    vacuum_rb_phi: float,
) -> EquilibriumShape | None:
    """Full shape/profile metric set, or ``None`` when the flux map has no plasma."""
    psi_arr = np.asarray(psi, dtype=float)
    psi_axis = float(np.max(psi_arr))
    if psi_axis <= 1e-12:
        return None
    psi_n = np.clip(1.0 - psi_arr / psi_axis, 0.0, 1.0)

    # Prefer a smooth sub-grid boundary contour (accurate triangularity); fall back
    # to the grid-cell plasma ring when contourpy is unavailable.
    boundary = largest_flux_contour(psi_n, R, Z, 0.98)
    if boundary is None or boundary.shape[0] < 4:
        boundary = plasma_boundary(psi_arr, R, Z)  # pragma: no cover - contourpy-absent grid-ring fallback
    if boundary.shape[0] < 4:
        return None  # pragma: no cover - defensive: positive-flux plasma always rings a boundary

    r0, a, kappa, delta_upper, delta_lower = boundary_geometry(boundary)
    li = internal_inductance(psi_arr, R, Z, ip, r0)
    beta_pol = poloidal_beta(psi_arr, psi_n, R, Z, p_coeffs, psi_axis, a, ip)
    q95 = safety_factor_q95(psi_arr, psi_n, R, Z, ff_coeffs, psi_axis, vacuum_rb_phi)
    if not np.isfinite(q95):
        # No flux-surface contour (e.g. contourpy unavailable): fall back to the
        # contour-free cylindrical estimate so q95 stays finite everywhere. The CI
        # coverage job has contourpy, so this branch is exercised only where it is absent.
        q95 = cylindrical_q95(a, kappa, r0, ip, vacuum_rb_phi)  # pragma: no cover - contourpy-absent fallback
    return EquilibriumShape(
        R0=r0,
        a=a,
        kappa=kappa,
        delta_upper=delta_upper,
        delta_lower=delta_lower,
        q95=q95,
        beta_pol=beta_pol,
        li=li,
    )
