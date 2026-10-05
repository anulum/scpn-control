# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — local magnetic saddle estimate.

"""Estimate an X-point from the local quadratic flux geometry."""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyFloatArray


def find_magnetic_xpoint(
    psi: AnyFloatArray, r_grid: AnyFloatArray, z_grid: AnyFloatArray
) -> tuple[float, float] | None:
    """Return an interior flux saddle or ``None`` when no local saddle exists.

    Central finite differences estimate the gradient and Hessian at each
    interior grid node. A Newton step to a stationary point is accepted only
    within one local grid interval and only when the Hessian has opposite-sign
    eigenvalues. This is a bounded local topology estimate, not separatrix
    certification.
    """
    r = np.asarray(r_grid, dtype=float)
    z = np.asarray(z_grid, dtype=float)
    flux = np.asarray(psi, dtype=float)
    if r.ndim != 1 or z.ndim != 1 or r.size < 3 or z.size < 3:
        raise ValueError("X-point grid must have at least three R and Z coordinates")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(z)):
        raise ValueError("X-point grid coordinates must be finite")
    if np.any(np.diff(r) <= 0.0) or np.any(np.diff(z) <= 0.0):
        raise ValueError("X-point grid coordinates must be strictly increasing")
    if flux.shape != (r.size, z.size):
        raise ValueError("psi shape must match the EFIT R/Z grid")
    if not np.all(np.isfinite(flux)):
        raise ValueError("psi must be finite")

    grad_r, grad_z = np.gradient(flux, r, z, edge_order=2)
    h_rr = np.gradient(grad_r, r, axis=0, edge_order=2)
    h_zz = np.gradient(grad_z, z, axis=1, edge_order=2)
    h_rz = 0.5 * (np.gradient(grad_r, z, axis=1, edge_order=2) + np.gradient(grad_z, r, axis=0, edge_order=2))
    if not all(np.all(np.isfinite(array)) for array in (grad_r, grad_z, h_rr, h_zz, h_rz)):
        raise ValueError("X-point flux derivatives must be finite")

    interior = np.s_[1:-1, 1:-1]
    g_r, g_z = grad_r[interior], grad_z[interior]
    h11, h22, h12 = h_rr[interior], h_zz[interior], h_rz[interior]
    determinant = h11 * h22 - h12 * h12
    if not np.all(np.isfinite(determinant)):
        raise ValueError("X-point flux Hessian must be finite")
    saddle = determinant < 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dr = (h12 * g_z - h22 * g_r) / determinant
        dz = (h12 * g_r - h11 * g_z) / determinant

    r_step = np.minimum(np.diff(r)[:-1], np.diff(r)[1:])[:, np.newaxis]
    z_step = np.minimum(np.diff(z)[:-1], np.diff(z)[1:])[np.newaxis, :]
    candidate_r = r[1:-1, np.newaxis] + dr
    candidate_z = z[np.newaxis, 1:-1] + dz
    valid = (
        saddle
        & np.isfinite(dr)
        & np.isfinite(dz)
        & (np.abs(dr) <= r_step)
        & (np.abs(dz) <= z_step)
        & (candidate_r >= r[1])
        & (candidate_r <= r[-2])
        & (candidate_z >= z[1])
        & (candidate_z <= z[-2])
    )
    if not np.any(valid):
        return None
    score = np.where(valid, np.abs(dr) / r_step + np.abs(dz) / z_step, np.inf)
    index = np.unravel_index(int(np.argmin(score)), score.shape)
    return float(candidate_r[index]), float(candidate_z[index])
