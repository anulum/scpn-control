# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Geometric multigrid primitives for GS*

"""Pure geometric multigrid operators for the Grad-Shafranov GS* elliptic form.

This leaf owns full-weighting restriction, bilinear prolongation, Red-Black
SOR smoothing, residual evaluation, and V-cycle recursion used by the CONTROL
equilibrium solver. The CONTROL
:class:`~scpn_control.core.fusion_kernel.FusionKernel` product surface remains
first-class under dual-home C and keeps thin wrappers that call these helpers.
"""

from __future__ import annotations

import numpy as np

from scpn_control._typing import FloatArray


def restrict_full_weight(fine: FloatArray) -> FloatArray:
    """Full-weighting restriction operator (fine → coarse).

    Standard 9-point stencil:
        coarse[i,j] = 1/16 * (4*fine[2i,2j]
                      + 2*(fine[2i-1,2j] + fine[2i+1,2j]
                           + fine[2i,2j-1] + fine[2i,2j+1])
                      + (fine[2i-1,2j-1] + fine[2i-1,2j+1]
                         + fine[2i+1,2j-1] + fine[2i+1,2j+1]))
    """
    nz_f, nr_f = fine.shape
    nz_c = (nz_f + 1) // 2
    nr_c = (nr_f + 1) // 2
    coarse = np.zeros((nz_c, nr_c))

    # Interior points via full-weighting
    for ic in range(1, nz_c - 1):
        for jc in range(1, nr_c - 1):
            i = 2 * ic
            j = 2 * jc
            coarse[ic, jc] = (
                4.0 * fine[i, j]
                + 2.0 * (fine[i - 1, j] + fine[i + 1, j] + fine[i, j - 1] + fine[i, j + 1])
                + (fine[i - 1, j - 1] + fine[i - 1, j + 1] + fine[i + 1, j - 1] + fine[i + 1, j + 1])
            ) / 16.0

    # Boundary: inject directly. Keep row/column extents separate so
    # rectangular odd grids do not cross-use nr_c for a Z-edge slice.
    coarse[0, :] = fine[0, ::2][:nr_c]
    coarse[-1, :] = fine[-1, ::2][:nr_c]
    coarse[:, 0] = fine[::2, 0][:nz_c]
    coarse[:, -1] = fine[::2, -1][:nz_c]

    return coarse


def prolongate_bilinear(coarse: FloatArray, nz_f: int, nr_f: int) -> FloatArray:
    """Bilinear prolongation operator (coarse → fine).

    Direct injection at coincident points, linear interpolation elsewhere.
    """
    nz_c, nr_c = coarse.shape
    fine = np.zeros((nz_f, nr_f))

    for ic in range(nz_c):
        for jc in range(nr_c):
            i = 2 * ic
            j = 2 * jc
            if i < nz_f and j < nr_f:
                fine[i, j] = coarse[ic, jc]

    # Interpolate horizontal midpoints
    for ic in range(nz_c):
        i = 2 * ic
        if i >= nz_f:
            continue
        for jc in range(nr_c - 1):
            j = 2 * jc + 1
            if j < nr_f:
                fine[i, j] = 0.5 * (coarse[ic, jc] + coarse[ic, jc + 1])

    # Interpolate vertical midpoints
    for ic in range(nz_c - 1):
        i = 2 * ic + 1
        if i >= nz_f:
            continue
        for jc in range(nr_c):
            j = 2 * jc
            if j < nr_f:
                fine[i, j] = 0.5 * (coarse[ic, jc] + coarse[ic + 1, jc])

    # Interpolate center points
    for ic in range(nz_c - 1):
        i = 2 * ic + 1
        if i >= nz_f:
            continue
        for jc in range(nr_c - 1):
            j = 2 * jc + 1
            if j < nr_f:
                fine[i, j] = 0.25 * (coarse[ic, jc] + coarse[ic + 1, jc] + coarse[ic, jc + 1] + coarse[ic + 1, jc + 1])

    return fine


def mg_smooth(
    Psi: FloatArray,
    Source: FloatArray,
    R_grid: FloatArray,
    dR: float,
    dZ: float,
    omega: float,
    n_sweeps: int,
) -> FloatArray:
    """Red-Black SOR smoother with toroidal 1/R stencil for multigrid.

    Works on arbitrary grid sizes (not just the root grid).
    """
    NZ, NR = Psi.shape
    dR2 = dR**2
    dZ2 = dZ**2

    R_int = R_grid[1:-1, 1:-1]
    R_safe = np.maximum(R_int, 1e-10)
    # GS* east/west coefficients (matches Rust sor.rs)
    a_E = 1.0 / dR2 - 1.0 / (2.0 * R_safe * dR)
    a_W = 1.0 / dR2 + 1.0 / (2.0 * R_safe * dR)
    a_NS = 1.0 / dZ2
    a_C = 2.0 / dR2 + 2.0 / dZ2

    ii, jj = np.mgrid[1 : NZ - 1, 1 : NR - 1]

    for _ in range(n_sweeps):
        for parity in (0, 1):
            mask = ((ii + jj) % 2) == parity
            gs_update = (
                a_E[mask] * Psi[1:-1, 2:][mask]
                + a_W[mask] * Psi[1:-1, 0:-2][mask]
                + a_NS * Psi[0:-2, 1:-1][mask]
                + a_NS * Psi[2:, 1:-1][mask]
                - Source[1:-1, 1:-1][mask]
            ) / a_C
            old_vals = Psi[1:-1, 1:-1][mask]
            interior = Psi[1:-1, 1:-1]
            interior[mask] = (1.0 - omega) * old_vals + omega * gs_update
            Psi[1:-1, 1:-1] = interior

    return Psi


def mg_residual(
    Psi: FloatArray,
    Source: FloatArray,
    R_grid: FloatArray,
    dR: float,
    dZ: float,
) -> FloatArray:
    """Compute GS* residual r = L*[Psi] - Source on given grid."""
    NZ, NR = Psi.shape
    dR2 = dR**2
    dZ2 = dZ**2

    residual = np.zeros_like(Psi)
    R_int = R_grid[1:-1, 1:-1]
    R_safe = np.maximum(R_int, 1e-10)

    d2R = (Psi[1:-1, 2:] - 2.0 * Psi[1:-1, 1:-1] + Psi[1:-1, 0:-2]) / dR2
    d1R = (Psi[1:-1, 2:] - Psi[1:-1, 0:-2]) / (2.0 * dR)
    d2Z = (Psi[2:, 1:-1] - 2.0 * Psi[1:-1, 1:-1] + Psi[0:-2, 1:-1]) / dZ2

    Lpsi = d2R - d1R / R_safe + d2Z
    residual[1:-1, 1:-1] = Lpsi - Source[1:-1, 1:-1]
    return residual


def multigrid_vcycle(
    Psi: FloatArray,
    Source: FloatArray,
    R_grid: FloatArray,
    dR: float,
    dZ: float,
    *,
    omega: float = 1.6,
    pre_smooth: int = 3,
    post_smooth: int = 3,
    min_grid: int = 5,
) -> FloatArray:
    """One V-cycle of geometric multigrid for the GS* operator.

    Parameters
    ----------
    Psi :
        Current solution estimate.
    Source :
        Right-hand-side source term.
    R_grid :
        R-coordinate meshgrid matching Psi shape.
    dR, dZ :
        Grid spacings.
    omega :
        SOR over-relaxation factor.
    pre_smooth, post_smooth :
        Number of smoothing sweeps before/after coarse correction.
    min_grid :
        Minimum grid dimension before switching to direct solve.

    Returns
    -------
    FloatArray
        Improved solution estimate.
    """
    NZ, NR = Psi.shape

    # Base case: grid too coarse — solve directly with many SOR sweeps
    if NZ <= min_grid or NR <= min_grid:
        return mg_smooth(Psi.copy(), Source, R_grid, dR, dZ, omega, n_sweeps=50)

    # 1. Pre-smooth
    Psi = mg_smooth(Psi.copy(), Source, R_grid, dR, dZ, omega, pre_smooth)

    # 2. Compute the defect d = Source - L*[Psi].  ``mg_residual`` returns
    #    the signed residual L*[Psi] - Source, so the coarse-grid correction
    #    equation L*[e] = d requires its negation. Feeding the unnegated
    #    residual inverts every correction (Psi <- Psi - e), which stalls the
    #    V-cycle and lets the interior error grow instead of decaying.
    defect = -mg_residual(Psi, Source, R_grid, dR, dZ)

    # 3. Restrict defect and R-grid to coarse level
    r_coarse = restrict_full_weight(defect)
    R_coarse = restrict_full_weight(R_grid)
    nz_c, nr_c = r_coarse.shape

    # Coarse grid spacings (doubled)
    dR_c = dR * 2.0
    dZ_c = dZ * 2.0

    # 4. Solve coarse-grid correction: L*[e] = r
    e_coarse = np.zeros((nz_c, nr_c))
    e_coarse[:] = multigrid_vcycle(
        e_coarse,
        r_coarse,
        R_coarse,
        dR_c,
        dZ_c,
        omega=omega,
        pre_smooth=pre_smooth,
        post_smooth=post_smooth,
        min_grid=min_grid,
    )

    # 5. Prolongate correction and apply
    correction = prolongate_bilinear(e_coarse, NZ, NR)
    Psi = Psi + correction

    # 6. Post-smooth
    Psi = mg_smooth(Psi, Source, R_grid, dR, dZ, omega, post_smooth)

    return Psi


# Historical private names used by FusionKernel wrappers.
_restrict_full_weight = restrict_full_weight
_prolongate_bilinear = prolongate_bilinear
_mg_smooth = mg_smooth
_mg_residual = mg_residual
_multigrid_vcycle = multigrid_vcycle
