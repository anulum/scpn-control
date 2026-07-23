# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Linear elliptic iterators for GS*

"""Pure Jacobi, Red-Black SOR, Anderson mixing, and Python elliptic solve.

This leaf owns the linear elliptic iteration steps used by the CONTROL
Grad-Shafranov solver (toroidal 1/R stencil). Multigrid V-cycles are delegated
to :mod:`scpn_control.core.gs_multigrid`. The CONTROL
:class:`~scpn_control.core.fusion_kernel.FusionKernel` product surface remains
first-class under dual-home C and keeps thin wrappers (including the HPC
offload branch of the elliptic solve).
"""

from __future__ import annotations

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.core.gs_multigrid import multigrid_vcycle


def jacobi_step(
    Psi: FloatArray,
    Source: FloatArray,
    RR: FloatArray,
    dR: float,
    dZ: float,
) -> FloatArray:
    """Perform one Jacobi iteration with toroidal 1/R stencil.

    Solves the GS* operator ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z² = Source
    using the same cylindrical coefficients as :func:`sor_step`.
    """
    Psi_new = Psi.copy()
    dR2 = dR**2
    dZ2 = dZ**2
    R_int = RR[1:-1, 1:-1]
    R_safe = np.maximum(R_int, 1e-10)

    # GS* operator: ∂²ψ/∂R² − (1/R)∂ψ/∂R + ∂²ψ/∂Z²
    # East (R+dR) coefficient: 1/dR² − 1/(2R·dR)
    # West (R−dR) coefficient: 1/dR² + 1/(2R·dR)
    a_E = 1.0 / dR2 - 1.0 / (2.0 * R_safe * dR)
    a_W = 1.0 / dR2 + 1.0 / (2.0 * R_safe * dR)
    a_NS = 1.0 / dZ2
    a_C = 2.0 / dR2 + 2.0 / dZ2

    Psi_new[1:-1, 1:-1] = (
        a_E * Psi[1:-1, 2:] + a_W * Psi[1:-1, 0:-2] + a_NS * (Psi[0:-2, 1:-1] + Psi[2:, 1:-1]) - Source[1:-1, 1:-1]
    ) / a_C
    return Psi_new


def sor_step(
    Psi: FloatArray,
    Source: FloatArray,
    RR: FloatArray,
    dR: float,
    dZ: float,
    omega: float = 1.6,
) -> FloatArray:
    """Vectorised Red-Black SOR iteration with toroidal 1/R stencil.

    The GS* operator in cylindrical (R, Z) coordinates is:

        ∂²ψ/∂R² - (1/R) ∂ψ/∂R + ∂²ψ/∂Z² = Source

    Discretised with central differences this gives R-dependent
    coefficients a_E, a_W (east/west neighbours in R) and constant
    a_N, a_S (north/south neighbours in Z).
    """
    Psi_new = Psi.copy()
    NZ, NR = Psi.shape
    dR2 = dR**2
    dZ2 = dZ**2

    # Toroidal stencil coefficients (arrays over interior grid)
    R_int = RR[1:-1, 1:-1]
    R_safe = np.maximum(R_int, 1e-10)
    # GS* east/west coefficients (matches Rust sor.rs)
    a_E = 1.0 / dR2 - 1.0 / (2.0 * R_safe * dR)  # (NZ-2, NR-2)
    a_W = 1.0 / dR2 + 1.0 / (2.0 * R_safe * dR)  # (NZ-2, NR-2)
    a_NS = 1.0 / dZ2  # scalar — same for north and south
    a_C = 2.0 / dR2 + 2.0 / dZ2  # scalar

    # Checkerboard mask for interior points
    ii, jj = np.mgrid[1 : NZ - 1, 1 : NR - 1]

    for parity in (0, 1):  # 0 = red, 1 = black
        mask = ((ii + jj) % 2) == parity
        gs_update = (
            a_E[mask] * Psi_new[1:-1, 2:][mask]
            + a_W[mask] * Psi_new[1:-1, 0:-2][mask]
            + a_NS * Psi_new[0:-2, 1:-1][mask]
            + a_NS * Psi_new[2:, 1:-1][mask]
            - Source[1:-1, 1:-1][mask]
        ) / a_C

        old_vals = Psi_new[1:-1, 1:-1][mask]
        interior = Psi_new[1:-1, 1:-1]
        interior[mask] = (1.0 - omega) * old_vals + omega * gs_update
        Psi_new[1:-1, 1:-1] = interior

    return Psi_new


def anderson_step(
    psi_history: list[FloatArray],
    res_history: list[FloatArray],
    m: int = 5,
) -> FloatArray:
    """Anderson acceleration (mixing) for the Picard iterate sequence.

    Computes optimal coefficients from the last *m* residuals via a
    least-squares solve, then returns the mixed iterate.
    """
    k = len(res_history)
    mk = min(m, k)

    if mk < 2:
        # Not enough history — fall back to latest iterate
        return psi_history[-1].copy()

    # Stack the last mk residuals as column vectors
    res_cols = [r.ravel() for r in res_history[-mk:]]
    F = np.column_stack(res_cols)  # (N, mk)

    # Solve min ||F @ alpha||^2 s.t. sum(alpha) = 1
    # via: delta_F[:,j] = F[:,j+1] - F[:,j], then solve normal equations
    dF = np.diff(F, axis=1)  # (N, mk-1)
    rhs = F[:, -1]  # latest residual

    # Tikhonov regularisation for numerical stability
    gram = dF.T @ dF
    gram += 1e-10 * np.eye(gram.shape[0])
    try:
        gamma = np.linalg.solve(gram, dF.T @ rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - Tikhonov term keeps the Gram matrix positive-definite
        return psi_history[-1].copy()

    # Reconstruct alpha from gamma
    alpha = np.zeros(mk)
    alpha[-1] = 1.0 - np.sum(gamma)
    alpha[:-1] -= gamma  # alpha_j -= gamma_j
    # But alpha must sum to 1; fix via normalisation
    alpha_sum = np.sum(alpha)
    if abs(alpha_sum) < 1e-12:  # pragma: no cover - regularised solve keeps alpha_sum away from zero
        return psi_history[-1].copy()
    alpha /= alpha_sum

    # Mix iterates
    mixed = np.zeros_like(psi_history[-1])
    psi_cols = psi_history[-mk:]
    for j in range(mk):
        mixed += alpha[j] * psi_cols[j]

    return mixed


def apply_boundary_conditions(Psi: FloatArray, Psi_bc: FloatArray) -> None:
    """Copy vacuum-field boundary values onto the edges of *Psi* (in place)."""
    Psi[0, :] = Psi_bc[0, :]
    Psi[-1, :] = Psi_bc[-1, :]
    Psi[:, 0] = Psi_bc[:, 0]
    Psi[:, -1] = Psi_bc[:, -1]


def elliptic_solve_python(
    method: str,
    Psi: FloatArray,
    Source: FloatArray,
    Psi_bc: FloatArray,
    RR: FloatArray,
    dR: float,
    dZ: float,
    omega: float = 1.6,
) -> FloatArray:
    """Run the pure-Python elliptic step and enforce Dirichlet boundaries.

    Method selection matches the CONTROL config key ``solver.solver_method``:

    - ``"jacobi"`` — single Jacobi sweep
    - ``"multigrid"`` — geometric V-cycle
    - ``"sor"`` / ``"anderson"`` / other — Red-Black SOR sweep
    """
    if method == "jacobi":
        Psi_new = jacobi_step(Psi, Source, RR, dR, dZ)
    elif method == "multigrid":
        Psi_new = multigrid_vcycle(
            Psi.copy(),
            Source,
            RR,
            dR,
            dZ,
            omega=omega,
        )
    else:
        # Both "sor" and "anderson" use SOR as the inner sweep
        Psi_new = sor_step(Psi, Source, RR, dR, dZ, omega=omega)

    apply_boundary_conditions(Psi_new, Psi_bc)
    return Psi_new


# Historical private names for FusionKernel wrappers / re-exports.
_jacobi_step = jacobi_step
_sor_step = sor_step
_anderson_step = anderson_step
_apply_boundary_conditions = apply_boundary_conditions
_elliptic_solve_python = elliptic_solve_python
