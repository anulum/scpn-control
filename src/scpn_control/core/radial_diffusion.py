# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Radial Diffusion PDE Numerics

"""Crank-Nicolson numerics for the 1-D radial transport diffusion equation.

Stateless discretisation helpers extracted from the integrated transport
solver: the explicit cylindrical diffusion operator, the Crank-Nicolson
tridiagonal assembly, and the Thomas tridiagonal solve. The radial grid
(``rho``, ``drho``, minor radius ``a``) is passed explicitly so the numerics
are independent of any solver state.
"""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.tridiagonal import solve_tridiagonal

__all__ = [
    "build_cn_tridiag",
    "explicit_diffusion_rhs",
    "thomas_solve",
]


def thomas_solve(a: AnyFloatArray, b: AnyFloatArray, c: AnyFloatArray, d: AnyFloatArray) -> FloatArray:
    """Solve a general tridiagonal system through pivoted banded LAPACK.

    Solves ``A x = d`` where ``A`` is tridiagonal with sub-diagonal *a*, main
    diagonal *b*, and super-diagonal *c*. The historical name is retained for
    callers; unlike the no-pivot Thomas algorithm, this accepts nonsingular
    systems whose first diagonal entry is zero. Invalid or singular inputs
    raise typed errors from :mod:`scpn_control.core.tridiagonal`.

    Parameters
    ----------
    a : array, length n-1
        Sub-diagonal.
    b : array, length n
        Main diagonal.
    c : array, length n-1
        Super-diagonal.
    d : array, length n
        Right-hand side.

    Returns
    -------
    x : array, length n
        Finite solution vector with a scale-aware residual check.
    """
    return solve_tridiagonal(a, b, c, d)


def _density_weights(density: AnyFloatArray | None, size: int) -> FloatArray:
    """Validate positive cell weights for conservative heat diffusion."""
    if density is None:
        return np.ones(size)
    raw = np.asarray(density)
    if raw.shape != (size,) or raw.dtype.kind not in "fiu":
        raise ValueError("density must be a real numeric vector matching the radial grid")
    weights = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("density must be finite and strictly positive")
    return weights


def explicit_diffusion_rhs(
    T: AnyFloatArray,
    chi: AnyFloatArray,
    rho: AnyFloatArray,
    drho: float,
    a_minor: float,
    *,
    density: AnyFloatArray | None = None,
) -> FloatArray:
    """Compute the explicit cylindrical diffusion operator L_h(T).

    ``L_h(T) = (1/a^2) * (1/rho) d/drho(rho chi dT/drho)``, evaluated with
    half-grid diffusivities and central differences on the interior. Returns an
    array of the same length as *T* (boundary points left at zero).

    When density is supplied, use (1/a^2)/(rho n) d/drho(rho n chi dT/drho),
    averaging n*chi at faces. This conserves density-weighted heat on the
    interior control volumes. chi is a thermal diffusivity in m^2/s; density
    must be a finite positive vector matching rho, in any consistent units.
    Omitting density retains the unit-capacity temperature diffusion operator.

    To evaluate an old-state conductive flux divided by a different new
    capacity, pass density=n_new and chi=chi_physical*n_old/n_new. Face products
    are then n_old*chi_physical while the cell denominator is n_new. The new
    density must be strictly positive; old density may be zero. This call only
    evaluates the explicit flux divergence: the caller must also scale the
    old storage by n_old/n_new when assembling an evolving-density heat step.
    """
    n = len(T)
    weights = _density_weights(density, n)
    conductivity = np.asarray(chi) * weights
    Lh = np.zeros(n)
    dr = drho

    # Precompute 1/a^2 factor for units [1/s]
    scale = 1.0 / max(a_minor**2, 1e-6)

    for i in range(1, n - 1):
        r = rho[i]
        # half-grid chi
        chi_ip = 0.5 * (conductivity[i] + conductivity[i + 1]) / weights[i]
        chi_im = 0.5 * (conductivity[i] + conductivity[i - 1]) / weights[i]
        r_ip = r + 0.5 * dr
        r_im = r - 0.5 * dr

        flux_ip = chi_ip * r_ip * (T[i + 1] - T[i]) / dr
        flux_im = chi_im * r_im * (T[i] - T[i - 1]) / dr

        Lh[i] = scale * (flux_ip - flux_im) / (r * dr)

    return Lh


def build_cn_tridiag(
    chi: AnyFloatArray,
    dt: float,
    rho: AnyFloatArray,
    drho: float,
    a_minor: float,
    *,
    density: AnyFloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build the Crank-Nicolson LHS tridiagonal coefficients.

    The implicit system is
    ``(I - 0.5*dt*L_h) T^{n+1} = (I + 0.5*dt*L_h) T^n + dt*(S - Sink)``.

    Returns ``(a, b, c)`` sub/main/super diagonals for the interior points,
    padded to full grid size (boundary conditions applied separately).

    Optional density has the same positive-vector contract and face n*chi
    averaging as explicit_diffusion_rhs. Omitting it uses unit cell weights.
    The displayed RHS applies to a density frozen over the step. This helper
    constructs only the implicit LHS; it does not assemble storage or sources.

    For evolving density, build this LHS with density=n_new and physical chi.
    Assemble the temperature RHS as n_old*T_old/n_new plus half a timestep of
    explicit_diffusion_rhs(T_old, chi*n_old/n_new, ..., density=n_new), plus
    dt*Q/(H*n_new). Here H is the channel energy conversion factor and Q is
    volumetric heat power. This gives old-density explicit conductivity,
    new-density implicit conductivity, and n_new*T_new-n_old*T_old storage.
    n_new must be finite and strictly positive; n_old may contain zeros.
    Boundary rows and their old/new-capacity energy accounting remain the
    caller's responsibility. Neither helper supplies particle heat convection,
    reaction energies or a physical source closure.
    """
    n = len(rho)
    weights = _density_weights(density, n)
    conductivity = np.asarray(chi) * weights
    dr = drho
    scale = 1.0 / max(a_minor**2, 1e-6)

    a = np.zeros(n - 1)  # sub-diagonal
    b = np.ones(n)  # main diagonal
    c = np.zeros(n - 1)  # super-diagonal

    for i in range(1, n - 1):
        r = rho[i]
        chi_ip = 0.5 * (conductivity[i] + conductivity[i + 1]) / weights[i]
        chi_im = 0.5 * (conductivity[i] + conductivity[i - 1]) / weights[i]
        r_ip = r + 0.5 * dr
        r_im = r - 0.5 * dr

        coeff_ip = scale * chi_ip * r_ip / (r * dr * dr)
        coeff_im = scale * chi_im * r_im / (r * dr * dr)

        # Crank-Nicolson LHS: (I - 0.5·dt·L_h)
        b[i] = 1.0 + 0.5 * dt * (coeff_ip + coeff_im)
        c[i] = -0.5 * dt * coeff_ip  # T_{i+1} coefficient
        a[i - 1] = -0.5 * dt * coeff_im  # T_{i-1} coefficient

    return a, b, c
