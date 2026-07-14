# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport Radial-Grid Geometry

"""Toroidal-geometry helpers for the transport radial grid.

Stateless geometry primitives extracted from the integrated transport solver:
the per-cell toroidal volume element used to power-normalise sources and
integrate profiles, an analytic plasma surface-area estimate for the Martin L-H
power-threshold scaling, and validation/repair of the normalised radial grid so
external mutation cannot leave the solver with a non-monotone or non-finite mesh.
"""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

__all__ = [
    "canonical_radial_grid",
    "estimate_plasma_surface_area_m2",
    "is_canonical_radial_grid",
    "rho_volume_element",
]


def rho_volume_element(rho: AnyFloatArray, drho: float, r_min: float, r_max: float) -> FloatArray:
    r"""Toroidal volume element per radial cell [m^3].

    Each cell of the normalised radial grid maps to a toroidal shell of volume
    ``dV = (2 pi R0)(2 pi a^2 rho drho)`` where ``R0`` is the major radius and
    ``a`` the minor radius derived from the machine bore.

    Parameters
    ----------
    rho : array
        Normalised radial coordinate in ``[0, 1]``.
    drho : float
        Uniform grid spacing of *rho*.
    r_min : float
        Inner major-radius bore of the plasma [m].
    r_max : float
        Outer major-radius bore of the plasma [m].

    Returns
    -------
    array
        Volume element per radial cell [m^3].
    """
    r0 = (r_min + r_max) / 2.0
    a_minor = (r_max - r_min) / 2.0
    return np.asarray(2.0 * np.pi * r0 * 2.0 * np.pi * rho * a_minor**2 * drho, dtype=np.float64)


def estimate_plasma_surface_area_m2(R0: float, a: float, kappa: float) -> float:
    """Estimate plasma surface area for Martin L-H threshold scaling.

    Approximates the last-closed-flux-surface area of an elongated torus as the
    product of the toroidal circumference ``2 pi R0`` and the poloidal perimeter
    of an ellipse with semi-axes ``a`` and ``kappa a`` (RMS-radius perimeter).

    Parameters
    ----------
    R0 : float
        Major radius [m]; floored at ``1e-6`` for degenerate geometry.
    a : float
        Minor radius [m]; floored at ``1e-6``.
    kappa : float
        Plasma elongation; floored at ``1e-6``.

    Returns
    -------
    float
        Estimated plasma surface area [m^2].
    """
    a_safe = max(float(a), 1e-6)
    r0_safe = max(float(R0), 1e-6)
    kappa_safe = max(float(kappa), 1e-6)
    b = a_safe * kappa_safe
    perimeter = 2.0 * np.pi * np.sqrt((a_safe * a_safe + b * b) / 2.0)
    return float(2.0 * np.pi * r0_safe * perimeter)


def is_canonical_radial_grid(rho: AnyFloatArray, nr: int, drho: float) -> bool:
    """Return ``True`` if *rho*/*drho* form a valid normalised radial grid.

    A valid grid has exactly *nr* finite, strictly increasing points spanning
    ``[0, 1]`` with a finite positive spacing.

    Parameters
    ----------
    rho : array
        Candidate normalised radial coordinate.
    nr : int
        Expected number of grid points.
    drho : float
        Expected grid spacing.

    Returns
    -------
    bool
        Whether the grid is canonical.
    """
    arr = np.asarray(rho, dtype=np.float64)
    return bool(
        arr.shape == (nr,)
        and nr >= 2
        and np.all(np.isfinite(arr))
        and np.isclose(arr[0], 0.0)
        and np.isclose(arr[-1], 1.0)
        and np.all(np.diff(arr) > 0.0)
        and np.isfinite(drho)
        and drho > 0.0
    )


def canonical_radial_grid(nr: int) -> tuple[FloatArray, float]:
    """Return the canonical normalised grid and its spacing.

    Parameters
    ----------
    nr : int
        Number of grid points; must be at least 2.

    Returns
    -------
    rho : array
        ``linspace(0, 1, nr)``.
    drho : float
        Uniform spacing ``1 / (nr - 1)``.

    Raises
    ------
    ValueError
        If *nr* is less than 2.
    """
    if nr < 2:
        raise ValueError(f"nr must be at least 2, got {nr!r}")
    rho = np.linspace(0.0, 1.0, nr, dtype=np.float64)
    return rho, 1.0 / (nr - 1)
