# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tridiagonal solver adapter.

"""Validate compact tridiagonal systems and solve them with banded LAPACK.

The four-array public contract is shared with the native CONTROL adapter.
Numerical formulation and reference vectors are owned by SCPN-FUSION-CORE.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from scpn_control._typing import AnyFloatArray, FloatArray


class InvalidShapeError(ValueError):
    """A tridiagonal input has an invalid rank, shape or real dtype."""


class NonFiniteInputError(ValueError):
    """A tridiagonal input contains NaN or infinity."""


class SingularFactorizationError(np.linalg.LinAlgError):
    """Pivoted factorisation found a singular tridiagonal system."""


class NumericalFailureError(ArithmeticError):
    """A factorisation returned a nonfinite or inaccurate solution."""


def validate_tridiagonal(
    lower: AnyFloatArray,
    diagonal: AnyFloatArray,
    upper: AnyFloatArray,
    rhs: AnyFloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return finite float64 copies of exact compact one-dimensional inputs.

    Parameters
    ----------
    lower, upper : array_like
        Sub- and super-diagonals, each of length ``n-1``.
    diagonal, rhs : array_like
        Main diagonal and right-hand side, each of length ``n >= 1``.

    Returns
    -------
    tuple of ndarray
        Independent float64 copies in input order.

    Raises
    ------
    InvalidShapeError
        A rank, length or dtype is outside the compact real-vector contract.
    NonFiniteInputError
        A converted value is not finite.
    """
    raw = tuple(np.asarray(value) for value in (lower, diagonal, upper, rhs))
    if any(value.ndim != 1 or value.dtype.kind not in "fiu" for value in raw):
        raise InvalidShapeError("tridiagonal inputs must be one-dimensional real numeric vectors")
    n = raw[1].size
    if n < 1 or raw[0].size != n - 1 or raw[2].size != n - 1 or raw[3].size != n:
        raise InvalidShapeError("expected lower[n-1], diagonal[n], upper[n-1], rhs[n], n>=1")
    sub = np.array(raw[0], dtype=np.float64, copy=True)
    diag = np.array(raw[1], dtype=np.float64, copy=True)
    sup = np.array(raw[2], dtype=np.float64, copy=True)
    right = np.array(raw[3], dtype=np.float64, copy=True)
    converted = (sub, diag, sup, right)
    if any(not np.all(np.isfinite(value)) for value in converted):
        raise NonFiniteInputError("tridiagonal inputs must be finite after float64 conversion")
    return converted


def solve_tridiagonal(
    lower: AnyFloatArray,
    diagonal: AnyFloatArray,
    upper: AnyFloatArray,
    rhs: AnyFloatArray,
) -> FloatArray:
    """Solve a finite real tridiagonal system with pivoted O(n) banded LAPACK.

    Parameters
    ----------
    lower, upper : array_like
        Compact off-diagonals, each of length ``n-1``.
    diagonal, rhs : array_like
        Main diagonal and right-hand side, each of length ``n >= 1``.

    Returns
    -------
    ndarray
        Finite float64 solution whose rowwise backward error is bounded.

    Raises
    ------
    InvalidShapeError
        A rank, length or dtype violates the compact vector contract.
    NonFiniteInputError
        An input is NaN or infinite.
    SingularFactorizationError
        Banded factorisation finds a zero pivot.
    NumericalFailureError
        The solution is nonfinite or fails the scale-aware residual check.
    """
    sub, diag, sup, right = validate_tridiagonal(lower, diagonal, upper, rhs)
    n = diag.size
    if n == 1:
        if diag[0] == 0.0:
            raise SingularFactorizationError("tridiagonal factorisation is singular")
        result = right / diag
    else:
        banded = np.zeros((3, n), dtype=np.float64)
        banded[1] = diag
        banded[0, 1:] = sup
        banded[2, :-1] = sub
        try:
            result = np.asarray(solve_banded((1, 1), banded, right, check_finite=False), dtype=np.float64)
        except np.linalg.LinAlgError as exc:
            raise SingularFactorizationError("tridiagonal factorisation is singular") from exc
    check_tridiagonal_result(sub, diag, sup, right, result)
    return result


def check_tridiagonal_result(
    sub: FloatArray, diag: FloatArray, sup: FloatArray, right: FloatArray, result: FloatArray
) -> None:
    """Reject a nonfinite or inaccurate result for already validated inputs."""
    n = diag.size
    if result.shape != (n,) or not np.all(np.isfinite(result)):
        raise NumericalFailureError("tridiagonal solution is not finite")
    extended = np.longdouble
    x = result.astype(extended)
    row = diag.astype(extended) * x
    scale = np.abs(diag.astype(extended) * x) + np.abs(right.astype(extended))
    if n > 1:
        below = sub.astype(extended) * x[:-1]
        above = sup.astype(extended) * x[1:]
        row[1:] += below
        row[:-1] += above
        scale[1:] += np.abs(below)
        scale[:-1] += np.abs(above)
    error = np.abs(row - right.astype(extended))
    tolerance = extended(64 * n * np.finfo(np.float64).eps)
    if not np.all(np.isfinite(error)) or np.any(error > tolerance * np.maximum(scale, extended(1e-300))):
        raise NumericalFailureError("tridiagonal solution fails the rowwise backward-error check")


def certify_diffusion_cn(
    temperature: AnyFloatArray,
    diffusivity: AnyFloatArray,
    source: AnyFloatArray,
    rho: AnyFloatArray,
    drho: float,
    dt: float,
    edge: float,
) -> None:
    """Certify the physical builder's finite strictly dominant CN matrix.

    The cylindrical builder has nonnegative off-diagonal magnitudes and a
    unit identity term when ``chi >= 0``, the grid is uniform and positive
    away from the axis, and both spacings are positive.
    """
    vectors = tuple(np.asarray(value) for value in (temperature, diffusivity, source, rho))
    if any(value.ndim != 1 or value.dtype.kind not in "fiu" for value in vectors):
        raise InvalidShapeError("CN inputs must be one-dimensional real vectors")
    n = vectors[0].size
    if n < 3 or any(value.size != n for value in vectors):
        raise InvalidShapeError("CN inputs must have matching length >= 3")
    if not all(np.isfinite(value).all() for value in vectors) or not np.isfinite([drho, dt, edge]).all():
        raise NonFiniteInputError("CN inputs must be finite")
    if drho <= 0.0 or dt <= 0.0 or np.any(vectors[1] < 0.0):
        raise InvalidShapeError("CN requires positive spacings and nonnegative diffusivity")
    grid = np.asarray(vectors[3], dtype=np.float64)
    if np.any(grid[1:-1] <= 0.5 * drho) or not np.allclose(np.diff(grid), drho, rtol=1e-10, atol=0.0):
        raise InvalidShapeError("CN requires a uniform radial grid with positive interior cells")
