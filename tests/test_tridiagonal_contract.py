# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tridiagonal contract tests.

"""Exercise the public compact numerical contract and its refusal cases."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.jax_solvers import has_jax
from scpn_control.core.jax_solvers import thomas_solve as dispatch_solve
from scpn_control.core.tridiagonal import (
    InvalidShapeError,
    NonFiniteInputError,
    SingularFactorizationError,
    solve_tridiagonal,
)


def test_pivoted_system_and_singleton_preserve_inputs() -> None:
    """General systems with a zero first pivot are solvable by row interchange."""
    lower = np.array([1.0])
    diagonal = np.array([0.0, 1.0])
    upper = np.array([1.0])
    rhs = np.array([1.0, 2.0])
    snapshots = tuple(value.copy() for value in (lower, diagonal, upper, rhs))
    np.testing.assert_allclose(solve_tridiagonal(lower, diagonal, upper, rhs), [1.0, 1.0])
    np.testing.assert_allclose(solve_tridiagonal([], [2.0], [], [4.0]), [2.0])
    for actual, before in zip((lower, diagonal, upper, rhs), snapshots, strict=True):
        np.testing.assert_array_equal(actual, before)


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_scale_aware_solution(scale: float) -> None:
    """A dominant system has the same solution across finite common scales."""
    x = solve_tridiagonal(
        scale * np.array([-1.0, -1.0]),
        scale * np.array([4.0, 4.0, 4.0]),
        scale * np.array([-1.0, -1.0]),
        scale * np.array([3.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(x, [1.0, 1.0, 1.0], rtol=1e-12)


def test_seeded_dominant_matrix_matches_dense_reference() -> None:
    """The O(n) public solver agrees with an independent dense solve."""
    generator = np.random.default_rng(20260923)
    size = 37
    lower = generator.uniform(-0.5, 0.5, size - 1)
    upper = generator.uniform(-0.5, 0.5, size - 1)
    diagonal = np.full(size, 2.0)
    rhs = generator.normal(size=size)
    dense = np.diag(diagonal) + np.diag(lower, -1) + np.diag(upper, 1)
    np.testing.assert_allclose(solve_tridiagonal(lower, diagonal, upper, rhs), np.linalg.solve(dense, rhs), rtol=1e-12)


@pytest.mark.parametrize(
    ("lower", "diagonal", "upper", "rhs"),
    [([], [], [], []), ([], [1.0], [1.0], [1.0]), ([[1.0]], [1.0, 1.0], [1.0], [1.0, 1.0])],
)
def test_invalid_shapes(lower: object, diagonal: object, upper: object, rhs: object) -> None:
    """The public API rejects rank and compact-length mismatch."""
    with pytest.raises(InvalidShapeError):
        solve_tridiagonal(lower, diagonal, upper, rhs)


def test_nonfinite_and_singular() -> None:
    """Refusal categories distinguish malformed physics from singular algebra."""
    with pytest.raises(NonFiniteInputError):
        solve_tridiagonal([], [np.nan], [], [1.0])
    with pytest.raises(SingularFactorizationError):
        solve_tridiagonal([], [0.0], [], [1.0])
    with pytest.raises(SingularFactorizationError):
        solve_tridiagonal([0.0], [1.0, 0.0], [0.0], [1.0, 1.0])


@pytest.mark.skipif(not has_jax(), reason="JAX is unavailable")
def test_jax_restricted_to_dominant_system() -> None:
    """The traced no-pivot path refuses the pivot-required general system."""
    with pytest.raises(InvalidShapeError, match="diagonal dominance"):
        dispatch_solve([1.0], [0.0, 1.0], [1.0], [1.0, 2.0], use_jax=True)
