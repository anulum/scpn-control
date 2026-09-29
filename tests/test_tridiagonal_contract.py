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

from scpn_control.core.jax_solvers import batched_crank_nicolson, has_jax
from scpn_control.core.jax_solvers import thomas_solve as dispatch_solve
from scpn_control.core.tridiagonal import (
    InvalidShapeError,
    NonFiniteInputError,
    NumericalFailureError,
    SingularFactorizationError,
    certify_diffusion_cn,
    check_tridiagonal_result,
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


def test_pivot_moved_identity_row_meets_rowwise_bound() -> None:
    """Refinement keeps a pivot-moved identity boundary row exact.

    Unrefined partial pivoting swaps row 0 below the dominant interior row and
    returns ``x[0] = 0.1 ± 1e-14``, which misses the rowwise backward-error
    bound; the transport CN pass reached this system after 47 steps.
    """
    solution = solve_tridiagonal([-100.0, 0.0], [1.0, 80101.0, 1.0], [0.0, -80000.0], [0.1, 50.0, 0.1])

    assert solution[0] == 0.1
    assert solution[2] == 0.1
    np.testing.assert_allclose(
        solution[1], (50.0 + 100.0 * 0.1 + 80000.0 * 0.1) / 80101.0, rtol=4 * np.finfo(float).eps
    )


def test_rcond_below_precision_is_admitted_by_the_rowwise_check() -> None:
    """LAPACK INFO = n+1 reports RCOND < eps; the exact rowwise solution stands.

    ``[[1, 1], [1, 1 + 2**-52]]`` has nonzero pivots but RCOND ~ 5.6e-17.
    """
    solution = solve_tridiagonal([1.0], [1.0, 1.0 + 2.0**-52], [1.0], [1.0, 1.0])

    assert solution.tolist() == [1.0, 0.0]


def test_result_check_refuses_nonfinite_misshapen_and_inaccurate_solutions() -> None:
    """The shared result check rejects any solution that does not solve the rows."""
    sub = np.array([-1.0, -1.0])
    diag = np.array([4.0, 4.0, 4.0])
    sup = np.array([-1.0, -1.0])
    right = np.array([3.0, 2.0, 3.0])
    check_tridiagonal_result(sub, diag, sup, right, np.ones(3))
    for bad in (np.array([1.0, np.nan, 1.0]), np.ones(2)):
        with pytest.raises(NumericalFailureError, match="not finite"):
            check_tridiagonal_result(sub, diag, sup, right, bad)
    with pytest.raises(NumericalFailureError, match="rowwise backward-error"):
        check_tridiagonal_result(sub, diag, sup, right, np.array([1.0, 1.0 + 1e-9, 1.0]))


_CN_VALID = (
    np.array([1.0, 0.5, 0.1]),
    np.array([1.0, 1.0, 1.0]),
    np.zeros(3),
    np.array([0.0, 0.5, 1.0]),
    0.5,
    0.01,
    0.1,
)


def _cn_with(index: int, value: object) -> tuple[object, ...]:
    """Return the valid CN arguments with one position replaced."""
    arguments: list[object] = list(_CN_VALID)
    arguments[index] = value
    return tuple(arguments)


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        (_cn_with(0, np.ones((3, 1))), InvalidShapeError, "one-dimensional real"),
        (_cn_with(1, np.array([True, True, True])), InvalidShapeError, "one-dimensional real"),
        (_cn_with(2, np.zeros(2)), InvalidShapeError, "matching length"),
        (_cn_with(0, np.array([1.0, np.nan, 0.1])), NonFiniteInputError, "finite"),
        (_cn_with(5, np.inf), NonFiniteInputError, "finite"),
        (_cn_with(4, 0.0), InvalidShapeError, "positive spacings"),
        (_cn_with(1, np.array([1.0, -1.0, 1.0])), InvalidShapeError, "nonnegative diffusivity"),
        (_cn_with(3, np.array([0.0, 0.3, 1.0])), InvalidShapeError, "uniform radial grid"),
        (_cn_with(3, np.array([-0.5, 0.0, 0.5])), InvalidShapeError, "positive interior cells"),
    ],
)
def test_cn_certificate_refuses_each_violated_premise(
    arguments: tuple[object, ...], error: type[Exception], message: str
) -> None:
    """Every premise of the dominant-CN certificate is enforced before solving."""
    certify_diffusion_cn(*_CN_VALID)
    with pytest.raises(error, match=message):
        certify_diffusion_cn(*arguments)


def test_batched_cn_refuses_an_empty_batch() -> None:
    """A batched CN step needs at least one two-dimensional profile row."""
    for batch in (np.empty((0, 3)), np.ones(3)):
        with pytest.raises(InvalidShapeError, match="nonempty two-dimensional"):
            batched_crank_nicolson(
                batch,
                *_CN_VALID[1:6],
                allow_numpy_fallback=True,
                allow_legacy_numpy_fallback=True,
            )


@pytest.mark.skipif(not has_jax(), reason="JAX is unavailable")
def test_jax_restricted_to_dominant_system() -> None:
    """The traced no-pivot path refuses the pivot-required general system."""
    with pytest.raises(InvalidShapeError, match="diagonal dominance"):
        dispatch_solve([1.0], [0.0, 1.0], [1.0], [1.0, 2.0], use_jax=True)
    np.testing.assert_array_equal(dispatch_solve([], [4.0], [], [2.0], use_jax=True), [0.5])
