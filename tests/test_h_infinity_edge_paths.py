# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — H-infinity fail-closed edge tests.

"""Numerical and validation edge paths for the public DGKF controller."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.h_infinity_controller as hinf_module
from scpn_control.control.h_infinity_controller import HInfinityController


def _plant() -> dict[str, np.ndarray]:
    return {
        "A": np.array([[0.0, 1.0], [1.0, -1.0]]),
        "B1": np.array([[0.0, 0.0], [0.5, 0.0]]),
        "B2": np.array([[0.0], [1.0]]),
        "C1": np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        "C2": np.array([[1.0, 0.0]]),
        "D12": np.array([[0.0], [0.0], [1.0]]),
        "D21": np.array([[0.0, 1.0]]),
    }


@pytest.mark.parametrize("gamma", [0.0, -1.0, np.nan, np.inf])
def test_gamma_domain_rejected(gamma: float) -> None:
    """Gamma must remain finite and strictly positive."""
    with pytest.raises(ValueError, match="gamma"):
        HInfinityController(**_plant(), gamma=gamma)


def test_gamma_search_domain_rejected() -> None:
    """Search bounds, tolerance, and iteration budget fail closed."""
    controller = HInfinityController(**_plant(), gamma=5.0)
    with pytest.raises(ValueError, match="gamma_min must be strictly below"):
        controller._find_optimal_gamma(gamma_min=2.0, gamma_max=1.0)
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        controller._find_optimal_gamma(max_iter=0)
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        controller._find_optimal_gamma(max_iter=True)


def test_gamma_search_upper_bound_fails_closed() -> None:
    """An infeasible upper search bound cannot fabricate a result."""
    controller = HInfinityController(**_plant(), gamma=5.0)
    with pytest.raises(ValueError, match="No feasible gamma found"):
        controller._find_optimal_gamma(gamma_min=0.1, gamma_max=0.2)


def test_gamma_search_can_exhaust_a_bounded_iteration_budget() -> None:
    """A bounded early stop still returns an explicitly padded feasible point."""
    controller = HInfinityController(**_plant(), gamma=5.0)
    gamma = controller._find_optimal_gamma(max_iter=1, rtol=1.0e-15)
    assert gamma > 0.0


def test_empty_and_rank_three_values_rejected() -> None:
    """Empty and rank-three arrays cannot masquerade as state matrices."""
    plant = _plant()
    with pytest.raises(ValueError, match="non-empty"):
        HInfinityController(**{**plant, "A": np.array([])})
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        HInfinityController(**{**plant, "A": np.zeros((1, 1, 1))})


def test_c1_detectability_and_b2_stabilizability_rejected() -> None:
    """Both remaining PBH standard-plant conditions are enforced."""
    plant = _plant()
    undetectable = {
        **plant,
        "A": np.diag([1.0, -1.0]),
        "B1": np.array([[1.0, 0.0], [0.0, 0.0]]),
        "C1": np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
        "C2": np.array([[1.0, 0.0]]),
    }
    with pytest.raises(ValueError, match=r"\(C1, A\) must be detectable"):
        HInfinityController(**undetectable)

    uncontrollable = {
        **plant,
        "A": np.diag([1.0, -1.0]),
        "B1": np.array([[1.0, 0.0], [0.0, 0.0]]),
        "B2": np.array([[0.0], [1.0]]),
        "C1": np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        "C2": np.array([[1.0, 0.0]]),
    }
    with pytest.raises(ValueError, match=r"\(A, B2\) must be stabilizable"):
        HInfinityController(**uncontrollable)


def test_scalar_matrix_is_rejected() -> None:
    """A scalar is not silently promoted to a one-state plant."""
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        HInfinityController(**{**_plant(), "A": 1.0})


def test_numerical_admission_failures_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-PSD dual Riccati result is rejected even after solver return."""
    controller = HInfinityController(**_plant(), gamma=5.0)
    original_eigvalsh = hinf_module.np.linalg.eigvalsh

    calls = 0

    def negative_y(matrix: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original_eigvalsh(matrix) if calls == 1 else np.array([-1.0])

    monkeypatch.setattr(hinf_module.np.linalg, "eigvalsh", negative_y)
    with pytest.raises(ValueError, match="Y is not positive semidefinite"):
        controller._synthesize(5.0)


@pytest.mark.parametrize(("unstable_call", "match"), [(1, "X is not"), (2, "Y is not")])
def test_nonstabilizing_riccati_solution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    unstable_call: int,
    match: str,
) -> None:
    """Only stabilizing primal and dual Riccati solutions are admitted."""
    controller = HInfinityController(**_plant(), gamma=5.0)
    original_eigvals = hinf_module.np.linalg.eigvals
    calls = 0

    def unstable_once(matrix: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == unstable_call:
            return np.array([0.0])
        return original_eigvals(matrix)

    monkeypatch.setattr(hinf_module.np.linalg, "eigvals", unstable_once)
    with pytest.raises(ValueError, match=match):
        controller._synthesize(5.0)


def test_singular_coupling_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A singular central-controller coupling fails without an explicit inverse."""
    controller = HInfinityController(**_plant(), gamma=5.0)

    def singular(*args: object, **kwargs: object) -> np.ndarray:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(hinf_module.np.linalg, "solve", singular)
    with pytest.raises(ValueError, match="coupling matrix is singular"):
        controller._synthesize(5.0)


def test_unstable_augmented_controller_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The final augmented internal-stability check is mandatory."""
    controller = HInfinityController(**_plant(), gamma=5.0)

    def unstable_realization(*args: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.array([[1.0]]), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))

    monkeypatch.setattr(controller, "_closed_loop_realization", unstable_realization)
    with pytest.raises(ValueError, match="does not internally stabilize"):
        controller._synthesize(5.0)
