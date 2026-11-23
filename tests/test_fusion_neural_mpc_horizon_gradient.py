# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Exact horizon gradient and stable step of the gradient MPC.
"""The gradient MPC optimises the whole horizon cost, not a sum of one-step costs.

Under ``x_{t+1} = x_t + B u_t`` every action moves all later states, so the
gradient must carry the later-stage errors. These tests check the adjoint
gradient against finite differences, the converged plan against the closed-form
optimum of the quadratic, and the step bound that keeps the descent stable.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.control.fusion_neural_mpc import (
    ModelPredictiveController,
    NeuralSurrogate,
    horizon_cost_and_gradient,
    horizon_coupling_spectral_radius,
)

FloatArray = NDArray[np.float64]

_B = np.array([[0.7, -0.2], [0.1, 0.5], [-0.3, 0.4]], dtype=np.float64)
_TARGET = np.array([0.4, -0.6, 0.2], dtype=np.float64)
_X0 = np.array([1.0, 0.5, -0.8], dtype=np.float64)


def _surrogate(scale: float) -> NeuralSurrogate:
    """Return the three-state, two-coil fixture surrogate with ``B`` scaled."""
    surrogate = NeuralSurrogate(n_coils=2, n_state=3, verbose=False)
    surrogate.B[:] = _B * scale
    return surrogate


def _sample_actions(horizon: int) -> FloatArray:
    """Return a non-trivial deterministic action sequence."""
    steps = np.arange(horizon, dtype=np.float64)
    return np.stack([0.1 * steps - 0.3, 0.2 - 0.05 * steps], axis=1)


def _closed_form_plan(b_matrix: FloatArray, horizon: int, regularization: float) -> FloatArray:
    """Solve the unconstrained horizon quadratic exactly by stacked least squares."""
    n_state, n_coils = b_matrix.shape
    stacked = np.zeros((horizon * n_state, horizon * n_coils), dtype=np.float64)
    rhs = np.zeros(horizon * n_state, dtype=np.float64)
    for t in range(horizon):
        for j in range(t + 1):
            stacked[t * n_state : (t + 1) * n_state, j * n_coils : (j + 1) * n_coils] = b_matrix
        rhs[t * n_state : (t + 1) * n_state] = _TARGET - _X0
    normal = stacked.T @ stacked + regularization * np.eye(horizon * n_coils)
    return np.asarray(np.linalg.solve(normal, stacked.T @ rhs).reshape(horizon, n_coils), dtype=np.float64)


@pytest.mark.parametrize("horizon", [1, 2, 3, 7, 10, 25])
def test_spectral_radius_matches_the_coupling_matrix_eigenvalue(horizon: int) -> None:
    """The closed form equals the largest eigenvalue of ``T_H[i, j] = H - max(i, j)``."""
    coupling = np.array(
        [[horizon - max(i, j) for j in range(horizon)] for i in range(horizon)],
        dtype=np.float64,
    )
    assert horizon_coupling_spectral_radius(horizon) == pytest.approx(
        float(np.linalg.eigvalsh(coupling).max()), rel=1e-12
    )


def test_gradient_matches_central_finite_difference() -> None:
    """The adjoint gradient equals a central finite difference of the horizon cost."""
    surrogate = _surrogate(1.0)
    actions = _sample_actions(6)
    _, gradient = horizon_cost_and_gradient(surrogate, _TARGET, _X0, actions, 0.1)
    step = 1e-6
    numeric = np.zeros_like(actions)
    for index in np.ndindex(actions.shape):
        plus, minus = actions.copy(), actions.copy()
        plus[index] += step
        minus[index] -= step
        cost_plus, _ = horizon_cost_and_gradient(surrogate, _TARGET, _X0, plus, 0.1)
        cost_minus, _ = horizon_cost_and_gradient(surrogate, _TARGET, _X0, minus, 0.1)
        numeric[index] = (cost_plus - cost_minus) / (2.0 * step)
    np.testing.assert_allclose(gradient, numeric, rtol=0.0, atol=1e-6)


def test_gradient_carries_later_stage_errors() -> None:
    """The first entry differs from the one-step term; the last entry is the one-step term."""
    surrogate = _surrogate(1.0)
    actions = _sample_actions(6)
    _, gradient = horizon_cost_and_gradient(surrogate, _TARGET, _X0, actions, 0.1)
    first_error = surrogate.predict(_X0, actions[0]) - _TARGET
    one_step = surrogate.B.T @ first_error + 0.1 * actions[0]
    assert float(np.max(np.abs(gradient[0] - one_step))) > 1e-3
    np.testing.assert_allclose(
        gradient[-1],
        surrogate.B.T @ (_X0 + surrogate.B @ actions.sum(axis=0) - _TARGET) + 0.1 * actions[-1],
        rtol=0.0,
        atol=1e-12,
    )


def test_cost_is_the_horizon_tracking_and_effort_sum() -> None:
    """The cost is the summed half squared tracking error plus the half-weighted effort."""
    surrogate = _surrogate(1.0)
    actions = _sample_actions(4)
    cost, _ = horizon_cost_and_gradient(surrogate, _TARGET, _X0, actions, 0.3)
    states = _X0 + np.cumsum(actions @ surrogate.B.T, axis=0)
    expected = 0.5 * float(np.sum((states - _TARGET) ** 2)) + 0.15 * float(np.sum(actions**2))
    assert cost == pytest.approx(expected, rel=1e-12)


def test_converged_plan_returns_the_first_closed_form_action() -> None:
    """Unclipped and converged, the plan returns the first action of the exact optimum."""
    surrogate = _surrogate(1.0)
    controller = ModelPredictiveController(
        surrogate,
        _TARGET,
        prediction_horizon=6,
        learning_rate=10.0,
        iterations=4000,
        action_limit=1e6,
        action_regularization=0.1,
    )
    action = controller.plan_trajectory(_X0)
    assert action.shape == (2,)
    np.testing.assert_allclose(action, _closed_form_plan(surrogate.B, 6, 0.1)[0], rtol=0.0, atol=1e-5)


def test_stable_step_decreases_cost_where_the_requested_rate_diverges() -> None:
    """For a strong surrogate the capped step descends where the requested rate overshoots."""
    surrogate = _surrogate(3.0)
    controller = ModelPredictiveController(surrogate, _TARGET, learning_rate=0.5)
    step = controller.stable_step()
    assert step < 0.5
    zero = np.zeros((controller.horizon, 2), dtype=np.float64)
    cost, gradient = horizon_cost_and_gradient(surrogate, _TARGET, _X0, zero, 0.1)
    stepped, _ = horizon_cost_and_gradient(surrogate, _TARGET, _X0, zero - step * gradient, 0.1)
    requested, _ = horizon_cost_and_gradient(surrogate, _TARGET, _X0, zero - 0.5 * gradient, 0.1)
    assert stepped < cost < requested


def test_plan_cost_never_increases_across_iterations() -> None:
    """The planner equals plain descent with the stable step, and its cost never rises."""
    surrogate = _surrogate(3.0)
    costs = []
    for iterations in (1, 2, 5, 20, 80):
        controller = ModelPredictiveController(
            surrogate, _TARGET, learning_rate=0.5, iterations=iterations, action_limit=1e6
        )
        step = controller.stable_step()
        planned = np.zeros((controller.horizon, 2), dtype=np.float64)
        for _ in range(iterations):
            _, gradient = horizon_cost_and_gradient(surrogate, _TARGET, _X0, planned, 0.1)
            planned -= step * gradient
        np.testing.assert_allclose(controller.plan_trajectory(_X0), planned[0], rtol=0.0, atol=1e-12)
        costs.append(horizon_cost_and_gradient(surrogate, _TARGET, _X0, planned, 0.1)[0])
    assert all(later <= earlier for earlier, later in zip(costs, costs[1:]))


def test_stable_step_keeps_the_requested_rate_for_a_weak_surrogate() -> None:
    """A weak surrogate leaves the requested rate in force."""
    controller = ModelPredictiveController(_surrogate(0.05), _TARGET, learning_rate=0.5)
    assert controller.stable_step() == 0.5


def test_stable_step_keeps_the_rate_when_the_cost_is_flat() -> None:
    """With no coupling and no penalty the gradient vanishes and the rate stands."""
    controller = ModelPredictiveController(_surrogate(0.0), _TARGET, learning_rate=0.7, action_regularization=0.0)
    assert controller.stable_step() == 0.7
    np.testing.assert_array_equal(controller.plan_trajectory(_X0), np.zeros(2))
