# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for FB tracking control law

"""Exercise coil corrections through the production free-boundary control law."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control.free_boundary_tracking_control_law as law
from scpn_control._typing import FloatArray


def test_compute_coil_correction_identity_response_and_clip() -> None:
    """Correction recovers the error on an identity plant and respects the clip."""
    n_obj = 3
    n_coils = 3
    target = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    observation = np.array([0.0, 2.0, 3.0], dtype=np.float64)
    response = np.eye(n_coils, dtype=np.float64)
    weights = np.ones(n_obj, dtype=np.float64)
    mask = np.ones(n_obj, dtype=np.float64)
    bias = np.zeros(n_obj, dtype=np.float64)
    currents = np.zeros(n_coils, dtype=np.float64)
    limits = np.full(n_coils, 100.0, dtype=np.float64)
    delta, penalties = law.compute_coil_correction(
        observation,
        target_vector=target,
        objective_bias_estimate=bias,
        control_objective_weights=weights,
        response_matrix=response,
        response_regularization=1e-9,
        correction_limit=10.0,
        control_mask=mask,
        coil_currents=currents,
        coil_current_limits=limits,
    )
    assert delta.shape == (n_coils,)
    assert penalties.shape == (n_coils,)
    # First channel error is +1.0 on an identity response → positive correction.
    assert float(delta[0]) == pytest.approx(1.0, abs=1e-3)
    clipped, _ = law.compute_coil_correction(
        observation,
        target_vector=target,
        objective_bias_estimate=bias,
        control_objective_weights=weights,
        response_matrix=response,
        response_regularization=1e-9,
        correction_limit=0.25,
        control_mask=mask,
        coil_currents=currents,
        coil_current_limits=limits,
    )
    assert float(np.max(np.abs(clipped))) == pytest.approx(0.25)


def test_negative_correction_respects_directional_coil_headroom() -> None:
    """A negative command near a negative coil limit receives a larger penalty."""
    delta, penalties = law.compute_coil_correction(
        np.zeros(2, dtype=np.float64),
        target_vector=np.array([-1.0, 0.0], dtype=np.float64),
        objective_bias_estimate=np.zeros(2, dtype=np.float64),
        control_objective_weights=np.ones(2, dtype=np.float64),
        response_matrix=np.eye(2, dtype=np.float64),
        response_regularization=1e-3,
        correction_limit=1.0,
        control_mask=np.ones(2, dtype=np.float64),
        coil_currents=np.array([-9.0, 0.0], dtype=np.float64),
        coil_current_limits=np.full(2, 10.0, dtype=np.float64),
    )
    assert delta[0] < 0.0
    assert penalties[0] > penalties[1]


@pytest.mark.parametrize("bad_call", [1, 2])
def test_nonfinite_linear_solver_result_cannot_become_coil_command(
    monkeypatch: pytest.MonkeyPatch, bad_call: int
) -> None:
    """A nonfinite backend answer at either solve stage fails before command output."""
    original_lstsq = np.linalg.lstsq
    calls = 0

    def nonfinite_lstsq(
        matrix: FloatArray, rhs: FloatArray, rcond: float | None = None
    ) -> tuple[FloatArray, FloatArray, int, FloatArray]:
        nonlocal calls
        calls += 1
        solution, residuals, rank, singular_values = original_lstsq(matrix, rhs, rcond=rcond)
        if calls == bad_call:
            solution = np.full_like(solution, np.nan)
        return (
            np.asarray(solution, dtype=np.float64),
            np.asarray(residuals, dtype=np.float64),
            int(rank),
            np.asarray(singular_values, dtype=np.float64),
        )

    monkeypatch.setattr(np.linalg, "lstsq", nonfinite_lstsq)
    with pytest.raises(ValueError, match="nonfinite"):
        law.compute_coil_correction(
            np.zeros(1, dtype=np.float64),
            target_vector=np.ones(1, dtype=np.float64),
            objective_bias_estimate=np.zeros(1, dtype=np.float64),
            control_objective_weights=np.ones(1, dtype=np.float64),
            response_matrix=np.eye(1, dtype=np.float64),
            response_regularization=1e-3,
            correction_limit=1.0,
            control_mask=np.ones(1, dtype=np.float64),
            coil_currents=np.zeros(1, dtype=np.float64),
            coil_current_limits=np.ones(1, dtype=np.float64),
        )
    assert calls == bad_call


@pytest.mark.parametrize("invalid_field", ["currents", "limits"])
def test_coil_correction_rejects_mismatched_coil_vectors(invalid_field: str) -> None:
    """Both coil vectors must match the response matrix before solving."""
    currents = np.zeros(2 if invalid_field == "currents" else 1, dtype=np.float64)
    limits = np.ones(2 if invalid_field == "limits" else 1, dtype=np.float64)
    with pytest.raises(ValueError, match="match the response coil count"):
        law.compute_coil_correction(
            np.zeros(1, dtype=np.float64),
            target_vector=np.ones(1, dtype=np.float64),
            objective_bias_estimate=np.zeros(1, dtype=np.float64),
            control_objective_weights=np.ones(1, dtype=np.float64),
            response_matrix=np.eye(1, dtype=np.float64),
            response_regularization=1e-3,
            correction_limit=1.0,
            control_mask=np.ones(1, dtype=np.float64),
            coil_currents=currents,
            coil_current_limits=limits,
        )


@pytest.mark.parametrize("observation", [float("nan"), -1e308])
def test_compute_coil_correction_refuses_nonfinite_or_overflowed_error(observation: float) -> None:
    """A nonfinite or overflowing objective error cannot become a coil command."""
    with pytest.raises(ValueError, match="nonfinite"):
        law.compute_coil_correction(
            np.array([observation], dtype=np.float64),
            target_vector=np.array([1e308], dtype=np.float64),
            objective_bias_estimate=np.zeros(1, dtype=np.float64),
            control_objective_weights=np.ones(1, dtype=np.float64),
            response_matrix=np.ones((1, 1), dtype=np.float64),
            response_regularization=1e-3,
            correction_limit=1.0,
            control_mask=np.ones(1, dtype=np.float64),
            coil_currents=np.zeros(1, dtype=np.float64),
            coil_current_limits=np.ones(1, dtype=np.float64),
        )


def test_compute_coil_correction_refuses_finite_weight_product_overflow() -> None:
    """Finite weights and response values cannot overflow into least squares."""
    with pytest.raises(ValueError, match="nonfinite"):
        law.compute_coil_correction(
            np.zeros(1, dtype=np.float64),
            target_vector=np.ones(1, dtype=np.float64),
            objective_bias_estimate=np.zeros(1, dtype=np.float64),
            control_objective_weights=np.array([1e308], dtype=np.float64),
            response_matrix=np.array([[1e308]], dtype=np.float64),
            response_regularization=1e-3,
            correction_limit=1.0,
            control_mask=np.ones(1, dtype=np.float64),
            coil_currents=np.zeros(1, dtype=np.float64),
            coil_current_limits=np.ones(1, dtype=np.float64),
        )


def test_compute_coil_correction_rejects_invalid_coil_limits() -> None:
    """A malformed coil envelope cannot reach headroom allocation."""
    for limits in (
        np.array([float("nan")], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    ):
        with pytest.raises(ValueError, match="coil_current_limits"):
            law.compute_coil_correction(
                np.zeros(1, dtype=np.float64),
                target_vector=np.ones(1, dtype=np.float64),
                objective_bias_estimate=np.zeros(1, dtype=np.float64),
                control_objective_weights=np.ones(1, dtype=np.float64),
                response_matrix=np.ones((1, 1), dtype=np.float64),
                response_regularization=1e-3,
                correction_limit=1.0,
                control_mask=np.ones(1, dtype=np.float64),
                coil_currents=np.zeros(1, dtype=np.float64),
                coil_current_limits=limits,
            )


def test_compute_coil_correction_refuses_headroom_penalty_overflow() -> None:
    """Extreme finite headroom ratios cannot enter the final least-squares solve."""
    with pytest.raises(ValueError, match="nonfinite"):
        law.compute_coil_correction(
            np.zeros(2, dtype=np.float64),
            target_vector=np.ones(2, dtype=np.float64),
            objective_bias_estimate=np.zeros(2, dtype=np.float64),
            control_objective_weights=np.ones(2, dtype=np.float64),
            response_matrix=np.eye(2, dtype=np.float64),
            response_regularization=1e-3,
            correction_limit=1.0,
            control_mask=np.ones(2, dtype=np.float64),
            coil_currents=np.zeros(2, dtype=np.float64),
            coil_current_limits=np.array([1e308, 1e-9], dtype=np.float64),
        )
