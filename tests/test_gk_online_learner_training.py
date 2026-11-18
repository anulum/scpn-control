# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK online learner training and admission behaviour.
"""Exact backpropagation, standardisation folding and same-slice admission of the GK learner."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.core._gk_learner_mlp import (
    LAYER_SHAPES,
    Standardiser,
    Weights,
    forward,
    loss_and_gradients,
    mean_squared_error,
)
from scpn_control.core.gk_online_learner import LearnerConfig, OnlineLearner

FloatArray = NDArray[np.float64]


def _random_weights(rng: np.random.Generator, scale: float = 0.3) -> Weights:
    """Return random weights of the learner's shapes."""
    return {name: rng.normal(0.0, scale, shape) for name, shape in LAYER_SHAPES.items()}


def _nonlinear_target(inputs: FloatArray) -> FloatArray:
    """Return a non-negative target that a linear read-out of random features cannot fit."""
    return np.stack(
        [
            np.abs(inputs[:, 0] - 0.5) + 0.2 * inputs[:, 1],
            inputs[:, 2] * inputs[:, 3] + 0.1,
            np.maximum(inputs[:, 4] - 0.3, 0.0) * 2.0,
        ],
        axis=1,
    )


def _filled(config: LearnerConfig, inputs: FloatArray, targets: FloatArray) -> OnlineLearner:
    """Return a learner whose buffer holds the given samples."""
    learner = OnlineLearner(config=config)
    for x, y in zip(inputs, targets, strict=True):
        learner.add_sample(x, y)
    return learner


def test_gradients_of_all_six_arrays_match_finite_differences() -> None:
    """Every weight array receives the exact gradient of the mean squared error."""
    rng = np.random.default_rng(7)
    weights = _random_weights(rng)
    inputs = rng.normal(size=(9, 10))
    targets = rng.normal(size=(9, 3))
    _, gradients = loss_and_gradients(weights, inputs, targets)
    step = 1e-6
    for name, shape in LAYER_SHAPES.items():
        probe = tuple(min(1, extent - 1) for extent in shape)
        plus = {key: value.copy() for key, value in weights.items()}
        minus = {key: value.copy() for key, value in weights.items()}
        plus[name][probe] += step
        minus[name][probe] -= step
        numeric = (mean_squared_error(plus, inputs, targets) - mean_squared_error(minus, inputs, targets)) / (2 * step)
        assert gradients[name][probe] == pytest.approx(numeric, rel=1e-5, abs=1e-9), name
        assert gradients[name].shape == shape


def test_folded_weights_map_raw_inputs_to_raw_targets() -> None:
    """Folding the standardisation into the outer layers is exact in both directions."""
    rng = np.random.default_rng(11)
    inputs = rng.normal(3.0, 2.0, size=(20, 10))
    targets = rng.normal(5.0, 4.0, size=(20, 3))
    scaler = Standardiser.fit(inputs, targets)
    standard = _random_weights(rng)
    raw = scaler.to_raw(standard)
    expected = forward(standard, scaler.inputs(inputs)) * scaler.y_scale + scaler.y_mean
    np.testing.assert_allclose(forward(raw, inputs), expected, rtol=1e-12, atol=1e-10)
    for name, array in scaler.to_standard(raw).items():
        np.testing.assert_allclose(array, standard[name], rtol=1e-12, atol=1e-12)


def test_constant_columns_keep_unit_scale() -> None:
    """A constant input or target column is centred but not divided by zero."""
    inputs = np.ones((4, 10))
    inputs[:, 0] = [0.0, 1.0, 2.0, 3.0]
    targets = np.full((4, 3), 0.25)
    scaler = Standardiser.fit(inputs, targets)
    assert np.all(scaler.x_scale[1:] == 1.0)
    assert np.all(scaler.y_scale == 1.0)
    assert np.all(np.isfinite(scaler.targets(targets)))


def test_retraining_updates_the_hidden_layers() -> None:
    """All six arrays move during retraining, not only the read-out layer."""
    rng = np.random.default_rng(3)
    inputs = rng.random((40, 10))
    learner = _filled(LearnerConfig(buffer_size=40, n_epochs=5, learning_rate=0.05), inputs, _nonlinear_target(inputs))
    learner.try_retrain()
    start = learner._weights_backup
    assert start is not None
    second = _filled(LearnerConfig(buffer_size=40, n_epochs=5, learning_rate=0.05), inputs, _nonlinear_target(inputs))
    trained = second.try_retrain(current_weights=start)
    assert trained is not None
    for name in LAYER_SHAPES:
        assert not np.allclose(trained[name], start[name]), name


def test_learned_model_beats_the_mean_predictor_on_a_nonlinear_target() -> None:
    """Training the hidden layers fits structure that the target mean cannot explain."""
    rng = np.random.default_rng(5)
    inputs = rng.random((200, 10))
    targets = _nonlinear_target(inputs)
    learner = _filled(LearnerConfig(buffer_size=200, n_epochs=400, learning_rate=0.05), inputs, targets)
    weights = learner.try_retrain()
    assert weights is not None
    decision = learner.latest_decision()
    assert decision is not None and decision.accepted
    # In standardised units the mean predictor scores about 1 on the validation slice.
    assert decision.val_loss < 0.5
    raw_error = float(np.mean((forward(weights, inputs) - targets) ** 2))
    assert raw_error < 0.5 * float(np.mean((targets - targets.mean(axis=0)) ** 2))


def test_decision_is_invariant_to_target_units() -> None:
    """Rescaling one transport coefficient's units leaves the standardised decision unchanged."""
    rng = np.random.default_rng(9)
    inputs = rng.random((30, 10))
    targets = _nonlinear_target(inputs)
    rescaled = targets * np.array([1.0e6, 1.0, 1.0e-3])
    plain = _filled(LearnerConfig(buffer_size=30, n_epochs=20, learning_rate=0.05), inputs, targets)
    scaled = _filled(LearnerConfig(buffer_size=30, n_epochs=20, learning_rate=0.05), inputs, rescaled)
    plain.try_retrain()
    scaled.try_retrain()
    assert scaled.retrain_history[-1]["val_loss"] == pytest.approx(plain.retrain_history[-1]["val_loss"], rel=1e-9)


def test_stale_best_from_another_slice_does_not_block_an_improving_candidate() -> None:
    """Admission compares with the incumbent on this slice, not with an earlier slice's loss."""
    rng = np.random.default_rng(13)
    inputs = rng.random((40, 10))
    targets = _nonlinear_target(inputs)
    learner = _filled(LearnerConfig(buffer_size=40, n_epochs=200, learning_rate=0.05), inputs, targets)
    learner._last_accepted_val_loss = 0.0  # an unbeatable value from some other slice
    incumbent = Standardiser.fit(inputs, targets).to_raw(_random_weights(np.random.default_rng(1), scale=0.1))
    accepted = learner.try_retrain(current_weights=incumbent)
    assert accepted is not None
    decision = learner.retrain_history[-1]
    assert decision["accepted"] is True
    assert decision["val_loss"] < decision["incumbent_val_loss"]


def test_min_validation_improvement_applies_against_the_incumbent() -> None:
    """A candidate that improves by less than the margin is rolled back."""
    rng = np.random.default_rng(17)
    inputs = rng.random((40, 10))
    targets = _nonlinear_target(inputs)
    learner = _filled(
        LearnerConfig(buffer_size=40, n_epochs=2, learning_rate=1e-4, min_validation_improvement=10.0), inputs, targets
    )
    incumbent = Standardiser.fit(inputs, targets).to_raw(_random_weights(np.random.default_rng(2), scale=0.1))
    assert learner.try_retrain(current_weights=incumbent) is None
    decision = learner.retrain_history[-1]
    assert decision["reason"] == "validation_loss_not_improved"
    assert decision["val_loss"] < decision["incumbent_val_loss"]


def test_all_non_finite_epochs_are_rejected() -> None:
    """An overflowing step never yields weights; the decision names the non-finite loss."""
    rng = np.random.default_rng(19)
    inputs = rng.random((10, 10))
    learner = _filled(LearnerConfig(buffer_size=10, n_epochs=2, learning_rate=1e305), inputs, _nonlinear_target(inputs))
    assert learner.try_retrain() is None
    decision = learner.retrain_history[-1]
    assert decision["reason"] == "validation_loss_not_finite"
    assert decision["accepted"] is False
    assert len(learner.buffer) == 10


def test_decision_generation_matches_the_learner_generation() -> None:
    """The first accepted decision is generation 1; a later rejection keeps that number."""
    rng = np.random.default_rng(23)
    inputs = rng.random((20, 10))
    targets = _nonlinear_target(inputs)
    learner = _filled(LearnerConfig(buffer_size=20, n_epochs=2, learning_rate=0.05), inputs, targets)
    weights = learner.try_retrain()
    assert weights is not None
    assert learner.generation == 1
    assert learner.retrain_history[-1]["generation"] == 1
    learner.config.learning_rate = 1e305
    for x, y in zip(inputs, targets, strict=True):
        learner.add_sample(x, y)
    assert learner.try_retrain(current_weights=weights) is None
    assert learner.retrain_history[-1]["generation"] == 1
    assert learner.retrain_history[-1]["accepted"] is False
