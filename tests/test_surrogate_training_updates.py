# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Update order of the NumPy surrogate trainers.
"""The turbulence and equilibrium MLP trainers update from gradients of one forward pass.

Each trainer previously changed layer i before propagating the error to layer
i-1 through the changed weights. These tests pin the corrected order: a step
equals the clipped exact gradient at the pre-step weights, and the equilibrium
model keeps the weights of its best validation epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.core import neural_turbulence
from scpn_control.core.neural_equilibrium import NeuralEqConfig, NeuralEquilibriumAccelerator, SimpleMLP
from scpn_control.core.neural_turbulence import NeuralTransportTrainer, QLKNNSurrogate, TrainingDataGenerator

FloatArray = NDArray[np.float64]
SPARC_DIR = Path(__file__).resolve().parents[1] / "validation" / "reference_data" / "sparc"


def _elu(x: FloatArray) -> FloatArray:
    """Return ELU(x) with unit scale."""
    return np.where(x > 0.0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


def _mse(weights: list[FloatArray], biases: list[FloatArray], x: FloatArray, y: FloatArray) -> float:
    """Return the mean squared error of an ELU network with a linear output layer."""
    out = x
    for w, b in zip(weights[:-1], biases[:-1], strict=True):
        out = _elu(out @ w + b)
    return float(np.mean((out @ weights[-1] + biases[-1] - y) ** 2))


def _numeric_gradients(
    weights: list[FloatArray], biases: list[FloatArray], x: FloatArray, y: FloatArray
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Return central finite-difference gradients of :func:`_mse` for every parameter."""
    step = 1e-6
    grads: list[list[FloatArray]] = [[], []]
    for slot, params in enumerate((weights, biases)):
        for layer in range(len(params)):
            grad = np.zeros_like(params[layer])
            for index in np.ndindex(params[layer].shape):
                saved = params[layer][index]
                params[layer][index] = saved + step
                up = _mse(weights, biases, x, y)
                params[layer][index] = saved - step
                down = _mse(weights, biases, x, y)
                params[layer][index] = saved
                grad[index] = (up - down) / (2 * step)
            grads[slot].append(grad)
    return grads[0], grads[1]


def test_transport_trainer_step_is_the_exact_gradient_at_the_pre_step_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """One full-batch epoch moves every layer by -lr times the gradient at the weights it started from."""
    rng = np.random.RandomState(7)
    model = QLKNNSurrogate(hidden_layers=[5, 4], activation="elu", pretrained=False)
    model.weights = [rng.randn(*w.shape) * 0.4 for w in model.weights]
    model.biases = [rng.randn(*b.shape) * 0.1 for b in model.biases]
    start_w = [w.copy() for w in model.weights]
    start_b = [b.copy() for b in model.biases]
    x = rng.randn(21, 10)
    y = model.forward(x) + 0.05 * rng.randn(21, 3)
    monkeypatch.setattr(neural_turbulence, "QLKNNSurrogate", lambda pretrained=False: model)

    grad_w, grad_b = _numeric_gradients(start_w, start_b, x[:-1], y[:-1])
    assert all(float(np.linalg.norm(g)) < 1.0 for g in grad_w), "the step must stay below the clipping norm"
    lr = 0.5
    trainer = NeuralTransportTrainer()
    history = trainer.train(x, y, epochs=1, lr=lr, val_frac=0.04)

    assert history["train_loss"] == [pytest.approx(_mse(start_w, start_b, x[:-1], y[:-1]), rel=1e-12)]
    for layer in range(3):
        np.testing.assert_allclose(model.weights[layer], start_w[layer] - lr * grad_w[layer], rtol=0, atol=1e-8)
        np.testing.assert_allclose(model.biases[layer], start_b[layer] - lr * grad_b[layer], rtol=0, atol=1e-8)


def test_pretrained_surrogate_follows_the_documented_schedule() -> None:
    """Construction replays 100 clipped full-batch steps of size 1e-3, each from one forward pass."""
    rng = np.random.RandomState(42)
    layers = [10, 128, 128, 64, 3]
    weights = [rng.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i]) for i in range(4)]
    biases = [np.zeros(layers[i + 1]) for i in range(4)]
    full_x = TrainingDataGenerator.generate_parameter_scan(500, rng=rng)
    x, y = full_x[:-50], TrainingDataGenerator.generate_analytic_targets(full_x)[:-50]
    for _ in range(100):
        pre: list[FloatArray] = []
        acts = [x]
        out = x
        for w, b in zip(weights[:-1], biases[:-1], strict=True):
            pre.append(out @ w + b)
            out = _elu(pre[-1])
            acts.append(out)
        delta = 2.0 * (out @ weights[-1] + biases[-1] - y) / y.size
        steps: list[tuple[int, FloatArray, FloatArray]] = []
        for layer in range(3, -1, -1):
            dw = acts[layer].T @ delta
            db = delta.sum(axis=0)
            if float(np.linalg.norm(dw)) > 1.0:
                dw = dw / float(np.linalg.norm(dw))
                db = db / max(float(np.linalg.norm(db)), 1e-8)
            steps.append((layer, dw, db))
            if layer > 0:
                delta = np.clip(
                    (delta @ weights[layer].T) * np.where(pre[layer - 1] > 0, 1.0, np.exp(pre[layer - 1])), -1e6, 1e6
                )
        for layer, dw, db in steps:
            weights[layer] = weights[layer] - 1e-3 * dw
            biases[layer] = biases[layer] - 1e-3 * db

    surrogate = QLKNNSurrogate()
    for layer in range(4):
        np.testing.assert_allclose(surrogate.weights[layer], weights[layer], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(surrogate.biases[layer], biases[layer], rtol=1e-10, atol=1e-12)


def test_equilibrium_training_keeps_the_best_validation_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The returned model predicts the validation set exactly as at the reported best epoch."""
    files = sorted(SPARC_DIR.glob("*.geqdsk"))[:2]
    assert len(files) == 2, "the SPARC reference equilibria are part of the repository"
    calls: list[tuple[Any, FloatArray]] = []
    original = SimpleMLP.forward

    def recording_forward(self: SimpleMLP, x: Any) -> Any:
        """Record each input object and its prediction."""
        out = original(self, x)
        calls.append((x, np.array(out, copy=True)))
        return out

    monkeypatch.setattr(SimpleMLP, "forward", recording_forward)
    accel = NeuralEquilibriumAccelerator(NeuralEqConfig(n_components=5, hidden_sizes=(16, 8)))
    result = accel.train_from_geqdsk(files, n_perturbations=3, seed=0)
    monkeypatch.undo()

    validation_input = calls[0][0]
    epochs = [prediction for given, prediction in calls if given is validation_input]
    assert 0 <= result.best_epoch < len(epochs)
    assert accel.mlp is not None
    np.testing.assert_array_equal(accel.mlp.forward(validation_input), epochs[result.best_epoch])
    if result.best_epoch < len(epochs) - 1:
        assert not np.array_equal(epochs[-1], epochs[result.best_epoch])
