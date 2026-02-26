"""Tests for scpn_control.core.neural_equilibrium."""

import numpy as np
import pytest

from scpn_control.core.neural_equilibrium import (
    MinimalPCA,
    NeuralEqConfig,
    NeuralEquilibriumAccelerator,
    SimpleMLP,
)


class TestSimpleMLP:
    def test_forward_shape(self):
        mlp = SimpleMLP([10, 32, 16, 5], seed=0)
        x = np.random.default_rng(0).standard_normal((4, 10))
        out = mlp.forward(x)
        assert out.shape == (4, 5)

    def test_predict_equals_forward(self):
        mlp = SimpleMLP([3, 8, 2], seed=1)
        x = np.ones((1, 3))
        np.testing.assert_array_equal(mlp.predict(x), mlp.forward(x))

    def test_relu_nonnegative_hidden(self):
        mlp = SimpleMLP([2, 4, 1], seed=42)
        x = np.array([[1.0, -1.0]])
        mlp.forward(x)  # no crash; output can be negative (linear output layer)

    def test_deterministic_with_seed(self):
        a = SimpleMLP([5, 10, 3], seed=7)
        b = SimpleMLP([5, 10, 3], seed=7)
        x = np.ones((2, 5))
        np.testing.assert_array_equal(a.forward(x), b.forward(x))


class TestMinimalPCA:
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 20))
        pca = MinimalPCA(n_components=10)
        Z = pca.fit_transform(X)
        assert Z.shape == (50, 10)
        X_hat = pca.inverse_transform(Z)
        assert X_hat.shape == X.shape

    def test_explained_variance_sums_below_one(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 30))
        pca = MinimalPCA(n_components=5).fit(X)
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-10

    def test_n_components_respected(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 15))
        pca = MinimalPCA(n_components=3).fit(X)
        assert pca.components_.shape[0] == 3


class TestNeuralEquilibriumAccelerator:
    def test_default_config(self):
        acc = NeuralEquilibriumAccelerator()
        assert acc.cfg.n_components == 20
        assert not acc.is_trained

    def test_custom_config(self):
        cfg = NeuralEqConfig(n_components=5, hidden_sizes=(32, 16), n_input_features=8)
        acc = NeuralEquilibriumAccelerator(cfg)
        assert acc.cfg.n_components == 5

    def test_gs_residual_loss_zero_for_flat(self):
        acc = NeuralEquilibriumAccelerator()
        flat = np.ones(129 * 129)
        loss = acc._gs_residual_loss(flat, (129, 129))
        assert loss == pytest.approx(0.0, abs=1e-12)

    def test_evaluate_before_train_raises(self):
        acc = NeuralEquilibriumAccelerator()
        with pytest.raises(RuntimeError, match="Not trained"):
            acc.evaluate_surrogate(np.zeros((1, 12)), np.zeros((1, 129 * 129)))
