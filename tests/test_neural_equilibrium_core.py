# Tests for neural_equilibrium pure-numpy components (no GEQDSK/files needed).

import numpy as np
import pytest

from scpn_control.core.neural_equilibrium import (
    MinimalPCA,
    NeuralEqConfig,
    SimpleMLP,
    TrainingResult,
)


class TestNeuralEqConfig:
    def test_defaults(self):
        cfg = NeuralEqConfig()
        assert cfg.n_components == 20
        assert cfg.hidden_sizes == (128, 64, 32)
        assert cfg.n_input_features == 12
        assert cfg.grid_shape == (129, 129)
        assert cfg.lambda_gs == pytest.approx(0.1)

    def test_custom(self):
        cfg = NeuralEqConfig(n_components=5, hidden_sizes=(64,), lambda_gs=0.5)
        assert cfg.n_components == 5
        assert cfg.hidden_sizes == (64,)


class TestTrainingResult:
    def test_fields(self):
        r = TrainingResult(
            n_samples=100, n_components=10, explained_variance=0.95,
            final_loss=0.01, train_time_s=5.0, weights_path="/tmp/w.npz",
        )
        assert r.n_samples == 100
        assert np.isnan(r.val_loss)
        assert np.isnan(r.test_mse)


class TestSimpleMLP:
    def test_forward_shape(self):
        mlp = SimpleMLP([4, 8, 3], seed=0)
        x = np.random.default_rng(1).normal(size=(5, 4))
        out = mlp.forward(x)
        assert out.shape == (5, 3)

    def test_predict_equals_forward(self):
        mlp = SimpleMLP([3, 6, 2], seed=42)
        x = np.ones((2, 3))
        np.testing.assert_array_equal(mlp.predict(x), mlp.forward(x))

    def test_single_hidden_layer(self):
        mlp = SimpleMLP([2, 5, 1], seed=7)
        assert len(mlp.weights) == 2
        assert mlp.weights[0].shape == (2, 5)
        assert mlp.weights[1].shape == (5, 1)

    def test_he_init_scale(self):
        mlp = SimpleMLP([100, 50], seed=0)
        # He init: std ~ sqrt(2/fan_in) = sqrt(2/100) ~ 0.14
        empirical_std = float(np.std(mlp.weights[0]))
        assert 0.05 < empirical_std < 0.30

    def test_relu_kills_negative(self):
        mlp = SimpleMLP([1, 4, 1], seed=0)
        # Force W0 to produce negative pre-activations
        mlp.weights[0] = -np.ones((1, 4))
        mlp.biases[0] = np.zeros(4)
        mlp.weights[1] = np.ones((4, 1))
        mlp.biases[1] = np.zeros(1)
        out = mlp.forward(np.array([[1.0]]))
        # All hidden activations negative → ReLU zeroes them → output = 0
        assert out[0, 0] == pytest.approx(0.0)


class TestMinimalPCA:
    def test_fit_transform_shape(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 20))
        pca = MinimalPCA(n_components=5)
        Z = pca.fit_transform(X)
        assert Z.shape == (50, 5)

    def test_inverse_roundtrip(self):
        rng = np.random.default_rng(0)
        # Low-rank data: should reconstruct near-exactly with enough components
        basis = rng.normal(size=(3, 30))
        X = rng.normal(size=(40, 3)) @ basis
        pca = MinimalPCA(n_components=3)
        Z = pca.fit_transform(X)
        X_recon = pca.inverse_transform(Z)
        np.testing.assert_allclose(X_recon, X, atol=1e-10)

    def test_explained_variance_sums_to_one_if_enough_components(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(30, 10))
        pca = MinimalPCA(n_components=10)
        pca.fit(X)
        total = float(np.sum(pca.explained_variance_ratio_))
        assert total == pytest.approx(1.0, abs=1e-8)

    def test_explained_variance_decreasing(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(100, 15))
        pca = MinimalPCA(n_components=10)
        pca.fit(X)
        ev = pca.explained_variance_ratio_
        for i in range(len(ev) - 1):
            assert ev[i] >= ev[i + 1] - 1e-12

    def test_mean_removed(self):
        rng = np.random.default_rng(5)
        X = rng.normal(loc=100.0, size=(20, 8))
        pca = MinimalPCA(n_components=3)
        pca.fit(X)
        np.testing.assert_allclose(pca.mean_, X.mean(axis=0), atol=1e-12)
