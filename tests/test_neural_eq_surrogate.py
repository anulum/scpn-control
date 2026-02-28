# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Neural Equilibrium Surrogate Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for neural_equilibrium: MLP, PCA, Accelerator, save/load."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.neural_equilibrium import (
    MinimalPCA,
    NeuralEqConfig,
    NeuralEquilibriumAccelerator,
    SimpleMLP,
    TrainingResult,
)


class TestSimpleMLP:
    def test_forward_shape(self):
        mlp = SimpleMLP([4, 8, 2], seed=0)
        x = np.ones((3, 4))
        y = mlp.forward(x)
        assert y.shape == (3, 2)

    def test_predict_equals_forward(self):
        mlp = SimpleMLP([5, 10, 3], seed=7)
        x = np.random.default_rng(0).normal(size=(6, 5))
        np.testing.assert_array_equal(mlp.predict(x), mlp.forward(x))

    def test_single_layer(self):
        mlp = SimpleMLP([3, 2], seed=0)
        assert len(mlp.weights) == 1
        x = np.ones((1, 3))
        y = mlp.forward(x)
        assert y.shape == (1, 2)

    def test_deterministic(self):
        m1 = SimpleMLP([4, 8, 2], seed=42)
        m2 = SimpleMLP([4, 8, 2], seed=42)
        np.testing.assert_array_equal(m1.weights[0], m2.weights[0])


class TestMinimalPCA:
    def test_fit_transform_shape(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 20))
        pca = MinimalPCA(n_components=5)
        Z = pca.fit_transform(X)
        assert Z.shape == (50, 5)

    def test_inverse_transform_recovers(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 10))
        pca = MinimalPCA(n_components=10)
        Z = pca.fit_transform(X)
        X_rec = pca.inverse_transform(Z)
        np.testing.assert_allclose(X_rec, X, atol=1e-8)

    def test_explained_variance(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 8))
        pca = MinimalPCA(n_components=3)
        pca.fit(X)
        assert pca.explained_variance_ratio_ is not None
        assert len(pca.explained_variance_ratio_) == 3
        assert all(v >= 0 for v in pca.explained_variance_ratio_)


class TestNeuralEquilibriumAccelerator:
    @pytest.fixture()
    def trained_accel(self, tmp_path):
        """Build a manually-trained accelerator on synthetic data."""
        rng = np.random.default_rng(42)
        cfg = NeuralEqConfig(
            n_components=5, hidden_sizes=(16, 8),
            n_input_features=12, grid_shape=(8, 8),
        )
        accel = NeuralEquilibriumAccelerator(cfg)

        n_samples = 30
        X = rng.normal(size=(n_samples, 12))
        Y = rng.normal(size=(n_samples, 64))

        accel._input_mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-10] = 1.0
        accel._input_std = std

        accel.pca.fit(Y)
        Y_c = accel.pca.transform(Y)

        layer_sizes = [12, 16, 8, 5]
        accel.mlp = SimpleMLP(layer_sizes, seed=0)
        accel.is_trained = True
        return accel, X, Y

    def test_predict_not_trained_raises(self):
        accel = NeuralEquilibriumAccelerator()
        with pytest.raises(RuntimeError, match="not trained"):
            accel.predict(np.ones(12))

    def test_predict_single(self, trained_accel):
        accel, X, _ = trained_accel
        psi = accel.predict(X[0])
        assert psi.shape == (8, 8)

    def test_predict_batch(self, trained_accel):
        accel, X, _ = trained_accel
        psi = accel.predict(X[:5])
        assert psi.shape == (5, 8, 8)

    def test_evaluate_not_trained_raises(self):
        accel = NeuralEquilibriumAccelerator()
        with pytest.raises(RuntimeError, match="Not trained"):
            accel.evaluate_surrogate(np.ones((2, 12)), np.ones((2, 64)))

    def test_evaluate_surrogate(self, trained_accel):
        accel, X, Y = trained_accel
        metrics = accel.evaluate_surrogate(X[:5], Y[:5])
        assert "mse" in metrics
        assert "max_error" in metrics
        assert "gs_residual" in metrics
        assert metrics["mse"] >= 0.0

    def test_gs_residual_loss(self, trained_accel):
        accel, _, _ = trained_accel
        psi_flat = np.random.default_rng(0).normal(size=64)
        loss = accel._gs_residual_loss(psi_flat, (8, 8))
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_save_load_roundtrip(self, trained_accel, tmp_path):
        accel, X, _ = trained_accel
        path = tmp_path / "weights.npz"
        accel.save_weights(path)
        assert path.exists()

        accel2 = NeuralEquilibriumAccelerator()
        accel2.load_weights(path)
        assert accel2.is_trained

        psi_orig = accel.predict(X[0])
        psi_loaded = accel2.predict(X[0])
        np.testing.assert_allclose(psi_loaded, psi_orig, atol=1e-10)

    def test_benchmark(self, trained_accel):
        accel, X, _ = trained_accel
        stats = accel.benchmark(X[0], n_runs=5)
        assert "mean_ms" in stats
        assert "median_ms" in stats
        assert "p95_ms" in stats
        assert stats["mean_ms"] >= 0.0


class TestTrainingResult:
    def test_dataclass_fields(self):
        r = TrainingResult(
            n_samples=100, n_components=20, explained_variance=0.99,
            final_loss=0.01, train_time_s=5.0, weights_path="/tmp/w.npz",
        )
        assert r.n_samples == 100
        assert np.isnan(r.val_loss)
        assert np.isnan(r.test_mse)
