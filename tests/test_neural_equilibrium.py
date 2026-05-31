# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Neural Equilibrium
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

"""Tests for scpn_control.core.neural_equilibrium."""

import numpy as np
import pytest

from scpn_control.core.neural_equilibrium import (
    NEURAL_EQ_FEATURE_NAMES,
    MinimalPCA,
    NeuralEqConfig,
    NeuralEquilibriumAccelerator,
    PretrainingResult,
    SimpleMLP,
    SyntheticEquilibriumCampaign,
    generate_synthetic_equilibrium_dataset,
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


class TestSyntheticPretraining:
    def test_synthetic_dataset_contract_is_finite_and_reproducible(self):
        x1, y1, campaign = generate_synthetic_equilibrium_dataset(32, grid_shape=(17, 19), seed=123)
        x2, y2, _ = generate_synthetic_equilibrium_dataset(32, grid_shape=(17, 19), seed=123)
        assert isinstance(campaign, SyntheticEquilibriumCampaign)
        assert campaign.feature_names == NEURAL_EQ_FEATURE_NAMES
        assert x1.shape == (32, 12)
        assert y1.shape == (32, 17 * 19)
        assert np.all(np.isfinite(x1))
        assert np.all(np.isfinite(y1))
        np.testing.assert_allclose(x1, x2)
        np.testing.assert_allclose(y1, y2)

    def test_pretraining_produces_jax_compatible_weights(self, tmp_path):
        acc = NeuralEquilibriumAccelerator(
            NeuralEqConfig(n_components=6, hidden_sizes=(), n_input_features=12, grid_shape=(17, 19))
        )
        path = tmp_path / "synthetic_pretrain.npz"
        result = acc.pretrain_from_synthetic_equilibria(160, seed=7, save_path=path)
        assert isinstance(result, PretrainingResult)
        assert result.evidence_kind == "synthetic_pretraining"
        assert result.n_samples == 160
        assert 0.0 < result.explained_variance <= 1.0
        assert result.test_mse >= 0.0
        assert np.isfinite(result.gs_residual)
        assert path.exists()

        reloaded = NeuralEquilibriumAccelerator()
        reloaded.load_weights(path)
        x, _, _ = generate_synthetic_equilibrium_dataset(2, grid_shape=(17, 19), seed=9)
        pred = reloaded.predict(x)
        assert pred.shape == (2, 17, 19)
        assert np.all(np.isfinite(pred))

    def test_real_efit_fine_tune_requires_reference_artifacts(self):
        acc = NeuralEquilibriumAccelerator()
        with pytest.raises(RuntimeError, match="requires passing neural equilibrium reference artifacts"):
            acc.fine_tune_from_efit_reconstructions([], reference_artifact_root="/does/not/exist")
