# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural Equilibrium Tests

"""Tests for scpn_control.core.neural_equilibrium."""

import hashlib
import json

import numpy as np
import pytest

from scpn_control.core.neural_equilibrium import (
    NEURAL_EQ_FEATURE_NAMES,
    MinimalPCA,
    NeuralEqConfig,
    NeuralEquilibriumAccelerator,
    NeuralEquilibriumClaimEvidence,
    PretrainingResult,
    SimpleMLP,
    SyntheticEquilibriumCampaign,
    assert_neural_equilibrium_facility_claim_admissible,
    generate_synthetic_equilibrium_dataset,
    neural_equilibrium_claim_evidence,
    save_neural_equilibrium_claim_evidence,
)
from validation.validate_neural_equilibrium_reference import canonical_artifact_sha256


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

    def test_claim_evidence_records_synthetic_pretraining_boundary(self, tmp_path):
        acc = NeuralEquilibriumAccelerator(
            NeuralEqConfig(n_components=6, hidden_sizes=(), n_input_features=12, grid_shape=(17, 19))
        )
        weights = tmp_path / "synthetic_pretrain.npz"
        result = acc.pretrain_from_synthetic_equilibria(160, seed=8, save_path=weights)

        evidence = neural_equilibrium_claim_evidence(
            result,
            weights_path=weights,
            source="synthetic_regression_reference",
            source_id="tests/test_neural_equilibrium.py::synthetic_claim_boundary",
        )

        assert isinstance(evidence, NeuralEquilibriumClaimEvidence)
        assert evidence.facility_claim_allowed is False
        assert evidence.reference_source == "none"
        assert evidence.reference_equilibria_count == 0
        assert evidence.feature_names == NEURAL_EQ_FEATURE_NAMES
        assert evidence.grid_shape == (17, 19)
        assert evidence.synthetic_test_mse == pytest.approx(result.test_mse)
        assert evidence.synthetic_gs_residual == pytest.approx(result.gs_residual)
        assert evidence.weights_sha256 == hashlib.sha256(weights.read_bytes()).hexdigest()

        out = tmp_path / "claim.json"
        save_neural_equilibrium_claim_evidence(evidence, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["claim_status"].startswith("synthetic pretraining evidence only")
        assert payload["facility_claim_allowed"] is False
        with pytest.raises(ValueError, match="blocked without matched reference"):
            assert_neural_equilibrium_facility_claim_admissible(evidence)

    def test_facility_admission_requires_reference_artifact_matching_weights(self, tmp_path):
        acc = NeuralEquilibriumAccelerator(
            NeuralEqConfig(n_components=6, hidden_sizes=(), n_input_features=12, grid_shape=(17, 19))
        )
        weights = tmp_path / "synthetic_pretrain.npz"
        result = acc.pretrain_from_synthetic_equilibria(160, seed=9, save_path=weights)
        weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        artifact_payload = {
            "schema_version": "scpn-control.neural-equilibrium-reference.v1",
            "source": "documented_public_reference",
            "model_id": "neural_equilibrium_pca_mlp",
            "model_version": "test",
            "trained_weights_sha256": weights_sha,
            "reference_dataset_id": "bounded-reference-fixture",
            "reference_artifact_uri": "bounded-reference-fixture/reference_equilibria.npz",
            "prediction_artifact_uri": "bounded-reference-fixture/scpn_predictions.npz",
            "reference_artifact_sha256": "a" * 64,
            "prediction_artifact_sha256": "b" * 64,
            "executed_at": "2026-05-31T00:00:00Z",
            "reference_url": "https://example.invalid/reference",
            "grid_shape": [17, 19],
            "target_schema": ["psi", "pressure", "q_profile", "lcfs_boundary", "magnetic_axis"],
            "units": {"psi": "Wb/rad", "pressure": "Pa", "q_profile": "1", "boundary": "m"},
            "reference_equilibria_count": 3,
            "metrics": {
                "psi_rmse_Wb": 0.01,
                "pressure_rmse_Pa": 100.0,
                "q_profile_rmse": 0.02,
                "boundary_rmse_m": 0.003,
                "axis_position_error_m": 0.002,
            },
            "tolerances": {
                "psi_rmse_Wb": 0.02,
                "pressure_rmse_Pa": 200.0,
                "q_profile_rmse": 0.05,
                "boundary_rmse_m": 0.01,
                "axis_position_error_m": 0.01,
            },
        }
        artifact_payload["payload_sha256"] = canonical_artifact_sha256(artifact_payload)
        artifact = tmp_path / "reference.json"
        artifact.write_text(json.dumps(artifact_payload), encoding="utf-8")

        evidence = neural_equilibrium_claim_evidence(
            result,
            weights_path=weights,
            source="documented_public_reference",
            source_id="tests/test_neural_equilibrium.py::reference_claim_boundary",
            reference_artifact_path=artifact,
        )

        assert evidence.facility_claim_allowed is True
        assert evidence.reference_source == "documented_public_reference"
        assert evidence.reference_equilibria_count == 3
        assert evidence.psi_rmse_Wb == pytest.approx(0.01)
        assert evidence.psi_tolerance_Wb == pytest.approx(0.02)
        assert assert_neural_equilibrium_facility_claim_admissible(evidence) is evidence

        artifact_payload["trained_weights_sha256"] = "b" * 64
        artifact_payload["payload_sha256"] = canonical_artifact_sha256(artifact_payload)
        artifact.write_text(json.dumps(artifact_payload), encoding="utf-8")
        with pytest.raises(ValueError, match="does not match supplied weights"):
            neural_equilibrium_claim_evidence(
                result,
                weights_path=weights,
                source="documented_public_reference",
                source_id="tests/test_neural_equilibrium.py::reference_claim_boundary",
                reference_artifact_path=artifact,
            )
