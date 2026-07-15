# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Neural Equilibrium Core
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# Tests for neural_equilibrium pure-numpy components (no GEQDSK/files needed).

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_control.core.neural_equilibrium import (
    NEURAL_EQ_FEATURE_NAMES,
    MinimalPCA,
    NeuralEqConfig,
    NeuralEquilibriumAccelerator,
    PretrainingResult,
    SimpleMLP,
    TrainingResult,
    _finite_nonnegative_or_none,
    _finite_positive_or_none,
    _non_empty_text,
    _require_positive_float,
    _require_positive_int,
    _synthetic_equilibrium_from_features,
    _validate_feature_matrix,
    _validate_psi_matrix,
    assert_neural_equilibrium_facility_claim_admissible,
    neural_equilibrium_claim_evidence,
    pretrain_neural_equilibrium_synthetic,
    save_neural_equilibrium_claim_evidence,
)
from validation.validate_neural_equilibrium_reference import canonical_artifact_sha256


class TestNeuralEqConfig:
    def test_defaults(self) -> None:
        cfg = NeuralEqConfig()
        assert cfg.n_components == 20
        assert cfg.hidden_sizes == (128, 64, 32)
        assert cfg.n_input_features == 12
        assert cfg.grid_shape == (129, 129)
        assert cfg.lambda_gs == pytest.approx(0.1)

    def test_custom(self) -> None:
        cfg = NeuralEqConfig(n_components=5, hidden_sizes=(64,), lambda_gs=0.5)
        assert cfg.n_components == 5
        assert cfg.hidden_sizes == (64,)


class TestTrainingResult:
    def test_fields(self) -> None:
        r = TrainingResult(
            n_samples=100,
            n_components=10,
            explained_variance=0.95,
            final_loss=0.01,
            train_time_s=5.0,
            weights_path="/tmp/w.npz",
        )
        assert r.n_samples == 100
        assert np.isnan(r.val_loss)
        assert np.isnan(r.test_mse)


class TestSimpleMLP:
    def test_forward_shape(self) -> None:
        mlp = SimpleMLP([4, 8, 3], seed=0)
        x = np.random.default_rng(1).normal(size=(5, 4))
        out = mlp.forward(x)
        assert out.shape == (5, 3)

    def test_predict_equals_forward(self) -> None:
        mlp = SimpleMLP([3, 6, 2], seed=42)
        x = np.ones((2, 3))
        np.testing.assert_array_equal(mlp.predict(x), mlp.forward(x))

    def test_single_hidden_layer(self) -> None:
        mlp = SimpleMLP([2, 5, 1], seed=7)
        assert len(mlp.weights) == 2
        assert mlp.weights[0].shape == (2, 5)
        assert mlp.weights[1].shape == (5, 1)

    def test_he_init_scale(self) -> None:
        mlp = SimpleMLP([100, 50], seed=0)
        # He init: std ~ sqrt(2/fan_in) = sqrt(2/100) ~ 0.14
        empirical_std = float(np.std(mlp.weights[0]))
        assert 0.05 < empirical_std < 0.30

    def test_relu_kills_negative(self) -> None:
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
    def test_fit_transform_shape(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 20))
        pca = MinimalPCA(n_components=5)
        Z = pca.fit_transform(X)
        assert Z.shape == (50, 5)

    def test_inverse_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        # Low-rank data: should reconstruct near-exactly with enough components
        basis = rng.normal(size=(3, 30))
        X = rng.normal(size=(40, 3)) @ basis
        pca = MinimalPCA(n_components=3)
        Z = pca.fit_transform(X)
        X_recon = pca.inverse_transform(Z)
        np.testing.assert_allclose(X_recon, X, atol=1e-10)

    def test_explained_variance_sums_to_one_if_enough_components(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(30, 10))
        pca = MinimalPCA(n_components=10)
        pca.fit(X)
        assert pca.explained_variance_ratio_ is not None
        total = float(np.sum(pca.explained_variance_ratio_))
        assert total == pytest.approx(1.0, abs=1e-8)

    def test_explained_variance_decreasing(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(100, 15))
        pca = MinimalPCA(n_components=10)
        pca.fit(X)
        assert pca.explained_variance_ratio_ is not None
        ev = pca.explained_variance_ratio_
        for i in range(len(ev) - 1):
            assert ev[i] >= ev[i + 1] - 1e-12

    def test_mean_removed(self) -> None:
        rng = np.random.default_rng(5)
        X = rng.normal(loc=100.0, size=(20, 8))
        pca = MinimalPCA(n_components=3)
        pca.fit(X)
        assert pca.mean_ is not None
        np.testing.assert_allclose(pca.mean_, X.mean(axis=0), atol=1e-12)


class TestNeuralEqValidationHelpers:
    def test_require_positive_int_rejects_non_integer(self) -> None:
        bad: Any = 2.5
        with pytest.raises(ValueError, match="must be an integer >= 1"):
            _require_positive_int("n", bad)

    def test_require_positive_int_rejects_below_one(self) -> None:
        with pytest.raises(ValueError, match="must be an integer >= 1"):
            _require_positive_int("n", 0)

    def test_require_positive_float_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="must be positive and finite"):
            _require_positive_float("x", 0.0)

    def test_non_empty_text_rejects_blank(self) -> None:
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _non_empty_text("name", "   ")

    def test_finite_nonnegative_or_none_rejects_bad_type(self) -> None:
        bad: Any = "nan"
        with pytest.raises(ValueError, match="must be finite and non-negative"):
            _finite_nonnegative_or_none("x", bad)

    def test_finite_nonnegative_or_none_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="must be finite and non-negative"):
            _finite_nonnegative_or_none("x", -1.0)

    def test_finite_nonnegative_or_none_passes_through_none(self) -> None:
        assert _finite_nonnegative_or_none("x", None) is None

    def test_finite_positive_or_none_rejects_bad_type(self) -> None:
        bad: Any = True
        with pytest.raises(ValueError, match="must be finite and positive"):
            _finite_positive_or_none("x", bad)

    def test_finite_positive_or_none_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="must be finite and positive"):
            _finite_positive_or_none("x", 0.0)

    def test_validate_feature_matrix_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="features must be finite with shape"):
            _validate_feature_matrix(np.zeros((4, len(NEURAL_EQ_FEATURE_NAMES) - 1)))

    def test_validate_psi_matrix_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="psi targets must be finite with shape"):
            _validate_psi_matrix(np.zeros((4, 5)), (3, 3))

    def test_synthetic_equilibrium_handles_degenerate_flux(self) -> None:
        # simag == sibry makes the flux denominator degenerate; the helper must
        # fall back to a unit denominator and still return a finite field.
        features = np.array([8.0, 5.0, 1.6, 0.0, 1.0, 1.0, 0.5, 0.5, 1.7, 0.4, 0.4, 4.0])
        psi = _synthetic_equilibrium_from_features(features, (9, 11))
        assert psi.shape == (9, 11)
        assert np.all(np.isfinite(psi))


def _pretrained_accelerator(tmp_path: Path, seed: int) -> tuple[NeuralEquilibriumAccelerator, Path, PretrainingResult]:
    accel = NeuralEquilibriumAccelerator(
        NeuralEqConfig(n_components=6, hidden_sizes=(), n_input_features=12, grid_shape=(17, 19))
    )
    weights = tmp_path / "weights.npz"
    result = accel.pretrain_from_synthetic_equilibria(160, seed=seed, save_path=weights)
    return accel, weights, result


def _reference_artifact_payload(weights_sha: str) -> dict[str, Any]:
    return {
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


class TestClaimEvidenceGuards:
    def test_claim_evidence_rejects_non_pretraining_result(self, tmp_path: Path) -> None:
        not_result: Any = object()
        with pytest.raises(ValueError, match="pretraining must be a PretrainingResult"):
            neural_equilibrium_claim_evidence(
                not_result, weights_path=tmp_path / "missing.npz", source="s", source_id="id"
            )

    def test_claim_evidence_rejects_missing_weights(self, tmp_path: Path) -> None:
        _, _, result = _pretrained_accelerator(tmp_path, seed=11)
        with pytest.raises(FileNotFoundError, match="weights not found"):
            neural_equilibrium_claim_evidence(
                result, weights_path=tmp_path / "absent.npz", source="synthetic_regression_reference", source_id="id"
            )

    def test_claim_evidence_rejects_reference_failing_strict_validation(self, tmp_path: Path) -> None:
        _, weights, result = _pretrained_accelerator(tmp_path, seed=12)
        weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        payload = _reference_artifact_payload(weights_sha)
        payload["metrics"]["psi_rmse_Wb"] = 0.5  # exceeds the declared tolerance -> strict validation fails
        payload["payload_sha256"] = canonical_artifact_sha256(payload)
        artifact = tmp_path / "reference.json"
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="reference artifact failed strict validation"):
            neural_equilibrium_claim_evidence(
                result,
                weights_path=weights,
                source="documented_public_reference",
                source_id="id",
                reference_artifact_path=artifact,
            )

    def test_assert_admissible_rejects_non_evidence(self) -> None:
        not_evidence: Any = object()
        with pytest.raises(ValueError, match="evidence must be NeuralEquilibriumClaimEvidence"):
            assert_neural_equilibrium_facility_claim_admissible(not_evidence)

    def test_assert_admissible_rejects_unsupported_schema_version(self, tmp_path: Path) -> None:
        _, weights, result = _pretrained_accelerator(tmp_path, seed=13)
        evidence = neural_equilibrium_claim_evidence(
            result, weights_path=weights, source="synthetic_regression_reference", source_id="id"
        )
        downgraded = dataclasses.replace(evidence, schema_version=evidence.schema_version + 1)
        with pytest.raises(ValueError, match="schema_version is unsupported"):
            assert_neural_equilibrium_facility_claim_admissible(downgraded)

    def test_save_claim_evidence_rejects_non_evidence(self, tmp_path: Path) -> None:
        not_evidence: Any = object()
        with pytest.raises(ValueError, match="evidence must be NeuralEquilibriumClaimEvidence"):
            save_neural_equilibrium_claim_evidence(not_evidence, tmp_path / "out.json")


class TestSyntheticPretrainConvenience:
    def test_pretrain_neural_equilibrium_synthetic_entry_point(self, tmp_path: Path) -> None:
        result = pretrain_neural_equilibrium_synthetic(
            n_samples=64,
            save_path=tmp_path / "synthetic.npz",
            grid_shape=(17, 19),
            n_components=6,
            seed=20240531,
        )
        assert isinstance(result, PretrainingResult)
        assert result.n_samples == 64
        assert (tmp_path / "synthetic.npz").exists()

    def test_pretrain_without_save_path_returns_unsaved_result(self) -> None:
        """Omitting save_path pretrains in memory and records an empty weights path."""
        accel = NeuralEquilibriumAccelerator(
            NeuralEqConfig(n_components=6, hidden_sizes=(), n_input_features=12, grid_shape=(17, 19))
        )
        result = accel.pretrain_from_synthetic_equilibria(160, seed=7)
        assert isinstance(result, PretrainingResult)
        assert result.n_samples == 160
        assert result.weights_path == ""
