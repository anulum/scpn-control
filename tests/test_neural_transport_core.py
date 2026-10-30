# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural Transport Core Tests
"""Tests for MLP internals, weight loading, and profile prediction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from scpn_control._typing import FloatArray
from scpn_control.core.neural_transport import (
    MLPWeights,
    NeuralTransportClaimEvidence,
    NeuralTransportClosureResult,
    NeuralTransportModel,
    TransportFluxes,
    TransportInputs,
    _mlp_forward,
    _relu,
    _softplus,
    assert_neural_transport_quantitative_claim_admissible,
    critical_gradient_model,
    cross_validate_neural_transport,
    neural_transport_claim_evidence,
    neural_transport_closure_profiles,
    save_neural_transport_claim_evidence,
)
from validation.validate_neural_transport_reference import canonical_artifact_sha256

Profiles = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]

# ── Activation functions ────────────────────────────────────────────


class TestRelu:
    """ReLU keeps positive inputs and zeroes negative ones."""

    def test_positive_passthrough(self) -> None:
        """Positive inputs pass through unchanged."""
        assert np.allclose(_relu(np.array([1.0, 2.0])), [1.0, 2.0])

    def test_negative_zeroed(self) -> None:
        """Negative inputs become zero."""
        assert np.allclose(_relu(np.array([-1.0, -0.5])), [0.0, 0.0])


class TestSoftplus:
    """Softplus is positive, above the identity for positive input, and finite far below zero."""

    def test_positive_input(self) -> None:
        """Softplus of a positive input exceeds the input."""
        out = _softplus(np.array([5.0]))
        assert out[0] > 5.0  # softplus(x) > x for x > 0

    def test_nonnegative(self) -> None:
        """Softplus is non-negative across signs."""
        out = _softplus(np.array([-10.0, 0.0, 10.0]))
        assert np.all(out >= 0.0)

    def test_large_negative_clipped(self) -> None:
        """A very negative input stays finite."""
        out = _softplus(np.array([-100.0]))
        assert np.isfinite(out[0])


# ── MLP forward pass ───────────────────────────────────────────────


def _make_weights(h1: int = 16, h2: int = 8) -> MLPWeights:
    """Build a small random MLP weight set for testing."""
    rng = np.random.default_rng(0)
    return MLPWeights(
        w1=rng.standard_normal((10, h1)),
        b1=np.zeros(h1),
        w2=rng.standard_normal((h1, h2)),
        b2=np.zeros(h2),
        w3=rng.standard_normal((h2, 3)),
        b3=np.zeros(3),
        input_mean=np.zeros(10),
        input_std=np.ones(10),
        output_scale=np.ones(3),
    )


def _save_weights(path: Path, *, version: int = 1) -> None:
    """Save the weights of :func:`_make_weights` as a format-``version`` archive."""
    w = _make_weights()
    np.savez(
        path,
        w1=w.w1,
        b1=w.b1,
        w2=w.w2,
        b2=w.b2,
        w3=w.w3,
        b3=w.b3,
        input_mean=w.input_mean,
        input_std=w.input_std,
        output_scale=w.output_scale,
        version=np.array(version),
    )


def _make_profiles(n: int = 32) -> Profiles:
    """Return smooth, monotone rho, Te, Ti, ne, q and shear profiles on ``n`` points."""
    rho = np.linspace(0.01, 0.99, n)
    te = 10.0 * (1 - rho**2)
    ti = 9.0 * (1 - rho**2)
    ne = 8.0 * (1 - 0.5 * rho**2)
    q = 1.0 + 2.0 * rho**2
    s_hat = 0.5 + 1.5 * rho
    return rho, te, ti, ne, q, s_hat


class TestMLPForward:
    """The three-layer MLP forward pass: shapes, sign, output scale and input normalisation."""

    def test_single_input_shape(self) -> None:
        """One input vector gives three outputs."""
        w = _make_weights()
        out = _mlp_forward(np.zeros(10), w)
        assert out.shape == (3,)

    def test_batch_input_shape(self) -> None:
        """A batch of five inputs gives a 5 x 3 output."""
        w = _make_weights()
        out = _mlp_forward(np.zeros((5, 10)), w)
        assert out.shape == (5, 3)

    def test_output_nonnegative(self) -> None:
        """The softplus output layer keeps every output non-negative."""
        w = _make_weights()
        out = _mlp_forward(np.random.default_rng(1).standard_normal(10), w)
        assert np.all(out >= 0.0)  # softplus ensures this

    def test_output_scale_applied(self) -> None:
        """The output scale multiplies each channel."""
        w = _make_weights()
        w_scaled = _make_weights()
        w_scaled.output_scale = np.array([2.0, 3.0, 4.0])
        out_base = _mlp_forward(np.zeros(10), w)
        out_scaled = _mlp_forward(np.zeros(10), w_scaled)
        assert np.allclose(out_scaled, out_base * [2.0, 3.0, 4.0])

    def test_input_normalisation(self) -> None:
        """Inputs are normalised by mean and standard deviation before the first layer."""
        w = _make_weights()
        w.input_mean = np.ones(10) * 5.0
        w.input_std = np.ones(10) * 2.0
        out = _mlp_forward(np.ones(10) * 5.0, w)
        out_zero = _mlp_forward(np.zeros(10), w)
        # (5 - 5) / 2 = 0 and (0 - 5) / 2 = -2.5 normalise differently.
        assert not np.allclose(out, out_zero)


# ── Weight loading via .npz ────────────────────────────────────────


class TestNeuralTransportModel:
    """Weight loading from ``.npz`` archives and the explicit fallback policy."""

    def test_fallback_mode(self) -> None:
        """Without discovery or a path, the model is not neural."""
        model = NeuralTransportModel(auto_discover=False)
        assert not model.is_neural

    def test_missing_path_falls_back(self) -> None:
        """A missing explicit weight file fails closed by default."""
        with pytest.raises(FileNotFoundError, match="allow_weight_load_fallback=True"):
            NeuralTransportModel("/nonexistent/path.npz")

    def test_missing_path_legacy_fallback_opt_in(self) -> None:
        """With both fallback flags, a missing weight file degrades to the analytic model."""
        model = NeuralTransportModel(
            "/nonexistent/path.npz",
            allow_weight_load_fallback=True,
            allow_legacy_weight_load_fallback=True,
        )
        assert not model.is_neural

    def test_load_valid_weights(self, tmp_path: Path) -> None:
        """A valid archive loads and records its checksum."""
        weights_path = tmp_path / "weights.npz"
        _save_weights(weights_path)
        model = NeuralTransportModel(weights_path)
        assert model.is_neural
        assert model.weights_checksum is not None

    def test_predict_neural(self, tmp_path: Path) -> None:
        """Loaded weights give non-negative fluxes."""
        weights_path = tmp_path / "weights.npz"
        _save_weights(weights_path)
        model = NeuralTransportModel(weights_path)
        fluxes = model.predict(TransportInputs(grad_ti=8.0))
        assert isinstance(fluxes, TransportFluxes)
        assert fluxes.chi_e >= 0
        assert fluxes.chi_i >= 0

    def test_predict_fallback(self) -> None:
        """Without weights, prediction equals the critical-gradient model."""
        model = NeuralTransportModel(auto_discover=False)
        fluxes = model.predict(TransportInputs(grad_ti=8.0))
        expected = critical_gradient_model(TransportInputs(grad_ti=8.0))
        assert fluxes.chi_i == expected.chi_i

    def test_wrong_version_falls_back(self, tmp_path: Path) -> None:
        """An unsupported archive version fails closed by default."""
        weights_path = tmp_path / "weights.npz"
        _save_weights(weights_path, version=99)
        with pytest.raises(
            RuntimeError,
            match="Failed to load explicit neural transport weights.*allow_weight_load_fallback=True",
        ):
            NeuralTransportModel(weights_path)

    def test_wrong_version_legacy_fallback_opt_in(self, tmp_path: Path) -> None:
        """With both fallback flags, an unsupported version degrades to the analytic model."""
        weights_path = tmp_path / "weights.npz"
        _save_weights(weights_path, version=99)
        model = NeuralTransportModel(
            weights_path,
            allow_weight_load_fallback=True,
            allow_legacy_weight_load_fallback=True,
        )
        assert not model.is_neural

    def test_missing_key_falls_back(self, tmp_path: Path) -> None:
        """An archive missing weight arrays fails closed by default."""
        weights_path = tmp_path / "weights.npz"
        np.savez(weights_path, w1=np.zeros((10, 16)))
        with pytest.raises(
            RuntimeError,
            match="Failed to load explicit neural transport weights.*allow_weight_load_fallback=True",
        ):
            NeuralTransportModel(weights_path)

    def test_missing_key_legacy_fallback_opt_in(self, tmp_path: Path) -> None:
        """With both fallback flags, missing arrays degrade to the analytic model."""
        weights_path = tmp_path / "weights.npz"
        np.savez(weights_path, w1=np.zeros((10, 16)))
        model = NeuralTransportModel(
            weights_path,
            allow_weight_load_fallback=True,
            allow_legacy_weight_load_fallback=True,
        )
        assert not model.is_neural

    def test_weight_load_legacy_fallback_requires_explicit_opt_in(self) -> None:
        """The fallback flag alone, without the legacy flag, is refused."""
        with pytest.raises(ValueError, match="allow_legacy_weight_load_fallback=True"):
            NeuralTransportModel(
                "/nonexistent/path.npz",
                allow_weight_load_fallback=True,
                allow_legacy_weight_load_fallback=False,
            )


# ── Profile prediction ─────────────────────────────────────────────


class TestPredictProfile:
    """Profile-wide prediction with and without loaded weights."""

    def test_fallback_profile_shape(self) -> None:
        """Analytic profile prediction keeps the radial shape."""
        model = NeuralTransportModel(auto_discover=False)
        rho, te, ti, ne, q, s_hat = _make_profiles()
        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        assert chi_e.shape == rho.shape
        assert chi_i.shape == rho.shape
        assert d_e.shape == rho.shape

    def test_fallback_profile_nonneg(self) -> None:
        """Analytic profile prediction is non-negative."""
        model = NeuralTransportModel(auto_discover=False)
        rho, te, ti, ne, q, s_hat = _make_profiles()
        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        assert np.all(chi_e >= 0)
        assert np.all(chi_i >= 0)
        assert np.all(d_e >= 0)

    def test_neural_profile_shape(self, tmp_path: Path) -> None:
        """Neural profile prediction keeps the radial shape."""
        weights_path = tmp_path / "weights.npz"
        _save_weights(weights_path)
        model = NeuralTransportModel(weights_path)
        rho, te, ti, ne, q, s_hat = _make_profiles()
        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        assert chi_e.shape == rho.shape
        assert chi_i.shape == rho.shape

    def test_d_e_uses_profile_density_gradient_and_shear(self) -> None:
        """D_e/chi_e varies with the profile instead of a fixed one third, within bounds."""
        model = NeuralTransportModel(auto_discover=False)
        rho, te, ti, ne, q, s_hat = _make_profiles()
        chi_e, _, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        mask = chi_e > 0
        if mask.any():
            ratio = d_e[mask] / chi_e[mask]
            assert not np.allclose(ratio, 1.0 / 3.0, atol=1e-10)
            assert np.all((0.05 <= ratio) & (ratio <= 0.65))


class TestNeuralTransportClosureProfiles:
    """The closure wrapper: provenance, neural requirement and profile validation."""

    def test_fallback_closure_matches_profile_prediction_with_provenance(self) -> None:
        """The analytic closure equals profile prediction and records its provenance."""
        model = NeuralTransportModel(auto_discover=False)
        rho, te, ti, ne, q, s_hat = _make_profiles()

        closure = neural_transport_closure_profiles(rho, te, ti, ne, q, s_hat, model=model)
        expected_chi_e, expected_chi_i, expected_d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)

        assert isinstance(closure, NeuralTransportClosureResult)
        assert closure.source == "analytic_fallback"
        assert closure.weights_checksum is None
        assert np.allclose(closure.chi_e, expected_chi_e)
        assert np.allclose(closure.chi_i, expected_chi_i)
        assert np.allclose(closure.d_e, expected_d_e)
        assert closure.channel_weights.shape == (3, rho.size)
        assert np.all(np.isfinite(closure.channel_weights))
        assert np.allclose(closure.channel_weights.sum(axis=0), np.ones_like(rho))

    def test_closure_requires_neural_weights_unless_degraded_mode_is_explicit(self) -> None:
        """Requiring the neural model without weights fails closed."""
        model = NeuralTransportModel(auto_discover=False)
        rho, te, ti, ne, q, s_hat = _make_profiles()

        with pytest.raises(RuntimeError, match="requires loaded neural transport weights"):
            neural_transport_closure_profiles(rho, te, ti, ne, q, s_hat, model=model, require_neural=True)

    def test_closure_allows_explicit_degraded_mode_for_missing_neural_weights(self) -> None:
        """Explicit degraded mode returns the analytic closure."""
        model = NeuralTransportModel(auto_discover=False)
        rho, te, ti, ne, q, s_hat = _make_profiles()

        closure = neural_transport_closure_profiles(
            rho,
            te,
            ti,
            ne,
            q,
            s_hat,
            model=model,
            require_neural=True,
            allow_fallback=True,
            allow_legacy_fallback=True,
        )

        assert closure.source == "analytic_fallback"
        assert np.all(closure.chi_e >= 0.0)
        assert np.all(closure.chi_i >= 0.0)
        assert np.all(closure.d_e >= 0.0)

    def test_neural_closure_reports_weight_checksum(self, tmp_path: Path) -> None:
        """A neural closure reports the checksum of its weights."""
        weights_path = tmp_path / "weights.npz"
        _save_weights(weights_path)
        model = NeuralTransportModel(weights_path)
        rho, te, ti, ne, q, s_hat = _make_profiles()

        closure = neural_transport_closure_profiles(rho, te, ti, ne, q, s_hat, model=model, require_neural=True)

        assert closure.source == "neural"
        assert closure.weights_checksum == model.weights_checksum
        assert np.all(closure.chi_e >= 0.0)
        assert np.all(closure.chi_i >= 0.0)
        assert np.all(closure.d_e >= 0.0)

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (lambda rho, te, ti, ne, q, s_hat: (rho[::-1], te, ti, ne, q, s_hat), "rho must be strictly increasing"),
            (lambda rho, te, ti, ne, q, s_hat: (rho, -te, ti, ne, q, s_hat), "te must be strictly positive"),
            (lambda rho, te, ti, ne, q, s_hat: (rho, te, ti[:-1], ne, q, s_hat), "same shape"),
        ],
    )
    def test_closure_rejects_invalid_profiles(self, mutate: Callable[..., Profiles], match: str) -> None:
        """Decreasing rho, negative Te and mismatched shapes are refused."""
        profiles = mutate(*_make_profiles())

        with pytest.raises(ValueError, match=match):
            neural_transport_closure_profiles(*profiles, model=NeuralTransportModel(auto_discover=False))


class TestNeuralTransportClaimEvidence:
    """Claim evidence: the local fallback boundary and reference admission."""

    def _reference_artifact(self, weights_sha: str) -> dict[str, object]:
        """Return a self-consistent reference artifact bound to ``weights_sha``."""
        payload: dict[str, object] = {
            "schema_version": "scpn-control.neural-transport-reference.v1",
            "source": "documented_public_reference",
            "model_id": "neural_transport_qlknn_facade",
            "model_version": "test",
            "trained_weights_sha256": weights_sha,
            "reference_dataset_id": "bounded-qlknn-fixture",
            "reference_artifact_uri": "bounded-qlknn-fixture/reference_targets.npz",
            "prediction_artifact_uri": "bounded-qlknn-fixture/scpn_predictions.npz",
            "reference_artifact_sha256": "c" * 64,
            "prediction_artifact_sha256": "d" * 64,
            "executed_at": "2026-05-31T00:00:00Z",
            "reference_url": "https://example.invalid/qlknn-reference",
            "feature_schema": [
                "R_LTi",
                "R_LTe",
                "R_Ln",
                "q",
                "s_hat",
                "alpha",
                "Ti_Te",
                "Zeff",
                "collisionality",
                "beta_e",
            ],
            "target_schema": ["chi_i", "chi_e", "D_e", "unstable_branch"],
            "units": {
                "chi_i": "m^2/s",
                "chi_e": "m^2/s",
                "D_e": "m^2/s",
                "input_gradients": "dimensionless",
            },
            "reference_sample_count": 64,
            "metrics": {
                "chi_i_rmse_m2_s": 0.02,
                "chi_e_rmse_m2_s": 0.03,
                "D_e_rmse_m2_s": 0.01,
                "chi_i_relative_mae": 0.04,
                "unstable_branch_accuracy": 0.96,
            },
            "tolerances": {
                "chi_i_rmse_m2_s": 0.05,
                "chi_e_rmse_m2_s": 0.05,
                "D_e_rmse_m2_s": 0.03,
                "chi_i_relative_mae": 0.08,
                "unstable_branch_accuracy_min": 0.90,
            },
        }
        payload["payload_sha256"] = canonical_artifact_sha256(payload)
        return payload

    def test_claim_evidence_records_local_fallback_boundary(self, tmp_path: Path) -> None:
        """Fallback evidence is local only, blocks quantitative claims and saves its status."""
        result = cross_validate_neural_transport(NeuralTransportModel(auto_discover=False))

        evidence = neural_transport_claim_evidence(
            result,
            source="synthetic_regression_reference",
            source_id="tests/test_neural_transport_core.py::fallback_claim_boundary",
        )

        assert isinstance(evidence, NeuralTransportClaimEvidence)
        assert evidence.quantitative_claim_allowed is False
        assert evidence.surrogate_mode == "analytic_fallback"
        assert evidence.reference_source == "none"
        assert evidence.local_case_count == result["n_cases"]
        assert evidence.local_channel_agreement == pytest.approx(1.0)
        assert evidence.local_max_abs_error == pytest.approx(0.0)
        assert evidence.local_per_channel_relative_rmse == pytest.approx((0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="blocked without matched reference"):
            assert_neural_transport_quantitative_claim_admissible(evidence)

        out = tmp_path / "neural_transport_claim.json"
        save_neural_transport_claim_evidence(evidence, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["claim_status"].startswith("local surrogate regression evidence only")

    def test_reference_admission_requires_matching_weight_checksum(self, tmp_path: Path) -> None:
        """A matching reference admits the claim; a mismatched weight checksum is refused."""
        weights = tmp_path / "weights.npz"
        _save_weights(weights)
        model = NeuralTransportModel(weights)
        result = cross_validate_neural_transport(model)
        weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        artifact = tmp_path / "reference.json"
        artifact.write_text(json.dumps(self._reference_artifact(weights_sha)), encoding="utf-8")

        evidence = neural_transport_claim_evidence(
            result,
            source="documented_public_reference",
            source_id="tests/test_neural_transport_core.py::reference_claim_boundary",
            weights_path=weights,
            reference_artifact_path=artifact,
        )

        assert evidence.quantitative_claim_allowed is True
        assert evidence.reference_source == "documented_public_reference"
        assert evidence.reference_sample_count == 64
        assert evidence.weights_sha256 == weights_sha
        assert evidence.chi_i_rmse_m2_s == pytest.approx(0.02)
        assert evidence.unstable_branch_accuracy == pytest.approx(0.96)
        assert assert_neural_transport_quantitative_claim_admissible(evidence) is evidence

        bad_payload = self._reference_artifact("d" * 64)
        artifact.write_text(json.dumps(bad_payload), encoding="utf-8")
        with pytest.raises(ValueError, match="does not match supplied weights"):
            neural_transport_claim_evidence(
                result,
                source="documented_public_reference",
                source_id="tests/test_neural_transport_core.py::reference_claim_boundary",
                weights_path=weights,
                reference_artifact_path=artifact,
            )
