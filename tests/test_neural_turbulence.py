# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural Turbulence Tests
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from scpn_control.core.neural_turbulence import (
    NeuralTurbulenceClaimEvidence,
    NeuralTransportTrainer,
    QLKNNSurrogate,
    QLKNNTransportModel,
    TrainingDataGenerator,
    TransportInputNormalizer,
    assert_neural_turbulence_quantitative_claim_admissible,
    cross_validate_neural_turbulence,
    neural_turbulence_claim_evidence,
    save_neural_turbulence_claim_evidence,
    _finite_nonnegative,
    _finite_nonnegative_or_none,
    _finite_positive_or_none,
    _finite_scalar,
    _non_empty_text,
    _profile_array,
    _unit_interval,
    _unit_interval_or_none,
)


def test_surrogate_forward_pass():
    model = QLKNNSurrogate(hidden_layers=[32, 16])
    x = np.random.randn(10, 10)
    out = model.forward(x)

    assert out.shape == (10, 3)
    assert np.all(np.isfinite(out))


def test_input_normalization():
    norm = TransportInputNormalizer()
    r = np.linspace(0.1, 2.0, 50)
    Te = 10.0 * (1.0 - (r / 2.0) ** 2)
    Ti = 10.0 * (1.0 - (r / 2.0) ** 2)
    ne = 5.0 * (1.0 - (r / 2.0) ** 2)
    q = 1.0 + 2.0 * (r / 2.0) ** 2

    inputs = norm.from_profiles(Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3, r=r)

    assert inputs.shape == (50, 10)
    # R/L_Te should be positive
    assert np.all(inputs[:, 1] >= 0.0)


def test_input_normalization_collisionality_uses_q_and_temperature_scaling():
    norm = TransportInputNormalizer()
    r = np.linspace(0.2, 2.0, 50)
    Te = np.ones_like(r) * 8.0
    Ti = np.ones_like(r) * 8.0
    ne = np.ones_like(r) * 5.0
    q_low = np.ones_like(r)
    q_high = np.ones_like(r) * 3.0

    nu_low_q = norm.from_profiles(Te, Ti, ne, q_low, R0=6.2, a=2.0, B0=5.3, r=r)[:, 7]
    nu_high_q = norm.from_profiles(Te, Ti, ne, q_high, R0=6.2, a=2.0, B0=5.3, r=r)[:, 7]
    nu_hot = norm.from_profiles(Te * 2.0, Ti, ne, q_low, R0=6.2, a=2.0, B0=5.3, r=r)[:, 7]

    assert np.all(nu_high_q > nu_low_q)
    assert np.all(nu_hot < nu_low_q)


def test_input_normalization_rejects_nonphysical_profile_domains():
    norm = TransportInputNormalizer()
    r = np.linspace(0.1, 2.0, 5)
    Te = np.ones_like(r) * 10.0
    Ti = np.ones_like(r) * 10.0
    ne = np.ones_like(r) * 5.0
    q = np.ones_like(r) * 2.0

    with pytest.raises(ValueError, match="r"):
        norm.from_profiles(Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3, r=r[::-1])
    with pytest.raises(ValueError, match="Te"):
        norm.from_profiles(Te[:-1], Ti, ne, q, R0=6.2, a=2.0, B0=5.3, r=r)
    with pytest.raises(ValueError, match="B0"):
        norm.from_profiles(Te, Ti, ne, q, R0=6.2, a=2.0, B0=0.0, r=r)


def test_analytic_targets_critical_gradient():
    gen = TrainingDataGenerator()

    # inputs = [R_L_Ti, R_L_Te, R_L_ne, q, s_hat, alpha_MHD, Ti_Te, nu_star, Z_eff, eps]
    X_sub = np.array([[1.0, 5.0, 1.0, 2.0, 1.0, 0.1, 1.0, 0.01, 1.5, 0.1]])  # Sub-critical R/L_Ti
    X_super = np.array([[10.0, 5.0, 1.0, 2.0, 1.0, 0.1, 1.0, 0.01, 1.5, 0.1]])  # Super-critical

    y_sub = gen.generate_analytic_targets(X_sub)
    y_super = gen.generate_analytic_targets(X_super)

    assert y_sub[0, 0] == 0.0  # Q_i should be 0 below threshold
    assert y_super[0, 0] > 0.0  # Q_i > 0 above threshold


def test_neural_transport_trainer():
    trainer = NeuralTransportTrainer()

    gen = TrainingDataGenerator()
    X = gen.generate_parameter_scan(200)
    y = gen.generate_analytic_targets(X)

    hist = trainer.train(X, y, epochs=50, lr=1e-3)

    assert len(hist["train_loss"]) == 50
    assert hist["train_loss"][-1] < hist["train_loss"][0]


def test_neural_transport_trainer_skips_clipping_for_small_gradients():
    """Sub-unit weight-gradient norms bypass the clipping rescale during training (branch 530->533)."""
    trainer = NeuralTransportTrainer()
    x_small = np.full((16, 10), 1e-4, dtype=np.float64)
    y_small = np.full((16, 3), 1e-4, dtype=np.float64)
    hist = trainer.train(x_small, y_small, epochs=1, lr=1e-3)
    assert len(hist["train_loss"]) == 1


def test_surrogate_save_load(tmp_path):
    model = QLKNNSurrogate(hidden_layers=[16, 8])
    x = np.random.randn(5, 10)
    out_before = model.forward(x)

    path = str(tmp_path / "weights.npz")
    model.save_weights(path)

    model2 = QLKNNSurrogate(hidden_layers=[16, 8])
    model2.load_weights(path)
    out_after = model2.forward(x)

    np.testing.assert_array_almost_equal(out_before, out_after)


def test_qlknn_transport_model_denormalization():
    model = QLKNNSurrogate()
    t_model = QLKNNTransportModel(model)

    r = np.linspace(0.1, 2.0, 50)
    Te = 10.0 * (1.0 - (r / 2.0) ** 2)
    Ti = 10.0 * (1.0 - (r / 2.0) ** 2)
    ne = 5.0 * (1.0 - (r / 2.0) ** 2)
    q = 1.0 + 2.0 * (r / 2.0) ** 2

    fluxes = t_model.compute_fluxes(Te, Ti, ne, q, R0=6.2, a=2.0, B0=5.3, r=r)

    assert fluxes.Q_i_W_m2.shape == (50,)
    assert fluxes.Q_e_W_m2.shape == (50,)
    assert fluxes.Gamma_e_inv_m2_s.shape == (50,)
    assert np.all(np.isfinite(fluxes.Q_i_W_m2))


def test_activate_relu_and_tanh():
    """Lines 49-53: _activate covers relu and tanh branches."""
    model_relu = QLKNNSurrogate(hidden_layers=[16], activation="relu", pretrained=False)
    x = np.array([[1.0] * 10, [-1.0] * 10])
    out_relu = model_relu.forward(x)
    assert out_relu.shape == (2, 3)
    assert np.all(np.isfinite(out_relu))

    model_tanh = QLKNNSurrogate(hidden_layers=[16], activation="tanh", pretrained=False)
    out_tanh = model_tanh.forward(x)
    assert out_tanh.shape == (2, 3)
    assert np.all(np.isfinite(out_tanh))


def test_activate_deriv_branches():
    """Lines 58-62: _activate_deriv covers relu and tanh branches."""
    model = QLKNNSurrogate(hidden_layers=[16], activation="relu", pretrained=False)
    x = np.array([1.0, -1.0, 0.0])
    d_relu = model._activate_deriv(x)
    assert d_relu[0] == 1.0
    assert d_relu[1] == 0.0

    model_tanh = QLKNNSurrogate(hidden_layers=[16], activation="tanh", pretrained=False)
    d_tanh = model_tanh._activate_deriv(x)
    assert np.all(np.isfinite(d_tanh))


def test_forward_single_sample():
    """Line 106: forward handles 1D input (single sample)."""
    model = QLKNNSurrogate(hidden_layers=[16], pretrained=False)
    x_1d = np.ones(10)
    out = model.forward(x_1d)
    assert out.shape == (1, 3)


def test_trainer_activate_deriv_relu_tanh():
    """Lines 254-258: NeuralTransportTrainer._activate_deriv relu and tanh."""
    trainer = NeuralTransportTrainer()
    x = np.array([1.0, -1.0, 0.0])
    d_relu = trainer._activate_deriv(x, "relu")
    assert d_relu[0] == 1.0
    assert d_relu[1] == 0.0

    d_tanh = trainer._activate_deriv(x, "tanh")
    assert np.all(np.isfinite(d_tanh))

    d_unknown = trainer._activate_deriv(x, "linear")
    assert np.allclose(d_unknown, 1.0)


def _reference_artifact(weights_sha: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "model_id": "neural_turbulence_qlknn_facade",
        "model_version": "test",
        "trained_weights_sha256": weights_sha,
        "reference_dataset_id": "bounded-gk-fixture",
        "reference_artifact_sha256": "e" * 64,
        "executed_at": "2026-05-31T00:00:00Z",
        "reference_url": "https://example.invalid/gk-reference",
        "feature_schema": [
            "R_LTi",
            "R_LTe",
            "R_Ln",
            "q",
            "s_hat",
            "alpha_MHD",
            "Ti_Te",
            "nu_star",
            "Z_eff",
            "epsilon",
        ],
        "units": {
            "Q_i": "gyroBohm",
            "Q_e": "gyroBohm",
            "Gamma_e": "gyroBohm",
            "input_gradients": "dimensionless",
        },
        "reference_sample_count": 96,
        "metrics": {
            "Q_i_rmse_gB": 0.04,
            "Q_e_rmse_gB": 0.03,
            "Gamma_e_rmse_gB": 0.02,
            "flux_relative_mae": 0.05,
            "critical_gradient_accuracy": 0.94,
        },
        "tolerances": {
            "Q_i_rmse_gB": 0.08,
            "Q_e_rmse_gB": 0.08,
            "Gamma_e_rmse_gB": 0.06,
            "flux_relative_mae": 0.10,
            "critical_gradient_accuracy_min": 0.90,
        },
    }


def test_neural_turbulence_claim_evidence_records_local_boundary(tmp_path):
    validation = cross_validate_neural_turbulence(QLKNNSurrogate(hidden_layers=[16], pretrained=True), n_samples=32)

    evidence = neural_turbulence_claim_evidence(
        validation,
        source="synthetic_regression_reference",
        source_id="tests/test_neural_turbulence.py::local_claim_boundary",
    )

    assert isinstance(evidence, NeuralTurbulenceClaimEvidence)
    assert evidence.quantitative_claim_allowed is False
    assert evidence.reference_source == "none"
    assert evidence.local_sample_count == 32
    assert evidence.local_q_i_rmse_gB >= 0.0
    assert 0.0 <= evidence.local_critical_gradient_accuracy <= 1.0
    with pytest.raises(ValueError, match="blocked without matched reference"):
        assert_neural_turbulence_quantitative_claim_admissible(evidence)

    out = tmp_path / "neural_turbulence_claim.json"
    save_neural_turbulence_claim_evidence(evidence, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["claim_status"].startswith("local analytic-target regression evidence only")


def test_neural_turbulence_reference_admission_requires_matching_weights(tmp_path):
    model = QLKNNSurrogate(hidden_layers=[16, 8], pretrained=True)
    weights = tmp_path / "weights.npz"
    model.save_weights(str(weights))
    validation = cross_validate_neural_turbulence(model, n_samples=32)
    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    artifact = tmp_path / "reference.json"
    artifact.write_text(json.dumps(_reference_artifact(weights_sha)), encoding="utf-8")

    evidence = neural_turbulence_claim_evidence(
        validation,
        source="documented_public_reference",
        source_id="tests/test_neural_turbulence.py::reference_claim_boundary",
        weights_path=weights,
        reference_artifact_path=artifact,
    )

    assert evidence.quantitative_claim_allowed is True
    assert evidence.reference_source == "documented_public_reference"
    assert evidence.reference_sample_count == 96
    assert evidence.weights_sha256 == weights_sha
    assert evidence.q_i_rmse_gB == pytest.approx(0.04)
    assert evidence.critical_gradient_accuracy == pytest.approx(0.94)
    assert assert_neural_turbulence_quantitative_claim_admissible(evidence) is evidence

    artifact.write_text(json.dumps(_reference_artifact("f" * 64)), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match supplied weights"):
        neural_turbulence_claim_evidence(
            validation,
            source="documented_public_reference",
            source_id="tests/test_neural_turbulence.py::reference_claim_boundary",
            weights_path=weights,
            reference_artifact_path=artifact,
        )


def test_reference_admission_requires_weights_path(tmp_path):
    validation = cross_validate_neural_turbulence(QLKNNSurrogate(hidden_layers=[16], pretrained=True), n_samples=32)
    artifact = tmp_path / "reference.json"
    artifact.write_text(json.dumps(_reference_artifact("a" * 64)), encoding="utf-8")
    with pytest.raises(ValueError, match="requires the exact neural-turbulence weights_path"):
        neural_turbulence_claim_evidence(
            validation,
            source="documented_public_reference",
            source_id="tests/test_neural_turbulence.py::missing_weights",
            reference_artifact_path=artifact,
        )


def test_reference_admission_rejects_artifact_that_fails_strict_validation(tmp_path):
    model = QLKNNSurrogate(hidden_layers=[16, 8], pretrained=True)
    weights = tmp_path / "weights.npz"
    model.save_weights(str(weights))
    validation = cross_validate_neural_turbulence(model, n_samples=32)
    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    bad = _reference_artifact(weights_sha)
    bad["metrics"]["Q_i_rmse_gB"] = 5.0  # exceeds the declared tolerance → validator fails
    artifact = tmp_path / "reference.json"
    artifact.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="failed strict validation"):
        neural_turbulence_claim_evidence(
            validation,
            source="documented_public_reference",
            source_id="tests/test_neural_turbulence.py::strict_fail",
            weights_path=weights,
            reference_artifact_path=artifact,
        )


def test_claim_evidence_requires_positive_local_sample_count():
    with pytest.raises(ValueError, match="positive n_samples"):
        neural_turbulence_claim_evidence(
            {"n_samples": 0},
            source="synthetic_regression_reference",
            source_id="tests/test_neural_turbulence.py::no_samples",
        )


def test_claim_evidence_rejects_missing_weights_file(tmp_path):
    validation = cross_validate_neural_turbulence(QLKNNSurrogate(hidden_layers=[16], pretrained=True), n_samples=32)
    with pytest.raises(FileNotFoundError, match="weights not found"):
        neural_turbulence_claim_evidence(
            validation,
            source="synthetic_regression_reference",
            source_id="tests/test_neural_turbulence.py::no_weights_file",
            weights_path=tmp_path / "absent.npz",
        )


def test_claim_evidence_rejects_blank_identity_text(tmp_path):
    validation = cross_validate_neural_turbulence(QLKNNSurrogate(hidden_layers=[16], pretrained=True), n_samples=32)
    with pytest.raises(ValueError, match="must be a non-empty string"):
        neural_turbulence_claim_evidence(
            validation,
            source="synthetic_regression_reference",
            source_id="   ",
        )


def test_assert_admissible_rejects_wrong_type_and_schema():
    with pytest.raises(ValueError, match="must be NeuralTurbulenceClaimEvidence"):
        assert_neural_turbulence_quantitative_claim_admissible({"not": "evidence"})


def test_save_claim_evidence_rejects_wrong_type(tmp_path):
    with pytest.raises(ValueError, match="must be NeuralTurbulenceClaimEvidence"):
        save_neural_turbulence_claim_evidence({"not": "evidence"}, tmp_path / "x.json")


# ── Numeric / profile validation helpers ─────────────────────────────


@pytest.mark.parametrize("value", ["", "   ", 5, None])
def test_non_empty_text_rejects_blank_or_non_string(value):
    with pytest.raises(ValueError, match="must be a non-empty string"):
        _non_empty_text("field", value)


@pytest.mark.parametrize("value", [True, "x", [1.0]])
def test_finite_nonnegative_or_none_rejects_non_numeric(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _finite_nonnegative_or_none("metric", value)


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_finite_nonnegative_or_none_rejects_negative_or_non_finite(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _finite_nonnegative_or_none("metric", value)


def test_finite_nonnegative_or_none_passes_through_none():
    assert _finite_nonnegative_or_none("metric", None) is None


@pytest.mark.parametrize("value", [True, "x"])
def test_finite_positive_or_none_rejects_non_numeric(value):
    with pytest.raises(ValueError, match="finite and positive"):
        _finite_positive_or_none("metric", value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf")])
def test_finite_positive_or_none_rejects_non_positive_or_non_finite(value):
    with pytest.raises(ValueError, match="finite and positive"):
        _finite_positive_or_none("metric", value)


@pytest.mark.parametrize("value", [True, "x"])
def test_unit_interval_or_none_rejects_non_numeric(value):
    with pytest.raises(ValueError, match=r"finite in \[0, 1\]"):
        _unit_interval_or_none("score", value)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_unit_interval_or_none_rejects_out_of_range(value):
    with pytest.raises(ValueError, match=r"finite in \[0, 1\]"):
        _unit_interval_or_none("score", value)


def test_finite_nonnegative_requires_present_value():
    with pytest.raises(ValueError, match="must be present"):
        _finite_nonnegative("metric", None)


def test_unit_interval_requires_present_value():
    with pytest.raises(ValueError, match="must be present"):
        _unit_interval("score", None)


def test_finite_scalar_rejects_non_finite():
    with pytest.raises(ValueError, match="must be finite"):
        _finite_scalar("scalar", float("inf"))


def test_profile_array_rejects_wrong_dimensions_and_size():
    with pytest.raises(ValueError, match="one-dimensional profile with at least two points"):
        _profile_array("p", [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="one-dimensional profile with at least two points"):
        _profile_array("p", [1.0])


def test_profile_array_rejects_non_finite_values():
    with pytest.raises(ValueError, match="only finite values"):
        _profile_array("p", [1.0, np.inf])


def test_profile_array_positive_with_boundary_zero_rejects_interior_and_negative_edge():
    with pytest.raises(ValueError, match="positive in the interior and non-negative at the boundary"):
        _profile_array("p", [1.0, 0.0, 1.0], positive=True, allow_last_zero=True)
    with pytest.raises(ValueError, match="positive in the interior and non-negative at the boundary"):
        _profile_array("p", [1.0, 2.0, -1.0], positive=True, allow_last_zero=True)


def test_profile_array_strictly_positive_rejects_non_positive():
    with pytest.raises(ValueError, match="positive everywhere"):
        _profile_array("p", [1.0, 0.0], positive=True)


def test_from_profiles_rejects_minor_radius_not_smaller_than_major():
    norm = TransportInputNormalizer()
    r = np.linspace(0.1, 2.0, 5)
    Te = np.ones_like(r) * 10.0
    Ti = np.ones_like(r) * 10.0
    ne = np.ones_like(r) * 5.0
    q = np.ones_like(r) * 2.0
    with pytest.raises(ValueError, match="a must be smaller than R0"):
        norm.from_profiles(Te, Ti, ne, q, R0=2.0, a=2.0, B0=5.3, r=r)


def test_activation_unknown_passes_through_for_forward_and_derivative():
    model = QLKNNSurrogate(hidden_layers=[8], activation="linear", pretrained=False)
    x = np.ones((3, 10))
    out = model.forward(x)
    assert out.shape == (3, 3)
    deriv = model._activate_deriv(np.array([1.0, -1.0, 0.0]))
    assert np.allclose(deriv, 1.0)


def test_cross_validate_requires_minimum_sample_count():
    with pytest.raises(ValueError, match="at least 8"):
        cross_validate_neural_turbulence(n_samples=4)


def test_cross_validate_rejects_non_finite_surrogate_output():
    class _NonFiniteSurrogate:
        def forward(self, inputs):
            return np.full((inputs.shape[0], 3), np.nan)

    with pytest.raises(RuntimeError, match="invalid local benchmark outputs"):
        cross_validate_neural_turbulence(_NonFiniteSurrogate(), n_samples=16)


def test_assert_admissible_rejects_unsupported_schema_version():
    validation = cross_validate_neural_turbulence(QLKNNSurrogate(hidden_layers=[16], pretrained=True), n_samples=32)
    evidence = neural_turbulence_claim_evidence(
        validation,
        source="synthetic_regression_reference",
        source_id="tests/test_neural_turbulence.py::schema_guard",
    )
    tampered = replace(evidence, schema_version="neural-turbulence-claim.v0")
    with pytest.raises(ValueError, match="schema_version is unsupported"):
        assert_neural_turbulence_quantitative_claim_admissible(tampered)
