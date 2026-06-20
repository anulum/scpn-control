# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural transport tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

"""Tests for scpn_control.core.neural_transport."""

import json
from pathlib import Path
from typing import Any

import numpy as np

import pytest

from scpn_control.core.neural_transport import (
    NeuralTransportModel,
    TransportInputs,
    critical_gradient_model,
)


def _valid_weight_arrays() -> dict[str, np.ndarray]:
    return {
        "w1": np.full((10, 4), 0.1),
        "b1": np.zeros(4),
        "w2": np.full((4, 4), 0.1),
        "b2": np.zeros(4),
        "w3": np.full((4, 3), 0.1),
        "b3": np.zeros(3),
        "input_mean": np.zeros(10),
        "input_std": np.ones(10),
        "output_scale": np.ones(3),
    }


def _write_weights(path: Path, **overrides: np.ndarray) -> Path:
    arrays = _valid_weight_arrays()
    arrays.update(overrides)
    np.savez(path, **arrays)
    return path


def _write_reference_artifact(path: Path, weights_sha256: str) -> Path:
    from validation.validate_neural_transport_reference import canonical_artifact_sha256

    payload: dict[str, Any] = {
        "schema_version": "scpn-control.neural-transport-reference.v1",
        "source": "documented_public_reference",
        "model_id": "neural_transport_qlknn_facade",
        "model_version": "1.0.0",
        "trained_weights_sha256": weights_sha256,
        "reference_dataset_id": "qlknn-10d-ref-001",
        "reference_artifact_uri": "https://example.org/ref.json",
        "prediction_artifact_uri": "https://example.org/pred.json",
        "reference_artifact_sha256": "a" * 64,
        "prediction_artifact_sha256": "b" * 64,
        "executed_at": "2026-06-20T00:00:00Z",
        "reference_url": "https://example.org/reference-paper",
        "target_schema": ["chi_i", "chi_e", "D_e", "unstable_branch"],
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
        "units": {
            "chi_i": "m^2/s",
            "chi_e": "m^2/s",
            "D_e": "m^2/s",
            "input_gradients": "dimensionless",
        },
        "reference_sample_count": 128,
        "metrics": {
            "chi_i_rmse_m2_s": 0.1,
            "chi_e_rmse_m2_s": 0.1,
            "D_e_rmse_m2_s": 0.05,
            "chi_i_relative_mae": 0.1,
            "unstable_branch_accuracy": 0.95,
        },
        "tolerances": {
            "chi_i_rmse_m2_s": 0.5,
            "chi_e_rmse_m2_s": 0.5,
            "D_e_rmse_m2_s": 0.5,
            "chi_i_relative_mae": 0.5,
            "unstable_branch_accuracy_min": 0.8,
        },
    }
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _claim_validation_result() -> dict[str, Any]:
    return {
        "mode": "neural",
        "n_cases": 7,
        "max_abs_error": 0.2,
        "channel_agreement": 0.9,
        "per_channel_relative_rmse": (0.1, 0.1, 0.1),
        "profile_per_channel_relative_rmse": (0.1, 0.1, 0.1),
        "weights_checksum": "deadbeefdeadbeef",
    }


class TestTransportInputs:
    def test_defaults(self):
        inp = TransportInputs()
        assert inp.rho == 0.5
        assert inp.te_kev == 5.0
        assert inp.q == 1.5

    def test_custom(self):
        inp = TransportInputs(rho=0.3, te_kev=10.0, grad_ti=8.0)
        assert inp.rho == 0.3
        assert inp.grad_ti == 8.0


class TestCriticalGradientModel:
    def test_below_threshold_stable(self):
        inp = TransportInputs(grad_ti=2.0, grad_te=2.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_i == 0.0
        assert fluxes.chi_e == 0.0
        assert fluxes.channel == "stable"

    def test_itg_dominant(self):
        inp = TransportInputs(grad_ti=8.0, grad_te=2.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_i > 0.0
        assert fluxes.chi_e == 0.0
        assert fluxes.channel == "ITG"

    def test_tem_dominant(self):
        inp = TransportInputs(grad_ti=2.0, grad_te=9.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_e > 0.0
        assert fluxes.chi_i == 0.0
        assert fluxes.channel == "TEM"

    def test_particle_diffusivity_uses_density_gradient_and_shear(self):
        inp = TransportInputs(grad_ti=2.0, grad_te=9.0)
        fluxes = critical_gradient_model(inp)
        steeper_density = critical_gradient_model(TransportInputs(grad_ti=2.0, grad_te=9.0, grad_ne=5.0))
        higher_shear = critical_gradient_model(TransportInputs(grad_ti=2.0, grad_te=9.0, s_hat=2.0))

        fixed_ratio_alias = 1.0 / 3.0
        assert fluxes.d_e != pytest.approx(fluxes.chi_e * fixed_ratio_alias)
        assert steeper_density.d_e > fluxes.d_e
        assert higher_shear.d_e > fluxes.d_e

    def test_stiffness_exponent(self):
        inp = TransportInputs(grad_ti=5.0, grad_te=2.0)
        fluxes = critical_gradient_model(inp)
        # chi_i = 1.0 * (5.0 - 4.0)^2.0 = 1.0
        assert fluxes.chi_i == pytest.approx(1.0)

    def test_both_channels_itg_wins(self):
        inp = TransportInputs(grad_ti=8.0, grad_te=7.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_i > fluxes.chi_e
        assert fluxes.channel == "ITG"

    def test_profile_fallback_particle_diffusivity_matches_local_model(self):
        rho = np.linspace(0.2, 0.8, 7)
        te = np.linspace(7.0, 4.0, 7)
        ti = np.linspace(5.0, 4.5, 7)
        ne = np.linspace(8.0, 3.0, 7)
        q_profile = 1.0 + rho
        s_hat_profile = np.linspace(0.2, 1.4, 7)

        model = NeuralTransportModel(auto_discover=False)
        chi_e, _, d_e = model.predict_profile(rho, te, ti, ne, q_profile, s_hat_profile, r_major=6.2)

        assert np.any(chi_e > 0.0)
        assert not np.allclose(d_e[chi_e > 0.0], chi_e[chi_e > 0.0] / 3.0)


def test_neural_transport_explicit_weights_fail_closed_without_legacy_fallback(tmp_path):
    from scpn_control.core.neural_transport import NeuralTransportModel

    missing = tmp_path / "missing_weights.npz"

    with pytest.raises(FileNotFoundError, match="weights not found"):
        NeuralTransportModel(weights_path=missing)

    model = NeuralTransportModel(
        weights_path=missing,
        allow_weight_load_fallback=True,
        allow_legacy_weight_load_fallback=True,
    )
    fluxes = model.predict(TransportInputs(grad_ti=8.0, grad_te=2.0))
    assert model.is_neural is False
    assert fluxes.channel == "ITG"


def test_neural_transport_closure_profiles_validate_radial_contracts():
    from scpn_control.core.neural_transport import neural_transport_closure_profiles

    rho = np.array([0.0, 0.5, 0.25])
    profile = np.array([6.0, 5.0, 4.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        neural_transport_closure_profiles(rho, profile, profile, profile, profile, profile)

    with pytest.raises(ValueError, match="strictly positive"):
        neural_transport_closure_profiles(
            np.array([0.0, 0.5, 1.0]),
            np.array([6.0, 0.0, 4.0]),
            profile,
            profile,
            profile,
            profile,
        )


def test_neural_transport_claim_evidence_blocks_quantitative_claim_without_reference():
    from scpn_control.core.neural_transport import (
        assert_neural_transport_quantitative_claim_admissible,
        neural_transport_claim_evidence,
    )

    validation_result = {
        "mode": "analytic_fallback",
        "n_cases": 2,
        "max_abs_error": 0.0,
        "channel_agreement": 1.0,
        "per_channel_relative_rmse": (0.0, 0.0, 0.0),
        "profile_per_channel_relative_rmse": (0.0, 0.0, 0.0),
    }

    evidence = neural_transport_claim_evidence(
        validation_result,
        source="local_regression_reference",
        source_id="tests/test_neural_transport.py::analytic_fallback",
    )

    assert evidence.quantitative_claim_allowed is False
    assert evidence.feature_schema[0] == "R_LTi"
    with pytest.raises(ValueError, match="blocked without matched reference"):
        assert_neural_transport_quantitative_claim_admissible(evidence)


class TestScalarAndTupleValidators:
    def test_non_empty_text_rejects_blank(self) -> None:
        from scpn_control.core.neural_transport import _non_empty_text

        with pytest.raises(ValueError, match="non-empty string"):
            _non_empty_text("x", "   ")
        assert _non_empty_text("x", " ok ") == "ok"

    def test_finite_nonnegative_or_none(self) -> None:
        from scpn_control.core.neural_transport import _finite_nonnegative_or_none

        assert _finite_nonnegative_or_none("x", None) is None
        assert _finite_nonnegative_or_none("x", 2.0) == 2.0
        with pytest.raises(ValueError, match="finite and non-negative"):
            _finite_nonnegative_or_none("x", True)
        with pytest.raises(ValueError, match="finite and non-negative"):
            _finite_nonnegative_or_none("x", -1.0)

    def test_finite_positive_or_none(self) -> None:
        from scpn_control.core.neural_transport import _finite_positive_or_none

        assert _finite_positive_or_none("x", None) is None
        assert _finite_positive_or_none("x", 3.0) == 3.0
        with pytest.raises(ValueError, match="finite and positive"):
            _finite_positive_or_none("x", "a")
        with pytest.raises(ValueError, match="finite and positive"):
            _finite_positive_or_none("x", 0.0)

    def test_unit_interval_or_none(self) -> None:
        from scpn_control.core.neural_transport import _unit_interval_or_none

        assert _unit_interval_or_none("x", None) is None
        assert _unit_interval_or_none("x", 0.5) == 0.5
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _unit_interval_or_none("x", True)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _unit_interval_or_none("x", 2.0)

    def test_three_float_tuple(self) -> None:
        from scpn_control.core.neural_transport import _three_float_tuple

        assert _three_float_tuple("x", [1.0, 2.0, 3.0]) == (1.0, 2.0, 3.0)
        with pytest.raises(ValueError, match="three channel values"):
            _three_float_tuple("x", [1.0, 2.0])
        with pytest.raises(ValueError, match="non-negative channel values"):
            _three_float_tuple("x", [1.0, -2.0, 3.0])

    def test_profile_array_guards(self) -> None:
        from scpn_control.core.neural_transport import _profile_array

        with pytest.raises(ValueError, match="one-dimensional"):
            _profile_array("x", np.zeros((2, 2)))
        with pytest.raises(ValueError, match="at least three"):
            _profile_array("x", np.zeros(2))
        with pytest.raises(ValueError, match="finite"):
            _profile_array("x", np.array([1.0, np.nan, 3.0]))


class TestWeightLoading:
    def test_init_rejects_fallback_without_legacy(self) -> None:
        with pytest.raises(ValueError, match="requires allow_legacy_weight_load_fallback"):
            NeuralTransportModel(allow_weight_load_fallback=True)

    def test_loads_synthetic_weights_and_predicts_both_channels(self, tmp_path: Path) -> None:
        wp = tmp_path / "w.npz"
        _write_weights(wp)
        model = NeuralTransportModel(weights_path=wp)
        assert model.is_neural is True
        assert model.weights_checksum is not None
        fluxes = model.predict(TransportInputs(grad_ti=8.0))
        assert np.isfinite(fluxes.chi_i)
        assert fluxes.channel in {"ITG", "TEM"}

        itg_wp = tmp_path / "itg.npz"
        _write_weights(itg_wp, output_scale=np.array([0.5, 2.0, 1.0]))
        itg_model = NeuralTransportModel(weights_path=itg_wp)
        assert itg_model.predict(TransportInputs(grad_ti=8.0)).channel == "ITG"

    def test_neural_predict_profile_batched(self, tmp_path: Path) -> None:
        wp = tmp_path / "w.npz"
        _write_weights(wp)
        model = NeuralTransportModel(weights_path=wp)
        rho = np.linspace(0.1, 0.9, 6)
        te = np.linspace(8.0, 4.0, 6)
        ti = te.copy()
        ne = np.linspace(8.0, 3.0, 6)
        q = 1.0 + rho
        s_hat = np.linspace(0.3, 1.2, 6)
        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        assert chi_e.shape == (6,)
        assert np.all(np.isfinite(chi_e))
        assert np.all(np.isfinite(chi_i))
        assert np.all(np.isfinite(d_e))

    def test_auto_discovers_default_weights(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import scpn_control.core.neural_transport as nt

        wp = tmp_path / "default.npz"
        _write_weights(wp)
        monkeypatch.setattr(nt, "_DEFAULT_WEIGHTS_PATH", wp)
        model = nt.NeuralTransportModel()
        assert model.is_neural is True

    def test_missing_key_strict_fails_closed(self, tmp_path: Path) -> None:
        wp = tmp_path / "w.npz"
        arrays = _valid_weight_arrays()
        del arrays["w2"]
        np.savez(wp, **arrays)
        # The missing-key ValueError is raised inside the load try-block and the
        # strict handler re-raises it as a RuntimeError fail-closed.
        with pytest.raises(RuntimeError, match="Failed to load"):
            NeuralTransportModel(weights_path=wp)

    def test_missing_key_legacy_fallback(self, tmp_path: Path) -> None:
        wp = tmp_path / "w.npz"
        arrays = _valid_weight_arrays()
        del arrays["w2"]
        np.savez(wp, **arrays)
        model = NeuralTransportModel(
            weights_path=wp,
            allow_weight_load_fallback=True,
            allow_legacy_weight_load_fallback=True,
        )
        assert model.is_neural is False

    def test_version_mismatch_strict_fails_closed(self, tmp_path: Path) -> None:
        wp = tmp_path / "w.npz"
        _write_weights(wp, version=np.array(99))
        # The version ValueError is raised inside the load try-block and the
        # strict handler re-raises it as a RuntimeError fail-closed.
        with pytest.raises(RuntimeError, match="Failed to load"):
            NeuralTransportModel(weights_path=wp)

    def test_version_mismatch_legacy_fallback(self, tmp_path: Path) -> None:
        wp = tmp_path / "w.npz"
        _write_weights(wp, version=np.array(99))
        model = NeuralTransportModel(
            weights_path=wp,
            allow_weight_load_fallback=True,
            allow_legacy_weight_load_fallback=True,
        )
        assert model.is_neural is False

    def test_corrupt_file_strict_fails_closed(self, tmp_path: Path) -> None:
        wp = tmp_path / "w.npz"
        wp.write_bytes(b"not a valid npz archive")
        with pytest.raises(RuntimeError, match="Failed to load"):
            NeuralTransportModel(weights_path=wp)

    def test_corrupt_file_legacy_fallback(self, tmp_path: Path) -> None:
        wp = tmp_path / "w.npz"
        wp.write_bytes(b"not a valid npz archive")
        model = NeuralTransportModel(
            weights_path=wp,
            allow_weight_load_fallback=True,
            allow_legacy_weight_load_fallback=True,
        )
        assert model.is_neural is False


class TestClosureProfiles:
    def test_shape_mismatch_and_rmajor_and_require_neural(self) -> None:
        from scpn_control.core.neural_transport import neural_transport_closure_profiles

        rho = np.array([0.0, 0.5, 1.0])
        good = np.array([6.0, 5.0, 4.0])
        with pytest.raises(ValueError, match="must have the same shape"):
            neural_transport_closure_profiles(rho, np.array([6.0, 5.0, 4.0, 3.0]), good, good, good, good)
        with pytest.raises(ValueError, match="r_major must be a positive"):
            neural_transport_closure_profiles(rho, good, good, good, good, good, r_major=0.0)
        # An analytic-only model fails closed when neural weights are required.
        analytic_model = NeuralTransportModel(auto_discover=False)
        with pytest.raises(RuntimeError, match="requires loaded neural transport weights"):
            neural_transport_closure_profiles(
                rho, good, good, good, good, good, model=analytic_model, require_neural=True
            )

    def test_rejects_invalid_model_output(self) -> None:
        from scpn_control.core.neural_transport import neural_transport_closure_profiles

        rho = np.linspace(0.1, 0.9, 5)
        good = np.linspace(8.0, 4.0, 5)
        q = 1.0 + rho
        s_hat = np.linspace(0.3, 1.0, 5)

        class _FakeModel:
            is_neural = False
            weights_checksum = None

            def __init__(self, output: tuple[Any, Any, Any]) -> None:
                self._output = output

            def predict_profile(self, *_args: Any, **_kwargs: Any) -> tuple[Any, Any, Any]:
                return self._output

        wrong_shape = _FakeModel((np.zeros(3), np.zeros(5), np.zeros(5)))
        with pytest.raises(RuntimeError, match="invalid chi_e shape"):
            neural_transport_closure_profiles(rho, good, good, good, q, s_hat, model=wrong_shape)

        non_finite = _FakeModel((np.full(5, np.nan), np.zeros(5), np.zeros(5)))
        with pytest.raises(RuntimeError, match="non-finite chi_e"):
            neural_transport_closure_profiles(rho, good, good, good, q, s_hat, model=non_finite)

        negative = _FakeModel((np.full(5, -1.0), np.zeros(5), np.zeros(5)))
        with pytest.raises(RuntimeError, match="negative chi_e"):
            neural_transport_closure_profiles(rho, good, good, good, q, s_hat, model=negative)

    def test_analytic_fallback_returns_active_channel_weights(self) -> None:
        from scpn_control.core.neural_transport import neural_transport_closure_profiles

        rho = np.linspace(0.1, 0.9, 5)
        te = np.linspace(8.0, 5.0, 5)
        ti = te.copy()
        ne = np.linspace(8.0, 4.0, 5)
        q = 1.0 + rho
        s_hat = np.linspace(0.3, 1.0, 5)
        model = NeuralTransportModel(auto_discover=False)
        result = neural_transport_closure_profiles(rho, te, ti, ne, q, s_hat, model=model)
        assert result.source == "analytic_fallback"
        assert result.channel_weights.shape == (3, 5)
        assert result.weights_checksum is None

    def test_constructs_default_model_when_none(self) -> None:
        from scpn_control.core.neural_transport import neural_transport_closure_profiles

        rho = np.linspace(0.1, 0.9, 5)
        te = np.linspace(8.0, 5.0, 5)
        ti = te.copy()
        ne = np.linspace(8.0, 4.0, 5)
        q = 1.0 + rho
        s_hat = np.linspace(0.3, 1.0, 5)
        result = neural_transport_closure_profiles(rho, te, ti, ne, q, s_hat)
        assert result.chi_e.shape == (5,)
        assert result.source in {"neural", "analytic_fallback"}

    def test_stable_profile_uses_uniform_channel_weights(self) -> None:
        from scpn_control.core.neural_transport import neural_transport_closure_profiles

        rho = np.linspace(0.1, 0.9, 5)
        flat = np.full(5, 5.0)
        model = NeuralTransportModel(auto_discover=False)
        result = neural_transport_closure_profiles(rho, flat, flat, flat, np.full(5, 2.0), np.full(5, 0.5), model=model)
        assert np.allclose(result.channel_weights, 1.0 / 3.0)


class TestCrossValidation:
    def test_cross_validate_returns_metric_block(self) -> None:
        from scpn_control.core.neural_transport import cross_validate_neural_transport

        result = cross_validate_neural_transport()
        assert result["n_cases"] == 7
        assert len(result["per_channel_rmse"]) == 3
        assert len(result["profile_per_channel_relative_rmse"]) == 3
        assert 0.0 <= result["channel_agreement"] <= 1.0
        assert result["mode"] in {"neural", "analytic_fallback"}

    def test_cross_validate_rejects_empty_cases(self) -> None:
        from scpn_control.core.neural_transport import cross_validate_neural_transport

        with pytest.raises(ValueError, match="at least one case"):
            cross_validate_neural_transport(benchmark_cases=())


class TestClaimEvidenceGuards:
    def test_rejects_invalid_validation_result(self) -> None:
        from scpn_control.core.neural_transport import neural_transport_claim_evidence

        base = _claim_validation_result()
        with pytest.raises(ValueError, match="at least one local benchmark"):
            neural_transport_claim_evidence({**base, "n_cases": 0}, source="s", source_id="i")
        with pytest.raises(ValueError, match="max_abs_error must be finite"):
            neural_transport_claim_evidence({**base, "max_abs_error": -1.0}, source="s", source_id="i")
        with pytest.raises(ValueError, match="channel_agreement must be finite"):
            neural_transport_claim_evidence({**base, "channel_agreement": 2.0}, source="s", source_id="i")

    def test_missing_weights_file_raises(self, tmp_path: Path) -> None:
        from scpn_control.core.neural_transport import neural_transport_claim_evidence

        with pytest.raises(FileNotFoundError, match="weights not found"):
            neural_transport_claim_evidence(
                _claim_validation_result(),
                source="s",
                source_id="i",
                weights_path=tmp_path / "nope.npz",
            )

    def test_reference_requires_weights_path(self, tmp_path: Path) -> None:
        from scpn_control.core.neural_transport import neural_transport_claim_evidence

        artifact = _write_reference_artifact(tmp_path / "ref.json", "a" * 64)
        with pytest.raises(ValueError, match="requires the exact neural-transport weights_path"):
            neural_transport_claim_evidence(
                _claim_validation_result(),
                source="s",
                source_id="i",
                reference_artifact_path=artifact,
            )

    def test_reference_artifact_failing_validation_is_rejected(self, tmp_path: Path) -> None:
        from scpn_control.core.neural_transport import _sha256_file, neural_transport_claim_evidence

        wp = tmp_path / "w.npz"
        _write_weights(wp)
        artifact = _write_reference_artifact(tmp_path / "ref.json", _sha256_file(wp))
        # Corrupt the tamper-evident digest so strict validation fails.
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["payload_sha256"] = "0" * 64
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="failed strict validation"):
            neural_transport_claim_evidence(
                _claim_validation_result(),
                source="real_qualikiz",
                source_id="i",
                weights_path=wp,
                reference_artifact_path=artifact,
            )

    def test_reference_artifact_weights_digest_mismatch_is_rejected(self, tmp_path: Path) -> None:
        from scpn_control.core.neural_transport import neural_transport_claim_evidence

        wp = tmp_path / "w.npz"
        _write_weights(wp)
        # Artifact references a different (valid-format) weights digest than wp.
        artifact = _write_reference_artifact(tmp_path / "ref.json", "c" * 64)
        with pytest.raises(ValueError, match="does not match supplied weights"):
            neural_transport_claim_evidence(
                _claim_validation_result(),
                source="real_qualikiz",
                source_id="i",
                weights_path=wp,
                reference_artifact_path=artifact,
            )

    def test_matched_reference_allows_quantitative_claim(self, tmp_path: Path) -> None:
        from scpn_control.core.neural_transport import (
            _sha256_file,
            assert_neural_transport_quantitative_claim_admissible,
            neural_transport_claim_evidence,
            save_neural_transport_claim_evidence,
        )

        wp = tmp_path / "w.npz"
        _write_weights(wp)
        artifact = _write_reference_artifact(tmp_path / "ref.json", _sha256_file(wp))
        evidence = neural_transport_claim_evidence(
            _claim_validation_result(),
            source="real_qualikiz",
            source_id="tests::matched_reference",
            weights_path=wp,
            reference_artifact_path=artifact,
        )
        assert evidence.quantitative_claim_allowed is True
        assert evidence.reference_source == "documented_public_reference"
        assert evidence.reference_sample_count == 128
        assert assert_neural_transport_quantitative_claim_admissible(evidence) is evidence

        out = tmp_path / "evidence.json"
        save_neural_transport_claim_evidence(evidence, out)
        assert out.exists()
        assert json.loads(out.read_text(encoding="utf-8"))["quantitative_claim_allowed"] is True

    def test_assert_admissible_rejects_bad_inputs(self) -> None:
        import dataclasses

        from scpn_control.core.neural_transport import (
            assert_neural_transport_quantitative_claim_admissible,
            neural_transport_claim_evidence,
            save_neural_transport_claim_evidence,
        )

        with pytest.raises(ValueError, match="must be NeuralTransportClaimEvidence"):
            assert_neural_transport_quantitative_claim_admissible(object())  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must be NeuralTransportClaimEvidence"):
            save_neural_transport_claim_evidence(object(), "unused.json")  # type: ignore[arg-type]

        evidence = neural_transport_claim_evidence(
            _claim_validation_result(),
            source="s",
            source_id="i",
        )
        bad_schema = dataclasses.replace(evidence, schema_version=999)
        with pytest.raises(ValueError, match="schema_version is unsupported"):
            assert_neural_transport_quantitative_claim_admissible(bad_schema)
