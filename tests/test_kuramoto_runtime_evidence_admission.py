# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Admission and validator behaviour for Kuramoto runtime evidence.

Exercises the field-by-field validation in
``_validate_kuramoto_runtime_payload`` and the numeric guards on
``order_parameter`` / ``kuramoto_sakaguchi_step`` / ``kuramoto_runtime_evidence``.

The tamper-evident digest in ``payload_sha256`` is checked before any field
validator runs, so to reach a downstream validator a hostile payload must be
re-sealed with ``_payload_sha256`` after the field is mutated; otherwise the
digest-mismatch guard masks the intended branch.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

import scpn_control.phase.kuramoto as kuramoto_module
from scpn_control.phase.kuramoto import (
    KURAMOTO_RUNTIME_EVIDENCE_BOUNDED,
    KURAMOTO_RUNTIME_EVIDENCE_QUALIFIED,
    KURAMOTO_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    KuramotoRuntimeEvidence,
    _payload_sha256,
    _utc_now,
    _validate_kuramoto_runtime_payload,
    assert_kuramoto_runtime_claim_admissible,
    kuramoto_runtime_evidence,
    kuramoto_sakaguchi_step,
    order_parameter,
    save_kuramoto_runtime_evidence,
)


def _reseal(payload: dict) -> dict:
    """Return a copy of ``payload`` with a self-consistent ``payload_sha256``."""
    sealed = dict(payload)
    sealed["payload_sha256"] = _payload_sha256(sealed)
    return sealed


def _valid_deployment_payload() -> dict:
    """A payload that passes every gate, including the deployment-claim gates.

    Mutating one field of this baseline and re-sealing isolates exactly one
    validator branch per test.
    """
    payload = {
        "schema_version": KURAMOTO_RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "generated_utc": "2026-05-31T00:00:00Z",
        "target_id": "lab-phase-runtime",
        "oscillator_count": 8,
        "deployment_target_oscillators": 4,
        "dt_s": 1.0e-3,
        "K": 1.5,
        "alpha": 0.0,
        "zeta": 0.2,
        "psi_mode": "external",
        "wrap": True,
        "input_sha256": "a" * 64,
        "python_reference_sha256": "b" * 64,
        "rust_available": True,
        "rust_parity_checked": True,
        "rust_theta_max_abs_error_rad": 1.0e-12,
        "rust_order_parameter_abs_error": 1.0e-12,
        "parity_tolerance": 1.0e-10,
        "parity_passed": True,
        "timestep_refinement_checked": True,
        "timestep_refinement_error_rad": 1.0e-6,
        "timestep_refinement_tolerance": 5.0e-3,
        "timestep_refinement_passed": True,
        "deployment_claim_allowed": True,
        "claim_status": KURAMOTO_RUNTIME_EVIDENCE_QUALIFIED,
        "payload_sha256": "",
    }
    return _reseal(payload)


class TestUtcNow:
    def test_returns_zulu_suffixed_iso_timestamp(self):
        stamp = _utc_now()
        assert stamp.endswith("Z")
        assert "+00:00" not in stamp
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None
        assert parsed.microsecond == 0


class TestOrderParameterShapeGuard:
    def test_rejects_two_dimensional_phase_array(self):
        with pytest.raises(ValueError, match="1D phase vector"):
            order_parameter(np.zeros((2, 3)))


class TestKuramotoStepArrayGuards:
    def test_rejects_two_dimensional_theta(self):
        with pytest.raises(ValueError, match="theta must be a 1D phase vector"):
            kuramoto_sakaguchi_step(np.zeros((2, 2)), np.zeros(4), dt=1.0e-3, K=1.0, psi_mode="mean_field")

    def test_rejects_two_dimensional_omega(self):
        with pytest.raises(ValueError, match="omega must be a 1D frequency vector"):
            kuramoto_sakaguchi_step(np.zeros(4), np.zeros((2, 2)), dt=1.0e-3, K=1.0, psi_mode="mean_field")

    def test_rejects_nonfinite_theta(self):
        theta = np.array([0.0, np.nan, 0.2])
        with pytest.raises(ValueError, match="theta must contain only finite"):
            kuramoto_sakaguchi_step(theta, np.zeros(3), dt=1.0e-3, K=1.0, psi_mode="mean_field")

    def test_rejects_nonfinite_omega(self):
        omega = np.array([0.0, np.inf, 0.2])
        with pytest.raises(ValueError, match="omega must contain only finite"):
            kuramoto_sakaguchi_step(np.zeros(3), omega, dt=1.0e-3, K=1.0, psi_mode="mean_field")


class TestRuntimeEvidenceInputGuards:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"theta": np.zeros((2, 2))}, "non-empty 1D phase vector"),
            ({"theta": np.array([])}, "non-empty 1D phase vector"),
            ({"omega": np.zeros(2)}, "1D frequency vector matching theta"),
            ({"omega": np.zeros((3, 1))}, "1D frequency vector matching theta"),
            ({"theta": np.array([0.0, np.nan, 0.2])}, "theta must contain only finite"),
            ({"omega": np.array([0.0, np.inf, 0.0])}, "omega must contain only finite"),
            ({"dt": 0.0}, "dt must be positive"),
            ({"dt": np.inf}, "dt must be positive"),
            ({"K": -1.0}, "K must be finite and non-negative"),
            ({"alpha": np.nan}, "alpha must be finite"),
            ({"zeta": np.inf}, "zeta must be finite"),
        ],
    )
    def test_rejects_unphysical_inputs(self, monkeypatch, overrides, match):
        monkeypatch.setattr(kuramoto_module, "RUST_KURAMOTO", False)
        kwargs = {
            "theta": np.array([0.1, 0.2, 0.3]),
            "omega": np.zeros(3),
            "dt": 1.0e-3,
            "K": 1.0,
        }
        kwargs.update(overrides)
        theta = kwargs.pop("theta")
        omega = kwargs.pop("omega")
        with pytest.raises(ValueError, match=match):
            kuramoto_runtime_evidence(theta, omega, **kwargs)


class TestPayloadFieldValidation:
    def test_baseline_payload_is_admissible_for_deployment(self):
        evidence = _validate_kuramoto_runtime_payload(_valid_deployment_payload(), require_deployment_claim=True)
        assert isinstance(evidence, KuramotoRuntimeEvidence)
        assert evidence.deployment_claim_allowed is True
        assert evidence.claim_status == KURAMOTO_RUNTIME_EVIDENCE_QUALIFIED

    def test_rejects_unsupported_schema_version(self):
        payload = _valid_deployment_payload()
        payload["schema_version"] = "scpn-control.kuramoto-runtime-evidence.v0"
        with pytest.raises(ValueError, match="schema_version is unsupported"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_malformed_payload_digest_field(self):
        payload = _valid_deployment_payload()
        payload["payload_sha256"] = "not-a-digest"
        with pytest.raises(ValueError, match="payload_sha256 must be a SHA-256"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_timestamp_without_zulu_suffix(self):
        payload = _reseal({**_valid_deployment_payload(), "generated_utc": "2026-05-31T00:00:00"})
        with pytest.raises(ValueError, match="must be a UTC timestamp ending in Z"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_blank_target_id(self):
        payload = _reseal({**_valid_deployment_payload(), "target_id": "   "})
        with pytest.raises(ValueError, match="target_id must be non-empty"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_nonpositive_oscillator_count(self):
        payload = _reseal({**_valid_deployment_payload(), "oscillator_count": 0})
        with pytest.raises(ValueError, match="oscillator_count must be a positive integer"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_nonpositive_timestep(self):
        payload = _reseal({**_valid_deployment_payload(), "dt_s": 0.0})
        with pytest.raises(ValueError, match="dt_s must be positive"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_non_numeric_alpha(self):
        payload = _reseal({**_valid_deployment_payload(), "alpha": [1.0]})
        with pytest.raises(ValueError, match="alpha must be finite"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_unparseable_string_alpha(self):
        payload = _reseal({**_valid_deployment_payload(), "alpha": "not_a_float"})
        with pytest.raises(ValueError, match="alpha must be finite"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_nonfinite_zeta(self):
        payload = _reseal({**_valid_deployment_payload(), "zeta": float("inf")})
        with pytest.raises(ValueError, match="zeta must be finite"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_negative_parity_tolerance(self):
        payload = _reseal({**_valid_deployment_payload(), "parity_tolerance": -1.0})
        with pytest.raises(ValueError, match="parity_tolerance must be non-negative"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_unsupported_psi_mode(self):
        payload = _reseal({**_valid_deployment_payload(), "psi_mode": "telepathic"})
        with pytest.raises(ValueError, match="psi_mode is unsupported"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_non_boolean_flag(self):
        payload = _reseal({**_valid_deployment_payload(), "wrap": 1})
        with pytest.raises(ValueError, match="wrap must be boolean"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_malformed_input_digest(self):
        payload = _reseal({**_valid_deployment_payload(), "input_sha256": "short"})
        with pytest.raises(ValueError, match="input_sha256 must be a SHA-256"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_malformed_reference_digest(self):
        payload = _reseal({**_valid_deployment_payload(), "python_reference_sha256": "short"})
        with pytest.raises(ValueError, match="python_reference_sha256 must be a SHA-256"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_claim_status_inconsistent_with_deployment_flag(self):
        payload = _reseal({**_valid_deployment_payload(), "claim_status": KURAMOTO_RUNTIME_EVIDENCE_BOUNDED})
        with pytest.raises(ValueError, match="claim_status does not match"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_parity_checked_without_error_metrics(self):
        payload = _reseal({**_valid_deployment_payload(), "rust_theta_max_abs_error_rad": None})
        with pytest.raises(ValueError, match="requires numerical error metrics"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)


class TestDeploymentClaimGates:
    def test_rejects_oscillator_count_below_deployment_target(self):
        payload = _reseal({**_valid_deployment_payload(), "oscillator_count": 2})
        with pytest.raises(ValueError, match="below the declared deployment oscillator count"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_deployment_without_rust_parity(self):
        payload = _reseal({**_valid_deployment_payload(), "rust_available": False})
        with pytest.raises(ValueError, match="requires Rust parity for deployment"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_deployment_when_parity_did_not_pass(self):
        payload = _reseal({**_valid_deployment_payload(), "parity_passed": False})
        with pytest.raises(ValueError, match="Rust parity did not pass"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_deployment_when_phase_parity_exceeds_tolerance(self):
        payload = _reseal({**_valid_deployment_payload(), "rust_theta_max_abs_error_rad": 1.0})
        with pytest.raises(ValueError, match="phase parity exceeds tolerance"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_deployment_when_order_parameter_parity_exceeds_tolerance(self):
        payload = _reseal({**_valid_deployment_payload(), "rust_order_parameter_abs_error": 1.0})
        with pytest.raises(ValueError, match="order-parameter parity exceeds tolerance"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)

    def test_rejects_deployment_when_timestep_refinement_exceeds_tolerance(self):
        payload = _reseal({**_valid_deployment_payload(), "timestep_refinement_error_rad": 1.0})
        with pytest.raises(ValueError, match="timestep refinement exceeds tolerance"):
            _validate_kuramoto_runtime_payload(payload, require_deployment_claim=False)


class TestEvidenceInstanceGuards:
    def test_admission_rejects_non_evidence_object(self):
        with pytest.raises(ValueError, match="evidence must be KuramotoRuntimeEvidence"):
            assert_kuramoto_runtime_claim_admissible({"schema_version": KURAMOTO_RUNTIME_EVIDENCE_SCHEMA_VERSION})

    def test_save_rejects_non_evidence_object(self, tmp_path):
        with pytest.raises(ValueError, match="evidence must be KuramotoRuntimeEvidence"):
            save_kuramoto_runtime_evidence({"not": "evidence"}, tmp_path / "out.json")

    def test_load_rejects_non_object_json(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        from scpn_control.phase.kuramoto import load_kuramoto_runtime_evidence

        with pytest.raises(ValueError, match="must be a JSON object"):
            load_kuramoto_runtime_evidence(path)
