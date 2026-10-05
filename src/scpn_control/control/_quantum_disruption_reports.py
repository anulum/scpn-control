# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Quantum Disruption Reports

"""Quantum disruption reports for the bounded quantum bridge."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from scpn_control.control._quantum_disruption_backend import (
    _backend_contract_attestation_digest,
    _validate_backend_contract_attestation,
)
from scpn_control.control._quantum_disruption_constants import (
    ADVISORY_DECISION_SCHEMA_VERSION,
    CERTIFICATE_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    CONTROL_FACADE_OWNER,
    KERNEL_SCHEMA_VERSION,
    QUANTUM_BACKEND_OWNER,
    REQUIRED_DOWNSTREAM_POLICY,
    RISK_BAND_THRESHOLDS,
    SCHEMA_VERSION,
)
from scpn_control.control._quantum_disruption_contract import (
    _dependency_contract_digest,
    _dependency_contract_schema_version,
    _validate_report_dependency_contract,
)
from scpn_control.control._quantum_disruption_features import (
    _classical_baseline_score,
    _validate_admission_evidence,
    _validate_mapping_payload,
)
from scpn_control.control._quantum_disruption_utils import (
    _advisory_decision_payload_digest,
    _bounded_score,
    _certificate_digest,
    _is_sha256,
    _payload_digest,
    _report_content_digest,
)


def validate_quantum_disruption_bridge_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a tamper-evident quantum disruption bridge report."""
    if not isinstance(payload, dict):
        raise ValueError("quantum disruption bridge report must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("quantum disruption bridge report schema_version is unsupported")
    if payload.get("status") not in {"advisory", "quantum-unavailable"}:
        raise ValueError("quantum disruption bridge report status is unsupported")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("quantum disruption bridge report claim_boundary is unsupported")
    if payload.get("control_facade_owner") != CONTROL_FACADE_OWNER:
        raise ValueError("quantum disruption bridge report control_facade_owner is unsupported")
    if payload.get("quantum_backend_owner") != QUANTUM_BACKEND_OWNER:
        raise ValueError("quantum disruption bridge report quantum_backend_owner is unsupported")
    if not isinstance(payload.get("created_at"), str) or not payload["created_at"]:
        raise ValueError("quantum disruption bridge report created_at must be non-empty")
    if payload.get("claim_status") not in {"bounded_model", "validation_gap"}:
        raise ValueError("quantum disruption bridge report claim_status is unsupported")
    if not isinstance(payload.get("quantum_available"), bool):
        raise ValueError("quantum disruption bridge report quantum_available must be a bool")
    if payload["status"] == "quantum-unavailable" and payload["quantum_available"]:
        raise ValueError("quantum disruption bridge unavailable status conflicts with quantum_available")
    if payload["status"] == "advisory" and not payload["quantum_available"]:
        raise ValueError("quantum disruption bridge advisory status requires quantum_available")
    if not isinstance(payload.get("admitted_for_control"), bool) or payload["admitted_for_control"]:
        raise ValueError("quantum disruption bridge report is not allowed to admit control action")
    if payload.get("human_review_required") is not True:
        raise ValueError("quantum disruption bridge report requires human review")
    feature_mapping = _validate_mapping_payload(payload.get("feature_mapping"))
    _validate_admission_evidence(
        payload.get("admission_evidence"),
        feature_mapping=feature_mapping,
        quantum_available=payload["quantum_available"],
    )
    classical_score = _bounded_score("classical_baseline_score", payload.get("classical_baseline_score"))
    quantum_score = payload.get("quantum_score")
    if quantum_score is not None:
        quantum_score = _bounded_score("quantum_score", quantum_score)
    risk_score = _bounded_score("risk_score", payload.get("risk_score"))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("quantum disruption bridge report config must be an object")
    for key in ("quantum_module", "backend_profile"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"quantum disruption bridge report {key} must be non-empty")
    _validate_report_dependency_contract(payload)
    backend_attestation = _validate_backend_contract_attestation(
        payload.get("backend_contract_attestation"), payload=payload
    )
    _validate_advisory_decision(payload.get("advisory_decision"), payload=payload)
    _validate_report_certificate(payload, report_kind="bridge-advisory")
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError("quantum disruption bridge report payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("quantum disruption bridge report payload_sha256 does not match payload")
    if payload["quantum_available"] != (quantum_score is not None):
        raise ValueError("quantum disruption bridge quantum_available must match quantum_score presence")
    backend_status = backend_attestation["status"]
    if (payload["quantum_available"] and backend_status not in {"matched", "not_exposed"}) or (
        not payload["quantum_available"] and backend_status != "backend_unavailable"
    ):
        raise ValueError("quantum disruption bridge backend attestation must match quantum availability")
    expected_classical_score = _classical_baseline_score(
        np.asarray(feature_mapping["normalized_iter_features"], dtype=np.float64)
    )
    if classical_score != expected_classical_score:
        raise ValueError("quantum disruption bridge classical_baseline_score must match feature mapping")
    if risk_score != (quantum_score if quantum_score is not None else classical_score):
        raise ValueError("quantum disruption bridge risk_score must match selected source score")
    return payload


def validate_quantum_disruption_kernel_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a tamper-evident quantum disruption kernel report."""
    if not isinstance(payload, dict):
        raise ValueError("quantum disruption kernel report must be an object")
    if payload.get("schema_version") != KERNEL_SCHEMA_VERSION:
        raise ValueError("quantum disruption kernel report schema_version is unsupported")
    if payload.get("status") != "advisory-kernel":
        raise ValueError("quantum disruption kernel report status is unsupported")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError("quantum disruption kernel report claim_boundary is unsupported")
    if payload.get("control_facade_owner") != CONTROL_FACADE_OWNER:
        raise ValueError("quantum disruption kernel report control_facade_owner is unsupported")
    if payload.get("quantum_backend_owner") != QUANTUM_BACKEND_OWNER:
        raise ValueError("quantum disruption kernel report quantum_backend_owner is unsupported")
    if not isinstance(payload.get("admitted_for_control"), bool) or payload["admitted_for_control"]:
        raise ValueError("quantum disruption kernel report is not allowed to admit control action")
    matrix = np.asarray(payload.get("kernel_matrix"), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (payload.get("samples_a_count"), payload.get("samples_b_count")):
        raise ValueError("quantum disruption kernel report matrix shape is inconsistent")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("quantum disruption kernel report matrix must be finite")
    if np.any(matrix < -1.0e-12) or np.any(matrix > 1.0 + 1.0e-12):
        raise ValueError("quantum disruption kernel report matrix values must be in [0, 1]")
    if matrix.shape[0] == matrix.shape[1]:
        if not np.allclose(matrix, matrix.T, atol=1.0e-12):
            raise ValueError("quantum disruption kernel report square matrix must be symmetric")
        if not np.allclose(np.diag(matrix), np.ones(matrix.shape[0]), atol=1.0e-12):
            raise ValueError("quantum disruption kernel report diagonal must be one")
    _validate_report_dependency_contract(payload)
    _validate_backend_contract_attestation(payload.get("backend_contract_attestation"), payload=payload)
    _validate_report_certificate(payload, report_kind="kernel-advisory")
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or not _is_sha256(declared_digest):
        raise ValueError("quantum disruption kernel report payload_sha256 must be a SHA-256 hex digest")
    if _payload_digest(payload) != declared_digest.lower():
        raise ValueError("quantum disruption kernel report payload_sha256 does not match payload")
    return payload


def _build_advisory_decision(
    *,
    classical_baseline_score: float,
    quantum_score: float | None,
    backend_contract_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    risk_score = _bounded_score(
        "risk_score",
        quantum_score if quantum_score is not None else classical_baseline_score,
    )
    score_basis = "quantum_score" if quantum_score is not None else "classical_baseline_score"
    backend_status = backend_contract_attestation.get("status")
    backend_contract_validated = backend_contract_attestation.get("backend_contract_validated") is True
    reasons = ["advisory_only", "external_validation_required", "control_admission_blocked"]
    if quantum_score is None:
        reasons.append("quantum_score_unavailable")
    if backend_status == "backend_unavailable":
        reasons.append("quantum_backend_unavailable")
    if not backend_contract_validated:
        reasons.append("backend_contract_not_validated")
    payload: dict[str, Any] = {
        "schema_version": ADVISORY_DECISION_SCHEMA_VERSION,
        "risk_score": risk_score,
        "score_basis": score_basis,
        "risk_band": _risk_band(risk_score),
        "thresholds": dict(RISK_BAND_THRESHOLDS),
        "control_action": "blocked",
        "admitted_for_control": False,
        "publication_safe": False,
        "human_review_required": True,
        "external_validation_required": True,
        "backend_contract_validated": backend_contract_validated,
        "reasons": reasons,
    }
    payload["decision_sha256"] = _advisory_decision_payload_digest(payload)
    return payload


def _build_report_certificate(payload: Mapping[str, Any], *, report_kind: str) -> dict[str, Any]:
    certificate: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "report_kind": report_kind,
        "report_schema_version": payload.get("schema_version"),
        "control_facade_owner": CONTROL_FACADE_OWNER,
        "quantum_backend_owner": QUANTUM_BACKEND_OWNER,
        "dependency_contract_schema_version": _dependency_contract_schema_version(payload),
        "dependency_contract_sha256": _dependency_contract_digest(payload),
        "backend_contract_attestation_sha256": _backend_contract_attestation_digest(payload),
        "claim_boundary_sha256": _payload_digest({"claim_boundary": payload.get("claim_boundary")}),
        "admitted_for_control": False,
        "publication_safe": False,
        "external_validation_required": True,
        "required_downstream_policy": list(REQUIRED_DOWNSTREAM_POLICY),
        "content_sha256": _report_content_digest(payload),
    }
    if report_kind == "bridge-advisory":
        certificate["advisory_decision_sha256"] = _advisory_decision_digest(payload)
    certificate["certificate_sha256"] = _certificate_digest(certificate)
    return certificate


def _validate_advisory_decision(value: object, *, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("quantum disruption advisory_decision must be an object")
    if value.get("schema_version") != ADVISORY_DECISION_SCHEMA_VERSION:
        raise ValueError("quantum disruption advisory_decision schema_version is unsupported")
    risk_score = _bounded_score("advisory_decision.risk_score", value.get("risk_score"))
    if risk_score != _bounded_score("risk_score", payload.get("risk_score")):
        raise ValueError("quantum disruption advisory_decision risk_score mismatch")
    expected_score_basis = "quantum_score" if payload.get("quantum_score") is not None else "classical_baseline_score"
    if value.get("score_basis") != expected_score_basis:
        raise ValueError("quantum disruption advisory_decision score_basis mismatch")
    if value.get("risk_band") != _risk_band(risk_score):
        raise ValueError("quantum disruption advisory_decision risk_band mismatch")
    if value.get("thresholds") != RISK_BAND_THRESHOLDS:
        raise ValueError("quantum disruption advisory_decision thresholds mismatch")
    if value.get("control_action") != "blocked":
        raise ValueError("quantum disruption advisory_decision control_action must be blocked")
    if value.get("admitted_for_control") is not False:
        raise ValueError("quantum disruption advisory_decision admitted_for_control must be false")
    if value.get("publication_safe") is not False:
        raise ValueError("quantum disruption advisory_decision publication_safe must be false")
    if value.get("human_review_required") is not True:
        raise ValueError("quantum disruption advisory_decision human_review_required must be true")
    if value.get("external_validation_required") is not True:
        raise ValueError("quantum disruption advisory_decision external_validation_required must be true")
    attestation = _validate_backend_contract_attestation(payload.get("backend_contract_attestation"), payload=payload)
    if value.get("backend_contract_validated") != attestation["backend_contract_validated"]:
        raise ValueError("quantum disruption advisory_decision backend_contract_validated mismatch")
    reasons = value.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) or not item for item in reasons):
        raise ValueError("quantum disruption advisory_decision reasons must be non-empty strings")
    for required_reason in ("advisory_only", "external_validation_required", "control_admission_blocked"):
        if required_reason not in reasons:
            raise ValueError(f"quantum disruption advisory_decision missing reason {required_reason}")
    if payload.get("quantum_score") is None and "quantum_score_unavailable" not in reasons:
        raise ValueError("quantum disruption advisory_decision must record quantum_score_unavailable")
    if attestation["status"] == "backend_unavailable" and "quantum_backend_unavailable" not in reasons:
        raise ValueError("quantum disruption advisory_decision must record quantum_backend_unavailable")
    if not attestation["backend_contract_validated"] and "backend_contract_not_validated" not in reasons:
        raise ValueError("quantum disruption advisory_decision must record backend_contract_not_validated")
    decision_digest = value.get("decision_sha256")
    if not isinstance(decision_digest, str) or not _is_sha256(decision_digest):
        raise ValueError("quantum disruption advisory_decision decision_sha256 must be a SHA-256")
    if decision_digest.lower() != _advisory_decision_payload_digest(value):
        raise ValueError("quantum disruption advisory_decision decision_sha256 mismatch")
    return value


def _advisory_decision_digest(payload: Mapping[str, Any]) -> str:
    decision = _validate_advisory_decision(payload.get("advisory_decision"), payload=payload)
    return str(decision["decision_sha256"])


def _validate_report_certificate(payload: Mapping[str, Any], *, report_kind: str) -> None:
    value = payload.get("report_certificate")
    if not isinstance(value, dict):
        raise ValueError("quantum disruption report_certificate must be an object")
    if value.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
        raise ValueError("quantum disruption report_certificate schema_version is unsupported")
    if value.get("report_kind") != report_kind:
        raise ValueError("quantum disruption report_certificate report_kind is unsupported")
    if value.get("report_schema_version") != payload.get("schema_version"):
        raise ValueError("quantum disruption report_certificate report_schema_version mismatch")
    if value.get("control_facade_owner") != CONTROL_FACADE_OWNER:
        raise ValueError("quantum disruption report_certificate control_facade_owner is unsupported")
    if value.get("quantum_backend_owner") != QUANTUM_BACKEND_OWNER:
        raise ValueError("quantum disruption report_certificate quantum_backend_owner is unsupported")
    if value.get("dependency_contract_schema_version") != _dependency_contract_schema_version(payload):
        raise ValueError("quantum disruption report_certificate dependency_contract_schema_version mismatch")
    dependency_digest = value.get("dependency_contract_sha256")
    if not isinstance(dependency_digest, str) or not _is_sha256(dependency_digest):
        raise ValueError(
            "quantum disruption report_certificate dependency_contract_sha256 must be a SHA-256 hex digest"
        )
    if dependency_digest != _dependency_contract_digest(payload):
        raise ValueError("quantum disruption report_certificate dependency_contract_sha256 mismatch")
    attestation_digest = value.get("backend_contract_attestation_sha256")
    if not isinstance(attestation_digest, str) or not _is_sha256(attestation_digest):
        raise ValueError(
            "quantum disruption report_certificate backend_contract_attestation_sha256 must be a SHA-256 hex digest"
        )
    if attestation_digest != _backend_contract_attestation_digest(payload):
        raise ValueError("quantum disruption report_certificate backend_contract_attestation_sha256 mismatch")
    if report_kind == "bridge-advisory":
        decision_digest = value.get("advisory_decision_sha256")
        if not isinstance(decision_digest, str) or not _is_sha256(decision_digest):
            raise ValueError(
                "quantum disruption report_certificate advisory_decision_sha256 must be a SHA-256 hex digest"
            )
        if decision_digest != _advisory_decision_digest(payload):
            raise ValueError("quantum disruption report_certificate advisory_decision_sha256 mismatch")
    elif "advisory_decision_sha256" in value:
        raise ValueError("quantum disruption report_certificate advisory_decision_sha256 is bridge-only")
    if value.get("admitted_for_control") is not False:
        raise ValueError("quantum disruption report_certificate admitted_for_control must be false")
    if value.get("publication_safe") is not False:
        raise ValueError("quantum disruption report_certificate publication_safe must be false")
    if value.get("external_validation_required") is not True:
        raise ValueError("quantum disruption report_certificate external_validation_required must be true")
    policy = value.get("required_downstream_policy")
    if not isinstance(policy, list) or any(not isinstance(item, str) or not item for item in policy):
        raise ValueError("quantum disruption report_certificate required_downstream_policy must be strings")
    for required_policy in REQUIRED_DOWNSTREAM_POLICY:
        if required_policy not in policy:
            raise ValueError(f"quantum disruption report_certificate missing downstream policy {required_policy}")
    claim_digest = value.get("claim_boundary_sha256")
    if not isinstance(claim_digest, str) or not _is_sha256(claim_digest):
        raise ValueError("quantum disruption report_certificate claim_boundary_sha256 must be a SHA-256 hex digest")
    if claim_digest != _payload_digest({"claim_boundary": payload.get("claim_boundary")}):
        raise ValueError("quantum disruption report_certificate claim_boundary_sha256 mismatch")
    content_digest = value.get("content_sha256")
    if not isinstance(content_digest, str) or not _is_sha256(content_digest):
        raise ValueError("quantum disruption report_certificate content_sha256 must be a SHA-256 hex digest")
    if content_digest != _report_content_digest(payload):
        raise ValueError("quantum disruption report_certificate content_sha256 mismatch")
    certificate_digest = value.get("certificate_sha256")
    if not isinstance(certificate_digest, str) or not _is_sha256(certificate_digest):
        raise ValueError("quantum disruption report_certificate certificate_sha256 must be a SHA-256 hex digest")
    if certificate_digest.lower() != _certificate_digest(value):
        raise ValueError("quantum disruption report_certificate certificate_sha256 mismatch")


def _risk_band(score: float) -> str:
    if score >= RISK_BAND_THRESHOLDS["high"]:
        return "high"
    if score >= RISK_BAND_THRESHOLDS["elevated"]:
        return "elevated"
    return "low"
