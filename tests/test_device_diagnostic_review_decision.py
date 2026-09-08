# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Device diagnostic review decision tests.

"""Canonical byte-contract tests for non-authorising review decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

from scpn_control.reactor_semantic_admission import (
    DEVICE_DIAGNOSTIC_REVIEW_DECISION_SCHEMA,
    DEVICE_DIAGNOSTIC_REVIEW_DECISION_VERSION,
    DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES,
    MAX_DEVICE_DIAGNOSTIC_REVIEW_DECISION_BYTES,
    SPO_CONTRACT_MODULE_SHA256,
    SPO_DISTRIBUTION_VERSION,
    SPO_RELEASE_WHEEL_SHA256,
    SPO_REVIEW_SCHEMA,
    SPO_REVIEW_SCHEMA_VERSION,
    TOKAMAK_CLOCK_CUSTODY_SHA256,
    TOKAMAK_CONFIGURATIONS,
    TOKAMAK_REVIEW_ID,
    TOKAMAK_REVIEW_SHA256,
    TOKAMAK_SOURCE_ARTIFACT_SHA256,
    TOKAMAK_SOURCE_ENVELOPE_SHA256,
    TOKAMAK_SOURCE_MANIFEST_SHA256,
    TOKAMAK_SOURCE_PLAN_SHA256,
    TOKAMAK_SOURCE_PROJECT,
    TOKAMAK_SOURCE_REVISION,
    DeviceDiagnosticReviewAdmissionDecision,
    DeviceDiagnosticReviewAdmissionStatus,
    device_diagnostic_review_decision_digest,
    device_diagnostic_review_decision_from_bytes,
    device_diagnostic_review_decision_to_bytes,
)


def _accepted() -> DeviceDiagnosticReviewAdmissionDecision:
    return DeviceDiagnosticReviewAdmissionDecision(
        decision=DeviceDiagnosticReviewAdmissionStatus.ACCEPTED_FOR_REVIEW,
        accepted_for_review=True,
        review_sha256=TOKAMAK_REVIEW_SHA256,
        review_id=TOKAMAK_REVIEW_ID,
        review_schema=SPO_REVIEW_SCHEMA,
        review_schema_version=SPO_REVIEW_SCHEMA_VERSION,
        source_project=TOKAMAK_SOURCE_PROJECT,
        source_revision=TOKAMAK_SOURCE_REVISION,
        source_artifact_sha256=TOKAMAK_SOURCE_ARTIFACT_SHA256,
        source_manifest_sha256=TOKAMAK_SOURCE_MANIFEST_SHA256,
        source_envelope_sha256=TOKAMAK_SOURCE_ENVELOPE_SHA256,
        source_plan_sha256=TOKAMAK_SOURCE_PLAN_SHA256,
        configurations=TOKAMAK_CONFIGURATIONS,
        clock_custody_sha256=TOKAMAK_CLOCK_CUSTODY_SHA256,
        spo_distribution_version=SPO_DISTRIBUTION_VERSION,
        spo_release_wheel_sha256=SPO_RELEASE_WHEEL_SHA256,
        spo_contract_module_sha256=SPO_CONTRACT_MODULE_SHA256,
        refusal_codes=(),
    )


def _decode_rejection() -> DeviceDiagnosticReviewAdmissionDecision:
    return DeviceDiagnosticReviewAdmissionDecision(
        decision=DeviceDiagnosticReviewAdmissionStatus.REJECTED,
        accepted_for_review=False,
        review_sha256="f" * 64,
        review_id=None,
        review_schema=SPO_REVIEW_SCHEMA,
        review_schema_version=SPO_REVIEW_SCHEMA_VERSION,
        source_project=None,
        source_revision=None,
        source_artifact_sha256=None,
        source_manifest_sha256=None,
        source_envelope_sha256=None,
        source_plan_sha256=None,
        configurations=(),
        clock_custody_sha256=None,
        spo_distribution_version=SPO_DISTRIBUTION_VERSION,
        spo_release_wheel_sha256=SPO_RELEASE_WHEEL_SHA256,
        spo_contract_module_sha256=SPO_CONTRACT_MODULE_SHA256,
        refusal_codes=("review_decode_failed",),
    )


def _reseal(record: dict[str, Any]) -> bytes:
    payload = record["payload"]
    canonical = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    record["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


@pytest.mark.parametrize("decision", [_accepted(), _decode_rejection()])
def test_decision_round_trips_with_both_digest_seals(
    decision: DeviceDiagnosticReviewAdmissionDecision,
) -> None:
    """Round-trip accepted and failed-decode decisions without drift."""
    encoded = device_diagnostic_review_decision_to_bytes(decision)
    record = json.loads(encoded)

    assert device_diagnostic_review_decision_from_bytes(encoded) == decision
    assert record["schema"] == DEVICE_DIAGNOSTIC_REVIEW_DECISION_SCHEMA
    assert record["schema_version"] == DEVICE_DIAGNOSTIC_REVIEW_DECISION_VERSION
    assert record["payload"]["decision_digest"] == decision.decision_digest
    assert len(record["payload_sha256"]) == 64
    assert device_diagnostic_review_decision_digest(decision) == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "must not be empty"),
        (b"\xff", "strict UTF-8"),
        (b"{", "valid JSON"),
        (b"[]", "must be an object"),
        (
            b"{" + b" " * MAX_DEVICE_DIAGNOSTIC_REVIEW_DECISION_BYTES + b"}",
            "size limit",
        ),
    ],
    ids=["empty", "invalid-utf8", "invalid-json", "non-object", "oversized"],
)
def test_decoder_rejects_invalid_byte_envelopes(payload: bytes, message: str) -> None:
    """Reject empty, malformed, non-object and oversized byte envelopes."""
    with pytest.raises(ValueError, match=message):
        device_diagnostic_review_decision_from_bytes(payload)


def test_decoder_rejects_non_bytes_and_duplicate_keys() -> None:
    """Require byte ingress and unique JSON object members."""
    with pytest.raises(TypeError, match="must be bytes"):
        device_diagnostic_review_decision_from_bytes(cast(bytes, "{}"))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        device_diagnostic_review_decision_from_bytes(b'{"payload":{},"payload":{}}')


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record.update(schema="drift"),
            "unsupported device diagnostic review decision schema",
        ),
        (
            lambda record: record.update(schema_version="2.0.0"),
            "unsupported device diagnostic review decision schema version",
        ),
        (
            lambda record: record.update(extra=True),
            "unsupported or missing fields",
        ),
        (
            lambda record: record["payload"].update(extra=True),
            "unsupported or missing fields",
        ),
        (
            lambda record: record.update(payload_sha256="0" * 64),
            "payload digest mismatch",
        ),
    ],
)
def test_decoder_rejects_schema_field_and_outer_digest_drift(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Reject every outer envelope and field-set mutation."""
    record = cast("dict[str, Any]", json.loads(device_diagnostic_review_decision_to_bytes(_accepted())))
    mutate(record)
    encoded = json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(ValueError, match=message):
        device_diagnostic_review_decision_from_bytes(encoded)


def test_decoder_rejects_noncanonical_and_inner_digest_tamper() -> None:
    """Reject alternate encodings and resealed decision mutation."""
    encoded = device_diagnostic_review_decision_to_bytes(_accepted())
    with pytest.raises(ValueError, match="not canonical"):
        device_diagnostic_review_decision_from_bytes(encoded + b"\n")

    record = cast("dict[str, Any]", json.loads(encoded))
    record["payload"]["decision_digest"] = "0" * 64
    with pytest.raises(ValueError, match="decision digest mismatch"):
        device_diagnostic_review_decision_from_bytes(_reseal(record))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"decision": "accepted_for_review"}, "DeviceDiagnosticReviewAdmissionStatus"),
        ({"accepted_for_review": 1}, "accepted_for_review must be boolean"),
        ({"review_only": 1}, "review_only must be boolean"),
        ({"evidence_claimed": 0}, "evidence_claimed must be boolean"),
        ({"observation_claimed": 0}, "observation_claimed must be boolean"),
        ({"measurement_claimed": 0}, "measurement_claimed must be boolean"),
        ({"facility_binding_claimed": 0}, "facility_binding_claimed must be boolean"),
        ({"classification_performed": 0}, "classification_performed must be boolean"),
        ({"semantic_ingress_declared": 0}, "semantic_ingress_declared must be boolean"),
        ({"control_intent_created": 0}, "control_intent_created must be boolean"),
        ({"actionable": 0}, "actionable must be boolean"),
        ({"execution_authorised": 0}, "execution_authorised must be boolean"),
        ({"actuation_authorised": 0}, "actuation_authorised must be boolean"),
        ({"review_only": False}, "cannot create operational authority"),
        ({"evidence_claimed": True}, "cannot create operational authority"),
        ({"observation_claimed": True}, "cannot create operational authority"),
        ({"measurement_claimed": True}, "cannot create operational authority"),
        ({"facility_binding_claimed": True}, "cannot create operational authority"),
        ({"classification_performed": True}, "cannot create operational authority"),
        ({"semantic_ingress_declared": True}, "cannot create operational authority"),
        ({"control_intent_created": True}, "cannot create operational authority"),
        ({"actionable": True}, "cannot create operational authority"),
        ({"execution_authorised": True}, "cannot create operational authority"),
        ({"actuation_authorised": True}, "cannot create operational authority"),
        ({"accepted_for_review": False}, "must match decision"),
        ({"refusal_codes": ["review_decode_failed"]}, "tuple of strings"),
        ({"refusal_codes": (1,)}, "tuple of strings"),
        (
            {"refusal_codes": ("review_decode_failed", "review_decode_failed")},
            "sorted and unique",
        ),
        ({"refusal_codes": ("not_closed",)}, "unknown code"),
        ({"review_sha256": "bad"}, "review_sha256"),
        ({"review_id": "bad"}, "review_id"),
        ({"source_revision": "bad"}, "source_revision"),
        ({"source_artifact_sha256": "bad"}, "source_artifact_sha256"),
        ({"source_manifest_sha256": "bad"}, "source_manifest_sha256"),
        ({"source_envelope_sha256": "bad"}, "source_envelope_sha256"),
        ({"source_plan_sha256": "bad"}, "source_plan_sha256"),
        ({"clock_custody_sha256": "bad"}, "clock_custody_sha256"),
        ({"spo_release_wheel_sha256": "bad"}, "spo_release_wheel_sha256"),
        ({"spo_contract_module_sha256": "bad"}, "spo_contract_module_sha256"),
        ({"review_schema": ""}, "review_schema"),
        ({"configurations": ["conventional_tokamak"]}, "tuple of non-empty strings"),
        ({"configurations": ("",)}, "tuple of non-empty strings"),
        (
            {"configurations": ("spherical_tokamak", "conventional_tokamak")},
            "sorted and unique",
        ),
        ({"review_sha256": None}, "accepted decisions require every decoded identity"),
        ({"review_id": None}, "accepted decisions require every decoded identity"),
        (
            {"spo_contract_module_sha256": None},
            "accepted decisions require every decoded identity",
        ),
        ({"configurations": ()}, "accepted decisions require every decoded identity"),
    ],
)
def test_decision_model_rejects_inconsistent_state(
    change: dict[str, object],
    message: str,
) -> None:
    """Reject unsealed, ambiguous and authority-bearing decision states."""
    with pytest.raises(ValueError, match=message):
        replace(_accepted(), **cast(Any, change))


def test_rejection_requires_reason_and_decode_failure_forbids_guesses() -> None:
    """Require one refusal and forbid inferred source identity after decode failure."""
    with pytest.raises(ValueError, match="require one"):
        replace(_decode_rejection(), refusal_codes=())
    with pytest.raises(ValueError, match="cannot guess"):
        replace(_decode_rejection(), source_project=TOKAMAK_SOURCE_PROJECT)
    with pytest.raises(ValueError, match="cannot guess"):
        replace(_decode_rejection(), review_id=TOKAMAK_REVIEW_ID)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("refusal_codes", "bad", "list of strings"),
        ("refusal_codes", [1], "list of strings"),
        ("configurations", "bad", "list of strings"),
        ("configurations", [1], "list of strings"),
        ("accepted_for_review", "true", "must be boolean"),
        ("review_only", "true", "must be boolean"),
        ("actionable", 0, "must be boolean"),
        ("decision", "unknown", "unknown device diagnostic review decision"),
        ("source_project", 1, "source_project must be non-empty text"),
        ("source_project", "", "source_project must be non-empty text"),
        ("decision_digest", 1, "decision digest mismatch"),
    ],
)
def test_decoder_rejects_invalid_payload_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject invalid field types after a valid outer reseal."""
    record = cast("dict[str, Any]", json.loads(device_diagnostic_review_decision_to_bytes(_accepted())))
    record["payload"][field] = value
    with pytest.raises(ValueError, match=message):
        device_diagnostic_review_decision_from_bytes(_reseal(record))


def test_refusal_vocabulary_is_closed_and_authority_free() -> None:
    """Keep the refusal set finite and the decision model non-authorising."""
    assert "review_decode_failed" in DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES
    assert "source_project_mismatch" in DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES
    assert "clock_custody_mismatch" in DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES
    assert "authority_escalation" in DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES
    assert all(code == code.lower() and " " not in code for code in DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES)
