# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Regime-assessment admission decision tests

"""Public byte-contract tests for review-only assessment decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

from scpn_control.reactor_semantic_admission import (
    MAX_REGIME_ASSESSMENT_ADMISSION_BYTES,
    REGIME_ASSESSMENT_ADMISSION_SCHEMA,
    REGIME_ASSESSMENT_ADMISSION_VERSION,
    REGIME_ASSESSMENT_REFUSAL_CODES,
    ReactorRegimeAssessmentAdmissionDecision,
    ReactorRegimeAssessmentAdmissionStatus,
    regime_assessment_admission_decision_digest,
    regime_assessment_admission_decision_from_bytes,
    regime_assessment_admission_decision_to_bytes,
)


def _admitted() -> ReactorRegimeAssessmentAdmissionDecision:
    return ReactorRegimeAssessmentAdmissionDecision(
        decision=ReactorRegimeAssessmentAdmissionStatus.ADMITTED_FOR_REVIEW,
        admitted=True,
        checked_at_ns=0,
        assessment_sha256="a" * 64,
        assessment_id="spo.assessment.abstaining.0123456789abcdef01234567",
        reactor_context_id="spo.mif.frc_compression.0123456789abcdef01234567",
        configuration="frc_compression_mif",
        event_id="mif-merge-compression-0001",
        producer_project="SCPN-PHASE-ORCHESTRATOR",
        producer_revision="1" * 40,
        producer_artifact_sha256="2" * 64,
        source_project="SCPN-MIF-CORE",
        source_revision="3" * 40,
        source_handoff_schema="scpn-phase-orchestrator.mif-merge-compression-handoff.v1",
        source_handoff_sha256="4" * 64,
        assessment_schema_version="1.0.0",
        registry_custody_sha256="5" * 64,
        clock_custody_sha256="6" * 64,
        axis_custody_sha256="7" * 64,
        refusal_codes=(),
    )


def _decode_rejection() -> ReactorRegimeAssessmentAdmissionDecision:
    return ReactorRegimeAssessmentAdmissionDecision(
        decision=ReactorRegimeAssessmentAdmissionStatus.REJECTED,
        admitted=False,
        checked_at_ns=0,
        assessment_sha256="f" * 64,
        assessment_id=None,
        reactor_context_id=None,
        configuration=None,
        event_id=None,
        producer_project=None,
        producer_revision=None,
        producer_artifact_sha256=None,
        source_project=None,
        source_revision=None,
        source_handoff_schema=None,
        source_handoff_sha256=None,
        assessment_schema_version=None,
        registry_custody_sha256=None,
        clock_custody_sha256=None,
        axis_custody_sha256=None,
        refusal_codes=("assessment_decode_failed",),
    )


def _reseal(record: dict[str, object]) -> bytes:
    payload = record["payload"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    record["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize("decision", [_admitted(), _decode_rejection()])
def test_decision_bytes_are_canonical_digest_sealed_and_round_trip(
    decision: ReactorRegimeAssessmentAdmissionDecision,
) -> None:
    """Round-trip admitted and undecodable-input decisions without drift."""
    encoded = regime_assessment_admission_decision_to_bytes(decision)
    record = json.loads(encoded)

    assert regime_assessment_admission_decision_from_bytes(encoded) == decision
    assert record["schema"] == REGIME_ASSESSMENT_ADMISSION_SCHEMA
    assert record["schema_version"] == REGIME_ASSESSMENT_ADMISSION_VERSION
    assert record["payload"]["decision_digest"] == decision.decision_digest
    assert len(record["payload_sha256"]) == 64
    assert regime_assessment_admission_decision_digest(decision) == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "must not be empty"),
        (b"\xff", "strict UTF-8"),
        (b"{", "valid JSON"),
        (b"[]", "unsupported or missing fields"),
        (b"{" + b" " * MAX_REGIME_ASSESSMENT_ADMISSION_BYTES + b"}", "size limit"),
    ],
    ids=["empty", "invalid-utf8", "invalid-json", "non-object", "oversized"],
)
def test_decoder_rejects_invalid_bytes(payload: bytes, message: str) -> None:
    """Reject empty, malformed, non-object, and oversized envelopes."""
    with pytest.raises(ValueError, match=message):
        regime_assessment_admission_decision_from_bytes(payload)


def test_decoder_rejects_non_bytes_and_duplicate_keys() -> None:
    """Require byte ingress and unique JSON object members."""
    with pytest.raises(TypeError, match="must be bytes"):
        regime_assessment_admission_decision_from_bytes(cast(bytes, "{}"))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        regime_assessment_admission_decision_from_bytes(b'{"payload":{},"payload":{}}')


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record.update(schema="drift"), "unsupported regime assessment admission schema"),
        (
            lambda record: record.update(schema_version="2.0.0"),
            "unsupported regime assessment admission schema version",
        ),
        (lambda record: record.update(extra=True), "unsupported or missing fields"),
        (lambda record: record["payload"].update(extra=True), "unsupported or missing fields"),
        (lambda record: record.update(payload_sha256="0" * 64), "payload digest mismatch"),
        (lambda record: record.update(payload_sha256="bad"), "payload_sha256"),
    ],
)
def test_decoder_rejects_envelope_drift(
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    """Reject schema, field-set, and envelope-seal drift."""
    record = json.loads(regime_assessment_admission_decision_to_bytes(_admitted()))
    mutate(record)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match=message):
        regime_assessment_admission_decision_from_bytes(encoded)


def test_decoder_rejects_noncanonical_and_inner_digest_tamper() -> None:
    """Reject alternate encodings and resealed decision mutation."""
    encoded = regime_assessment_admission_decision_to_bytes(_admitted())
    with pytest.raises(ValueError, match="unique canonical encoding"):
        regime_assessment_admission_decision_from_bytes(encoded + b"\n")

    record = json.loads(encoded)
    record["payload"]["decision_digest"] = "0" * 64
    with pytest.raises(ValueError, match="decision digest mismatch"):
        regime_assessment_admission_decision_from_bytes(_reseal(record))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"decision": "admitted_for_review"}, "ReactorRegimeAssessmentAdmissionStatus"),
        ({"admitted": 1}, "admitted must be boolean"),
        ({"review_only": 1}, "review_only must be boolean"),
        ({"actionable": 0}, "actionable must be boolean"),
        ({"review_only": False}, "review-only"),
        ({"actionable": True}, "review-only"),
        ({"admitted": False}, "admitted must match"),
        ({"checked_at_ns": True}, "non-negative integer"),
        ({"checked_at_ns": 1.5}, "non-negative integer"),
        ({"checked_at_ns": -1}, "non-negative integer"),
        ({"refusal_codes": ["assessment_decode_failed"]}, "tuple of strings"),
        ({"refusal_codes": (1,)}, "tuple of strings"),
        (
            {"refusal_codes": ("assessment_decode_failed", "assessment_decode_failed")},
            "sorted and unique",
        ),
        ({"refusal_codes": ("not_closed",)}, "unknown code"),
        ({"assessment_sha256": "bad"}, "assessment_sha256"),
        ({"producer_artifact_sha256": "bad"}, "producer_artifact_sha256"),
        ({"source_handoff_sha256": "bad"}, "source_handoff_sha256"),
        ({"registry_custody_sha256": "bad"}, "registry_custody_sha256"),
        ({"clock_custody_sha256": "bad"}, "clock_custody_sha256"),
        ({"axis_custody_sha256": "bad"}, "axis_custody_sha256"),
        ({"producer_revision": "bad"}, "producer_revision"),
        ({"producer_revision": 1}, "producer_revision"),
        ({"source_revision": "bad"}, "source_revision"),
        ({"assessment_id": ""}, "assessment_id must be non-empty text"),
        ({"assessment_id": None}, "every decoded identity"),
    ],
)
def test_decision_model_rejects_inconsistent_state(
    change: dict[str, object],
    message: str,
) -> None:
    """Reject ambiguous, authority-bearing, or incomplete decisions."""
    with pytest.raises(ValueError, match=message):
        replace(_admitted(), **cast(Any, change))


def test_rejection_requires_reason_and_decoder_failure_forbids_guesses() -> None:
    """Make every rejection explicit and keep failed-decode identity empty."""
    with pytest.raises(ValueError, match="require one"):
        replace(_decode_rejection(), refusal_codes=())
    with pytest.raises(ValueError, match="cannot guess"):
        replace(_decode_rejection(), event_id="guessed")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("refusal_codes", "bad", "list of strings"),
        ("refusal_codes", [1], "list of strings"),
        ("checked_at_ns", True, "must be an integer"),
        ("admitted", "true", "must be boolean"),
        ("review_only", "true", "must be boolean"),
        ("actionable", 0, "must be boolean"),
        ("decision", "unknown", "unknown regime assessment admission decision"),
        ("assessment_id", 1, "assessment_id must be non-empty text"),
        ("assessment_id", "", "assessment_id must be non-empty text"),
        ("decision_digest", 1, "decision digest mismatch"),
    ],
)
def test_decoder_rejects_invalid_payload_fields(field: str, value: object, message: str) -> None:
    """Reject invalid payload fields after a valid outer reseal."""
    record = json.loads(regime_assessment_admission_decision_to_bytes(_admitted()))
    record["payload"][field] = value
    with pytest.raises(ValueError, match=message):
        regime_assessment_admission_decision_from_bytes(_reseal(record))


def test_refusal_vocabulary_is_closed_and_non_actuating() -> None:
    """Keep refusal vocabulary finite and explicitly non-actuating."""
    assert "assessment_decode_failed" in REGIME_ASSESSMENT_REFUSAL_CODES
    assert "assessment_not_abstaining" in REGIME_ASSESSMENT_REFUSAL_CODES
    assert "assessment_schema_version_mismatch" in REGIME_ASSESSMENT_REFUSAL_CODES
    assert "outside_common_validity" in REGIME_ASSESSMENT_REFUSAL_CODES
    assert all(code == code.lower() and " " not in code for code in REGIME_ASSESSMENT_REFUSAL_CODES)
