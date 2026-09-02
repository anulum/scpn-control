# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Portable reactor semantic admission decision tests

"""Public-surface tests for portable CONTROL admission decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from scpn_control.reactor_semantic_admission import (
    ADMISSION_SCHEMA,
    ADMISSION_SCHEMA_VERSION,
    MAX_ADMISSION_BYTES,
    REFUSAL_CODES,
    ReactorSemanticAdmissionDecision,
    ReactorSemanticAdmissionStatus,
    admission_decision_digest,
    admission_decision_from_bytes,
    admission_decision_to_bytes,
)


def _admitted() -> ReactorSemanticAdmissionDecision:
    return ReactorSemanticAdmissionDecision(
        decision=ReactorSemanticAdmissionStatus.ADMITTED_FOR_REVIEW,
        admitted=True,
        checked_at_ns=20_000_000,
        handoff_sha256="3" * 64,
        event_id="fce10-primary-0001",
        context_id="fusion.torax.circular_iter_scale_comparison",
        source_schema="scpn-fusion-core.torax-runtime-review-envelope.v1",
        source_revision="3" * 40,
        source_envelope_sha256="b" * 64,
        handoff_schema_version="1.0.0",
        u0_schema_version="1.0.0",
        registry_version="1.0.0",
        registry_digest="7" * 64,
        refusal_codes=(),
    )


def _rejected_decode() -> ReactorSemanticAdmissionDecision:
    return ReactorSemanticAdmissionDecision(
        decision=ReactorSemanticAdmissionStatus.REJECTED,
        admitted=False,
        checked_at_ns=20_000_000,
        handoff_sha256="f" * 64,
        event_id=None,
        context_id=None,
        source_schema=None,
        source_revision=None,
        source_envelope_sha256=None,
        handoff_schema_version=None,
        u0_schema_version=None,
        registry_version=None,
        registry_digest=None,
        refusal_codes=("handoff_decode_failed",),
    )


def _reseal(record: dict[str, object]) -> bytes:
    payload = record["payload"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    record["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize("decision", [_admitted(), _rejected_decode()])
def test_public_decision_bytes_are_canonical_and_round_trip(
    decision: ReactorSemanticAdmissionDecision,
) -> None:
    """Round-trip both admitted and decoder-failure decisions exactly."""
    encoded = admission_decision_to_bytes(decision)
    record = json.loads(encoded)

    assert admission_decision_from_bytes(encoded) == decision
    assert encoded == admission_decision_to_bytes(admission_decision_from_bytes(encoded))
    assert record["schema"] == ADMISSION_SCHEMA
    assert record["schema_version"] == ADMISSION_SCHEMA_VERSION
    assert record["payload"]["decision_digest"] == admission_decision_digest(decision)
    assert len(record["payload_sha256"]) == len(decision.decision_digest) == 64


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "must not be empty"),
        (b"\xff", "strict UTF-8"),
        (b"{", "valid JSON"),
        (b"[]", "must be an object"),
        (b"{" + b" " * MAX_ADMISSION_BYTES + b"}", "size limit"),
    ],
    ids=["empty", "invalid-utf8", "invalid-json", "non-object", "oversized"],
)
def test_public_decoder_rejects_invalid_portable_bytes(payload: bytes, message: str) -> None:
    """Reject empty, oversized, non-UTF-8, non-JSON, and non-object bytes."""
    with pytest.raises(ValueError, match=message):
        admission_decision_from_bytes(payload)


def test_public_decoder_rejects_non_bytes_and_duplicate_keys() -> None:
    """Reject non-byte input and duplicate object members."""
    with pytest.raises(TypeError, match="must be bytes"):
        admission_decision_from_bytes(cast(bytes, "{}"))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        admission_decision_from_bytes(b'{"payload":{},"payload":{}}')


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record.update(schema="drift"), "unsupported admission schema"),
        (
            lambda record: record.update(schema_version="2.0.0"),
            "unsupported admission schema version",
        ),
        (lambda record: record.update(extra=True), "unsupported or missing fields"),
        (
            lambda record: record["payload"].update(extra=True),
            "unsupported or missing fields",
        ),
        (lambda record: record.update(payload_sha256="0" * 64), "payload digest mismatch"),
    ],
)
def test_public_decoder_rejects_outer_contract_drift(
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    """Reject schema, field-set, and outer payload-seal drift."""
    record = json.loads(admission_decision_to_bytes(_admitted()))
    mutate(record)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match=message):
        admission_decision_from_bytes(payload)


def test_public_decoder_rejects_noncanonical_and_decision_digest_tamper() -> None:
    """Reject alternate bytes and a resealed false decision digest."""
    encoded = admission_decision_to_bytes(_admitted())
    with pytest.raises(ValueError, match="unique canonical encoding"):
        admission_decision_from_bytes(encoded + b"\n")

    record = json.loads(encoded)
    record["payload"]["decision_digest"] = "0" * 64
    unsigned = json.dumps(record["payload"], sort_keys=True, separators=(",", ":")).encode()
    record["payload_sha256"] = hashlib.sha256(unsigned).hexdigest()
    tampered = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="decision digest mismatch"):
        admission_decision_from_bytes(tampered)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"checked_at_ns": -1}, "non-negative integer"),
        ({"checked_at_ns": True}, "non-negative integer"),
        ({"checked_at_ns": 1.5}, "non-negative integer"),
        ({"decision": "admitted_for_review"}, "ReactorSemanticAdmissionStatus"),
        ({"admitted": 1}, "admitted must be boolean"),
        ({"review_only": 1}, "review_only must be boolean"),
        ({"actionable": 0}, "actionable must be boolean"),
        ({"review_only": False}, "review-only"),
        ({"actionable": True}, "review-only"),
        ({"admitted": False}, "admitted must match"),
        ({"refusal_codes": ["handoff_decode_failed"]}, "tuple of strings"),
        ({"refusal_codes": (1,)}, "tuple of strings"),
        ({"refusal_codes": ("handoff_decode_failed", "handoff_decode_failed")}, "sorted and unique"),
        ({"refusal_codes": ("not_closed",)}, "unknown code"),
        ({"handoff_sha256": "x"}, "handoff_sha256"),
        ({"source_envelope_sha256": "x"}, "source_envelope_sha256"),
        ({"registry_digest": "x"}, "registry_digest"),
        ({"source_revision": "x"}, "source_revision"),
        ({"event_id": None}, "every decoded identity"),
        ({"event_id": ""}, "event_id must be non-empty text"),
        ({"context_id": ""}, "context_id must be non-empty text"),
        ({"source_schema": ""}, "source_schema must be non-empty text"),
        ({"handoff_schema_version": ""}, "handoff_schema_version must be non-empty text"),
        ({"u0_schema_version": ""}, "u0_schema_version must be non-empty text"),
        ({"registry_version": ""}, "registry_version must be non-empty text"),
    ],
)
def test_decision_model_rejects_inconsistent_state(
    change: dict[str, object],
    message: str,
) -> None:
    """Reject inconsistent decision object construction."""
    with pytest.raises(ValueError, match=message):
        replace(_admitted(), **change)


def test_decoder_failure_rejects_guessed_identity_and_empty_refusal() -> None:
    """Require a refusal and forbid guessed upstream identity after failure."""
    with pytest.raises(ValueError, match="cannot guess"):
        replace(_rejected_decode(), event_id="guessed")
    with pytest.raises(ValueError, match="require one"):
        replace(_rejected_decode(), refusal_codes=())


def test_refusal_vocabulary_is_closed_and_descriptive() -> None:
    """Keep refusal codes finite, lowercase, and action-boundary aware."""
    assert "handoff_decode_failed" in REFUSAL_CODES
    assert "handoff_digest_mismatch" in REFUSAL_CODES
    assert "observable_validity_unobservable" in REFUSAL_CODES
    assert all(code == code.lower() and " " not in code for code in REFUSAL_CODES)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("refusal_codes", "not-a-list", "list of strings"),
        ("refusal_codes", [1], "list of strings"),
        ("checked_at_ns", True, "must be an integer"),
        ("admitted", "true", "admitted must be boolean"),
        ("review_only", "true", "review_only must be boolean"),
        ("actionable", 0, "actionable must be boolean"),
        ("decision", "unknown", "unsupported admission decision"),
        ("decision_digest", "bad", "decision_digest"),
        ("event_id", 1, "event_id must be non-empty text"),
        ("context_id", "", "context_id must be non-empty text"),
    ],
)
def test_public_decoder_rejects_invalid_decision_field_types(
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject invalid field types even when the outer payload is resealed."""
    record = json.loads(admission_decision_to_bytes(_admitted()))
    record["payload"][field] = value
    with pytest.raises(ValueError, match=message):
        admission_decision_from_bytes(_reseal(record))


def test_public_decoder_requires_payload_object() -> None:
    """Reject a non-object payload before interpreting decision fields."""
    record = json.loads(admission_decision_to_bytes(_admitted()))
    record["payload"] = []
    record["payload_sha256"] = hashlib.sha256(b"[]").hexdigest()
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="admission payload must be an object"):
        admission_decision_from_bytes(encoded)
