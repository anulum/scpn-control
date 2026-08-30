# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Portable reactor semantic admission decisions

"""Canonical, digest-sealed review-admission decisions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, cast

ADMISSION_SCHEMA: Final = "scpn-control.reactor-semantic-admission.v1"
ADMISSION_SCHEMA_VERSION: Final = "1.0.0"
MAX_ADMISSION_BYTES: Final = 1024 * 1024

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_OUTER_KEYS = frozenset({"payload", "payload_sha256", "schema", "schema_version"})
_PAYLOAD_KEYS = frozenset(
    {
        "actionable",
        "admitted",
        "checked_at_ns",
        "context_id",
        "decision",
        "decision_digest",
        "event_id",
        "handoff_schema_version",
        "handoff_sha256",
        "refusal_codes",
        "registry_digest",
        "registry_version",
        "review_only",
        "source_envelope_sha256",
        "source_revision",
        "source_schema",
        "u0_schema_version",
    }
)


class ReactorSemanticAdmissionStatus(str, Enum):
    """Outcome of CONTROL's non-actuating semantic review gate."""

    ADMITTED_FOR_REVIEW = "admitted_for_review"
    REJECTED = "rejected"


REFUSAL_CODES: Final = frozenset(
    {
        "calibration_id_not_allowed",
        "calibration_stale",
        "clock_reference_mismatch",
        "evidence_from_future",
        "evidence_stale",
        "handoff_decode_failed",
        "handoff_digest_mismatch",
        "observable_degradation_not_allowed",
        "observable_quality_flags_not_allowed",
        "observable_quality_flags_undeclared",
        "observable_quality_invalid",
        "observable_quality_unknown",
        "observable_validity_invalid",
        "observable_validity_out_of_distribution",
        "observable_validity_stale",
        "observable_validity_unknown",
        "observable_validity_unobservable",
        "provenance_chain_mismatch",
        "source_envelope_digest_mismatch",
        "source_revision_mismatch",
        "source_schema_mismatch",
        "transfer_function_id_not_allowed",
    }
)


@dataclass(frozen=True, slots=True)
class ReactorSemanticAdmissionDecision:
    """Deterministic result of admitting one SPO handoff for review.

    Parameters
    ----------
    decision:
        Review admission state.
    admitted:
        Whether the evidence passed every configured review check.
    checked_at_ns:
        Explicit caller-supplied reference timestamp; never wall time.
    handoff_sha256:
        SHA-256 of the exact portable input bytes, when bytes were supplied.
    event_id, context_id, source_schema, source_revision:
        Decoded upstream identities, or ``None`` when decoding failed.
    source_envelope_sha256:
        Digest of the embedded canonical FUSION envelope, when decoded.
    handoff_schema_version, u0_schema_version:
        Decoded SPO and U0 contract versions, when decoded.
    registry_version, registry_digest:
        Decoded SPO registry identity, when decoded.
    refusal_codes:
        Sorted, unique members of :data:`REFUSAL_CODES`.
    """

    decision: ReactorSemanticAdmissionStatus
    admitted: bool
    checked_at_ns: int
    handoff_sha256: str | None
    event_id: str | None
    context_id: str | None
    source_schema: str | None
    source_revision: str | None
    source_envelope_sha256: str | None
    handoff_schema_version: str | None
    u0_schema_version: str | None
    registry_version: str | None
    registry_digest: str | None
    refusal_codes: tuple[str, ...]
    review_only: bool = True
    actionable: bool = False

    def __post_init__(self) -> None:
        """Validate the closed decision contract."""
        if not isinstance(self.decision, ReactorSemanticAdmissionStatus):
            raise ValueError("decision must be a ReactorSemanticAdmissionStatus")
        for bool_field, bool_value in (
            ("admitted", self.admitted),
            ("review_only", self.review_only),
            ("actionable", self.actionable),
        ):
            if not isinstance(bool_value, bool):
                raise ValueError(f"{bool_field} must be boolean")
        if isinstance(self.checked_at_ns, bool) or not isinstance(self.checked_at_ns, int) or self.checked_at_ns < 0:
            raise ValueError("checked_at_ns must be a non-negative integer")
        if self.review_only is not True or self.actionable is not False:
            raise ValueError("reactor semantic admission is review-only and non-actionable")
        if self.admitted != (self.decision is ReactorSemanticAdmissionStatus.ADMITTED_FOR_REVIEW):
            raise ValueError("admitted must match decision")
        if not isinstance(self.refusal_codes, tuple) or any(not isinstance(code, str) for code in self.refusal_codes):
            raise ValueError("refusal_codes must be a tuple of strings")
        if tuple(sorted(set(self.refusal_codes))) != self.refusal_codes:
            raise ValueError("refusal_codes must be sorted and unique")
        if not set(self.refusal_codes) <= REFUSAL_CODES:
            raise ValueError("refusal_codes contain an unknown code")
        if self.admitted == bool(self.refusal_codes):
            raise ValueError("admitted decisions have no refusals; rejected decisions require one")
        optional_text_fields = (
            ("handoff_sha256", self.handoff_sha256),
            ("event_id", self.event_id),
            ("context_id", self.context_id),
            ("source_schema", self.source_schema),
            ("source_revision", self.source_revision),
            ("source_envelope_sha256", self.source_envelope_sha256),
            ("handoff_schema_version", self.handoff_schema_version),
            ("u0_schema_version", self.u0_schema_version),
            ("registry_version", self.registry_version),
            ("registry_digest", self.registry_digest),
        )
        for text_field, text_value in optional_text_fields:
            _optional_text(text_value, text_field)
        _optional_digest(self.handoff_sha256, "handoff_sha256")
        _optional_digest(self.source_envelope_sha256, "source_envelope_sha256")
        _optional_digest(self.registry_digest, "registry_digest")
        if self.source_revision is not None and _HEX_40.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a lowercase 40-character Git commit")
        identity_fields = (
            self.event_id,
            self.context_id,
            self.source_schema,
            self.source_revision,
            self.source_envelope_sha256,
            self.handoff_schema_version,
            self.u0_schema_version,
            self.registry_version,
            self.registry_digest,
        )
        if self.admitted and any(value is None for value in identity_fields):
            raise ValueError("admitted decisions require every decoded identity")
        if "handoff_decode_failed" in self.refusal_codes and any(value is not None for value in identity_fields):
            raise ValueError("decoder failures cannot guess upstream identities")

    @property
    def decision_digest(self) -> str:
        """Return the digest of the decision payload with an empty seal field."""
        return _payload_decision_digest(_payload_record(self, decision_digest=None))


def admission_decision_to_bytes(decision: ReactorSemanticAdmissionDecision) -> bytes:
    """Encode a decision as unique canonical UTF-8 JSON bytes."""
    payload = _payload_record(decision, decision_digest=decision.decision_digest)
    record = {
        "payload": payload,
        "payload_sha256": _digest_record(payload),
        "schema": ADMISSION_SCHEMA,
        "schema_version": ADMISSION_SCHEMA_VERSION,
    }
    encoded = _canonical_json(record).encode("utf-8")
    return encoded


def admission_decision_from_bytes(payload: bytes) -> ReactorSemanticAdmissionDecision:
    """Decode only canonical, duplicate-key-free admission bytes."""
    if not isinstance(payload, bytes):
        raise TypeError("admission payload must be bytes")
    if not payload:
        raise ValueError("admission payload must not be empty")
    if len(payload) > MAX_ADMISSION_BYTES:
        raise ValueError("admission payload exceeds the portable size limit")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("admission payload must be strict UTF-8") from exc
    try:
        raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("admission payload must be valid JSON") from exc
    record = _mapping(raw, "admission record")
    _exact_keys(record, _OUTER_KEYS, "admission record")
    if record["schema"] != ADMISSION_SCHEMA:
        raise ValueError("unsupported admission schema")
    if record["schema_version"] != ADMISSION_SCHEMA_VERSION:
        raise ValueError("unsupported admission schema version")
    body = _mapping(record["payload"], "admission payload")
    _exact_keys(body, _PAYLOAD_KEYS, "admission payload")
    if record["payload_sha256"] != _digest_record(body):
        raise ValueError("admission payload digest mismatch")
    decision = _decision_from_payload(body)
    if body["decision_digest"] != decision.decision_digest:
        raise ValueError("admission decision digest mismatch")
    if admission_decision_to_bytes(decision) != payload:
        raise ValueError("admission payload is not the unique canonical encoding")
    return decision


def admission_decision_digest(decision: ReactorSemanticAdmissionDecision) -> str:
    """Return the CONTROL-owned decision digest."""
    return decision.decision_digest


def _payload_record(
    decision: ReactorSemanticAdmissionDecision,
    *,
    decision_digest: str | None,
) -> dict[str, object]:
    return {
        "actionable": decision.actionable,
        "admitted": decision.admitted,
        "checked_at_ns": decision.checked_at_ns,
        "context_id": decision.context_id,
        "decision": decision.decision.value,
        "decision_digest": decision_digest,
        "event_id": decision.event_id,
        "handoff_schema_version": decision.handoff_schema_version,
        "handoff_sha256": decision.handoff_sha256,
        "refusal_codes": list(decision.refusal_codes),
        "registry_digest": decision.registry_digest,
        "registry_version": decision.registry_version,
        "review_only": decision.review_only,
        "source_envelope_sha256": decision.source_envelope_sha256,
        "source_revision": decision.source_revision,
        "source_schema": decision.source_schema,
        "u0_schema_version": decision.u0_schema_version,
    }


def _decision_from_payload(payload: Mapping[str, object]) -> ReactorSemanticAdmissionDecision:
    refusal_codes = payload["refusal_codes"]
    if not isinstance(refusal_codes, list) or any(not isinstance(item, str) for item in refusal_codes):
        raise ValueError("refusal_codes must be a list of strings")
    checked_at_ns = payload["checked_at_ns"]
    if isinstance(checked_at_ns, bool) or not isinstance(checked_at_ns, int):
        raise ValueError("checked_at_ns must be an integer")
    admitted = _boolean(payload["admitted"], "admitted")
    review_only = _boolean(payload["review_only"], "review_only")
    actionable = _boolean(payload["actionable"], "actionable")
    try:
        status = ReactorSemanticAdmissionStatus(_text(payload["decision"], "decision"))
    except ValueError as exc:
        raise ValueError("unsupported admission decision") from exc
    digest = payload["decision_digest"]
    if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
        raise ValueError("decision_digest must be a lowercase SHA-256 digest")
    return ReactorSemanticAdmissionDecision(
        decision=status,
        admitted=admitted,
        checked_at_ns=checked_at_ns,
        handoff_sha256=_optional_text(payload["handoff_sha256"], "handoff_sha256"),
        event_id=_optional_text(payload["event_id"], "event_id"),
        context_id=_optional_text(payload["context_id"], "context_id"),
        source_schema=_optional_text(payload["source_schema"], "source_schema"),
        source_revision=_optional_text(payload["source_revision"], "source_revision"),
        source_envelope_sha256=_optional_text(payload["source_envelope_sha256"], "source_envelope_sha256"),
        handoff_schema_version=_optional_text(payload["handoff_schema_version"], "handoff_schema_version"),
        u0_schema_version=_optional_text(payload["u0_schema_version"], "u0_schema_version"),
        registry_version=_optional_text(payload["registry_version"], "registry_version"),
        registry_digest=_optional_text(payload["registry_digest"], "registry_digest"),
        refusal_codes=tuple(cast(list[str], refusal_codes)),
        review_only=review_only,
        actionable=actionable,
    )


def _payload_decision_digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _digest_record(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} has unsupported or missing fields")


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _optional_digest(value: str | None, field: str) -> None:
    if value is not None and _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


__all__ = [
    "ADMISSION_SCHEMA",
    "ADMISSION_SCHEMA_VERSION",
    "MAX_ADMISSION_BYTES",
    "REFUSAL_CODES",
    "ReactorSemanticAdmissionDecision",
    "ReactorSemanticAdmissionStatus",
    "admission_decision_digest",
    "admission_decision_from_bytes",
    "admission_decision_to_bytes",
]
