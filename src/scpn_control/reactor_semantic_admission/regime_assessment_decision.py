# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reactor regime assessment admission decisions

"""Canonical sealed decisions for non-actuating regime-assessment review."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

REGIME_ASSESSMENT_ADMISSION_SCHEMA: Final = "scpn-control.reactor-regime-assessment-admission.v1"
REGIME_ASSESSMENT_ADMISSION_VERSION: Final = "1.0.0"
MAX_REGIME_ASSESSMENT_ADMISSION_BYTES: Final = 1024 * 1024

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_OUTER_KEYS = frozenset({"payload", "payload_sha256", "schema", "schema_version"})
_PAYLOAD_KEYS = frozenset(
    {
        "actionable",
        "admitted",
        "assessment_id",
        "assessment_schema_version",
        "assessment_sha256",
        "axis_custody_sha256",
        "checked_at_ns",
        "clock_custody_sha256",
        "configuration",
        "decision",
        "decision_digest",
        "event_id",
        "producer_artifact_sha256",
        "producer_project",
        "producer_revision",
        "reactor_context_id",
        "refusal_codes",
        "registry_custody_sha256",
        "review_only",
        "source_handoff_schema",
        "source_handoff_sha256",
        "source_project",
        "source_revision",
    }
)


class ReactorRegimeAssessmentAdmissionStatus(StrEnum):
    """Outcome of CONTROL's regime-assessment review gate."""

    ADMITTED_FOR_REVIEW = "admitted_for_review"
    REJECTED = "rejected"


REGIME_ASSESSMENT_REFUSAL_CODES: Final = frozenset(
    {
        "assessment_decode_failed",
        "assessment_digest_mismatch",
        "assessment_identity_mismatch",
        "assessment_not_abstaining",
        "assessment_schema_version_mismatch",
        "axis_custody_mismatch",
        "axis_evidence_policy_mismatch",
        "axis_observability_policy_mismatch",
        "axis_provenance_policy_mismatch",
        "axis_quality_policy_mismatch",
        "clock_custody_mismatch",
        "evidence_from_future",
        "evidence_stale",
        "event_context_mismatch",
        "outside_common_validity",
        "producer_identity_mismatch",
        "registry_custody_mismatch",
        "source_handoff_digest_mismatch",
        "source_identity_mismatch",
        "source_semantic_identity_mismatch",
    }
)


@dataclass(frozen=True, slots=True)
class ReactorRegimeAssessmentAdmissionDecision:
    """Digest-sealed result of admitting one exact assessment for review.

    Every optional identity is populated only after the public SPO decoder has
    accepted canonical bytes. Decoder failures retain only the raw byte digest;
    they never infer upstream identity. ``review_only`` and ``actionable`` are
    hard invariants, not caller options.

    Attributes
    ----------
    decision : ReactorRegimeAssessmentAdmissionStatus
        Closed admitted-for-review or rejected outcome.
    admitted : bool
        Whether every policy check passed.
    checked_at_ns : int
        Explicit deterministic check time supplied by the policy.
    assessment_sha256, assessment_id : str | None
        Raw input digest and safely decoded assessment identity.
    reactor_context_id, configuration, event_id : str | None
        Safely decoded reactor and event identity.
    producer_project, producer_revision, producer_artifact_sha256 : str | None
        Safely decoded producer identity.
    source_project, source_revision, source_handoff_schema : str | None
        Safely decoded plant-truth source identity.
    source_handoff_sha256, assessment_schema_version : str | None
        Safely decoded source-handoff digest and schema version.
    registry_custody_sha256, clock_custody_sha256, axis_custody_sha256 : str | None
        CONTROL custody digests, present only after successful SPO decoding.
    refusal_codes : tuple[str, ...]
        Sorted unique codes from the closed refusal vocabulary.
    review_only, actionable : bool
        Hard-fixed authority boundary: true and false, respectively.
    """

    decision: ReactorRegimeAssessmentAdmissionStatus
    admitted: bool
    checked_at_ns: int
    assessment_sha256: str | None
    assessment_id: str | None
    reactor_context_id: str | None
    configuration: str | None
    event_id: str | None
    producer_project: str | None
    producer_revision: str | None
    producer_artifact_sha256: str | None
    source_project: str | None
    source_revision: str | None
    source_handoff_schema: str | None
    source_handoff_sha256: str | None
    assessment_schema_version: str | None
    registry_custody_sha256: str | None
    clock_custody_sha256: str | None
    axis_custody_sha256: str | None
    refusal_codes: tuple[str, ...]
    review_only: bool = True
    actionable: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous, unsealed, or authority-bearing decisions."""
        if not isinstance(self.decision, ReactorRegimeAssessmentAdmissionStatus):
            raise ValueError("decision must be a ReactorRegimeAssessmentAdmissionStatus")
        for name, bool_value in (
            ("admitted", self.admitted),
            ("review_only", self.review_only),
            ("actionable", self.actionable),
        ):
            if not isinstance(bool_value, bool):
                raise ValueError(f"{name} must be boolean")
        if self.review_only is not True or self.actionable is not False:
            raise ValueError("regime assessment admission is review-only and non-actionable")
        if self.admitted != (self.decision is ReactorRegimeAssessmentAdmissionStatus.ADMITTED_FOR_REVIEW):
            raise ValueError("admitted must match decision")
        if isinstance(self.checked_at_ns, bool) or not isinstance(self.checked_at_ns, int):
            raise ValueError("checked_at_ns must be a non-negative integer")
        if self.checked_at_ns < 0:
            raise ValueError("checked_at_ns must be a non-negative integer")
        if not isinstance(self.refusal_codes, tuple) or any(not isinstance(code, str) for code in self.refusal_codes):
            raise ValueError("refusal_codes must be a tuple of strings")
        if tuple(sorted(set(self.refusal_codes))) != self.refusal_codes:
            raise ValueError("refusal_codes must be sorted and unique")
        if not set(self.refusal_codes) <= REGIME_ASSESSMENT_REFUSAL_CODES:
            raise ValueError("refusal_codes contain an unknown code")
        if self.admitted == bool(self.refusal_codes):
            raise ValueError("admitted decisions have no refusals; rejected decisions require one")
        identities = _identity_items(self)
        for name, identity_value in identities:
            _optional_text(identity_value, name)
        for name in (
            "assessment_sha256",
            "producer_artifact_sha256",
            "source_handoff_sha256",
            "registry_custody_sha256",
            "clock_custody_sha256",
            "axis_custody_sha256",
        ):
            _optional_digest(cast(str | None, getattr(self, name)), name)
        for name in ("producer_revision", "source_revision"):
            revision = getattr(self, name)
            if revision is not None and (not isinstance(revision, str) or _HEX_40.fullmatch(revision) is None):
                raise ValueError(f"{name} must be a lowercase 40-character Git commit")
        decoded = tuple(value for name, value in identities if name != "assessment_sha256")
        if self.admitted and any(identity_value is None for _name, identity_value in identities):
            raise ValueError("admitted decisions require every decoded identity")
        if "assessment_decode_failed" in self.refusal_codes and any(value is not None for value in decoded):
            raise ValueError("decoder failures cannot guess assessment identities")

    @property
    def decision_digest(self) -> str:
        """Return the SHA-256 seal over the payload with an empty seal field."""
        return _payload_digest(_payload_record(self, decision_digest=""))


def regime_assessment_admission_decision_to_bytes(
    decision: ReactorRegimeAssessmentAdmissionDecision,
) -> bytes:
    """Encode one decision as canonical digest-sealed UTF-8 JSON bytes.

    Parameters
    ----------
    decision : ReactorRegimeAssessmentAdmissionDecision
        Validated admission decision.

    Returns
    -------
    bytes
        Canonical compact JSON envelope with inner and outer seals.
    """
    payload = _payload_record(decision, decision_digest=decision.decision_digest)
    envelope = {
        "payload": payload,
        "payload_sha256": _payload_digest(payload),
        "schema": REGIME_ASSESSMENT_ADMISSION_SCHEMA,
        "schema_version": REGIME_ASSESSMENT_ADMISSION_VERSION,
    }
    return _canonical_bytes(envelope)


def regime_assessment_admission_decision_from_bytes(
    data: bytes,
) -> ReactorRegimeAssessmentAdmissionDecision:
    """Decode strict canonical decision bytes and verify both digest seals.

    Parameters
    ----------
    data : bytes
        Candidate canonical decision envelope.

    Returns
    -------
    ReactorRegimeAssessmentAdmissionDecision
        Validated immutable decision.

    Raises
    ------
    TypeError
        If ``data`` is not bytes.
    ValueError
        If size, UTF-8, JSON, fields, schema, invariants, seals, or canonical
        encoding are invalid.
    """
    if not isinstance(data, bytes):
        raise TypeError("regime assessment admission decision must be bytes")
    if not data:
        raise ValueError("regime assessment admission decision must not be empty")
    if len(data) > MAX_REGIME_ASSESSMENT_ADMISSION_BYTES:
        raise ValueError("regime assessment admission decision exceeds size limit")
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise ValueError("regime assessment admission decision must be strict UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("regime assessment admission decision must be valid JSON") from exc
    envelope = _exact_mapping(raw, _OUTER_KEYS)
    if envelope["schema"] != REGIME_ASSESSMENT_ADMISSION_SCHEMA:
        raise ValueError("unsupported regime assessment admission schema")
    if envelope["schema_version"] != REGIME_ASSESSMENT_ADMISSION_VERSION:
        raise ValueError("unsupported regime assessment admission schema version")
    payload = _exact_mapping(envelope["payload"], _PAYLOAD_KEYS)
    supplied_payload_digest = _digest(envelope["payload_sha256"], "payload_sha256")
    if supplied_payload_digest != _payload_digest(payload):
        raise ValueError("regime assessment admission payload digest mismatch")
    decision = _decision_from_payload(payload)
    if payload["decision_digest"] != decision.decision_digest:
        raise ValueError("regime assessment admission decision digest mismatch")
    if regime_assessment_admission_decision_to_bytes(decision) != data:
        raise ValueError("regime assessment admission decision requires unique canonical encoding")
    return decision


def regime_assessment_admission_decision_digest(
    decision: ReactorRegimeAssessmentAdmissionDecision,
) -> str:
    """Return SHA-256 of the complete canonical decision envelope.

    Parameters
    ----------
    decision : ReactorRegimeAssessmentAdmissionDecision
        Validated admission decision.

    Returns
    -------
    str
        Lowercase SHA-256 of the encoded envelope bytes.
    """
    return hashlib.sha256(regime_assessment_admission_decision_to_bytes(decision)).hexdigest()


def _identity_items(
    decision: ReactorRegimeAssessmentAdmissionDecision,
) -> tuple[tuple[str, str | None], ...]:
    names = (
        "assessment_sha256",
        "assessment_id",
        "reactor_context_id",
        "configuration",
        "event_id",
        "producer_project",
        "producer_revision",
        "producer_artifact_sha256",
        "source_project",
        "source_revision",
        "source_handoff_schema",
        "source_handoff_sha256",
        "assessment_schema_version",
        "registry_custody_sha256",
        "clock_custody_sha256",
        "axis_custody_sha256",
    )
    return tuple((name, cast(str | None, getattr(decision, name))) for name in names)


def _payload_record(
    decision: ReactorRegimeAssessmentAdmissionDecision,
    *,
    decision_digest: str,
) -> dict[str, object]:
    return {
        "actionable": decision.actionable,
        "admitted": decision.admitted,
        "assessment_id": decision.assessment_id,
        "assessment_schema_version": decision.assessment_schema_version,
        "assessment_sha256": decision.assessment_sha256,
        "axis_custody_sha256": decision.axis_custody_sha256,
        "checked_at_ns": decision.checked_at_ns,
        "clock_custody_sha256": decision.clock_custody_sha256,
        "configuration": decision.configuration,
        "decision": decision.decision.value,
        "decision_digest": decision_digest,
        "event_id": decision.event_id,
        "producer_artifact_sha256": decision.producer_artifact_sha256,
        "producer_project": decision.producer_project,
        "producer_revision": decision.producer_revision,
        "reactor_context_id": decision.reactor_context_id,
        "refusal_codes": list(decision.refusal_codes),
        "registry_custody_sha256": decision.registry_custody_sha256,
        "review_only": decision.review_only,
        "source_handoff_schema": decision.source_handoff_schema,
        "source_handoff_sha256": decision.source_handoff_sha256,
        "source_project": decision.source_project,
        "source_revision": decision.source_revision,
    }


def _decision_from_payload(payload: Mapping[str, object]) -> ReactorRegimeAssessmentAdmissionDecision:
    try:
        status = ReactorRegimeAssessmentAdmissionStatus(_text(payload["decision"], "decision"))
    except ValueError as exc:
        raise ValueError("unknown regime assessment admission decision") from exc
    refusal_raw = payload["refusal_codes"]
    if not isinstance(refusal_raw, list) or any(not isinstance(item, str) for item in refusal_raw):
        raise ValueError("refusal_codes must be a list of strings")
    return ReactorRegimeAssessmentAdmissionDecision(
        decision=status,
        admitted=_boolean(payload["admitted"], "admitted"),
        checked_at_ns=_integer(payload["checked_at_ns"], "checked_at_ns"),
        assessment_sha256=_optional_payload_text(payload["assessment_sha256"], "assessment_sha256"),
        assessment_id=_optional_payload_text(payload["assessment_id"], "assessment_id"),
        reactor_context_id=_optional_payload_text(payload["reactor_context_id"], "reactor_context_id"),
        configuration=_optional_payload_text(payload["configuration"], "configuration"),
        event_id=_optional_payload_text(payload["event_id"], "event_id"),
        producer_project=_optional_payload_text(payload["producer_project"], "producer_project"),
        producer_revision=_optional_payload_text(payload["producer_revision"], "producer_revision"),
        producer_artifact_sha256=_optional_payload_text(
            payload["producer_artifact_sha256"], "producer_artifact_sha256"
        ),
        source_project=_optional_payload_text(payload["source_project"], "source_project"),
        source_revision=_optional_payload_text(payload["source_revision"], "source_revision"),
        source_handoff_schema=_optional_payload_text(payload["source_handoff_schema"], "source_handoff_schema"),
        source_handoff_sha256=_optional_payload_text(payload["source_handoff_sha256"], "source_handoff_sha256"),
        assessment_schema_version=_optional_payload_text(
            payload["assessment_schema_version"], "assessment_schema_version"
        ),
        registry_custody_sha256=_optional_payload_text(payload["registry_custody_sha256"], "registry_custody_sha256"),
        clock_custody_sha256=_optional_payload_text(payload["clock_custody_sha256"], "clock_custody_sha256"),
        axis_custody_sha256=_optional_payload_text(payload["axis_custody_sha256"], "axis_custody_sha256"),
        refusal_codes=tuple(refusal_raw),
        review_only=_boolean(payload["review_only"], "review_only"),
        actionable=_boolean(payload["actionable"], "actionable"),
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact_mapping(value: object, keys: frozenset[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("regime assessment admission has unsupported or missing fields")
    return cast(dict[str, object], value)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _optional_payload_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _optional_text(value: str | None, field: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{field} must be non-empty text when present")


def _optional_digest(value: str | None, field: str) -> None:
    if value is not None:
        _digest(value, field)


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "MAX_REGIME_ASSESSMENT_ADMISSION_BYTES",
    "REGIME_ASSESSMENT_ADMISSION_SCHEMA",
    "REGIME_ASSESSMENT_ADMISSION_VERSION",
    "REGIME_ASSESSMENT_REFUSAL_CODES",
    "ReactorRegimeAssessmentAdmissionDecision",
    "ReactorRegimeAssessmentAdmissionStatus",
    "regime_assessment_admission_decision_digest",
    "regime_assessment_admission_decision_from_bytes",
    "regime_assessment_admission_decision_to_bytes",
]
