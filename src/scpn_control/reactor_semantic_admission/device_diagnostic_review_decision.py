# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Sealed device diagnostic review admission decisions.

"""Canonical CONTROL decisions over sealed SPO diagnostic reviews."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

DEVICE_DIAGNOSTIC_REVIEW_DECISION_SCHEMA: Final = "scpn-control.device-diagnostic-review-admission.v1"
DEVICE_DIAGNOSTIC_REVIEW_DECISION_VERSION: Final = "1.0.0"
MAX_DEVICE_DIAGNOSTIC_REVIEW_DECISION_BYTES: Final = 1024 * 1024

_HEX_40: Final = re.compile(r"^[0-9a-f]{40}$")
_HEX_64: Final = re.compile(r"^[0-9a-f]{64}$")
_OUTER_KEYS: Final = frozenset({"payload", "payload_sha256", "schema", "schema_version"})
_PAYLOAD_KEYS: Final = frozenset(
    {
        "accepted_for_review",
        "actionable",
        "actuation_authorised",
        "classification_performed",
        "clock_custody_sha256",
        "configurations",
        "control_intent_created",
        "decision",
        "decision_digest",
        "evidence_claimed",
        "execution_authorised",
        "facility_binding_claimed",
        "measurement_claimed",
        "observation_claimed",
        "refusal_codes",
        "review_id",
        "review_only",
        "review_schema",
        "review_schema_version",
        "review_sha256",
        "semantic_ingress_declared",
        "source_artifact_sha256",
        "source_envelope_sha256",
        "source_manifest_sha256",
        "source_plan_sha256",
        "source_project",
        "source_revision",
        "spo_contract_module_sha256",
        "spo_distribution_version",
        "spo_release_wheel_sha256",
    }
)


class DeviceDiagnosticReviewAdmissionStatus(StrEnum):
    """Outcome of CONTROL's sealed diagnostic-review gate."""

    ACCEPTED_FOR_REVIEW = "accepted_for_review"
    REJECTED = "rejected"


DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES: Final = frozenset(
    {
        "authority_escalation",
        "clock_custody_mismatch",
        "configuration_mismatch",
        "review_decode_failed",
        "review_digest_mismatch",
        "review_identity_mismatch",
        "source_artifact_mismatch",
        "source_envelope_digest_mismatch",
        "source_manifest_digest_mismatch",
        "source_plan_digest_mismatch",
        "source_project_mismatch",
        "source_revision_mismatch",
        "spo_contract_module_digest_mismatch",
        "spo_contract_unavailable",
        "spo_distribution_version_mismatch",
    }
)


@dataclass(frozen=True, slots=True)
class DeviceDiagnosticReviewAdmissionDecision:
    """Digest-sealed, non-authorising result for one exact SPO review.

    Decoded source identities are absent when the SPO public decoder cannot
    establish them. Every field capable of implying operational authority is a
    fixed false invariant.

    Parameters
    ----------
    decision : DeviceDiagnosticReviewAdmissionStatus
        Closed accepted-for-review or rejected outcome.
    accepted_for_review : bool
        Whether the exact review and all caller-owned bindings matched.
    review_sha256, review_id : str | None
        Exact input digest and decoded upstream review identity.
    review_schema, review_schema_version : str
        SPO portable review contract bound by this decision.
    source_project, source_revision, source_artifact_sha256 : str | None
        Decoded producer custody, absent after decoder failure.
    source_manifest_sha256, source_envelope_sha256, source_plan_sha256 : str | None
        Decoded source-document custody.
    configurations : tuple[str, ...]
        Exact decoded reactor configurations, or empty after decoder failure.
    clock_custody_sha256 : str | None
        CONTROL digest over every decoded clock field.
    spo_distribution_version, spo_release_wheel_sha256 : str
        Exact public SPO release identity selected by CONTROL.
    spo_contract_module_sha256 : str | None
        Digest observed from the installed public decoder module.
    refusal_codes : tuple[str, ...]
        Sorted unique members of the closed refusal vocabulary.

    """

    decision: DeviceDiagnosticReviewAdmissionStatus
    accepted_for_review: bool
    review_sha256: str | None
    review_id: str | None
    review_schema: str
    review_schema_version: str
    source_project: str | None
    source_revision: str | None
    source_artifact_sha256: str | None
    source_manifest_sha256: str | None
    source_envelope_sha256: str | None
    source_plan_sha256: str | None
    configurations: tuple[str, ...]
    clock_custody_sha256: str | None
    spo_distribution_version: str
    spo_release_wheel_sha256: str
    spo_contract_module_sha256: str | None
    refusal_codes: tuple[str, ...]
    review_only: bool = True
    evidence_claimed: bool = False
    observation_claimed: bool = False
    measurement_claimed: bool = False
    facility_binding_claimed: bool = False
    classification_performed: bool = False
    semantic_ingress_declared: bool = False
    control_intent_created: bool = False
    actionable: bool = False
    execution_authorised: bool = False
    actuation_authorised: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous, unsealed, or authority-bearing decisions."""
        if not isinstance(self.decision, DeviceDiagnosticReviewAdmissionStatus):
            raise ValueError("decision must be a DeviceDiagnosticReviewAdmissionStatus")
        bool_fields = (
            ("accepted_for_review", self.accepted_for_review),
            ("review_only", self.review_only),
            ("evidence_claimed", self.evidence_claimed),
            ("observation_claimed", self.observation_claimed),
            ("measurement_claimed", self.measurement_claimed),
            ("facility_binding_claimed", self.facility_binding_claimed),
            ("classification_performed", self.classification_performed),
            ("semantic_ingress_declared", self.semantic_ingress_declared),
            ("control_intent_created", self.control_intent_created),
            ("actionable", self.actionable),
            ("execution_authorised", self.execution_authorised),
            ("actuation_authorised", self.actuation_authorised),
        )
        for name, value in bool_fields:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        if self.review_only is not True or any(value for _name, value in bool_fields[2:]):
            raise ValueError("device diagnostic review cannot create operational authority")
        if self.accepted_for_review != (self.decision is DeviceDiagnosticReviewAdmissionStatus.ACCEPTED_FOR_REVIEW):
            raise ValueError("accepted_for_review must match decision")
        if not isinstance(self.refusal_codes, tuple) or any(not isinstance(code, str) for code in self.refusal_codes):
            raise ValueError("refusal_codes must be a tuple of strings")
        if tuple(sorted(set(self.refusal_codes))) != self.refusal_codes:
            raise ValueError("refusal_codes must be sorted and unique")
        if not set(self.refusal_codes) <= DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES:
            raise ValueError("refusal_codes contain an unknown code")
        if self.accepted_for_review == bool(self.refusal_codes):
            raise ValueError("accepted decisions have no refusals; rejected decisions require one")

        _text(self.review_schema, "review_schema")
        _text(self.review_schema_version, "review_schema_version")
        _text(self.spo_distribution_version, "spo_distribution_version")
        _digest(self.spo_release_wheel_sha256, "spo_release_wheel_sha256")
        optional_text = (
            ("review_id", self.review_id),
            ("source_project", self.source_project),
            ("source_revision", self.source_revision),
        )
        for text_name, text_value in optional_text:
            _optional_text(text_value, text_name)
        optional_digests = (
            ("review_sha256", self.review_sha256),
            ("source_artifact_sha256", self.source_artifact_sha256),
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("source_envelope_sha256", self.source_envelope_sha256),
            ("source_plan_sha256", self.source_plan_sha256),
            ("clock_custody_sha256", self.clock_custody_sha256),
            ("spo_contract_module_sha256", self.spo_contract_module_sha256),
        )
        for digest_name, digest_value in optional_digests:
            _optional_digest(digest_value, digest_name)
        if self.review_id is not None:
            _digest(self.review_id, "review_id")
        if self.source_revision is not None and _HEX_40.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a lowercase 40-character Git commit")
        if not isinstance(self.configurations, tuple) or any(
            not isinstance(value, str) or not value for value in self.configurations
        ):
            raise ValueError("configurations must be a tuple of non-empty strings")
        if tuple(sorted(set(self.configurations))) != self.configurations:
            raise ValueError("configurations must be sorted and unique")

        decoded_upstream_fields = (
            self.review_id,
            self.source_project,
            self.source_revision,
            self.source_artifact_sha256,
            self.source_manifest_sha256,
            self.source_envelope_sha256,
            self.source_plan_sha256,
            self.clock_custody_sha256,
        )
        accepted_binding_fields = (
            self.review_sha256,
            *decoded_upstream_fields,
            self.spo_contract_module_sha256,
        )
        if self.accepted_for_review and (
            any(value is None for value in accepted_binding_fields) or not self.configurations
        ):
            raise ValueError("accepted decisions require every decoded identity")
        if "review_decode_failed" in self.refusal_codes and (
            any(value is not None for value in decoded_upstream_fields) or self.configurations
        ):
            raise ValueError("decoder failures cannot guess upstream identities")

    @property
    def decision_digest(self) -> str:
        """Return SHA-256 of the payload with an empty seal field."""
        return _digest_record(_payload_record(self, decision_digest=None))


def device_diagnostic_review_decision_to_bytes(
    decision: DeviceDiagnosticReviewAdmissionDecision,
) -> bytes:
    """Encode one decision as unique canonical UTF-8 JSON bytes."""
    payload = _payload_record(decision, decision_digest=decision.decision_digest)
    return _canonical_json(
        {
            "payload": payload,
            "payload_sha256": _digest_record(payload),
            "schema": DEVICE_DIAGNOSTIC_REVIEW_DECISION_SCHEMA,
            "schema_version": DEVICE_DIAGNOSTIC_REVIEW_DECISION_VERSION,
        }
    )


def device_diagnostic_review_decision_from_bytes(
    data: bytes,
) -> DeviceDiagnosticReviewAdmissionDecision:
    """Decode only canonical, duplicate-key-free CONTROL decision bytes."""
    if not isinstance(data, bytes):
        raise TypeError("device diagnostic review decision must be bytes")
    if not data:
        raise ValueError("device diagnostic review decision must not be empty")
    if len(data) > MAX_DEVICE_DIAGNOSTIC_REVIEW_DECISION_BYTES:
        raise ValueError("device diagnostic review decision exceeds the size limit")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("device diagnostic review decision must be strict UTF-8") from exc
    try:
        raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("device diagnostic review decision must be valid JSON") from exc
    record = _mapping(raw, "decision envelope")
    _exact_keys(record, _OUTER_KEYS, "decision envelope")
    if record["schema"] != DEVICE_DIAGNOSTIC_REVIEW_DECISION_SCHEMA:
        raise ValueError("unsupported device diagnostic review decision schema")
    if record["schema_version"] != DEVICE_DIAGNOSTIC_REVIEW_DECISION_VERSION:
        raise ValueError("unsupported device diagnostic review decision schema version")
    payload = _mapping(record["payload"], "decision payload")
    _exact_keys(payload, _PAYLOAD_KEYS, "decision payload")
    if record["payload_sha256"] != _digest_record(payload):
        raise ValueError("device diagnostic review decision payload digest mismatch")
    decision = _decision_from_payload(payload)
    if payload["decision_digest"] != decision.decision_digest:
        raise ValueError("device diagnostic review decision digest mismatch")
    if device_diagnostic_review_decision_to_bytes(decision) != data:
        raise ValueError("device diagnostic review decision is not canonical")
    return decision


def device_diagnostic_review_decision_digest(
    decision: DeviceDiagnosticReviewAdmissionDecision,
) -> str:
    """Return SHA-256 of the complete canonical CONTROL decision bytes."""
    return hashlib.sha256(device_diagnostic_review_decision_to_bytes(decision)).hexdigest()


def _payload_record(
    decision: DeviceDiagnosticReviewAdmissionDecision,
    *,
    decision_digest: str | None,
) -> dict[str, object]:
    return {
        "accepted_for_review": decision.accepted_for_review,
        "actionable": decision.actionable,
        "actuation_authorised": decision.actuation_authorised,
        "classification_performed": decision.classification_performed,
        "clock_custody_sha256": decision.clock_custody_sha256,
        "configurations": list(decision.configurations),
        "control_intent_created": decision.control_intent_created,
        "decision": decision.decision.value,
        "decision_digest": decision_digest,
        "evidence_claimed": decision.evidence_claimed,
        "execution_authorised": decision.execution_authorised,
        "facility_binding_claimed": decision.facility_binding_claimed,
        "measurement_claimed": decision.measurement_claimed,
        "observation_claimed": decision.observation_claimed,
        "refusal_codes": list(decision.refusal_codes),
        "review_id": decision.review_id,
        "review_only": decision.review_only,
        "review_schema": decision.review_schema,
        "review_schema_version": decision.review_schema_version,
        "review_sha256": decision.review_sha256,
        "semantic_ingress_declared": decision.semantic_ingress_declared,
        "source_artifact_sha256": decision.source_artifact_sha256,
        "source_envelope_sha256": decision.source_envelope_sha256,
        "source_manifest_sha256": decision.source_manifest_sha256,
        "source_plan_sha256": decision.source_plan_sha256,
        "source_project": decision.source_project,
        "source_revision": decision.source_revision,
        "spo_contract_module_sha256": decision.spo_contract_module_sha256,
        "spo_distribution_version": decision.spo_distribution_version,
        "spo_release_wheel_sha256": decision.spo_release_wheel_sha256,
    }


def _decision_from_payload(
    payload: Mapping[str, object],
) -> DeviceDiagnosticReviewAdmissionDecision:
    refusals = _string_tuple(payload["refusal_codes"], "refusal_codes")
    configurations = _string_tuple(payload["configurations"], "configurations")
    try:
        status = DeviceDiagnosticReviewAdmissionStatus(_text(payload["decision"], "decision"))
    except ValueError as exc:
        raise ValueError("unknown device diagnostic review decision") from exc
    return DeviceDiagnosticReviewAdmissionDecision(
        decision=status,
        accepted_for_review=_boolean(payload["accepted_for_review"], "accepted_for_review"),
        review_sha256=_optional_text_value(payload["review_sha256"], "review_sha256"),
        review_id=_optional_text_value(payload["review_id"], "review_id"),
        review_schema=_text(payload["review_schema"], "review_schema"),
        review_schema_version=_text(payload["review_schema_version"], "review_schema_version"),
        source_project=_optional_text_value(payload["source_project"], "source_project"),
        source_revision=_optional_text_value(payload["source_revision"], "source_revision"),
        source_artifact_sha256=_optional_text_value(payload["source_artifact_sha256"], "source_artifact_sha256"),
        source_manifest_sha256=_optional_text_value(payload["source_manifest_sha256"], "source_manifest_sha256"),
        source_envelope_sha256=_optional_text_value(payload["source_envelope_sha256"], "source_envelope_sha256"),
        source_plan_sha256=_optional_text_value(payload["source_plan_sha256"], "source_plan_sha256"),
        configurations=configurations,
        clock_custody_sha256=_optional_text_value(payload["clock_custody_sha256"], "clock_custody_sha256"),
        spo_distribution_version=_text(payload["spo_distribution_version"], "spo_distribution_version"),
        spo_release_wheel_sha256=_text(payload["spo_release_wheel_sha256"], "spo_release_wheel_sha256"),
        spo_contract_module_sha256=_optional_text_value(
            payload["spo_contract_module_sha256"], "spo_contract_module_sha256"
        ),
        refusal_codes=refusals,
        review_only=_boolean(payload["review_only"], "review_only"),
        evidence_claimed=_boolean(payload["evidence_claimed"], "evidence_claimed"),
        observation_claimed=_boolean(payload["observation_claimed"], "observation_claimed"),
        measurement_claimed=_boolean(payload["measurement_claimed"], "measurement_claimed"),
        facility_binding_claimed=_boolean(payload["facility_binding_claimed"], "facility_binding_claimed"),
        classification_performed=_boolean(payload["classification_performed"], "classification_performed"),
        semantic_ingress_declared=_boolean(payload["semantic_ingress_declared"], "semantic_ingress_declared"),
        control_intent_created=_boolean(payload["control_intent_created"], "control_intent_created"),
        actionable=_boolean(payload["actionable"], "actionable"),
        execution_authorised=_boolean(payload["execution_authorised"], "execution_authorised"),
        actuation_authorised=_boolean(payload["actuation_authorised"], "actuation_authorised"),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _digest_record(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} has unsupported or missing fields")


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _optional_text_value(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return tuple(value)


def _optional_text(value: str | None, name: str) -> None:
    if value is not None:
        _text(value, name)


def _digest(value: str, name: str) -> None:
    if _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _optional_digest(value: str | None, name: str) -> None:
    if value is not None:
        _digest(value, name)


__all__ = [
    "DEVICE_DIAGNOSTIC_REVIEW_DECISION_SCHEMA",
    "DEVICE_DIAGNOSTIC_REVIEW_DECISION_VERSION",
    "DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES",
    "MAX_DEVICE_DIAGNOSTIC_REVIEW_DECISION_BYTES",
    "DeviceDiagnosticReviewAdmissionDecision",
    "DeviceDiagnosticReviewAdmissionStatus",
    "device_diagnostic_review_decision_digest",
    "device_diagnostic_review_decision_from_bytes",
    "device_diagnostic_review_decision_to_bytes",
]
