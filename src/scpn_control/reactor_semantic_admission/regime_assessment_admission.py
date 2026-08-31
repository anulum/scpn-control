# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reactor regime assessment review admission

"""Fail-closed admission of exact public SPO regime-assessment bytes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Final

from scpn_phase_orchestrator.reactor_semantics import (
    EvidenceClass,
    QualityState,
    ReactorRegimeAssessment,
    ReactorRegimeAxisDisposition,
    ValidityState,
    regime_assessment_from_bytes,
)

from .regime_assessment_decision import (
    ReactorRegimeAssessmentAdmissionDecision,
    ReactorRegimeAssessmentAdmissionStatus,
)

_HEX_40: Final = re.compile(r"^[0-9a-f]{40}$")
_HEX_64: Final = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReactorRegimeAssessmentAdmissionPolicy:
    """Caller-owned exact expectations for one abstaining assessment.

    The three custody digests cover the complete installed-registry bindings,
    clock/validity record, and ordered eight-axis records. Individual identity
    fields remain explicit so rejection diagnostics do not reduce every drift
    to one opaque digest mismatch.

    Attributes
    ----------
    expected_assessment_sha256 : str
        SHA-256 of the exact canonical SPO assessment bytes.
    expected_assessment_id : str
        Expected upstream assessment identifier.
    expected_reactor_context_id, expected_configuration, expected_event_id : str
        Expected reactor and event identity.
    expected_producer_project, expected_producer_revision : str
        Expected SPO producer project and exact Git revision.
    expected_producer_artifact_sha256 : str
        SHA-256 of the producer artifact used to build the assessment.
    expected_source_project, expected_source_revision : str
        Expected plant-truth owner and exact source revision.
    expected_source_handoff_schema, expected_source_handoff_sha256 : str
        Expected source handoff contract and canonical-byte digest.
    expected_source_semantic_ids : tuple[str, ...]
        Sorted complete set of upstream semantic identifiers.
    expected_assessment_schema_version : str
        Expected SPO assessment schema version.
    expected_registry_custody_sha256 : str
        Digest over all four installed registry bindings.
    expected_clock_custody_sha256 : str
        Digest over clock, timing, latency, and validity custody.
    expected_axis_custody_sha256 : str
        Digest over every field in all eight ordered axis records.
    expected_axis_ids : tuple[str, ...]
        Sorted canonical set of eight axis identifiers.
    expected_axis_provenance : tuple[tuple[str, str], ...]
        Sorted axis-to-provenance identity pairs.
    checked_at_ns : int
        Explicit deterministic admission time in the assessment clock domain.
    max_evidence_age_ns : int
        Inclusive maximum evidence age in nanoseconds.
    """

    expected_assessment_sha256: str
    expected_assessment_id: str
    expected_reactor_context_id: str
    expected_configuration: str
    expected_event_id: str
    expected_producer_project: str
    expected_producer_revision: str
    expected_producer_artifact_sha256: str
    expected_source_project: str
    expected_source_revision: str
    expected_source_handoff_schema: str
    expected_source_handoff_sha256: str
    expected_source_semantic_ids: tuple[str, ...]
    expected_assessment_schema_version: str
    expected_registry_custody_sha256: str
    expected_clock_custody_sha256: str
    expected_axis_custody_sha256: str
    expected_axis_ids: tuple[str, ...]
    expected_axis_provenance: tuple[tuple[str, str], ...]
    checked_at_ns: int
    max_evidence_age_ns: int

    def __post_init__(self) -> None:
        """Validate deterministic policy fields without reading wall time."""
        for name in (
            "expected_assessment_sha256",
            "expected_producer_artifact_sha256",
            "expected_source_handoff_sha256",
            "expected_registry_custody_sha256",
            "expected_clock_custody_sha256",
            "expected_axis_custody_sha256",
        ):
            _digest(getattr(self, name), name)
        for name in ("expected_producer_revision", "expected_source_revision"):
            revision = getattr(self, name)
            if not isinstance(revision, str) or _HEX_40.fullmatch(revision) is None:
                raise ValueError(f"{name} must be a lowercase 40-character Git commit")
        for name in (
            "expected_assessment_id",
            "expected_reactor_context_id",
            "expected_configuration",
            "expected_event_id",
            "expected_producer_project",
            "expected_source_project",
            "expected_source_handoff_schema",
            "expected_assessment_schema_version",
        ):
            _text(getattr(self, name), name)
        _canonical_strings(self.expected_source_semantic_ids, "expected_source_semantic_ids")
        axis_ids = _canonical_strings(self.expected_axis_ids, "expected_axis_ids")
        if len(axis_ids) != 8:
            raise ValueError("expected_axis_ids must contain exactly eight axes")
        if not isinstance(self.expected_axis_provenance, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            for item in self.expected_axis_provenance
        ):
            raise ValueError("expected_axis_provenance must contain text pairs")
        provenance = self.expected_axis_provenance
        for axis, value in provenance:
            _text(axis, "expected_axis_provenance axis")
            _text(value, "expected_axis_provenance identity")
        if tuple(sorted(set(provenance))) != provenance:
            raise ValueError("expected_axis_provenance must be sorted and unique")
        if tuple(axis for axis, _value in provenance) != axis_ids:
            raise ValueError("expected_axis_provenance must cover every expected axis")
        for name in ("checked_at_ns", "max_evidence_age_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def admit_reactor_regime_assessment(
    payload: bytes,
    *,
    policy: ReactorRegimeAssessmentAdmissionPolicy,
) -> ReactorRegimeAssessmentAdmissionDecision:
    """Decode and admit exact SPO assessment bytes for review only.

    The sole ingress is SPO's public canonical decoder. A malformed, resealed,
    authority-bearing, classified, stale, identity-drifted, or non-abstaining
    record returns a deterministic non-actionable refusal.

    Parameters
    ----------
    payload : bytes
        Candidate canonical SPO assessment envelope.
    policy : ReactorRegimeAssessmentAdmissionPolicy
        Exact caller-owned identity, custody, and freshness expectations.

    Returns
    -------
    ReactorRegimeAssessmentAdmissionDecision
        Sealed review-only admission or deterministic rejection.
    """
    raw_digest = hashlib.sha256(payload).hexdigest() if isinstance(payload, bytes) else None
    try:
        assessment = regime_assessment_from_bytes(payload)
    except (TypeError, ValueError):
        return _decision(
            policy=policy,
            assessment=None,
            assessment_sha256=raw_digest,
            refusals={"assessment_decode_failed"},
        )
    refusals = _evaluate_assessment(
        assessment,
        assessment_sha256=raw_digest,
        policy=policy,
    )
    return _decision(
        policy=policy,
        assessment=assessment,
        assessment_sha256=raw_digest,
        refusals=refusals,
    )


def regime_assessment_registry_custody_digest(assessment: ReactorRegimeAssessment) -> str:
    """Digest all four versioned registry bindings in one canonical record.

    Parameters
    ----------
    assessment : ReactorRegimeAssessment
        Validated public SPO assessment.

    Returns
    -------
    str
        Lowercase SHA-256 over the canonical registry-custody record.
    """
    return _canonical_digest(
        {
            "observability": [
                assessment.observability_registry_version,
                assessment.observability_registry_digest,
            ],
            "ontology": [assessment.ontology_version, assessment.ontology_digest],
            "reactor": [assessment.reactor_registry_version, assessment.reactor_registry_digest],
            "semantic_profile": [
                assessment.semantic_profile_registry_version,
                assessment.semantic_profile_registry_digest,
            ],
        }
    )


def regime_assessment_clock_custody_digest(assessment: ReactorRegimeAssessment) -> str:
    """Digest clock, freshness and common-validity custody.

    Parameters
    ----------
    assessment : ReactorRegimeAssessment
        Validated public SPO assessment.

    Returns
    -------
    str
        Lowercase SHA-256 over the canonical clock-custody record.
    """
    return _canonical_digest(
        {
            "assessed_at_ns": assessment.assessed_at_ns,
            "clock_domain": assessment.clock_domain,
            "clock_epoch": assessment.clock_epoch,
            "clock_kind": assessment.clock_kind.value,
            "clock_synchronization_id": assessment.clock_synchronization_id,
            "evidence_timestamp_ns": assessment.evidence_timestamp_ns,
            "latency_s": assessment.latency_s,
            "sample_rate_hz": assessment.sample_rate_hz,
            "timestamp_offset_ps": assessment.timestamp_offset_ps,
            "valid_from_ns": assessment.valid_from_ns,
            "valid_until_ns": assessment.valid_until_ns,
        }
    )


def regime_assessment_axis_custody_digest(assessment: ReactorRegimeAssessment) -> str:
    """Digest every ordered field of all eight public SPO axis records.

    Parameters
    ----------
    assessment : ReactorRegimeAssessment
        Validated public SPO assessment.

    Returns
    -------
    str
        Lowercase SHA-256 over the ordered canonical axis records.
    """
    return _canonical_digest([axis.to_record() for axis in assessment.axes])


def _evaluate_assessment(
    assessment: ReactorRegimeAssessment,
    *,
    assessment_sha256: str | None,
    policy: ReactorRegimeAssessmentAdmissionPolicy,
) -> set[str]:
    refusals: set[str] = set()
    if assessment_sha256 != policy.expected_assessment_sha256:
        refusals.add("assessment_digest_mismatch")
    if assessment.assessment_id != policy.expected_assessment_id:
        refusals.add("assessment_identity_mismatch")
    if (
        assessment.event_id != policy.expected_event_id
        or assessment.reactor_context_id != policy.expected_reactor_context_id
        or assessment.configuration != policy.expected_configuration
    ):
        refusals.add("event_context_mismatch")
    if (
        assessment.producer_project != policy.expected_producer_project
        or assessment.producer_revision != policy.expected_producer_revision
        or assessment.producer_artifact_sha256 != policy.expected_producer_artifact_sha256
    ):
        refusals.add("producer_identity_mismatch")
    if (
        assessment.source_project != policy.expected_source_project
        or assessment.source_revision != policy.expected_source_revision
        or assessment.source_handoff_schema != policy.expected_source_handoff_schema
    ):
        refusals.add("source_identity_mismatch")
    if assessment.schema_version != policy.expected_assessment_schema_version:
        refusals.add("assessment_schema_version_mismatch")
    if assessment.source_handoff_sha256 != policy.expected_source_handoff_sha256:
        refusals.add("source_handoff_digest_mismatch")
    if assessment.source_semantic_ids != policy.expected_source_semantic_ids:
        refusals.add("source_semantic_identity_mismatch")
    _check_custody(assessment, policy=policy, refusals=refusals)
    _check_time(assessment, policy=policy, refusals=refusals)
    _check_abstention(assessment, policy=policy, refusals=refusals)
    return refusals


def _check_custody(
    assessment: ReactorRegimeAssessment,
    *,
    policy: ReactorRegimeAssessmentAdmissionPolicy,
    refusals: set[str],
) -> None:
    if regime_assessment_registry_custody_digest(assessment) != policy.expected_registry_custody_sha256:
        refusals.add("registry_custody_mismatch")
    if regime_assessment_clock_custody_digest(assessment) != policy.expected_clock_custody_sha256:
        refusals.add("clock_custody_mismatch")
    if regime_assessment_axis_custody_digest(assessment) != policy.expected_axis_custody_sha256:
        refusals.add("axis_custody_mismatch")


def _check_time(
    assessment: ReactorRegimeAssessment,
    *,
    policy: ReactorRegimeAssessmentAdmissionPolicy,
    refusals: set[str],
) -> None:
    age = policy.checked_at_ns - assessment.evidence_timestamp_ns
    if age < 0:
        refusals.add("evidence_from_future")
    elif age > policy.max_evidence_age_ns:
        refusals.add("evidence_stale")
    if not assessment.valid_from_ns <= policy.checked_at_ns <= assessment.valid_until_ns:
        refusals.add("outside_common_validity")


def _check_abstention(
    assessment: ReactorRegimeAssessment,
    *,
    policy: ReactorRegimeAssessmentAdmissionPolicy,
    refusals: set[str],
) -> None:
    axes = assessment.axes
    if any(axis.disposition is ReactorRegimeAxisDisposition.CLASSIFIED for axis in axes):
        refusals.add("assessment_not_abstaining")
    if tuple(axis.axis_id for axis in axes) != policy.expected_axis_ids:
        refusals.add("axis_custody_mismatch")
    if tuple((axis.axis_id, axis.provenance_id) for axis in axes) != policy.expected_axis_provenance:
        refusals.add("axis_provenance_policy_mismatch")
    for axis in axes:
        if axis.evidence_ids or axis.evidence_bindings:
            refusals.add("axis_evidence_policy_mismatch")
        if axis.confidence != 0.0 or axis.observability != 0.0:
            refusals.add("axis_observability_policy_mismatch")
        if axis.disposition is ReactorRegimeAxisDisposition.UNKNOWN:
            if (
                axis.evidence_class is not EvidenceClass.UNKNOWN
                or axis.validity is not ValidityState.UNKNOWN
                or axis.quality is not QualityState.UNKNOWN
            ):
                refusals.add("axis_quality_policy_mismatch")


def _decision(
    *,
    policy: ReactorRegimeAssessmentAdmissionPolicy,
    assessment: ReactorRegimeAssessment | None,
    assessment_sha256: str | None,
    refusals: set[str],
) -> ReactorRegimeAssessmentAdmissionDecision:
    admitted = not refusals
    return ReactorRegimeAssessmentAdmissionDecision(
        decision=(
            ReactorRegimeAssessmentAdmissionStatus.ADMITTED_FOR_REVIEW
            if admitted
            else ReactorRegimeAssessmentAdmissionStatus.REJECTED
        ),
        admitted=admitted,
        checked_at_ns=policy.checked_at_ns,
        assessment_sha256=assessment_sha256,
        assessment_id=None if assessment is None else assessment.assessment_id,
        reactor_context_id=None if assessment is None else assessment.reactor_context_id,
        configuration=None if assessment is None else assessment.configuration,
        event_id=None if assessment is None else assessment.event_id,
        producer_project=None if assessment is None else assessment.producer_project,
        producer_revision=None if assessment is None else assessment.producer_revision,
        producer_artifact_sha256=(None if assessment is None else assessment.producer_artifact_sha256),
        source_project=None if assessment is None else assessment.source_project,
        source_revision=None if assessment is None else assessment.source_revision,
        source_handoff_schema=None if assessment is None else assessment.source_handoff_schema,
        source_handoff_sha256=None if assessment is None else assessment.source_handoff_sha256,
        assessment_schema_version=None if assessment is None else assessment.schema_version,
        registry_custody_sha256=(None if assessment is None else regime_assessment_registry_custody_digest(assessment)),
        clock_custody_sha256=(None if assessment is None else regime_assessment_clock_custody_digest(assessment)),
        axis_custody_sha256=(None if assessment is None else regime_assessment_axis_custody_digest(assessment)),
        refusal_codes=tuple(sorted(refusals)),
    )


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _canonical_strings(values: object, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{field} must be a tuple of non-empty strings")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{field} must be sorted and unique")
    return values


__all__ = [
    "ReactorRegimeAssessmentAdmissionPolicy",
    "admit_reactor_regime_assessment",
    "regime_assessment_axis_custody_digest",
    "regime_assessment_clock_custody_digest",
    "regime_assessment_registry_custody_digest",
]
