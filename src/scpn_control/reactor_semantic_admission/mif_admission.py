# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MIF merge-compression semantic review admission

"""Fail-closed admission of the dedicated SPO MIF semantic handoff."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from scpn_phase_orchestrator.reactor_semantics import (
    ClockReference,
    EvidenceClass,
    MIFMergeCompressionHandoff,
    PhaseSemanticRecord,
    QualityState,
    SemanticCarrier,
    ValidityState,
    mif_merge_compression_handoff_from_bytes,
)

from .decision import ReactorSemanticAdmissionDecision, ReactorSemanticAdmissionStatus

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class MIFReactorSemanticAdmissionPolicy:
    """Caller-owned expectations for one review-only MIF handoff.

    Parameters
    ----------
    expected_handoff_sha256, expected_source_envelope_sha256:
        SHA-256 pins for the exact SPO handoff and embedded MIF envelope.
    expected_source_schema, expected_source_revision:
        Exact producer schema and 40-character MIF revision.
    expected_event_id, expected_context_id:
        Shot/event and SPO reactor-context identities.
    expected_registry_version, expected_registry_digest:
        Exact SPO registry identity used to resolve ``frc_compression_mif``.
    expected_observation_clock:
        Complete expected source sample clock, including timestamp and rate.
    reference_clock:
        Explicit review clock used for freshness and ``checked_at_ns``.
    expected_observable_ids, expected_semantic_carriers:
        Complete closed sets admitted by this policy.
    required_numerical_phase_ids:
        Exact semantic IDs that may carry numerical phase.
    required_provenance_attributes:
        Producer attributes that every observable must retain.
    min_numerical_observability, min_numerical_confidence:
        Inclusive lower bounds for numerical-phase evidence in ``[0, 1]``.
    max_numerical_circular_std_rad:
        Inclusive upper bound for numerical-phase circular uncertainty in
        radians.
    allowed_degradation_reasons, allowed_quality_flags:
        Exact closed allowlists for explicitly degraded evidence. Empty
        defaults reject every degradation reason and quality flag.

    Notes
    -----
    Construction validates the policy shape and identity pins. Admission still
    evaluates the supplied handoff independently and never confers actuation
    authority.
    """

    expected_handoff_sha256: str
    expected_source_schema: str
    expected_source_revision: str
    expected_source_envelope_sha256: str
    expected_event_id: str
    expected_context_id: str
    expected_registry_version: str
    expected_registry_digest: str
    expected_observation_clock: ClockReference
    reference_clock: ClockReference
    max_evidence_age_ns: int
    max_calibration_age_ns: int
    allowed_calibration_ids: frozenset[str]
    allowed_transfer_function_ids: frozenset[str]
    expected_observable_ids: frozenset[str]
    expected_semantic_carriers: tuple[tuple[str, SemanticCarrier], ...]
    required_numerical_phase_ids: frozenset[str]
    required_provenance_attributes: tuple[tuple[str, str], ...]
    min_numerical_observability: float
    min_numerical_confidence: float
    max_numerical_circular_std_rad: float
    allowed_degradation_reasons: frozenset[str] = frozenset()
    allowed_quality_flags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Reject malformed, duplicate, or unbounded policy inputs."""
        _digest(self.expected_handoff_sha256, "expected_handoff_sha256")
        _digest(
            self.expected_source_envelope_sha256,
            "expected_source_envelope_sha256",
        )
        _digest(self.expected_registry_digest, "expected_registry_digest")
        if _HEX_40.fullmatch(self.expected_source_revision) is None:
            raise ValueError("expected_source_revision must be a lowercase Git commit")
        for field in (
            "expected_source_schema",
            "expected_event_id",
            "expected_context_id",
            "expected_registry_version",
        ):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"{field} must be a non-empty string")
        for field in ("expected_observation_clock", "reference_clock"):
            if not isinstance(getattr(self, field), ClockReference):
                raise ValueError(f"{field} must be a ClockReference")
        for field in ("max_evidence_age_ns", "max_calibration_age_ns"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        for field in (
            "allowed_calibration_ids",
            "allowed_transfer_function_ids",
            "expected_observable_ids",
            "required_numerical_phase_ids",
            "allowed_degradation_reasons",
            "allowed_quality_flags",
        ):
            _string_set(getattr(self, field), field)
        _probability(self.min_numerical_observability, "min_numerical_observability")
        _probability(self.min_numerical_confidence, "min_numerical_confidence")
        if (
            isinstance(self.max_numerical_circular_std_rad, bool)
            or not isinstance(self.max_numerical_circular_std_rad, (int, float))
            or isinstance(self.max_numerical_circular_std_rad, float)
            and not math.isfinite(self.max_numerical_circular_std_rad)
            or self.max_numerical_circular_std_rad < 0.0
        ):
            raise ValueError("max_numerical_circular_std_rad must be a non-negative number")
        _semantic_carriers(self.expected_semantic_carriers)
        _string_pairs(
            self.required_provenance_attributes,
            "required_provenance_attributes",
        )
        semantic_ids = {item[0] for item in self.expected_semantic_carriers}
        if not self.required_numerical_phase_ids <= semantic_ids:
            raise ValueError("required_numerical_phase_ids must be present in expected semantics")


def admit_mif_reactor_semantic_handoff(
    payload: bytes,
    *,
    policy: MIFReactorSemanticAdmissionPolicy,
) -> ReactorSemanticAdmissionDecision:
    """Evaluate one exact SPO MIF handoff for non-actuating review.

    Parameters
    ----------
    payload:
        Canonical SPO MIF merge-compression handoff bytes.
    policy:
        Caller-owned identity, clock, provenance, evidence, and uncertainty
        bounds for the expected handoff.

    Returns
    -------
    ReactorSemanticAdmissionDecision
        Digest-sealed review decision. Decoder or policy mismatches are
        represented by refusal codes; the result always remains review-only
        and non-actionable.
    """
    raw_digest = hashlib.sha256(payload).hexdigest() if isinstance(payload, bytes) else None
    try:
        handoff = mif_merge_compression_handoff_from_bytes(payload)
    except (TypeError, ValueError):
        return _decision(
            policy=policy,
            handoff=None,
            handoff_sha256=raw_digest,
            refusal_codes={"handoff_decode_failed"},
        )
    refusals = _evaluate_handoff(
        handoff,
        handoff_sha256=raw_digest,
        policy=policy,
    )
    return _decision(
        policy=policy,
        handoff=handoff,
        handoff_sha256=raw_digest,
        refusal_codes=refusals,
    )


def _evaluate_handoff(
    handoff: MIFMergeCompressionHandoff,
    *,
    handoff_sha256: str | None,
    policy: MIFReactorSemanticAdmissionPolicy,
) -> set[str]:
    refusals: set[str] = set()
    _check_identity(
        handoff,
        handoff_sha256=handoff_sha256,
        policy=policy,
        refusals=refusals,
    )
    if frozenset(item.observable_id for item in handoff.observables) != (policy.expected_observable_ids):
        refusals.add("observable_set_mismatch")
    for observable in handoff.observables:
        if observable.clock != policy.expected_observation_clock:
            refusals.add("observation_clock_mismatch")
        _check_freshness(
            observable.clock,
            calibrated_at_ns=observable.calibration.calibrated_at_ns,
            policy=policy,
            refusals=refusals,
        )
        if observable.calibration.calibration_id not in policy.allowed_calibration_ids:
            refusals.add("calibration_id_not_allowed")
        if observable.calibration.transfer_function_id not in policy.allowed_transfer_function_ids:
            refusals.add("transfer_function_id_not_allowed")
        _check_validity(
            observable.validity.state,
            observable.validity.reasons,
            policy=policy,
            prefix="observable",
            refusals=refusals,
        )
        _check_quality(
            observable.quality.state,
            observable.quality.flags,
            policy=policy,
            prefix="observable",
            refusals=refusals,
        )
        attributes = dict(observable.provenance.attributes)
        required_attributes = tuple(attributes.get(key) for key, _ in policy.required_provenance_attributes)
        expected_attributes = tuple(value for _, value in policy.required_provenance_attributes)
        provenance_identity = (
            observable.provenance.sha256,
            observable.provenance.artifact_uri,
            attributes.get("event_id"),
            attributes.get("producer_revision"),
            required_attributes,
        )
        expected_provenance_identity = (
            handoff.source_envelope_sha256,
            f"artifact:sha256:{handoff.source_envelope_sha256}",
            handoff.event_id,
            handoff.source_revision,
            expected_attributes,
        )
        if provenance_identity != expected_provenance_identity:
            refusals.add("provenance_chain_mismatch")
    _check_semantics(handoff, policy=policy, refusals=refusals)
    return refusals


def _check_identity(
    handoff: MIFMergeCompressionHandoff,
    *,
    handoff_sha256: str | None,
    policy: MIFReactorSemanticAdmissionPolicy,
    refusals: set[str],
) -> None:
    if handoff_sha256 != policy.expected_handoff_sha256:
        refusals.add("handoff_digest_mismatch")
    if handoff.source_schema != policy.expected_source_schema:
        refusals.add("source_schema_mismatch")
    if handoff.source_revision != policy.expected_source_revision:
        refusals.add("source_revision_mismatch")
    if handoff.source_envelope_sha256 != policy.expected_source_envelope_sha256:
        refusals.add("source_envelope_digest_mismatch")
    if handoff.event_id != policy.expected_event_id:
        refusals.add("event_id_mismatch")
    if handoff.context.context_id != policy.expected_context_id:
        refusals.add("context_id_mismatch")
    if handoff.context.evidence_class is not EvidenceClass.SIMULATION:
        refusals.add("evidence_class_mismatch")
    if (
        handoff.context.registry_version != policy.expected_registry_version
        or handoff.context.registry_digest != policy.expected_registry_digest
    ):
        refusals.add("registry_identity_mismatch")


def _check_freshness(
    clock: ClockReference,
    *,
    calibrated_at_ns: int,
    policy: MIFReactorSemanticAdmissionPolicy,
    refusals: set[str],
) -> None:
    reference = policy.reference_clock
    if clock.domain != reference.domain or clock.kind is not reference.kind or clock.epoch != reference.epoch:
        refusals.add("clock_reference_mismatch")
    evidence_age = reference.timestamp_ns - clock.timestamp_ns
    if evidence_age < 0:
        refusals.add("evidence_from_future")
    elif evidence_age > policy.max_evidence_age_ns:
        refusals.add("evidence_stale")
    if clock.timestamp_ns - calibrated_at_ns > policy.max_calibration_age_ns:
        refusals.add("calibration_stale")


def _check_semantics(
    handoff: MIFMergeCompressionHandoff,
    *,
    policy: MIFReactorSemanticAdmissionPolicy,
    refusals: set[str],
) -> None:
    actual = tuple(sorted((item.phase_id, item.carrier_type) for item in handoff.semantics))
    expected = tuple(sorted(policy.expected_semantic_carriers))
    if tuple(item[0] for item in actual) != tuple(item[0] for item in expected):
        refusals.add("semantic_identity_mismatch")
    elif actual != expected:
        refusals.add("semantic_carrier_mismatch")
    numerical = {item.phase_id for item in handoff.semantics if item.carrier_type is SemanticCarrier.NUMERICAL_PHASE}
    if numerical != policy.required_numerical_phase_ids:
        refusals.add("numerical_phase_set_mismatch")
    for semantic in handoff.semantics:
        if semantic.evidence_class is not EvidenceClass.SIMULATION:
            refusals.add("evidence_class_mismatch")
        if semantic.carrier_type is SemanticCarrier.NUMERICAL_PHASE:
            _check_numerical_semantic(semantic, policy=policy, refusals=refusals)


def _check_numerical_semantic(
    semantic: PhaseSemanticRecord,
    *,
    policy: MIFReactorSemanticAdmissionPolicy,
    refusals: set[str],
) -> None:
    if semantic.observability < policy.min_numerical_observability:
        refusals.add("numerical_phase_observability_below_policy")
    if semantic.confidence < policy.min_numerical_confidence:
        refusals.add("numerical_phase_confidence_below_policy")
    circular_std = semantic.uncertainty.circular_std_rad
    if circular_std is None or circular_std > policy.max_numerical_circular_std_rad:
        refusals.add("numerical_phase_uncertainty_above_policy")
    _check_validity(
        semantic.validity.state,
        semantic.validity.reasons,
        policy=policy,
        prefix="semantic",
        refusals=refusals,
    )
    _check_quality(
        semantic.quality.state,
        semantic.quality.flags,
        policy=policy,
        prefix="semantic",
        refusals=refusals,
    )


def _check_validity(
    state: ValidityState,
    reasons: tuple[str, ...],
    *,
    policy: MIFReactorSemanticAdmissionPolicy,
    prefix: str,
    refusals: set[str],
) -> None:
    if state is ValidityState.VALID:
        return
    if state is ValidityState.DEGRADED and set(reasons) <= policy.allowed_degradation_reasons:
        return
    if prefix == "semantic":
        refusals.add("semantic_validity_not_usable")
        return
    if state is ValidityState.DEGRADED:
        refusals.add("observable_degradation_not_allowed")
        return
    refusals.add(
        {
            ValidityState.UNKNOWN: "observable_validity_unknown",
            ValidityState.STALE: "observable_validity_stale",
            ValidityState.OUT_OF_DISTRIBUTION: "observable_validity_out_of_distribution",
            ValidityState.UNOBSERVABLE: "observable_validity_unobservable",
            ValidityState.INVALID: "observable_validity_invalid",
        }[state]
    )


def _check_quality(
    state: QualityState,
    flags: tuple[str, ...],
    *,
    policy: MIFReactorSemanticAdmissionPolicy,
    prefix: str,
    refusals: set[str],
) -> None:
    if state is QualityState.VALID:
        return
    if state is QualityState.DEGRADED and flags and set(flags) <= policy.allowed_quality_flags:
        return
    if prefix == "semantic":
        refusals.add("semantic_quality_not_usable")
        return
    if state is QualityState.DEGRADED:
        refusals.add("observable_quality_flags_undeclared" if not flags else "observable_quality_flags_not_allowed")
        return
    refusals.add(
        {
            QualityState.UNKNOWN: "observable_quality_unknown",
            QualityState.INVALID: "observable_quality_invalid",
        }[state]
    )


def _decision(
    *,
    policy: MIFReactorSemanticAdmissionPolicy,
    handoff: MIFMergeCompressionHandoff | None,
    handoff_sha256: str | None,
    refusal_codes: set[str],
) -> ReactorSemanticAdmissionDecision:
    admitted = not refusal_codes
    return ReactorSemanticAdmissionDecision(
        decision=(
            ReactorSemanticAdmissionStatus.ADMITTED_FOR_REVIEW if admitted else ReactorSemanticAdmissionStatus.REJECTED
        ),
        admitted=admitted,
        checked_at_ns=policy.reference_clock.timestamp_ns,
        handoff_sha256=handoff_sha256,
        event_id=None if handoff is None else handoff.event_id,
        context_id=None if handoff is None else handoff.context.context_id,
        source_schema=None if handoff is None else handoff.source_schema,
        source_revision=None if handoff is None else handoff.source_revision,
        source_envelope_sha256=(None if handoff is None else handoff.source_envelope_sha256),
        handoff_schema_version=None if handoff is None else handoff.schema_version,
        u0_schema_version=None if handoff is None else handoff.context.schema_version,
        registry_version=None if handoff is None else handoff.context.registry_version,
        registry_digest=None if handoff is None else handoff.context.registry_digest,
        refusal_codes=tuple(sorted(refusal_codes)),
    )


def _digest(value: str, field: str) -> None:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _string_set(value: object, field: str) -> None:
    if not isinstance(value, frozenset) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be a frozenset of non-empty strings")


def _string_pairs(value: object, field: str) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 or any(not isinstance(part, str) or not part for part in item)
        for item in value
    ):
        raise ValueError(f"{field} must be a tuple of non-empty string pairs")
    keys = [item[0] for item in value]
    if len(set(keys)) != len(keys) or tuple(sorted(value)) != value:
        raise ValueError(f"{field} must have sorted unique keys")


def _semantic_carriers(value: object) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not item[0]
        or not isinstance(item[1], SemanticCarrier)
        for item in value
    ):
        raise ValueError("expected_semantic_carriers must contain semantic ID and carrier pairs")
    keys = [item[0] for item in value]
    if len(set(keys)) != len(keys) or tuple(sorted(value)) != value:
        raise ValueError("expected_semantic_carriers must have sorted unique IDs")


def _probability(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be a probability")


__all__ = [
    "MIFReactorSemanticAdmissionPolicy",
    "admit_mif_reactor_semantic_handoff",
]
