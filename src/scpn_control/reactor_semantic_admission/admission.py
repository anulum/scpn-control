# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reactor semantic review admission

"""Fail-closed admission of portable SPO reactor semantic handoffs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final

from scpn_phase_orchestrator.reactor_semantics import (
    ClockReference,
    QualityState,
    ReactorSemanticHandoff,
    ValidityState,
    handoff_digest,
    handoff_from_bytes,
)

from .decision import ReactorSemanticAdmissionDecision, ReactorSemanticAdmissionStatus

_FUSION_PROJECT: Final = "SCPN-FUSION-CORE"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReactorSemanticAdmissionPolicy:
    """Caller-owned deterministic admission expectations.

    Parameters
    ----------
    expected_handoff_sha256:
        Digest of the exact canonical SPO handoff bytes.
    expected_source_schema:
        Exact FUSION source-envelope schema.
    expected_source_revision:
        Exact 40-character FUSION producer commit.
    expected_source_envelope_sha256:
        Digest of the embedded canonical FUSION envelope.
    reference_clock:
        Explicit evaluation clock. Its domain, kind, and epoch must match every
        observable; its timestamp supplies ``checked_at_ns``.
    max_evidence_age_ns, max_calibration_age_ns:
        Inclusive non-negative freshness limits.
    allowed_calibration_ids, allowed_transfer_function_ids:
        Complete allowlists for every observable's declared calibration.
    allowed_degradation_reasons, allowed_quality_flags:
        Explicit allowlists for degraded evidence. Empty sets reject degraded
        evidence; they do not affect the expected nonphase semantic records.
    """

    expected_handoff_sha256: str
    expected_source_schema: str
    expected_source_revision: str
    expected_source_envelope_sha256: str
    reference_clock: ClockReference
    max_evidence_age_ns: int
    max_calibration_age_ns: int
    allowed_calibration_ids: frozenset[str]
    allowed_transfer_function_ids: frozenset[str]
    allowed_degradation_reasons: frozenset[str] = frozenset()
    allowed_quality_flags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate caller policy without consulting wall time."""
        _digest(self.expected_handoff_sha256, "expected_handoff_sha256")
        _digest(
            self.expected_source_envelope_sha256,
            "expected_source_envelope_sha256",
        )
        if _HEX_40.fullmatch(self.expected_source_revision) is None:
            raise ValueError("expected_source_revision must be a lowercase Git commit")
        if not self.expected_source_schema:
            raise ValueError("expected_source_schema must be non-empty")
        for field, value in (
            ("max_evidence_age_ns", self.max_evidence_age_ns),
            ("max_calibration_age_ns", self.max_calibration_age_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        for field, values in (
            ("allowed_calibration_ids", self.allowed_calibration_ids),
            ("allowed_transfer_function_ids", self.allowed_transfer_function_ids),
            ("allowed_degradation_reasons", self.allowed_degradation_reasons),
            ("allowed_quality_flags", self.allowed_quality_flags),
        ):
            if not isinstance(values, frozenset) or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{field} must be a frozenset of non-empty strings")


def admit_reactor_semantic_handoff(
    payload: bytes,
    *,
    policy: ReactorSemanticAdmissionPolicy,
) -> ReactorSemanticAdmissionDecision:
    """Admit one exact SPO handoff for non-actuating CONTROL review.

    The function calls SPO's public :func:`handoff_from_bytes` decoder as the
    sole portable ingress. Decoder failures produce a sealed rejected decision
    with no guessed upstream identity. No wall clock or control-action surface
    is read, constructed, forwarded, or serialized.
    """
    raw_digest = hashlib.sha256(payload).hexdigest() if isinstance(payload, bytes) else None
    try:
        handoff = handoff_from_bytes(payload)
    except (TypeError, ValueError):
        return _decision(
            policy=policy,
            handoff=None,
            handoff_sha256=raw_digest,
            refusal_codes={"handoff_decode_failed"},
        )

    refusal_codes = _evaluate_handoff(
        handoff,
        handoff_sha256=raw_digest,
        policy=policy,
    )
    return _decision(
        policy=policy,
        handoff=handoff,
        handoff_sha256=raw_digest,
        refusal_codes=refusal_codes,
    )


def _evaluate_handoff(
    handoff: ReactorSemanticHandoff,
    *,
    handoff_sha256: str | None,
    policy: ReactorSemanticAdmissionPolicy,
) -> set[str]:
    refusals: set[str] = set()
    _check_identity(handoff, handoff_sha256=handoff_sha256, policy=policy, refusals=refusals)
    for observable in handoff.observables:
        clock = observable.clock
        if (
            clock.domain != policy.reference_clock.domain
            or clock.kind is not policy.reference_clock.kind
            or clock.epoch != policy.reference_clock.epoch
        ):
            refusals.add("clock_reference_mismatch")
        evidence_age = policy.reference_clock.timestamp_ns - clock.timestamp_ns
        if evidence_age < 0:
            refusals.add("evidence_from_future")
        elif evidence_age > policy.max_evidence_age_ns:
            refusals.add("evidence_stale")
        calibrated_age = clock.timestamp_ns - observable.calibration.calibrated_at_ns
        if calibrated_age > policy.max_calibration_age_ns:
            refusals.add("calibration_stale")
        if observable.calibration.calibration_id not in policy.allowed_calibration_ids:
            refusals.add("calibration_id_not_allowed")
        if observable.calibration.transfer_function_id not in policy.allowed_transfer_function_ids:
            refusals.add("transfer_function_id_not_allowed")
        _check_observable_validity(observable.validity.state, observable.validity.reasons, policy, refusals)
        _check_observable_quality(observable.quality.state, observable.quality.flags, policy, refusals)
        _check_provenance(handoff, observable.provenance, refusals=refusals)
    return refusals


def _check_identity(
    handoff: ReactorSemanticHandoff,
    *,
    handoff_sha256: str | None,
    policy: ReactorSemanticAdmissionPolicy,
    refusals: set[str],
) -> None:
    if handoff_sha256 != policy.expected_handoff_sha256 or handoff_digest(handoff) != policy.expected_handoff_sha256:
        refusals.add("handoff_digest_mismatch")
    if handoff.source_schema != policy.expected_source_schema:
        refusals.add("source_schema_mismatch")
    if handoff.source_revision != policy.expected_source_revision:
        refusals.add("source_revision_mismatch")
    if handoff.source_envelope_sha256 != policy.expected_source_envelope_sha256:
        refusals.add("source_envelope_digest_mismatch")


def _check_observable_validity(
    state: ValidityState,
    reasons: tuple[str, ...],
    policy: ReactorSemanticAdmissionPolicy,
    refusals: set[str],
) -> None:
    if state is ValidityState.VALID:
        return
    if state is ValidityState.DEGRADED:
        if not set(reasons) <= policy.allowed_degradation_reasons:
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


def _check_observable_quality(
    state: QualityState,
    flags: tuple[str, ...],
    policy: ReactorSemanticAdmissionPolicy,
    refusals: set[str],
) -> None:
    if state is QualityState.VALID:
        return
    if state is QualityState.DEGRADED:
        if not flags:
            refusals.add("observable_quality_flags_undeclared")
        elif not set(flags) <= policy.allowed_quality_flags:
            refusals.add("observable_quality_flags_not_allowed")
        return
    refusals.add(
        {
            QualityState.UNKNOWN: "observable_quality_unknown",
            QualityState.INVALID: "observable_quality_invalid",
        }[state]
    )


def _check_provenance(
    handoff: ReactorSemanticHandoff,
    provenance: object,
    *,
    refusals: set[str],
) -> None:
    source_project = getattr(provenance, "source_project", None)
    sha256 = getattr(provenance, "sha256", None)
    attributes = dict(getattr(provenance, "attributes", ()))
    expected = {
        "calibration_basis": "simulation_declared_units",
        "calibration_empirical": "false",
        "event_id": handoff.event_id,
        "fuel_class_basis": "deuterium_only_input_no_fusion_power_or_burn_model",
        "producer_revision": handoff.source_revision,
        "source_envelope_sha256": handoff.source_envelope_sha256,
        "transfer": "identity",
    }
    if (
        source_project != _FUSION_PROJECT
        or sha256 != handoff.source_envelope_sha256
        or any(attributes.get(key) != value for key, value in expected.items())
    ):
        refusals.add("provenance_chain_mismatch")


def _decision(
    *,
    policy: ReactorSemanticAdmissionPolicy,
    handoff: ReactorSemanticHandoff | None,
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
    if _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


__all__ = [
    "ReactorSemanticAdmissionPolicy",
    "admit_reactor_semantic_handoff",
]
