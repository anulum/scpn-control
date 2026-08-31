# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MIF semantic review admission tests

"""Policy and refusal tests for CONTROL's public MIF admission function."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from scpn_phase_orchestrator.reactor_semantics import (
    ClockKind,
    EvidenceClass,
    MIFMergeCompressionHandoff,
    QualityAssessment,
    QualityState,
    SemanticCarrier,
    Uncertainty,
    ValidityState,
    ValidityWindow,
    mif_merge_compression_handoff_from_mif_bytes,
    mif_merge_compression_handoff_to_bytes,
)

from scpn_control.reactor_semantic_admission.mif_admission import (
    MIFReactorSemanticAdmissionPolicy,
    admit_mif_reactor_semantic_handoff,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures/reactor_semantic/mif_merge_compression_observation_v1.json"
SOURCE_SHA256 = "c780706abd5a0b185a95e85767e623248388664da61126d196fcb3d528b0c0ca"
HANDOFF_SHA256 = "c0f03b7c49346c39342598275556e8ac28c93138ba14f6e21d6739400e0edeb2"
SOURCE_REVISION = "f60dbae4b2ea3344ac0cb086a3b7d248d65cf92f"
EVENT_ID = "shot_2026_08_30.event_0001"
CONTEXT_ID = "spo.mif.frc_compression.c780706abd5a0b185a95e857"
REGISTRY_DIGEST = "786d9542ce76c56dd7748fa948b17efed6c073525e527ce90e6d5e29a2d00090"
CALIBRATION_ID = "mif.merge_compression.model_declared_units.v1"
TRANSFER_ID = "mif.merge_compression.identity_projection.v1"


def _handoff() -> MIFMergeCompressionHandoff:
    source = FIXTURE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    return mif_merge_compression_handoff_from_mif_bytes(
        source,
        expected_sha256=SOURCE_SHA256,
    )


def _handoff_bytes(handoff: MIFMergeCompressionHandoff | None = None) -> bytes:
    resolved = _handoff() if handoff is None else handoff
    return mif_merge_compression_handoff_to_bytes(resolved)


def _policy(
    payload: bytes | None = None,
) -> MIFReactorSemanticAdmissionPolicy:
    handoff = _handoff()
    encoded = _handoff_bytes(handoff) if payload is None else payload
    observables = handoff.observables
    semantics = handoff.semantics
    return MIFReactorSemanticAdmissionPolicy(
        expected_handoff_sha256=hashlib.sha256(encoded).hexdigest(),
        expected_source_schema="scpn-mif-core.merge-compression-observation.v1",
        expected_source_revision=SOURCE_REVISION,
        expected_source_envelope_sha256=SOURCE_SHA256,
        expected_event_id=EVENT_ID,
        expected_context_id=CONTEXT_ID,
        expected_registry_version="1.0.0",
        expected_registry_digest=REGISTRY_DIGEST,
        expected_observation_clock=observables[0].clock,
        reference_clock=observables[0].clock,
        max_evidence_age_ns=0,
        max_calibration_age_ns=0,
        allowed_calibration_ids=frozenset({CALIBRATION_ID}),
        allowed_transfer_function_ids=frozenset({TRANSFER_ID}),
        expected_observable_ids=frozenset(item.observable_id for item in observables),
        expected_semantic_carriers=tuple(sorted((item.phase_id, item.carrier_type) for item in semantics)),
        required_numerical_phase_ids=frozenset(
            item.phase_id for item in semantics if item.carrier_type is SemanticCarrier.NUMERICAL_PHASE
        ),
        required_provenance_attributes=(
            ("backend", "python"),
            ("backend_version", "0.1.1"),
            ("uncertainty_basis", "serialized_model_state_not_physical_uncertainty"),
        ),
        min_numerical_observability=1.0,
        min_numerical_confidence=1.0,
        max_numerical_circular_std_rad=0.0,
    )


def test_exact_mif_handoff_is_admitted_for_review_only() -> None:
    """Admit the exact digest-pinned MIF handoff without action authority."""
    encoded = _handoff_bytes()
    assert hashlib.sha256(encoded).hexdigest() == HANDOFF_SHA256

    decision = admit_mif_reactor_semantic_handoff(encoded, policy=_policy(encoded))

    assert decision.admitted is True
    assert decision.review_only is True
    assert decision.actionable is False
    assert decision.refusal_codes == ()
    assert decision.event_id == EVENT_ID
    assert decision.context_id == CONTEXT_ID


@pytest.mark.parametrize("payload", [b"", b"{}", b"\xef\xbb\xbf{}", "not-bytes"])
def test_strict_decoder_failures_reject_without_guessed_identity(
    payload: bytes | str,
) -> None:
    """Reject malformed public bytes and keep decoded identities absent."""
    decision = admit_mif_reactor_semantic_handoff(
        cast(bytes, payload),
        policy=_policy(),
    )

    assert decision.admitted is False
    assert decision.refusal_codes == ("handoff_decode_failed",)
    assert decision.event_id is None
    assert decision.context_id is None


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"expected_handoff_sha256": "0" * 64}, "handoff_digest_mismatch"),
        ({"expected_source_schema": "example.invalid.v1"}, "source_schema_mismatch"),
        ({"expected_source_revision": "0" * 40}, "source_revision_mismatch"),
        ({"expected_source_envelope_sha256": "0" * 64}, "source_envelope_digest_mismatch"),
        ({"expected_event_id": "shot_other"}, "event_id_mismatch"),
        ({"expected_context_id": "spo.mif.other"}, "context_id_mismatch"),
        ({"expected_registry_version": "2.0.0"}, "registry_identity_mismatch"),
        ({"expected_registry_digest": "0" * 64}, "registry_identity_mismatch"),
    ],
)
def test_exact_identity_policy_mismatches_fail_closed(
    change: dict[str, object],
    expected: str,
) -> None:
    """Reject digest, producer, event, context, and registry drift."""
    encoded = _handoff_bytes()
    decision = admit_mif_reactor_semantic_handoff(
        encoded,
        policy=replace(_policy(encoded), **cast(Any, change)),
    )

    assert expected in decision.refusal_codes


def test_clock_freshness_calibration_and_allowlists_are_independent() -> None:
    """Report each clock, freshness, calibration, and allowlist failure."""
    encoded = _handoff_bytes()
    base = _policy(encoded)
    mismatched_clock = replace(base.expected_observation_clock, sample_rate_hz=2_000.0)
    wrong_reference = replace(base.reference_clock, domain="other_model_time")
    wrong_kind = replace(base.reference_clock, kind=ClockKind.PLANT_MONOTONIC)
    wrong_epoch = replace(base.reference_clock, epoch="other_epoch")
    stale_reference = replace(base.reference_clock, timestamp_ns=1)

    decisions = (
        admit_mif_reactor_semantic_handoff(
            encoded,
            policy=replace(base, expected_observation_clock=mismatched_clock),
        ),
        admit_mif_reactor_semantic_handoff(
            encoded,
            policy=replace(base, reference_clock=wrong_reference),
        ),
        admit_mif_reactor_semantic_handoff(
            encoded,
            policy=replace(base, reference_clock=wrong_kind),
        ),
        admit_mif_reactor_semantic_handoff(
            encoded,
            policy=replace(base, reference_clock=wrong_epoch),
        ),
        admit_mif_reactor_semantic_handoff(
            encoded,
            policy=replace(base, reference_clock=stale_reference),
        ),
        admit_mif_reactor_semantic_handoff(
            encoded,
            policy=replace(base, allowed_calibration_ids=frozenset({"other.calibration"})),
        ),
        admit_mif_reactor_semantic_handoff(
            encoded,
            policy=replace(
                base,
                allowed_transfer_function_ids=frozenset({"other.transfer"}),
            ),
        ),
    )

    assert "observation_clock_mismatch" in decisions[0].refusal_codes
    assert "clock_reference_mismatch" in decisions[1].refusal_codes
    assert "clock_reference_mismatch" in decisions[2].refusal_codes
    assert "clock_reference_mismatch" in decisions[3].refusal_codes
    assert "evidence_stale" in decisions[4].refusal_codes
    assert "calibration_id_not_allowed" in decisions[5].refusal_codes
    assert "transfer_function_id_not_allowed" in decisions[6].refusal_codes


def test_future_and_stale_calibration_reject() -> None:
    """Reject samples from the future and calibration older than policy."""
    handoff = _handoff()
    shifted_clock = replace(handoff.observables[0].clock, timestamp_ns=10)
    shifted_observables = tuple(replace(item, clock=shifted_clock) for item in handoff.observables)
    shifted = replace(handoff, observables=shifted_observables)
    encoded = _handoff_bytes(shifted)
    policy = replace(
        _policy(encoded),
        expected_observation_clock=shifted_clock,
        reference_clock=replace(shifted_clock, timestamp_ns=9),
        max_calibration_age_ns=0,
    )

    decision = admit_mif_reactor_semantic_handoff(encoded, policy=policy)

    assert "evidence_from_future" in decision.refusal_codes
    assert "calibration_stale" in decision.refusal_codes


def test_observable_set_and_provenance_must_match_policy() -> None:
    """Reject incomplete expected sets and producer-provenance drift."""
    handoff = _handoff()
    encoded = _handoff_bytes(handoff)
    base = _policy(encoded)
    reduced_ids = frozenset(sorted(base.expected_observable_ids)[1:])
    missing = admit_mif_reactor_semantic_handoff(
        encoded,
        policy=replace(base, expected_observable_ids=reduced_ids),
    )
    first = handoff.observables[0]
    changed_provenance = replace(
        first.provenance,
        attributes=tuple(
            (key, "changed" if key == "backend_version" else value) for key, value in first.provenance.attributes
        ),
    )
    changed = replace(
        handoff,
        observables=(replace(first, provenance=changed_provenance), *handoff.observables[1:]),
    )
    changed_bytes = _handoff_bytes(changed)
    provenance = admit_mif_reactor_semantic_handoff(
        changed_bytes,
        policy=_policy(changed_bytes),
    )

    assert "observable_set_mismatch" in missing.refusal_codes
    assert "provenance_chain_mismatch" in provenance.refusal_codes


def test_context_and_semantic_evidence_must_remain_simulation() -> None:
    """Reject context or semantic maturity that exceeds the MIF source claim."""
    handoff = _handoff()
    changed_context = replace(
        handoff.context,
        evidence_class=EvidenceClass.EXPERIMENTAL,
    )
    changed_observables = tuple(replace(item, reactor_context=changed_context) for item in handoff.observables)
    context_handoff = replace(
        handoff,
        context=changed_context,
        observables=changed_observables,
    )
    context_bytes = _handoff_bytes(context_handoff)
    context_decision = admit_mif_reactor_semantic_handoff(
        context_bytes,
        policy=_policy(context_bytes),
    )
    nonphase = handoff.semantics[2]
    semantic_handoff = replace(
        handoff,
        semantics=(
            *handoff.semantics[:2],
            replace(nonphase, evidence_class=EvidenceClass.EXPERIMENTAL),
            *handoff.semantics[3:],
        ),
    )
    semantic_bytes = _handoff_bytes(semantic_handoff)
    semantic_decision = admit_mif_reactor_semantic_handoff(
        semantic_bytes,
        policy=_policy(semantic_bytes),
    )

    assert "evidence_class_mismatch" in context_decision.refusal_codes
    assert "evidence_class_mismatch" in semantic_decision.refusal_codes


@pytest.mark.parametrize(
    ("validity", "quality", "expected"),
    [
        (ValidityState.DEGRADED, QualityState.VALID, "observable_degradation_not_allowed"),
        (ValidityState.STALE, QualityState.VALID, "observable_validity_stale"),
        (ValidityState.UNKNOWN, QualityState.VALID, "observable_validity_unknown"),
        (ValidityState.OUT_OF_DISTRIBUTION, QualityState.VALID, "observable_validity_out_of_distribution"),
        (ValidityState.UNOBSERVABLE, QualityState.VALID, "observable_validity_unobservable"),
        (ValidityState.INVALID, QualityState.VALID, "observable_validity_invalid"),
        (ValidityState.UNKNOWN, QualityState.UNKNOWN, "observable_quality_unknown"),
        (ValidityState.UNKNOWN, QualityState.INVALID, "observable_quality_invalid"),
    ],
)
def test_nonusable_observable_states_reject(
    validity: ValidityState,
    quality: QualityState,
    expected: str,
) -> None:
    """Reject every nonusable source-observable state."""
    handoff = _handoff()
    first = handoff.observables[0]
    changed = replace(
        first,
        validity=ValidityWindow(
            validity,
            valid_from_ns=0,
            valid_until_ns=10_000_000,
            reasons=("declared",) if validity is not ValidityState.VALID else (),
        ),
        quality=QualityAssessment(
            quality,
            flags=("declared",) if quality is not QualityState.VALID else (),
        ),
    )
    modified = replace(
        handoff,
        observables=(changed, *handoff.observables[1:]),
    )
    encoded = _handoff_bytes(modified)
    decision = admit_mif_reactor_semantic_handoff(encoded, policy=_policy(encoded))

    assert expected in decision.refusal_codes


def test_degraded_observable_requires_declared_allowlists() -> None:
    """Allow degraded source evidence only under exact reason and flag policy."""
    handoff = _handoff()
    first = handoff.observables[0]
    changed = replace(
        first,
        validity=ValidityWindow(
            ValidityState.DEGRADED,
            valid_from_ns=0,
            valid_until_ns=10_000_000,
            reasons=("model_review",),
        ),
        quality=QualityAssessment(QualityState.DEGRADED, flags=("model_review",)),
    )
    modified = replace(
        handoff,
        observables=(changed, *handoff.observables[1:]),
    )
    encoded = _handoff_bytes(modified)
    decision = admit_mif_reactor_semantic_handoff(
        encoded,
        policy=replace(
            _policy(encoded),
            allowed_degradation_reasons=frozenset({"model_review"}),
            allowed_quality_flags=frozenset({"model_review"}),
        ),
    )

    assert decision.admitted is True


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((), "observable_quality_flags_undeclared"),
        (("model_review",), "observable_quality_flags_not_allowed"),
    ],
)
def test_degraded_observable_quality_requires_allowed_flags(
    flags: tuple[str, ...],
    expected: str,
) -> None:
    """Distinguish undeclared degraded quality from disallowed flags."""
    handoff = _handoff()
    first = handoff.observables[0]
    changed = replace(
        first,
        quality=QualityAssessment(QualityState.DEGRADED, flags=flags),
    )
    modified = replace(
        handoff,
        observables=(changed, *handoff.observables[1:]),
    )
    encoded = _handoff_bytes(modified)
    decision = admit_mif_reactor_semantic_handoff(encoded, policy=_policy(encoded))

    assert expected in decision.refusal_codes


def test_semantic_identity_carriers_and_numerical_set_are_closed() -> None:
    """Reject semantic ID, carrier, and numerical-phase allowlist drift."""
    encoded = _handoff_bytes()
    base = _policy(encoded)
    carriers = list(base.expected_semantic_carriers)
    first_id, _ = carriers[0]
    wrong_carriers = tuple(sorted(((first_id, SemanticCarrier.CATEGORICAL_STATE), *carriers[1:])))
    wrong_identity = tuple(sorted((("spo.mif.unexpected", carriers[0][1]), *carriers[1:])))

    carrier_decision = admit_mif_reactor_semantic_handoff(
        encoded,
        policy=replace(base, expected_semantic_carriers=wrong_carriers),
    )
    identity_decision = admit_mif_reactor_semantic_handoff(
        encoded,
        policy=replace(base, expected_semantic_carriers=wrong_identity),
    )
    numerical_decision = admit_mif_reactor_semantic_handoff(
        encoded,
        policy=replace(base, required_numerical_phase_ids=frozenset()),
    )

    assert "semantic_carrier_mismatch" in carrier_decision.refusal_codes
    assert "semantic_identity_mismatch" in identity_decision.refusal_codes
    assert "numerical_phase_set_mismatch" in numerical_decision.refusal_codes


def test_numerical_phase_metrics_uncertainty_and_state_are_policy_gated() -> None:
    """Reject numerical phase that falls below the review evidence policy."""
    handoff = _handoff()
    first = handoff.semantics[0]
    changed = replace(
        first,
        confidence=0.5,
        observability=0.5,
        observability_threshold=0.5,
        uncertainty=Uncertainty(
            standard_deviation=0.25,
            confidence_level=0.5,
            circular_std_rad=0.25,
        ),
        validity=ValidityWindow(
            ValidityState.DEGRADED,
            valid_from_ns=0,
            valid_until_ns=10_000_000,
            reasons=("model_review",),
        ),
        quality=QualityAssessment(QualityState.DEGRADED, flags=("model_review",)),
    )
    modified = replace(
        handoff,
        semantics=(changed, *handoff.semantics[1:]),
    )
    encoded = _handoff_bytes(modified)
    decision = admit_mif_reactor_semantic_handoff(encoded, policy=_policy(encoded))

    assert {
        "numerical_phase_confidence_below_policy",
        "numerical_phase_observability_below_policy",
        "numerical_phase_uncertainty_above_policy",
        "semantic_quality_not_usable",
        "semantic_validity_not_usable",
    } <= set(decision.refusal_codes)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"expected_handoff_sha256": "x"}, "expected_handoff_sha256"),
        ({"expected_source_revision": "x"}, "expected_source_revision"),
        ({"expected_source_schema": ""}, "expected_source_schema"),
        ({"expected_observation_clock": object()}, "expected_observation_clock must be a ClockReference"),
        ({"reference_clock": object()}, "reference_clock must be a ClockReference"),
        ({"max_evidence_age_ns": -1}, "max_evidence_age_ns"),
        ({"max_evidence_age_ns": True}, "max_evidence_age_ns"),
        ({"allowed_calibration_ids": {CALIBRATION_ID}}, "frozenset"),
        ({"allowed_calibration_ids": frozenset({""})}, "non-empty strings"),
        ({"min_numerical_confidence": 2.0}, "probability"),
        ({"min_numerical_confidence": True}, "probability"),
        ({"min_numerical_confidence": "x"}, "probability"),
        ({"max_numerical_circular_std_rad": -1.0}, "non-negative number"),
        ({"max_numerical_circular_std_rad": float("nan")}, "non-negative number"),
        ({"max_numerical_circular_std_rad": float("inf")}, "non-negative number"),
        ({"max_numerical_circular_std_rad": float("-inf")}, "non-negative number"),
        ({"max_numerical_circular_std_rad": True}, "non-negative number"),
        ({"max_numerical_circular_std_rad": "x"}, "non-negative number"),
        ({"required_provenance_attributes": []}, "string pairs"),
        ({"required_provenance_attributes": (("x", ""),)}, "string pairs"),
        ({"required_provenance_attributes": (("z", "1"), ("a", "2"))}, "sorted unique"),
        ({"expected_semantic_carriers": []}, "semantic ID"),
        ({"expected_semantic_carriers": (("x", "numerical_phase"),)}, "semantic ID"),
        (
            {
                "expected_semantic_carriers": (
                    ("z", SemanticCarrier.NUMERICAL_PHASE),
                    ("a", SemanticCarrier.NUMERICAL_PHASE),
                )
            },
            "sorted unique IDs",
        ),
        ({"required_numerical_phase_ids": frozenset({"missing"})}, "present in expected semantics"),
    ],
)
def test_policy_rejects_ambiguous_or_unbounded_inputs(
    change: dict[str, object],
    message: str,
) -> None:
    """Reject malformed identity pins, sets, thresholds, and mappings."""
    with pytest.raises(ValueError, match=message):
        replace(_policy(), **cast(Any, change))
