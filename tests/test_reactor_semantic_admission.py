# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reactor semantic admission policy tests

"""Real-handoff tests for CONTROL's public reactor semantic admission."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from scpn_phase_orchestrator.reactor_semantics import (
    ClockKind,
    ClockReference,
    QualityAssessment,
    QualityState,
    ValidityState,
    ValidityWindow,
    coupled_transport_handoff_from_fusion_bytes,
    handoff_to_bytes,
)

from scpn_control.reactor_semantic_admission import (
    ReactorSemanticAdmissionPolicy,
    ReactorSemanticAdmissionStatus,
    admit_reactor_semantic_handoff,
)

FUSION_FIXTURE = Path(__file__).resolve().parent / "fixtures/reactor_semantic/torax_runtime_review_envelope_v1.json"
SOURCE_SCHEMA = "scpn-fusion-core.torax-runtime-review-envelope.v1"
SOURCE_REVISION = "314463489c95692d851cf6b9102ca733d878ca8a"
SOURCE_DIGEST = "b594e2f8b72056426d628b638f6a849ef39e75daddc827305002b109365596c4"
CALIBRATION_ID = "fusion.torax.simulation_declared_units.v1"
TRANSFER_ID = "fusion.torax.identity_projection.v1"


def _handoff_bytes() -> bytes:
    source = FUSION_FIXTURE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_DIGEST
    handoff = coupled_transport_handoff_from_fusion_bytes(
        source,
        expected_sha256=SOURCE_DIGEST,
    )
    return handoff_to_bytes(handoff)


def _policy(
    handoff: bytes,
    *,
    reference_timestamp_ns: int = 20_000_000,
    max_evidence_age_ns: int = 0,
    max_calibration_age_ns: int = 20_000_000,
    allowed_calibration_ids: frozenset[str] = frozenset({CALIBRATION_ID}),
    allowed_transfer_function_ids: frozenset[str] = frozenset({TRANSFER_ID}),
    allowed_degradation_reasons: frozenset[str] = frozenset(),
    allowed_quality_flags: frozenset[str] = frozenset(),
) -> ReactorSemanticAdmissionPolicy:
    return ReactorSemanticAdmissionPolicy(
        expected_handoff_sha256=hashlib.sha256(handoff).hexdigest(),
        expected_source_schema=SOURCE_SCHEMA,
        expected_source_revision=SOURCE_REVISION,
        expected_source_envelope_sha256=SOURCE_DIGEST,
        reference_clock=ClockReference(
            domain="simulation_monotonic",
            kind=ClockKind.SIMULATION_MONOTONIC,
            epoch="scenario_start",
            timestamp_ns=reference_timestamp_ns,
            sample_rate_hz=100.0,
            latency_s=0.0,
            picosecond_offset=0,
            synchronized_to=None,
        ),
        max_evidence_age_ns=max_evidence_age_ns,
        max_calibration_age_ns=max_calibration_age_ns,
        allowed_calibration_ids=allowed_calibration_ids,
        allowed_transfer_function_ids=allowed_transfer_function_ids,
        allowed_degradation_reasons=allowed_degradation_reasons,
        allowed_quality_flags=allowed_quality_flags,
    )


def test_real_handoff_is_admitted_only_for_non_actuating_review() -> None:
    """Admit the immutable real handoff without granting action authority."""
    handoff = _handoff_bytes()
    decision = admit_reactor_semantic_handoff(handoff, policy=_policy(handoff))

    assert decision.decision is ReactorSemanticAdmissionStatus.ADMITTED_FOR_REVIEW
    assert decision.admitted is True
    assert decision.review_only is True
    assert decision.actionable is False
    assert decision.refusal_codes == ()
    assert decision.handoff_sha256 == hashlib.sha256(handoff).hexdigest()
    assert decision.source_envelope_sha256 == SOURCE_DIGEST
    assert decision.source_revision == SOURCE_REVISION


@pytest.mark.parametrize(
    "payload",
    [b"", b"{}", b'{"x":1,"x":2}', b"\xff"],
)
def test_decoder_failures_return_no_guessed_upstream_identity(payload: bytes) -> None:
    """Keep decoded identity fields empty after any public-decoder failure."""
    decision = admit_reactor_semantic_handoff(payload, policy=_policy(_handoff_bytes()))

    assert decision.decision is ReactorSemanticAdmissionStatus.REJECTED
    assert decision.refusal_codes == ("handoff_decode_failed",)
    assert decision.event_id is None
    assert decision.context_id is None
    assert decision.source_schema is None
    assert decision.source_revision is None
    assert decision.source_envelope_sha256 is None


@pytest.mark.parametrize(
    ("policy_change", "expected_code"),
    [
        ({"expected_handoff_sha256": "0" * 64}, "handoff_digest_mismatch"),
        ({"expected_source_schema": "drift"}, "source_schema_mismatch"),
        ({"expected_source_revision": "0" * 40}, "source_revision_mismatch"),
        ({"expected_source_envelope_sha256": "0" * 64}, "source_envelope_digest_mismatch"),
    ],
)
def test_independent_expected_identity_mismatches_reject(
    policy_change: dict[str, object],
    expected_code: str,
) -> None:
    """Reject caller expectations that differ from the decoded receipt."""
    handoff = _handoff_bytes()
    policy = replace(_policy(handoff), **policy_change)
    decision = admit_reactor_semantic_handoff(handoff, policy=policy)

    assert decision.admitted is False
    assert decision.refusal_codes == (expected_code,)


@pytest.mark.parametrize(
    ("policy", "expected_code"),
    [
        (
            {"reference_timestamp_ns": 19_999_999},
            "evidence_from_future",
        ),
        (
            {"reference_timestamp_ns": 20_000_001},
            "evidence_stale",
        ),
        (
            {"max_calibration_age_ns": 19_999_999},
            "calibration_stale",
        ),
        (
            {"allowed_calibration_ids": frozenset()},
            "calibration_id_not_allowed",
        ),
        (
            {"allowed_transfer_function_ids": frozenset()},
            "transfer_function_id_not_allowed",
        ),
    ],
)
def test_freshness_and_calibration_policy_fail_closed(
    policy: dict[str, object],
    expected_code: str,
) -> None:
    """Reject evidence outside explicit time and calibration bounds."""
    handoff = _handoff_bytes()
    decision = admit_reactor_semantic_handoff(handoff, policy=_policy(handoff, **policy))

    assert expected_code in decision.refusal_codes
    assert decision.admitted is False


@pytest.mark.parametrize(
    "clock_change",
    [
        {"domain": "another_domain"},
        {"kind": ClockKind.MODEL_TICK},
        {"epoch": "another-scenario"},
    ],
)
def test_reference_clock_domain_kind_and_epoch_must_match(
    clock_change: dict[str, object],
) -> None:
    """Reject caller clock domain, kind, or epoch mismatch."""
    handoff = _handoff_bytes()
    base = _policy(handoff)
    mismatched = replace(
        base,
        reference_clock=replace(base.reference_clock, **clock_change),
    )
    decision = admit_reactor_semantic_handoff(handoff, policy=mismatched)

    assert decision.refusal_codes == ("clock_reference_mismatch",)


def test_explicitly_allowlisted_degraded_observable_can_be_reviewed() -> None:
    """Admit degraded evidence only when every declaration is allowlisted."""
    source = FUSION_FIXTURE.read_bytes()
    handoff = coupled_transport_handoff_from_fusion_bytes(source, expected_sha256=SOURCE_DIGEST)
    first = handoff.observables[0]
    degraded = replace(
        first,
        validity=ValidityWindow(
            state=ValidityState.DEGRADED,
            valid_from_ns=first.clock.timestamp_ns,
            valid_until_ns=first.clock.timestamp_ns,
            reasons=("bounded_numerical_difference",),
        ),
        quality=QualityAssessment(
            state=QualityState.DEGRADED,
            flags=("reviewed_numerical_refinement",),
        ),
    )
    changed = replace(handoff, observables=(degraded, *handoff.observables[1:]))
    encoded = handoff_to_bytes(changed)

    rejected = admit_reactor_semantic_handoff(encoded, policy=_policy(encoded))
    admitted = admit_reactor_semantic_handoff(
        encoded,
        policy=_policy(
            encoded,
            allowed_degradation_reasons=frozenset({"bounded_numerical_difference"}),
            allowed_quality_flags=frozenset({"reviewed_numerical_refinement"}),
        ),
    )

    assert rejected.refusal_codes == (
        "observable_degradation_not_allowed",
        "observable_quality_flags_not_allowed",
    )
    assert admitted.admitted is True


def test_degraded_quality_without_flags_is_rejected() -> None:
    """Reject degraded quality that declares no caller-reviewable flags."""
    source = FUSION_FIXTURE.read_bytes()
    handoff = coupled_transport_handoff_from_fusion_bytes(source, expected_sha256=SOURCE_DIGEST)
    first = handoff.observables[0]
    changed = replace(
        handoff,
        observables=(
            replace(
                first,
                quality=QualityAssessment(state=QualityState.DEGRADED, flags=()),
            ),
            *handoff.observables[1:],
        ),
    )
    encoded = handoff_to_bytes(changed)

    decision = admit_reactor_semantic_handoff(encoded, policy=_policy(encoded))

    assert decision.refusal_codes == ("observable_quality_flags_undeclared",)


@pytest.mark.parametrize("provenance_change", ["digest", "attribute"])
def test_observable_provenance_must_retain_the_fusion_digest_chain(
    provenance_change: str,
) -> None:
    """Reject a valid SPO handoff whose observable lineage was weakened."""
    source = FUSION_FIXTURE.read_bytes()
    handoff = coupled_transport_handoff_from_fusion_bytes(source, expected_sha256=SOURCE_DIGEST)
    first = handoff.observables[0]
    if provenance_change == "digest":
        provenance = replace(first.provenance, sha256="0" * 64)
    else:
        provenance = replace(
            first.provenance,
            attributes=tuple(
                (key, "non_identity" if key == "transfer" else value) for key, value in first.provenance.attributes
            ),
        )
    changed = replace(
        handoff,
        observables=(replace(first, provenance=provenance), *handoff.observables[1:]),
    )
    encoded = handoff_to_bytes(changed)

    decision = admit_reactor_semantic_handoff(encoded, policy=_policy(encoded))

    assert decision.refusal_codes == ("provenance_chain_mismatch",)


@pytest.mark.parametrize(
    ("validity", "quality", "expected_code"),
    [
        (ValidityState.UNKNOWN, QualityState.VALID, "observable_validity_unknown"),
        (ValidityState.STALE, QualityState.VALID, "observable_validity_stale"),
        (
            ValidityState.OUT_OF_DISTRIBUTION,
            QualityState.VALID,
            "observable_validity_out_of_distribution",
        ),
        (ValidityState.UNOBSERVABLE, QualityState.VALID, "observable_validity_unobservable"),
        (ValidityState.INVALID, QualityState.VALID, "observable_validity_invalid"),
        (ValidityState.UNKNOWN, QualityState.UNKNOWN, "observable_quality_unknown"),
        (ValidityState.UNKNOWN, QualityState.INVALID, "observable_quality_invalid"),
    ],
)
def test_nonusable_observable_states_always_reject(
    validity: ValidityState,
    quality: QualityState,
    expected_code: str,
) -> None:
    """Reject every nonusable observable validity or quality state."""
    source = FUSION_FIXTURE.read_bytes()
    handoff = coupled_transport_handoff_from_fusion_bytes(source, expected_sha256=SOURCE_DIGEST)
    first = handoff.observables[0]
    changed_observable = replace(
        first,
        validity=ValidityWindow(
            state=validity,
            valid_from_ns=first.clock.timestamp_ns,
            valid_until_ns=first.clock.timestamp_ns,
            reasons=("declared",) if validity is not ValidityState.VALID else (),
        ),
        quality=QualityAssessment(
            state=quality,
            flags=("declared",) if quality is not QualityState.VALID else (),
        ),
    )
    encoded = handoff_to_bytes(replace(handoff, observables=(changed_observable, *handoff.observables[1:])))
    decision = admit_reactor_semantic_handoff(encoded, policy=_policy(encoded))

    assert expected_code in decision.refusal_codes


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"expected_handoff_sha256": "x"}, "expected_handoff_sha256"),
        ({"expected_source_envelope_sha256": "x"}, "expected_source_envelope_sha256"),
        ({"expected_source_revision": "x"}, "expected_source_revision"),
        ({"expected_source_schema": ""}, "expected_source_schema"),
        ({"max_evidence_age_ns": -1}, "max_evidence_age_ns"),
        ({"max_evidence_age_ns": True}, "max_evidence_age_ns"),
        ({"max_calibration_age_ns": -1}, "max_calibration_age_ns"),
        ({"allowed_calibration_ids": {CALIBRATION_ID}}, "frozenset"),
        ({"allowed_transfer_function_ids": frozenset({""})}, "non-empty strings"),
        ({"allowed_quality_flags": frozenset({1})}, "non-empty strings"),
    ],
)
def test_policy_rejects_ambiguous_or_unbounded_inputs(
    change: dict[str, object],
    message: str,
) -> None:
    """Reject malformed digests, limits, identities, and allowlists."""
    with pytest.raises(ValueError, match=message):
        replace(_policy(_handoff_bytes()), **change)
