# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reactor regime assessment admission tests

"""Real public-SPO tests for exact abstaining-assessment admission."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any, cast

import pytest
from scpn_phase_orchestrator.reactor_semantics import (
    EvidenceClass,
    QualityState,
    ReactorRegimeAssessment,
    ReactorRegimeAxisDisposition,
    ReactorRegimeEvidenceBinding,
    ValidityState,
    build_abstaining_regime_assessment,
    mif_merge_compression_handoff_from_mif_bytes,
    regime_assessment_to_bytes,
)

from scpn_control.reactor_semantic_admission import (
    ReactorRegimeAssessmentAdmissionPolicy,
    ReactorRegimeAssessmentAdmissionStatus,
    admit_reactor_regime_assessment,
    regime_assessment_axis_custody_digest,
    regime_assessment_clock_custody_digest,
    regime_assessment_registry_custody_digest,
)

MIF_FIXTURE = Path(__file__).resolve().parent / "fixtures/reactor_semantic/mif_merge_compression_observation_v1.json"
MIF_SOURCE_SHA256 = "c780706abd5a0b185a95e85767e623248388664da61126d196fcb3d528b0c0ca"
SPO_REVISION = "c2a7581d58819060806c6f173da941c822103695"
SPO_WHEEL_SHA256 = "c2d7c0a5c0ad47f420fee02e54ccc28122bf8d128eb3b80ca51ba5f034320274"
ASSESSMENT_SHA256 = "3a5077b95d8b94b23a647d57a8b25f80cb798f712f00d0a34e71b95c600b154b"


@cache
def _assessment() -> tuple[ReactorRegimeAssessment, bytes]:
    source = MIF_FIXTURE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == MIF_SOURCE_SHA256
    handoff = mif_merge_compression_handoff_from_mif_bytes(
        source,
        expected_sha256=MIF_SOURCE_SHA256,
    )
    assessment = build_abstaining_regime_assessment(
        handoff,
        producer_revision=SPO_REVISION,
        producer_artifact_sha256=SPO_WHEEL_SHA256,
    )
    payload = regime_assessment_to_bytes(assessment)
    assert hashlib.sha256(payload).hexdigest() == ASSESSMENT_SHA256
    return assessment, payload


def _policy(
    assessment: ReactorRegimeAssessment,
    payload: bytes,
    *,
    checked_at_ns: int = 0,
    max_evidence_age_ns: int = 0,
) -> ReactorRegimeAssessmentAdmissionPolicy:
    return ReactorRegimeAssessmentAdmissionPolicy(
        expected_assessment_sha256=hashlib.sha256(payload).hexdigest(),
        expected_assessment_id=assessment.assessment_id,
        expected_reactor_context_id=assessment.reactor_context_id,
        expected_configuration=assessment.configuration,
        expected_event_id=assessment.event_id,
        expected_producer_project=assessment.producer_project,
        expected_producer_revision=assessment.producer_revision,
        expected_producer_artifact_sha256=assessment.producer_artifact_sha256,
        expected_source_project=assessment.source_project,
        expected_source_revision=assessment.source_revision,
        expected_source_handoff_schema=assessment.source_handoff_schema,
        expected_source_handoff_sha256=assessment.source_handoff_sha256,
        expected_source_semantic_ids=assessment.source_semantic_ids,
        expected_assessment_schema_version=assessment.schema_version,
        expected_registry_custody_sha256=regime_assessment_registry_custody_digest(assessment),
        expected_clock_custody_sha256=regime_assessment_clock_custody_digest(assessment),
        expected_axis_custody_sha256=regime_assessment_axis_custody_digest(assessment),
        expected_axis_ids=tuple(axis.axis_id for axis in assessment.axes),
        expected_axis_provenance=tuple((axis.axis_id, axis.provenance_id) for axis in assessment.axes),
        checked_at_ns=checked_at_ns,
        max_evidence_age_ns=max_evidence_age_ns,
    )


def _encoded(assessment: ReactorRegimeAssessment) -> bytes:
    return regime_assessment_to_bytes(assessment)


def test_real_public_spo_assessment_is_admitted_for_review_only() -> None:
    """Admit the exact published SPO-builder result without action authority."""
    assessment, payload = _assessment()
    decision = admit_reactor_regime_assessment(payload, policy=_policy(assessment, payload))

    assert decision.decision is ReactorRegimeAssessmentAdmissionStatus.ADMITTED_FOR_REVIEW
    assert decision.admitted is True
    assert decision.review_only is True
    assert decision.actionable is False
    assert decision.refusal_codes == ()
    assert decision.assessment_sha256 == ASSESSMENT_SHA256
    assert decision.assessment_id == assessment.assessment_id
    assert decision.source_handoff_sha256 == assessment.source_handoff_sha256
    assert decision.registry_custody_sha256 == regime_assessment_registry_custody_digest(assessment)
    assert decision.clock_custody_sha256 == regime_assessment_clock_custody_digest(assessment)
    assert decision.axis_custody_sha256 == regime_assessment_axis_custody_digest(assessment)


@pytest.mark.parametrize("payload", [b"", b"{}", b'{"x":1,"x":2}', b"\xff"])
def test_decode_failure_returns_no_guessed_assessment_identity(payload: bytes) -> None:
    """Retain only the raw digest when SPO cannot decode the input."""
    assessment, valid = _assessment()
    decision = admit_reactor_regime_assessment(payload, policy=_policy(assessment, valid))

    assert decision.refusal_codes == ("assessment_decode_failed",)
    assert decision.assessment_sha256 == hashlib.sha256(payload).hexdigest()
    assert decision.assessment_id is None
    assert decision.producer_project is None
    assert decision.source_project is None
    assert decision.registry_custody_sha256 is None


def test_non_byte_ingress_fails_closed_without_a_raw_digest() -> None:
    """Reject non-byte input without manufacturing a byte identity."""
    assessment, valid = _assessment()
    decision = admit_reactor_regime_assessment(
        cast(bytes, "not-bytes"),
        policy=_policy(assessment, valid),
    )

    assert decision.refusal_codes == ("assessment_decode_failed",)
    assert decision.assessment_sha256 is None


@pytest.mark.parametrize(
    ("change", "expected_code"),
    [
        ({"expected_assessment_sha256": "0" * 64}, "assessment_digest_mismatch"),
        ({"expected_assessment_id": "other.assessment"}, "assessment_identity_mismatch"),
        ({"expected_event_id": "other.event"}, "event_context_mismatch"),
        ({"expected_reactor_context_id": "other.context"}, "event_context_mismatch"),
        ({"expected_configuration": "stellarator"}, "event_context_mismatch"),
        ({"expected_producer_project": "OTHER"}, "producer_identity_mismatch"),
        ({"expected_producer_revision": "0" * 40}, "producer_identity_mismatch"),
        ({"expected_producer_artifact_sha256": "0" * 64}, "producer_identity_mismatch"),
        ({"expected_source_project": "SCPN-FUSION-CORE"}, "source_identity_mismatch"),
        ({"expected_source_revision": "0" * 40}, "source_identity_mismatch"),
        ({"expected_source_handoff_schema": "other.schema.v1"}, "source_identity_mismatch"),
        (
            {"expected_assessment_schema_version": "1.0.1"},
            "assessment_schema_version_mismatch",
        ),
        ({"expected_source_handoff_sha256": "0" * 64}, "source_handoff_digest_mismatch"),
        (
            {"expected_source_semantic_ids": ("different.semantic",)},
            "source_semantic_identity_mismatch",
        ),
        ({"expected_registry_custody_sha256": "0" * 64}, "registry_custody_mismatch"),
        ({"expected_clock_custody_sha256": "0" * 64}, "clock_custody_mismatch"),
        ({"expected_axis_custody_sha256": "0" * 64}, "axis_custody_mismatch"),
    ],
)
def test_exact_identity_and_custody_expectations_fail_independently(
    change: dict[str, object],
    expected_code: str,
) -> None:
    """Reject each caller-owned identity or custody drift explicitly."""
    assessment, payload = _assessment()
    policy = replace(_policy(assessment, payload), **cast(Any, change))
    decision = admit_reactor_regime_assessment(payload, policy=policy)

    assert decision.admitted is False
    assert expected_code in decision.refusal_codes


def test_future_stale_and_outside_validity_times_fail_closed() -> None:
    """Apply caller time deterministically without reading wall time."""
    base, _payload = _assessment()
    future = replace(base, evidence_timestamp_ns=1, assessed_at_ns=1)
    future_bytes = _encoded(future)
    future_decision = admit_reactor_regime_assessment(
        future_bytes,
        policy=_policy(future, future_bytes, checked_at_ns=0),
    )
    assert future_decision.refusal_codes == ("evidence_from_future",)

    base_bytes = _encoded(base)
    stale = admit_reactor_regime_assessment(
        base_bytes,
        policy=_policy(base, base_bytes, checked_at_ns=1, max_evidence_age_ns=0),
    )
    assert stale.refusal_codes == ("evidence_stale",)

    outside = admit_reactor_regime_assessment(
        base_bytes,
        policy=_policy(
            base,
            base_bytes,
            checked_at_ns=base.valid_until_ns + 1,
            max_evidence_age_ns=base.valid_until_ns + 1,
        ),
    )
    assert outside.refusal_codes == ("outside_common_validity",)


def test_axis_id_and_provenance_policy_drift_is_explicit() -> None:
    """Reject ordered axis identity and per-axis provenance drift."""
    assessment, payload = _assessment()
    base = _policy(assessment, payload)
    changed_ids = tuple(f"x.{axis}" for axis in base.expected_axis_ids)
    changed_provenance = tuple((axis, f"p.{axis}") for axis in changed_ids)
    id_policy = replace(
        base,
        expected_axis_ids=changed_ids,
        expected_axis_provenance=changed_provenance,
    )
    assert admit_reactor_regime_assessment(payload, policy=id_policy).refusal_codes == (
        "axis_custody_mismatch",
        "axis_provenance_policy_mismatch",
    )

    provenance_policy = replace(
        base,
        expected_axis_provenance=tuple((axis, f"other.{axis}") for axis in base.expected_axis_ids),
    )
    assert admit_reactor_regime_assessment(payload, policy=provenance_policy).refusal_codes == (
        "axis_provenance_policy_mismatch",
    )


def test_unknown_axis_with_supplied_evidence_is_not_the_abstaining_builder_profile() -> None:
    """Reject otherwise valid unknown-axis evidence, quality, and observability drift."""
    assessment, _payload = _assessment()
    index = next(
        index for index, axis in enumerate(assessment.axes) if axis.disposition is ReactorRegimeAxisDisposition.UNKNOWN
    )
    axis = assessment.axes[index]
    changed_axis = replace(
        axis,
        evidence_ids=("evidence.synthetic.1",),
        observability=0.5,
        evidence_class=EvidenceClass.SIMULATION,
        validity=ValidityState.VALID,
        quality=QualityState.VALID,
    )
    axes = assessment.axes[:index] + (changed_axis,) + assessment.axes[index + 1 :]
    changed = replace(assessment, axes=axes)
    payload = _encoded(changed)
    decision = admit_reactor_regime_assessment(payload, policy=_policy(changed, payload))

    assert decision.refusal_codes == (
        "axis_evidence_policy_mismatch",
        "axis_observability_policy_mismatch",
        "axis_quality_policy_mismatch",
    )


def test_valid_classified_axis_is_refused_even_when_exactly_expected() -> None:
    """Keep classification results outside this abstaining review lane."""
    assessment, _payload = _assessment()
    index = next(index for index, axis in enumerate(assessment.axes) if axis.axis_id == "plant_readiness")
    axis = assessment.axes[index]
    roles = ("owner_declaration", "provenance", "validity")
    bindings = tuple(ReactorRegimeEvidenceBinding(role_id=role, reference_id=f"evidence.{role}") for role in roles)
    classified = replace(
        axis,
        disposition=ReactorRegimeAxisDisposition.CLASSIFIED,
        label="experimental",
        confidence=0.8,
        observability=0.9,
        uncertainty_probability=0.2,
        uncertainty_basis_id="uncertainty.test.v1",
        evidence_ids=tuple(binding.reference_id for binding in bindings),
        evidence_bindings=bindings,
        evidence_class=EvidenceClass.SIMULATION,
        validity=ValidityState.VALID,
        quality=QualityState.VALID,
        unknown_reason_id=None,
    )
    axes = assessment.axes[:index] + (classified,) + assessment.axes[index + 1 :]
    changed = replace(assessment, axes=axes)
    payload = _encoded(changed)
    decision = admit_reactor_regime_assessment(payload, policy=_policy(changed, payload))

    assert "assessment_not_abstaining" in decision.refusal_codes
    assert decision.admitted is False
    assert decision.actionable is False


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"expected_assessment_sha256": "bad"}, "expected_assessment_sha256"),
        ({"expected_producer_revision": "bad"}, "expected_producer_revision"),
        ({"expected_producer_revision": cast(Any, 1)}, "expected_producer_revision"),
        ({"expected_assessment_id": ""}, "expected_assessment_id"),
        ({"expected_source_semantic_ids": cast(Any, ["not-a-tuple"])}, "tuple of non-empty strings"),
        ({"expected_source_semantic_ids": ("",)}, "tuple of non-empty strings"),
        ({"expected_source_semantic_ids": ("b", "a")}, "sorted and unique"),
        ({"expected_axis_ids": ("only.one",)}, "exactly eight axes"),
        (
            {"expected_axis_provenance": (("duplicate", "a"), ("duplicate", "a"))},
            "sorted and unique",
        ),
        ({"expected_axis_provenance": cast(Any, [("axis", "value")])}, "text pairs"),
        ({"expected_axis_provenance": cast(Any, (("axis",),))}, "text pairs"),
        ({"expected_axis_provenance": cast(Any, ((1, "value"),))}, "text pairs"),
        ({"checked_at_ns": True}, "non-negative integer"),
        ({"max_evidence_age_ns": -1}, "non-negative integer"),
    ],
)
def test_policy_rejects_invalid_configuration(change: dict[str, object], message: str) -> None:
    """Reject malformed caller policy before any assessment is admitted."""
    assessment, payload = _assessment()
    with pytest.raises(ValueError, match=message):
        replace(_policy(assessment, payload), **cast(Any, change))


def test_policy_requires_complete_axis_provenance_and_nonempty_identity() -> None:
    """Bind every expected axis to one non-empty provenance identity."""
    assessment, payload = _assessment()
    base = _policy(assessment, payload)
    wrong_axes = tuple((f"x.{axis}", value) for axis, value in base.expected_axis_provenance)
    with pytest.raises(ValueError, match="cover every expected axis"):
        replace(base, expected_axis_provenance=wrong_axes)

    empty = tuple(
        (axis, "" if index == 0 else value) for index, (axis, value) in enumerate(base.expected_axis_provenance)
    )
    with pytest.raises(ValueError, match="identity must be non-empty text"):
        replace(base, expected_axis_provenance=empty)
