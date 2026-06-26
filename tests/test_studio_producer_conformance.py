# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — studio producer-conformance vectors
"""Producer-conformance vectors for the CONTROL studio evidence surface.

These assert that EVERY ``EvidenceBundle`` the CONTROL studio can emit satisfies the
ratified honesty invariants — exhaustively across the whole representative surface,
not just per-builder spot-checks — so a future evidence builder cannot silently emit a
bundle that renders as validated without being reference-validated AND admitted. This
is the producer side of the studio conformance contract (the keeper's
``validate_studio_bundle`` is the federation-gate counterpart). The vectors cover the
additive SDK axes CONTROL consumes: ``falsified`` / ``refuted`` parity findings,
``producer-asserted`` ceiling behavior, and the freshness floor that prevents stale
or unchecked evidence from rendering as validated.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.evidence import (  # noqa: E402
    AdmissionDecision,
    ClaimBoundary,
    ClaimStatus,
    EvidenceBundle,
    EvidenceKind,
    EvidenceLevel,
    Freshness,
    ProvActivity,
    ProvAgent,
    ProvEntity,
    validate_studio_bundle,
)

from scpn_control.studio.feed import FEED_SCHEMA, claim_summary, studio_feed  # noqa: E402
from scpn_control.studio.feed import representative_bundles  # noqa: E402

_BUNDLES = representative_bundles()
_IDS = [bundle.schema for bundle in _BUNDLES]


def test_representative_surface_is_one_bundle_per_schema() -> None:
    schemas = [bundle.schema for bundle in _BUNDLES]
    assert len(schemas) == len(set(schemas)), "schemas must be unique (one bundle per verb)"
    assert len(schemas) >= 12, "the representative surface must cover the 12-verb vertical"
    assert all(s.startswith("studio.") and s.endswith(".v1") for s in schemas)


@pytest.mark.parametrize("bundle", _BUNDLES, ids=_IDS)
def test_renders_as_validated_is_exactly_the_admitted_reference_validated_pair(bundle: EvidenceBundle) -> None:
    # The load-bearing cross-field invariant, stated as an iff for every emittable bundle:
    # a bundle renders as validated if and only if it is BOTH reference-validated AND admitted.
    boundary = bundle.claim_boundary
    expected = boundary.status is ClaimStatus.REFERENCE_VALIDATED and boundary.admission is AdmissionDecision.ADMITTED
    assert bundle.renders_as_validated is expected, (
        f"{bundle.schema}: renders_as_validated={bundle.renders_as_validated} but "
        f"status={boundary.status.value}, admission={boundary.admission.value}"
    )


@pytest.mark.parametrize("bundle", _BUNDLES, ids=_IDS)
def test_every_axis_is_a_valid_enum_member(bundle: EvidenceBundle) -> None:
    boundary = bundle.claim_boundary
    assert isinstance(boundary.status, ClaimStatus)
    assert isinstance(boundary.admission, AdmissionDecision)
    assert isinstance(bundle.evidence_kind, EvidenceKind)
    assert isinstance(bundle.freshness, Freshness)


def test_nothing_reaches_validated_by_any_other_path() -> None:
    # The honesty floor stated negatively: no bundle on the whole surface is validated
    # unless it carries the full admitted + reference-validated pair.
    for bundle in _BUNDLES:
        if bundle.renders_as_validated:
            boundary = bundle.claim_boundary
            assert boundary.status is ClaimStatus.REFERENCE_VALIDATED
            assert boundary.admission is AdmissionDecision.ADMITTED


def test_surface_is_honest_by_default_only_a_formal_proof_validates() -> None:
    # Regression guard: a change that lets the surface render validated more broadly is a
    # honesty regression. Today exactly the held safety certificate validates, and it is
    # backed by a formal proof.
    validated = sorted(bundle.schema for bundle in _BUNDLES if bundle.renders_as_validated)
    assert validated == ["studio.safety-certificate.v1"], validated
    proven_and_validated = [
        bundle
        for bundle in _BUNDLES
        if bundle.renders_as_validated and bundle.evidence_kind is EvidenceKind.FORMALLY_PROVEN
    ]
    assert proven_and_validated, "the one validated bundle must be backed by a formal proof"


@pytest.mark.parametrize("bundle", _BUNDLES, ids=_IDS)
def test_federation_gate_agrees_with_producer_rendering(bundle: EvidenceBundle) -> None:
    verdict = validate_studio_bundle(bundle.to_dict())
    assert verdict.admitted is True
    assert verdict.rejections == ()
    if bundle.evidence_kind is EvidenceKind.FALSIFIED or bundle.claim_boundary.status is ClaimStatus.REFUTED:
        assert verdict.mode == "refuted"
    elif bundle.renders_as_validated:
        assert bundle.freshness is Freshness.VERIFIED_AT_SOURCE
        assert verdict.mode == "validated"
    else:
        assert verdict.mode == "boundary"


def test_refuted_parity_is_a_fresh_traceable_negative_finding() -> None:
    parity = next(bundle for bundle in _BUNDLES if bundle.schema == "studio.parity-refutation.v1")
    assert parity.evidence_kind is EvidenceKind.FALSIFIED
    assert parity.claim_boundary.status is ClaimStatus.REFUTED
    assert parity.claim_boundary.admission is AdmissionDecision.REJECTED
    assert parity.freshness is Freshness.TRACEABLE_UNCHECKED
    assert parity.renders_as_validated is False
    assert validate_studio_bundle(parity.to_dict()).mode == "refuted"


def test_validated_certificate_declares_verified_at_source_freshness() -> None:
    certificate = next(bundle for bundle in _BUNDLES if bundle.schema == "studio.safety-certificate.v1")
    assert certificate.evidence_kind is EvidenceKind.FORMALLY_PROVEN
    assert certificate.claim_boundary.status is ClaimStatus.REFERENCE_VALIDATED
    assert certificate.claim_boundary.admission is AdmissionDecision.ADMITTED
    assert certificate.freshness is Freshness.VERIFIED_AT_SOURCE
    assert validate_studio_bundle(certificate.to_dict()).mode == "validated"


def test_unchecked_freshness_floors_reference_validated_claims() -> None:
    bundle = EvidenceBundle(
        schema="studio.synthetic-freshness-floor.v1",
        entity=ProvEntity(entity_id="scpn-control/freshness-floor/demo", digest="a" * 64),
        activity=ProvActivity(
            verb="validate",
            studio="scpn-control",
            started="2026-06-26T00:00:00Z",
            ended="2026-06-26T00:00:01Z",
        ),
        agent=ProvAgent(studio_version="test", operator="opaque:test"),
        evidence_level=EvidenceLevel.ENGINEERING_VERIFIED,
        evidence_kind=EvidenceKind.MEASURED,
        claim_boundary=ClaimBoundary(status=ClaimStatus.REFERENCE_VALIDATED, admission=AdmissionDecision.ADMITTED),
        freshness=Freshness.TRACEABLE_UNCHECKED,
    )
    assert bundle.renders_as_validated is True
    assert validate_studio_bundle(bundle.to_dict()).mode == "boundary"


def test_producer_asserted_claims_are_never_validated() -> None:
    bundle = EvidenceBundle(
        schema="studio.synthetic-producer-asserted.v1",
        entity=ProvEntity(entity_id="scpn-control/producer-asserted/demo", digest="b" * 64),
        activity=ProvActivity(
            verb="validate",
            studio="scpn-control",
            started="2026-06-26T00:00:00Z",
            ended="2026-06-26T00:00:01Z",
        ),
        agent=ProvAgent(studio_version="test", operator="opaque:test"),
        evidence_level=EvidenceLevel.TAXONOMY,
        evidence_kind=EvidenceKind.PRODUCER_ASSERTED,
        claim_boundary=ClaimBoundary(status=ClaimStatus.BOUNDED_SUPPORT, admission=AdmissionDecision.REJECTED),
        freshness=Freshness.TRACEABLE_UNCHECKED,
    )
    assert bundle.renders_as_validated is False
    assert validate_studio_bundle(bundle.to_dict()).mode == "boundary"


def test_claim_summary_and_feed_preserve_freshness_axis() -> None:
    certificate = next(bundle for bundle in _BUNDLES if bundle.schema == "studio.safety-certificate.v1")
    assert claim_summary(certificate)["freshness"] == "verified-at-source"

    feed = studio_feed(_BUNDLES, content_digest="sha256:test")
    assert feed["feed_schema"] == FEED_SCHEMA
    claims = feed["claims"]
    assert isinstance(claims, list)
    assert all("freshness" in claim for claim in claims)
