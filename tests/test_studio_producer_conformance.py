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
``validate_studio_bundle`` is the federation-gate counterpart).

Staged for the SDK additive increment (CONTROL pin is ``>=0.2,<0.3`` today, so these
members are not yet importable): once the keeper ships ``EvidenceKind.FALSIFIED`` /
``ClaimStatus.REFUTED`` / ``EvidenceKind.PRODUCER_ASSERTED`` and the freshness axis,
add vectors for ``falsified + reference-validated => rejected`` (INV-2/LOCK-4),
``producer-asserted => claim-status ceiling bounded-support`` (INV-6), and the
``verified-at-source / traceable-unchecked / untraceable`` freshness floor.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scpn_studio_platform")

from scpn_studio_platform.evidence import (  # noqa: E402
    AdmissionDecision,
    ClaimStatus,
    EvidenceKind,
)

from scpn_control.studio.feed import representative_bundles  # noqa: E402

_BUNDLES = representative_bundles()
_IDS = [bundle.schema for bundle in _BUNDLES]


def test_representative_surface_is_one_bundle_per_schema() -> None:
    schemas = [bundle.schema for bundle in _BUNDLES]
    assert len(schemas) == len(set(schemas)), "schemas must be unique (one bundle per verb)"
    assert len(schemas) >= 11, "the representative surface must cover the 11-verb vertical"
    assert all(s.startswith("studio.") and s.endswith(".v1") for s in schemas)


@pytest.mark.parametrize("bundle", _BUNDLES, ids=_IDS)
def test_renders_as_validated_is_exactly_the_admitted_reference_validated_pair(bundle) -> None:
    # The load-bearing cross-field invariant, stated as an iff for every emittable bundle:
    # a bundle renders as validated if and only if it is BOTH reference-validated AND admitted.
    boundary = bundle.claim_boundary
    expected = boundary.status is ClaimStatus.REFERENCE_VALIDATED and boundary.admission is AdmissionDecision.ADMITTED
    assert bundle.renders_as_validated is expected, (
        f"{bundle.schema}: renders_as_validated={bundle.renders_as_validated} but "
        f"status={boundary.status.value}, admission={boundary.admission.value}"
    )


@pytest.mark.parametrize("bundle", _BUNDLES, ids=_IDS)
def test_every_axis_is_a_valid_enum_member(bundle) -> None:
    boundary = bundle.claim_boundary
    assert isinstance(boundary.status, ClaimStatus)
    assert isinstance(boundary.admission, AdmissionDecision)
    assert isinstance(bundle.evidence_kind, EvidenceKind)


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
