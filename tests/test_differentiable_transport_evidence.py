# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for differentiable transport evidence leaf
"""Real-surface ownership tests for the transport evidence leaf module."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from scpn_control.core import differentiable_transport as facade
from scpn_control.core import differentiable_transport_evidence as evidence


def test_evidence_leaf_owns_campaign_and_claim_types() -> None:
    """Facade re-exports the leaf types without duplicating class objects."""
    assert facade.TransportCampaignMetadata is evidence.TransportCampaignMetadata
    assert facade.TransportDifferentiabilityEvidence is evidence.TransportDifferentiabilityEvidence
    assert facade.TransportFullFidelityReadinessEvidence is evidence.TransportFullFidelityReadinessEvidence
    assert facade.TransportGradientAudit is evidence.TransportGradientAudit
    assert facade.save_transport_campaign_metadata is evidence.save_transport_campaign_metadata
    assert facade.transport_differentiability_evidence is evidence.transport_differentiability_evidence


def test_campaign_metadata_round_trip_through_evidence_leaf(tmp_path) -> None:
    """Leaf save/load path preserves campaign provenance fail-closed."""
    n_rho = 8
    rho = np.linspace(0.0, 1.0, n_rho)
    profiles = np.ones((evidence.CHANNEL_COUNT, n_rho))
    chi = np.full((evidence.CHANNEL_COUNT, n_rho), 0.1)
    sources = np.zeros((evidence.CHANNEL_COUNT, n_rho))
    edge = np.ones(evidence.CHANNEL_COUNT)
    metadata = facade.transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        1.0e-3,
        edge,
        backend="numpy",
        gradient_tolerance=1.0e-6,
    )
    path = tmp_path / "campaign.json"
    evidence.save_transport_campaign_metadata(metadata, path)
    loaded = evidence.load_transport_campaign_metadata(path)
    assert asdict(loaded) == asdict(metadata)


def test_differentiability_evidence_fails_closed_on_failed_audit() -> None:
    """Leaf claim builder rejects failed finite-difference audits."""
    n_rho = 6
    rho = np.linspace(0.0, 1.0, n_rho)
    profiles = np.ones((evidence.CHANNEL_COUNT, n_rho))
    chi = np.full((evidence.CHANNEL_COUNT, n_rho), 0.2)
    sources = np.zeros((evidence.CHANNEL_COUNT, n_rho))
    edge = np.ones(evidence.CHANNEL_COUNT)
    metadata = facade.transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        1.0e-3,
        edge,
        backend="jax",
        gradient_tolerance=1.0e-5,
    )
    failed_audit = evidence.TransportGradientAudit(
        loss=1.0,
        epsilon=1.0e-6,
        tolerance=1.0e-5,
        checked_indices=((0, 0),),
        chi_max_abs_error=1.0,
        source_max_abs_error=0.0,
        passed=False,
    )
    built = evidence.transport_differentiability_evidence(metadata, failed_audit)
    assert built.audit_passed is False
    with pytest.raises(ValueError, match="passed audit"):
        evidence.assert_transport_differentiability_claim_admissible(built, metadata, failed_audit)
