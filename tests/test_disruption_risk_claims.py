# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for disruption risk claims leaf

"""Drive production disruption claim-boundary leaf contracts."""

from __future__ import annotations

import scpn_control.control.disruption_predictor as owner
import scpn_control.control.disruption_risk_claims as leaf


def test_owner_claim_boundary_symbols_bind_to_leaf() -> None:
    """Owner re-exports the production disruption claim-boundary leaf."""
    assert owner.DisruptionRiskClaimBoundary is leaf.DisruptionRiskClaimBoundary
    assert owner.disruption_risk_claim_boundary is leaf.disruption_risk_claim_boundary
    assert owner._attach_disruption_claim_boundary is leaf._attach_disruption_claim_boundary
    assert owner.DISRUPTION_FEATURE_CONTRACT is leaf.DISRUPTION_FEATURE_CONTRACT


def test_attach_claim_boundary_enriches_metadata() -> None:
    """Attachment adds the fixed claim boundary without mutating the input map."""
    base = {"status": "ok"}
    out = leaf._attach_disruption_claim_boundary(base)
    assert base == {"status": "ok"}
    assert out["status"] == "ok"
    boundary = out["claim_boundary"]
    assert boundary["public_claim_allowed"] is False
    assert boundary["facility_roc_validated"] is False
    assert boundary["predictor_id"] == "predict_disruption_risk"
    assert boundary["feature_contract"] == list(leaf.DISRUPTION_FEATURE_CONTRACT)
