# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption risk public claim boundary

"""Machine-readable claim boundary for public disruption-risk scores.

This leaf owns :class:`DisruptionRiskClaimBoundary`, the fixed
:func:`disruption_risk_claim_boundary` factory, heuristic provenance constants,
and metadata attachment (CTL-G07 R7-S1). Physics proxies, fault campaigns,
checkpoint integrity, and optional torch training remain on the owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DISRUPTION_FEATURE_CONTRACT: tuple[str, ...] = (
    "mean",
    "std",
    "max",
    "slope",
    "energy",
    "last",
    "toroidal_n1_amp",
    "toroidal_n2_amp",
    "toroidal_n3_amp",
    "toroidal_asymmetry_index",
    "toroidal_radial_spread",
)
DISRUPTION_HEURISTIC_SCORE_SOURCE = "fixed_weight_logistic_heuristic"
DISRUPTION_HEURISTIC_TRAINING_PROVENANCE = "hand_chosen_weights_no_real_disruption_database_fit"
DISRUPTION_HEURISTIC_VALIDATION_PROVENANCE = (
    "synthetic_sanity_check:validation/reports/disruption_replay_pipeline_benchmark.md"
)
DISRUPTION_HEURISTIC_REQUIRED_ACTION = (
    "Train or fit on an admitted real disruption database before any facility disruption-prediction claim."
)

# Minimum required alarm lead time.
# Lehnen et al. 2015, J. Nucl. Mater. 463, 39 — τ_warning > τ_TQ + τ_mitigation.
# For ITER τ_TQ ≈ 1–5 ms, τ_mitigation ≈ 10–20 ms → lower bound 10 ms.
TAU_WARNING_MIN_S: float = 0.010  # s

# Locked-mode amplitude threshold above which disruption alarm is raised.
# de Vries et al. 2011, Nucl. Fusion 51, 053018, §3 — locked-mode onset
# is identified as the dominant disruption precursor across JET, DIII-D, AUG.
LOCKED_MODE_ALARM_THRESHOLD: float = 0.15  # normalised units


@dataclass(frozen=True, slots=True)
class DisruptionRiskClaimBoundary:
    """Machine-readable claim boundary for the public disruption-risk score.

    The boundary is intentionally narrow: ``predict_disruption_risk`` emits a
    deterministic fixed-weight logistic heuristic. It is not trained on, fitted
    to, or ROC-validated against a real disruption database.
    """

    predictor_id: str
    score_source: str
    feature_contract: tuple[str, ...]
    training_provenance: str
    validation_provenance: str
    public_claim_allowed: bool
    facility_roc_validated: bool
    required_action: str

    def __post_init__(self) -> None:
        """Reject empty fields and any widened facility-claim boundary."""
        for name in (
            "predictor_id",
            "score_source",
            "training_provenance",
            "validation_provenance",
            "required_action",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"DisruptionRiskClaimBoundary.{name} must be a non-empty string")
        if not self.feature_contract or any(not item.strip() for item in self.feature_contract):
            raise ValueError("DisruptionRiskClaimBoundary.feature_contract must contain non-empty feature names")
        if self.public_claim_allowed:
            raise ValueError("DisruptionRiskClaimBoundary.public_claim_allowed must remain false")
        if self.facility_roc_validated:
            raise ValueError("DisruptionRiskClaimBoundary.facility_roc_validated must remain false")

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serialisable metadata representation."""
        return {
            "predictor_id": self.predictor_id,
            "score_source": self.score_source,
            "feature_contract": list(self.feature_contract),
            "training_provenance": self.training_provenance,
            "validation_provenance": self.validation_provenance,
            "public_claim_allowed": self.public_claim_allowed,
            "facility_roc_validated": self.facility_roc_validated,
            "required_action": self.required_action,
        }


def disruption_risk_claim_boundary() -> DisruptionRiskClaimBoundary:
    """Return the fixed public boundary for ``predict_disruption_risk`` claims."""
    return DisruptionRiskClaimBoundary(
        predictor_id="predict_disruption_risk",
        score_source=DISRUPTION_HEURISTIC_SCORE_SOURCE,
        feature_contract=DISRUPTION_FEATURE_CONTRACT,
        training_provenance=DISRUPTION_HEURISTIC_TRAINING_PROVENANCE,
        validation_provenance=DISRUPTION_HEURISTIC_VALIDATION_PROVENANCE,
        public_claim_allowed=False,
        facility_roc_validated=False,
        required_action=DISRUPTION_HEURISTIC_REQUIRED_ACTION,
    )


def _attach_disruption_claim_boundary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Attach the fixed disruption-risk claim boundary to an output mapping."""
    enriched = dict(metadata)
    enriched["claim_boundary"] = disruption_risk_claim_boundary().to_metadata()
    return enriched
