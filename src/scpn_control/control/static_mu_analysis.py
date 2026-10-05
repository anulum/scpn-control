# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static structured-mu analysis
"""Stable public facade for static structured-mu analysis and claim evidence."""

from __future__ import annotations

from scpn_control.control._static_mu_bounds import compute_static_mu_upper_bound
from scpn_control.control._static_mu_claims import (
    StaticMuAnalysisClaimEvidence,
    assert_static_mu_analysis_validated_claim_admissible,
    load_static_mu_analysis_claim_evidence,
    save_static_mu_analysis_claim_evidence,
    static_mu_analysis_claim_evidence,
)
from scpn_control.control._static_mu_claims import (
    _claim_payload_sha256 as _claim_payload_sha256,
)
from scpn_control.control._static_mu_claims import (
    _nonnegative_reference_scalar as _nonnegative_reference_scalar,
)
from scpn_control.control._static_mu_claims import (
    _positive_reference_scalar as _positive_reference_scalar,
)
from scpn_control.control._static_mu_claims import (
    _require_bool as _require_bool,
)
from scpn_control.control._static_mu_claims import (
    _require_positive_claim_int as _require_positive_claim_int,
)
from scpn_control.control._static_mu_claims import (
    _validate_claim_structure as _validate_claim_structure,
)
from scpn_control.control._static_mu_claims import (
    _validate_static_mu_analysis_claim_payload as _validate_static_mu_analysis_claim_payload,
)
from scpn_control.control._static_mu_claims import (
    _with_payload_digest as _with_payload_digest,
)
from scpn_control.control._static_mu_riccati import (
    RiccatiStateFeedbackController,
    StaticMuAnalysisResult,
    design_riccati_state_feedback_with_static_mu_analysis,
)
from scpn_control.control._static_mu_riccati import (
    _closed_loop_dc_uncertainty_map as _closed_loop_dc_uncertainty_map,
)
from scpn_control.control._static_mu_riccati import (
    _riccati_state_feedback as _riccati_state_feedback,
)
from scpn_control.control._static_mu_structure import (
    StructuredUncertainty,
    UncertaintyBlock,
)
from scpn_control.control._static_mu_structure import (
    _finite_scalar as _finite_scalar,
)
from scpn_control.control._static_mu_structure import (
    _positive_int as _positive_int,
)

__all__ = [
    "RiccatiStateFeedbackController",
    "StaticMuAnalysisClaimEvidence",
    "StaticMuAnalysisResult",
    "StructuredUncertainty",
    "UncertaintyBlock",
    "assert_static_mu_analysis_validated_claim_admissible",
    "compute_static_mu_upper_bound",
    "design_riccati_state_feedback_with_static_mu_analysis",
    "load_static_mu_analysis_claim_evidence",
    "static_mu_analysis_claim_evidence",
    "save_static_mu_analysis_claim_evidence",
]
