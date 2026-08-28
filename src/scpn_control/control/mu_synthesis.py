# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Deprecated mu-synthesis compatibility facade.

"""Compatibility facade for the former mu-synthesis API.

The implementation was renamed in version 0.23.0 because it designs one CARE
state-feedback gain and evaluates one static zero-frequency structured-mu upper
bound. It does not execute H-infinity synthesis, a frequency sweep, dynamic
D-scale fitting, or alternating D-K iterations.

Use :mod:`scpn_control.control.static_mu_analysis`. Compatibility symbols in
this module are scheduled for removal in version 0.25.0.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from scpn_control._typing import AnyComplexArray, AnyFloatArray, FloatArray
from scpn_control.control.static_mu_analysis import (
    RiccatiStateFeedbackController,
    StaticMuAnalysisClaimEvidence,
    UncertaintyBlock,
    _finite_scalar,
    _positive_int,
    assert_static_mu_analysis_validated_claim_admissible,
    compute_static_mu_upper_bound,
    design_riccati_state_feedback_with_static_mu_analysis,
    load_static_mu_analysis_claim_evidence,
    save_static_mu_analysis_claim_evidence,
    static_mu_analysis_claim_evidence,
)
from scpn_control.control.static_mu_analysis import (
    StructuredUncertainty as _StructuredUncertainty,
)

_REMOVAL_VERSION = "0.25.0"
_MIGRATION_TARGET = "scpn_control.control.static_mu_analysis"


def _warn(old_symbol: str, new_symbol: str, *, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    warnings.warn(
        f"{old_symbol} is deprecated and will be removed in {_REMOVAL_VERSION}; "
        f"use {_MIGRATION_TARGET}.{new_symbol}.{suffix}",
        DeprecationWarning,
        stacklevel=3,
    )


class StructuredUncertainty(_StructuredUncertainty):
    """Compatibility subclass retaining the former method spelling."""

    def build_Delta_structure(self) -> list[tuple[int, str]]:
        """Return the block structure through the deprecated method spelling."""
        _warn("StructuredUncertainty.build_Delta_structure", "StructuredUncertainty.build_delta_structure")
        return self.build_delta_structure()


def compute_mu_upper_bound(
    M: AnyFloatArray | AnyComplexArray,
    delta_structure: list[tuple[int, str]],
) -> float:
    """Compatibility wrapper for :func:`compute_static_mu_upper_bound`."""
    _warn("compute_mu_upper_bound", "compute_static_mu_upper_bound")
    return compute_static_mu_upper_bound(M, delta_structure)


def dk_iteration(
    plant_ss: tuple[AnyFloatArray, AnyFloatArray, AnyFloatArray, AnyFloatArray],
    uncertainty: _StructuredUncertainty,
    n_iter: int = 5,
    gamma_bisect_tol: float = 0.01,
) -> tuple[FloatArray, float, FloatArray]:
    """Run the former API as one static design-and-analysis pass.

    ``n_iter`` and ``gamma_bisect_tol`` are validated for compatibility but do
    not alter the result. The implementation does not execute D-K iterations or
    an H-infinity gamma bisection.
    """
    _positive_int("n_iter", n_iter)
    _finite_scalar("gamma_bisect_tol", gamma_bisect_tol, positive=True)
    _warn(
        "dk_iteration",
        "design_riccati_state_feedback_with_static_mu_analysis",
        detail="The compatibility call does not execute D-K iterations.",
    )
    result = design_riccati_state_feedback_with_static_mu_analysis(plant_ss, uncertainty)
    return (
        result.controller_gain.copy(),
        float(result.mu_upper_bound),
        result.d_scalings.copy(),
    )


class MuSynthesisController(RiccatiStateFeedbackController):
    """Deprecated controller name forwarding to the truthful static contract."""

    def __init__(
        self,
        plant_ss: tuple[AnyFloatArray, AnyFloatArray, AnyFloatArray, AnyFloatArray],
        uncertainty: _StructuredUncertainty,
    ) -> None:
        _warn("MuSynthesisController", "RiccatiStateFeedbackController")
        self._legacy_mu_peak_override: float | None = None
        super().__init__(plant_ss, uncertainty)

    @property
    def K(self) -> FloatArray | None:
        """Compatibility view of the designed controller gain."""
        if self.analysis_result is None:
            return None
        return self.analysis_result.controller_gain.copy()

    @property
    def mu_peak(self) -> float:
        """Compatibility view of the static, zero-frequency upper bound."""
        if self._legacy_mu_peak_override is not None:
            return self._legacy_mu_peak_override
        if self.analysis_result is None:
            return float("inf")
        return float(self.analysis_result.mu_upper_bound)

    @mu_peak.setter
    def mu_peak(self, value: float) -> None:
        self._legacy_mu_peak_override = float(value)

    @property
    def D_scalings(self) -> FloatArray | None:
        """Compatibility view of the fitted static D scalings."""
        if self.analysis_result is None:
            return None
        return self.analysis_result.d_scalings.copy()

    def synthesize(self, n_dk_iter: int = 5) -> None:
        """Design once while retaining the former method signature."""
        _positive_int("n_dk_iter", n_dk_iter)
        _warn(
            "MuSynthesisController.synthesize",
            "RiccatiStateFeedbackController.design",
            detail="The compatibility call does not execute D-K iterations.",
        )
        self.design()
        self._legacy_mu_peak_override = None

    def robustness_margin(self) -> float:
        """Return the deprecated name for the reciprocal static upper bound."""
        _warn("MuSynthesisController.robustness_margin", "inverse_static_mu_upper_bound")
        if self.mu_peak <= 0.0:
            return float("inf")
        return 1.0 / self.mu_peak


MuSynthesisClaimEvidence = StaticMuAnalysisClaimEvidence


def mu_synthesis_claim_evidence(
    controller: RiccatiStateFeedbackController,
    **kwargs: Any,
) -> StaticMuAnalysisClaimEvidence:
    """Compatibility wrapper for static claim-evidence construction."""
    _warn("mu_synthesis_claim_evidence", "static_mu_analysis_claim_evidence")
    return static_mu_analysis_claim_evidence(controller, **kwargs)


def assert_mu_synthesis_validated_claim_admissible(
    evidence: StaticMuAnalysisClaimEvidence,
) -> StaticMuAnalysisClaimEvidence:
    """Compatibility wrapper for static claim admission."""
    _warn(
        "assert_mu_synthesis_validated_claim_admissible",
        "assert_static_mu_analysis_validated_claim_admissible",
    )
    return assert_static_mu_analysis_validated_claim_admissible(evidence)


def save_mu_synthesis_claim_evidence(
    evidence: StaticMuAnalysisClaimEvidence,
    path: str | Path,
) -> None:
    """Compatibility wrapper for static evidence persistence."""
    _warn("save_mu_synthesis_claim_evidence", "save_static_mu_analysis_claim_evidence")
    save_static_mu_analysis_claim_evidence(evidence, path)


def load_mu_synthesis_claim_evidence(
    path: str | Path,
    *,
    require_validated_claim: bool = False,
) -> StaticMuAnalysisClaimEvidence:
    """Compatibility wrapper for static evidence loading."""
    _warn("load_mu_synthesis_claim_evidence", "load_static_mu_analysis_claim_evidence")
    return load_static_mu_analysis_claim_evidence(
        path,
        require_validated_claim=require_validated_claim,
    )


__all__ = [
    "MuSynthesisClaimEvidence",
    "MuSynthesisController",
    "StructuredUncertainty",
    "UncertaintyBlock",
    "assert_mu_synthesis_validated_claim_admissible",
    "compute_mu_upper_bound",
    "dk_iteration",
    "load_mu_synthesis_claim_evidence",
    "mu_synthesis_claim_evidence",
    "save_mu_synthesis_claim_evidence",
]
