# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reactor semantic admission facade

"""Public review-only admission of portable SPO reactor semantics."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .admission import (
        ReactorSemanticAdmissionPolicy,
        admit_reactor_semantic_handoff,
    )
    from .decision import (
        ADMISSION_SCHEMA,
        ADMISSION_SCHEMA_VERSION,
        MAX_ADMISSION_BYTES,
        REFUSAL_CODES,
        ReactorSemanticAdmissionDecision,
        ReactorSemanticAdmissionStatus,
        admission_decision_digest,
        admission_decision_from_bytes,
        admission_decision_to_bytes,
    )
    from .device_diagnostic_review_admission import (
        SPO_CONTRACT_MODULE_SHA256,
        SPO_DISTRIBUTION,
        SPO_DISTRIBUTION_VERSION,
        SPO_RELEASE_WHEEL_SHA256,
        SPO_REVIEW_SCHEMA,
        SPO_REVIEW_SCHEMA_VERSION,
        TOKAMAK_CLOCK_CUSTODY_SHA256,
        TOKAMAK_CONFIGURATIONS,
        TOKAMAK_REVIEW_ID,
        TOKAMAK_REVIEW_SHA256,
        TOKAMAK_SOURCE_ARTIFACT_SHA256,
        TOKAMAK_SOURCE_ENVELOPE_SHA256,
        TOKAMAK_SOURCE_MANIFEST_SHA256,
        TOKAMAK_SOURCE_PLAN_SHA256,
        TOKAMAK_SOURCE_PROJECT,
        TOKAMAK_SOURCE_REVISION,
        DeviceDiagnosticReviewAdmissionPolicy,
        admit_device_diagnostic_plan_review,
        device_diagnostic_review_clock_custody_digest,
        tokamak_device_diagnostic_review_policy,
    )
    from .device_diagnostic_review_decision import (
        DEVICE_DIAGNOSTIC_REVIEW_DECISION_SCHEMA,
        DEVICE_DIAGNOSTIC_REVIEW_DECISION_VERSION,
        DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES,
        MAX_DEVICE_DIAGNOSTIC_REVIEW_DECISION_BYTES,
        DeviceDiagnosticReviewAdmissionDecision,
        DeviceDiagnosticReviewAdmissionStatus,
        device_diagnostic_review_decision_digest,
        device_diagnostic_review_decision_from_bytes,
        device_diagnostic_review_decision_to_bytes,
    )
    from .mif_admission import (
        MIFReactorSemanticAdmissionPolicy,
        admit_mif_reactor_semantic_handoff,
    )
    from .regime_assessment_admission import (
        ReactorRegimeAssessmentAdmissionPolicy,
        admit_reactor_regime_assessment,
        regime_assessment_axis_custody_digest,
        regime_assessment_clock_custody_digest,
        regime_assessment_registry_custody_digest,
    )
    from .regime_assessment_decision import (
        MAX_REGIME_ASSESSMENT_ADMISSION_BYTES,
        REGIME_ASSESSMENT_ADMISSION_SCHEMA,
        REGIME_ASSESSMENT_ADMISSION_VERSION,
        REGIME_ASSESSMENT_REFUSAL_CODES,
        ReactorRegimeAssessmentAdmissionDecision,
        ReactorRegimeAssessmentAdmissionStatus,
        regime_assessment_admission_decision_digest,
        regime_assessment_admission_decision_from_bytes,
        regime_assessment_admission_decision_to_bytes,
    )

_EXPORT_MODULES = {
    "ADMISSION_SCHEMA": "decision",
    "ADMISSION_SCHEMA_VERSION": "decision",
    "DEVICE_DIAGNOSTIC_REVIEW_DECISION_SCHEMA": "device_diagnostic_review_decision",
    "DEVICE_DIAGNOSTIC_REVIEW_DECISION_VERSION": "device_diagnostic_review_decision",
    "DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES": "device_diagnostic_review_decision",
    "DeviceDiagnosticReviewAdmissionDecision": "device_diagnostic_review_decision",
    "DeviceDiagnosticReviewAdmissionPolicy": "device_diagnostic_review_admission",
    "DeviceDiagnosticReviewAdmissionStatus": "device_diagnostic_review_decision",
    "MAX_ADMISSION_BYTES": "decision",
    "MAX_DEVICE_DIAGNOSTIC_REVIEW_DECISION_BYTES": "device_diagnostic_review_decision",
    "MAX_REGIME_ASSESSMENT_ADMISSION_BYTES": "regime_assessment_decision",
    "MIFReactorSemanticAdmissionPolicy": "mif_admission",
    "REFUSAL_CODES": "decision",
    "REGIME_ASSESSMENT_ADMISSION_SCHEMA": "regime_assessment_decision",
    "REGIME_ASSESSMENT_ADMISSION_VERSION": "regime_assessment_decision",
    "REGIME_ASSESSMENT_REFUSAL_CODES": "regime_assessment_decision",
    "ReactorRegimeAssessmentAdmissionDecision": "regime_assessment_decision",
    "ReactorRegimeAssessmentAdmissionPolicy": "regime_assessment_admission",
    "ReactorRegimeAssessmentAdmissionStatus": "regime_assessment_decision",
    "ReactorSemanticAdmissionDecision": "decision",
    "ReactorSemanticAdmissionPolicy": "admission",
    "ReactorSemanticAdmissionStatus": "decision",
    "SPO_CONTRACT_MODULE_SHA256": "device_diagnostic_review_admission",
    "SPO_DISTRIBUTION": "device_diagnostic_review_admission",
    "SPO_DISTRIBUTION_VERSION": "device_diagnostic_review_admission",
    "SPO_RELEASE_WHEEL_SHA256": "device_diagnostic_review_admission",
    "SPO_REVIEW_SCHEMA": "device_diagnostic_review_admission",
    "SPO_REVIEW_SCHEMA_VERSION": "device_diagnostic_review_admission",
    "TOKAMAK_CLOCK_CUSTODY_SHA256": "device_diagnostic_review_admission",
    "TOKAMAK_CONFIGURATIONS": "device_diagnostic_review_admission",
    "TOKAMAK_REVIEW_ID": "device_diagnostic_review_admission",
    "TOKAMAK_REVIEW_SHA256": "device_diagnostic_review_admission",
    "TOKAMAK_SOURCE_ARTIFACT_SHA256": "device_diagnostic_review_admission",
    "TOKAMAK_SOURCE_ENVELOPE_SHA256": "device_diagnostic_review_admission",
    "TOKAMAK_SOURCE_MANIFEST_SHA256": "device_diagnostic_review_admission",
    "TOKAMAK_SOURCE_PLAN_SHA256": "device_diagnostic_review_admission",
    "TOKAMAK_SOURCE_PROJECT": "device_diagnostic_review_admission",
    "TOKAMAK_SOURCE_REVISION": "device_diagnostic_review_admission",
    "admission_decision_digest": "decision",
    "admission_decision_from_bytes": "decision",
    "admission_decision_to_bytes": "decision",
    "admit_device_diagnostic_plan_review": "device_diagnostic_review_admission",
    "admit_mif_reactor_semantic_handoff": "mif_admission",
    "admit_reactor_regime_assessment": "regime_assessment_admission",
    "admit_reactor_semantic_handoff": "admission",
    "device_diagnostic_review_clock_custody_digest": "device_diagnostic_review_admission",
    "device_diagnostic_review_decision_digest": "device_diagnostic_review_decision",
    "device_diagnostic_review_decision_from_bytes": "device_diagnostic_review_decision",
    "device_diagnostic_review_decision_to_bytes": "device_diagnostic_review_decision",
    "regime_assessment_admission_decision_digest": "regime_assessment_decision",
    "regime_assessment_admission_decision_from_bytes": "regime_assessment_decision",
    "regime_assessment_admission_decision_to_bytes": "regime_assessment_decision",
    "regime_assessment_axis_custody_digest": "regime_assessment_admission",
    "regime_assessment_clock_custody_digest": "regime_assessment_admission",
    "regime_assessment_registry_custody_digest": "regime_assessment_admission",
    "tokamak_device_diagnostic_review_policy": "device_diagnostic_review_admission",
}

__all__ = [
    "ADMISSION_SCHEMA",
    "ADMISSION_SCHEMA_VERSION",
    "DEVICE_DIAGNOSTIC_REVIEW_DECISION_SCHEMA",
    "DEVICE_DIAGNOSTIC_REVIEW_DECISION_VERSION",
    "DEVICE_DIAGNOSTIC_REVIEW_REFUSAL_CODES",
    "MAX_ADMISSION_BYTES",
    "MAX_DEVICE_DIAGNOSTIC_REVIEW_DECISION_BYTES",
    "MAX_REGIME_ASSESSMENT_ADMISSION_BYTES",
    "MIFReactorSemanticAdmissionPolicy",
    "REGIME_ASSESSMENT_ADMISSION_SCHEMA",
    "REGIME_ASSESSMENT_ADMISSION_VERSION",
    "REGIME_ASSESSMENT_REFUSAL_CODES",
    "REFUSAL_CODES",
    "SPO_CONTRACT_MODULE_SHA256",
    "SPO_DISTRIBUTION",
    "SPO_DISTRIBUTION_VERSION",
    "SPO_RELEASE_WHEEL_SHA256",
    "SPO_REVIEW_SCHEMA",
    "SPO_REVIEW_SCHEMA_VERSION",
    "TOKAMAK_CLOCK_CUSTODY_SHA256",
    "TOKAMAK_CONFIGURATIONS",
    "TOKAMAK_REVIEW_ID",
    "TOKAMAK_REVIEW_SHA256",
    "TOKAMAK_SOURCE_ARTIFACT_SHA256",
    "TOKAMAK_SOURCE_ENVELOPE_SHA256",
    "TOKAMAK_SOURCE_MANIFEST_SHA256",
    "TOKAMAK_SOURCE_PLAN_SHA256",
    "TOKAMAK_SOURCE_PROJECT",
    "TOKAMAK_SOURCE_REVISION",
    "DeviceDiagnosticReviewAdmissionDecision",
    "DeviceDiagnosticReviewAdmissionPolicy",
    "DeviceDiagnosticReviewAdmissionStatus",
    "ReactorRegimeAssessmentAdmissionDecision",
    "ReactorRegimeAssessmentAdmissionPolicy",
    "ReactorRegimeAssessmentAdmissionStatus",
    "ReactorSemanticAdmissionDecision",
    "ReactorSemanticAdmissionPolicy",
    "ReactorSemanticAdmissionStatus",
    "admission_decision_digest",
    "admission_decision_from_bytes",
    "admission_decision_to_bytes",
    "admit_device_diagnostic_plan_review",
    "admit_mif_reactor_semantic_handoff",
    "admit_reactor_regime_assessment",
    "admit_reactor_semantic_handoff",
    "device_diagnostic_review_clock_custody_digest",
    "device_diagnostic_review_decision_digest",
    "device_diagnostic_review_decision_from_bytes",
    "device_diagnostic_review_decision_to_bytes",
    "regime_assessment_admission_decision_digest",
    "regime_assessment_admission_decision_from_bytes",
    "regime_assessment_admission_decision_to_bytes",
    "regime_assessment_axis_custody_digest",
    "regime_assessment_clock_custody_digest",
    "regime_assessment_registry_custody_digest",
    "tokamak_device_diagnostic_review_policy",
]


def __getattr__(name: str) -> Any:
    """Load one public owner without importing unrelated upstream dependencies."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORT_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List every public admission symbol before its owner is loaded."""
    return sorted(set(globals()) | set(__all__))
