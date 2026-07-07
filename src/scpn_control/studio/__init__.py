# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Studio vertical
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
"""CONTROL's studio vertical, built on the locked ``scpn-studio-platform`` SDK.

This package is CONTROL's federated studio surface for the SCPN STUDIO Hub. It
consumes the domain-neutral platform SDK (it never forks it): it declares CONTROL's
verbs as platform :class:`~scpn_studio_platform.verbs.Verb` records
(:mod:`scpn_control.studio.verbs`), maps CONTROL's provenance-graded result surfaces
onto platform :class:`~scpn_studio_platform.evidence.EvidenceBundle` records
(:mod:`scpn_control.studio.evidence`), and authors the schema-A
:class:`~scpn_studio_platform.manifest.CapabilityManifest`
(:mod:`scpn_control.studio.manifest`).

The platform SDK is an optional dependency (install the ``studio`` extra). Public
exports resolve lazily: accessing any SDK-backed symbol without the SDK raises
:class:`ModuleNotFoundError` at first attribute access, which is the intended
fail-closed behaviour for an optional studio surface. SDK-free submodules —
:mod:`scpn_control.studio.sealed_claim`, whose artefacts are plain JSON handed to
the Hub keeper — stay importable from a checkout without the ``studio`` extra.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapters import (
        controller_latency_evidence_from_measurement,
        controller_run_evidence_from_simulation,
        disruption_mitigation_evidence_from_run,
        disruption_prediction_evidence_from_risk,
        efit_evidence_from_reconstruction,
        equilibrium_analysis_evidence_from_shape,
        phase_sync_monitor_evidence_from_snapshot,
        physics_validation_evidences_from_registry,
        replay_evidence_from_geometry_neutral,
        safety_certificate_evidence_from_certificate,
        scenario_simulation_evidence_from_audit,
    )
    from .evidence import (
        ControllerLatencyResult,
        ControllerRunResult,
        DisruptionPrediction,
        EfitReconstructionResult,
        EquilibriumAnalysis,
        MitigationRun,
        MonitorSnapshot,
        ReplaySummary,
        SafetyCertificateResult,
        ScenarioSimulationRun,
        TraceabilityClaim,
        canonical_digest,
        controller_latency_evidence,
        controller_run_evidence,
        disruption_mitigation_evidence,
        disruption_prediction_evidence,
        efit_reconstruction_evidence,
        equilibrium_analysis_evidence,
        geometry_neutral_replay_evidence_bundle,
        phase_sync_monitor_evidence,
        physics_validation_evidence,
        safety_certificate_evidence,
        scenario_simulation_evidence,
    )
    from .feed import (
        FEED_SCHEMA,
        claim_summary,
        render_feed_json,
        representative_bundles,
        representative_feed,
        studio_feed,
        verb_summary,
    )
    from .manifest import build_manifest, declared_surface
    from .sealed_claim import (
        SEALED_SAFETY_CLAIM_SCHEMA,
        assert_jcs_safe,
        build_safety_certificate_sealed_claim,
        render_sealed_claim_json,
        write_sealed_claim,
    )
    from .verbs import (
        CONTROL_VERBS,
        STUDIO_ID,
        core_verbs,
        domain_verbs,
        evidence_schemas,
    )

__all__ = [
    "CONTROL_VERBS",
    "FEED_SCHEMA",
    "SEALED_SAFETY_CLAIM_SCHEMA",
    "STUDIO_ID",
    "ControllerLatencyResult",
    "ControllerRunResult",
    "DisruptionPrediction",
    "EfitReconstructionResult",
    "EquilibriumAnalysis",
    "MitigationRun",
    "MonitorSnapshot",
    "ReplaySummary",
    "SafetyCertificateResult",
    "ScenarioSimulationRun",
    "TraceabilityClaim",
    "assert_jcs_safe",
    "build_manifest",
    "build_safety_certificate_sealed_claim",
    "canonical_digest",
    "claim_summary",
    "controller_latency_evidence",
    "controller_latency_evidence_from_measurement",
    "controller_run_evidence",
    "controller_run_evidence_from_simulation",
    "core_verbs",
    "declared_surface",
    "disruption_mitigation_evidence",
    "disruption_mitigation_evidence_from_run",
    "disruption_prediction_evidence",
    "disruption_prediction_evidence_from_risk",
    "domain_verbs",
    "efit_evidence_from_reconstruction",
    "efit_reconstruction_evidence",
    "equilibrium_analysis_evidence",
    "equilibrium_analysis_evidence_from_shape",
    "evidence_schemas",
    "geometry_neutral_replay_evidence_bundle",
    "phase_sync_monitor_evidence",
    "phase_sync_monitor_evidence_from_snapshot",
    "physics_validation_evidence",
    "physics_validation_evidences_from_registry",
    "render_feed_json",
    "render_sealed_claim_json",
    "replay_evidence_from_geometry_neutral",
    "representative_bundles",
    "representative_feed",
    "safety_certificate_evidence",
    "safety_certificate_evidence_from_certificate",
    "scenario_simulation_evidence",
    "scenario_simulation_evidence_from_audit",
    "studio_feed",
    "verb_summary",
    "write_sealed_claim",
]

_EXPORT_MODULES: dict[str, str] = {
    "controller_latency_evidence_from_measurement": "scpn_control.studio.adapters",
    "controller_run_evidence_from_simulation": "scpn_control.studio.adapters",
    "disruption_mitigation_evidence_from_run": "scpn_control.studio.adapters",
    "disruption_prediction_evidence_from_risk": "scpn_control.studio.adapters",
    "efit_evidence_from_reconstruction": "scpn_control.studio.adapters",
    "equilibrium_analysis_evidence_from_shape": "scpn_control.studio.adapters",
    "phase_sync_monitor_evidence_from_snapshot": "scpn_control.studio.adapters",
    "physics_validation_evidences_from_registry": "scpn_control.studio.adapters",
    "replay_evidence_from_geometry_neutral": "scpn_control.studio.adapters",
    "safety_certificate_evidence_from_certificate": "scpn_control.studio.adapters",
    "scenario_simulation_evidence_from_audit": "scpn_control.studio.adapters",
    "ControllerLatencyResult": "scpn_control.studio.evidence",
    "ControllerRunResult": "scpn_control.studio.evidence",
    "DisruptionPrediction": "scpn_control.studio.evidence",
    "EfitReconstructionResult": "scpn_control.studio.evidence",
    "EquilibriumAnalysis": "scpn_control.studio.evidence",
    "MitigationRun": "scpn_control.studio.evidence",
    "MonitorSnapshot": "scpn_control.studio.evidence",
    "ReplaySummary": "scpn_control.studio.evidence",
    "SafetyCertificateResult": "scpn_control.studio.evidence",
    "ScenarioSimulationRun": "scpn_control.studio.evidence",
    "TraceabilityClaim": "scpn_control.studio.evidence",
    "canonical_digest": "scpn_control.studio.evidence",
    "controller_latency_evidence": "scpn_control.studio.evidence",
    "controller_run_evidence": "scpn_control.studio.evidence",
    "disruption_mitigation_evidence": "scpn_control.studio.evidence",
    "disruption_prediction_evidence": "scpn_control.studio.evidence",
    "efit_reconstruction_evidence": "scpn_control.studio.evidence",
    "equilibrium_analysis_evidence": "scpn_control.studio.evidence",
    "geometry_neutral_replay_evidence_bundle": "scpn_control.studio.evidence",
    "phase_sync_monitor_evidence": "scpn_control.studio.evidence",
    "physics_validation_evidence": "scpn_control.studio.evidence",
    "safety_certificate_evidence": "scpn_control.studio.evidence",
    "scenario_simulation_evidence": "scpn_control.studio.evidence",
    "FEED_SCHEMA": "scpn_control.studio.feed",
    "claim_summary": "scpn_control.studio.feed",
    "render_feed_json": "scpn_control.studio.feed",
    "representative_bundles": "scpn_control.studio.feed",
    "representative_feed": "scpn_control.studio.feed",
    "studio_feed": "scpn_control.studio.feed",
    "verb_summary": "scpn_control.studio.feed",
    "build_manifest": "scpn_control.studio.manifest",
    "declared_surface": "scpn_control.studio.manifest",
    "CONTROL_VERBS": "scpn_control.studio.verbs",
    "STUDIO_ID": "scpn_control.studio.verbs",
    "core_verbs": "scpn_control.studio.verbs",
    "domain_verbs": "scpn_control.studio.verbs",
    "evidence_schemas": "scpn_control.studio.verbs",
    "SEALED_SAFETY_CLAIM_SCHEMA": "scpn_control.studio.sealed_claim",
    "assert_jcs_safe": "scpn_control.studio.sealed_claim",
    "build_safety_certificate_sealed_claim": "scpn_control.studio.sealed_claim",
    "render_sealed_claim_json": "scpn_control.studio.sealed_claim",
    "write_sealed_claim": "scpn_control.studio.sealed_claim",
}


def __getattr__(name: str) -> Any:
    """Resolve public exports lazily so SDK-free submodules stay importable.

    SDK-backed symbols raise :class:`ModuleNotFoundError` on first access when
    the ``studio`` extra is absent — the package keeps its fail-closed contract
    at the symbol boundary instead of at package import time.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'scpn_control.studio' has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package attributes including lazy public exports."""
    return sorted(set(globals()) | set(__all__))
