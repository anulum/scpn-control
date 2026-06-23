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

The platform SDK is an optional dependency (install the ``studio`` extra); importing
this package without it raises :class:`ModuleNotFoundError` at import time, which is
the intended fail-closed behaviour for an optional studio surface.
"""

from __future__ import annotations

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
    "build_manifest",
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
    "replay_evidence_from_geometry_neutral",
    "representative_bundles",
    "representative_feed",
    "safety_certificate_evidence",
    "safety_certificate_evidence_from_certificate",
    "scenario_simulation_evidence",
    "scenario_simulation_evidence_from_audit",
    "studio_feed",
    "verb_summary",
]
