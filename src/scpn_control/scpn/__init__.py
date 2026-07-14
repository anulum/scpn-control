# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SCPN package public exports.
"""SCPN Petri net to SNN compilation pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only imports: typing the lazy-import facades below with their real
    # parameter/return types (instead of `object`) removes the per-call
    # arg-type suppression comments without forcing the heavy geometry_neutral_replay
    # module to import at package-import time (it is still imported lazily, inside
    # each function body). `from __future__ import annotations` keeps these strings.
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Any

    from scpn_control.scpn.geometry_neutral_replay import GeometryNeutralReplayEvidence

from scpn_control.scpn.artifact import (
    Artifact,
    get_artifact_json_schema,
    load_artifact,
    save_artifact,
    validate_artifact,
)
from scpn_control.scpn.compiler import CompiledNet, FusionCompiler
from scpn_control.scpn.contracts import (
    ControlAction,
    ControlObservation,
    ControlScales,
    ControlTargets,
    FeatureAxisSpec,
    decode_action_vector,
    decode_actions,
    extract_features,
    feature_error_components,
)
from scpn_control.scpn.controller import NeuroSymbolicController
from scpn_control.scpn.formal_safety_certificate import (
    SafetyCertificateBundlePolicy,
    SafetyCertificatePolicy,
    admit_safety_certificate_bundle_artifact,
    build_safety_certificate_bundle_artifact,
    build_safety_certificate_bundle_payload,
    build_safety_certificate_payload,
    generate_safety_certificate,
    validate_safety_certificate_bundle_artifact,
    validate_safety_certificate_bundle_payload,
    validate_safety_certificate_payload,
    write_safety_certificate,
    write_safety_certificate_bundle,
)
from scpn_control.scpn.formal_verification import (
    AlwaysBounded,
    AlwaysEventuallyMarked,
    CTLFormula,
    EventuallyFires,
    FireLeadsToMarking,
    FormalPetriNetVerifier,
    FormalPropertyReport,
    FormalVerificationReport,
    FormalViolation,
    LTLFormula,
    NeverCoMarked,
    PlaceInvariant,
    ReachabilityReport,
    ReachableMarking,
    verify_formal_contracts,
)
from scpn_control.scpn.fpga_export import (
    FPGAConfig,
    HDLExportEvidence,
    assert_hdl_export_claim_admissible,
    compile_to_verilog,
    compile_to_vhdl,
    estimate_resources,
    export_bitstream_project,
    hdl_export_evidence,
    load_hdl_export_evidence,
    save_hdl_export_evidence,
)
from scpn_control.scpn.geometry_neutral_contracts import (
    ActuatorChannel,
    ActuatorSet,
    ControlObjective,
    DiagnosticChannel,
    DiagnosticFrame,
    MagneticConfiguration,
    ReplayScenario,
)
from scpn_control.scpn.lean_verification import (
    LeanFormalVerificationError,
    LeanFormalVerificationReport,
    build_lean_formal_report_payload,
    compute_assumption_sha256,
    load_lean_formal_report,
    validate_bounded_proof_assumptions,
    validate_lean_formal_report_payload,
    validate_required_contract_theorem_coverage,
    write_lean_formal_report,
)
from scpn_control.scpn.observation import (
    AERControlObservation,
    DecodeStrategy,
    FeatureNormalisation,
    SpikeBuffer,
    SpikeEvent,
    decode_isi,
    decode_rate,
    decode_temporal,
)
from scpn_control.scpn.runtime_safety_certificate import (
    CertificateReplayResult,
    ControllerRuntimeBinding,
    RuntimeTarget,
    TimingEnvelope,
    assert_runtime_certificate_admissible,
    compute_petri_topology_digest,
    issue_runtime_safety_certificate,
    replay_runtime_safety_certificate,
    validate_runtime_safety_certificate_payload,
)
from scpn_control.scpn.structure import StochasticPetriNet

GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay.v1"
GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION_V1_1 = "scpn-control.geometry-neutral-replay.v1.1"
SUPPORTED_GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSIONS = (
    GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION,
    GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION_V1_1,
)
GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay-manifest.v1"
GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay-evidence.v1"
GEOMETRY_NEUTRAL_REPLAY_AER_ADMISSION_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay-aer-admission.v1"


def generate_geometry_neutral_report(*, steps: int = 12, seed: int = 314159) -> dict[str, object]:
    """Generate a deterministic geometry-neutral replay report (lazy import)."""
    from scpn_control.scpn.geometry_neutral_replay import generate_geometry_neutral_report as _impl

    return _impl(steps=steps, seed=seed)


def validate_geometry_neutral_report(report: Mapping[str, Any]) -> None:
    """Validate a geometry-neutral replay report, raising on any violation."""
    from scpn_control.scpn.geometry_neutral_replay import validate_geometry_neutral_report as _impl

    _impl(report)


def load_geometry_neutral_replay_schema(version: str) -> dict[str, object]:
    """Load the geometry-neutral replay JSON schema for the given version."""
    from scpn_control.scpn.geometry_neutral_replay import load_replay_schema as _impl

    return _impl(version)


def register_geometry_neutral_replay_v1_1_schema() -> dict[str, object]:
    """Register and return the v1.1 geometry-neutral replay schema."""
    from scpn_control.scpn.geometry_neutral_replay import register_v1_1_schema as _impl

    return _impl()


def assert_geometry_neutral_v1_replay_loadable_under_v1_1_schema_bundle(
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Assert a v1 replay report loads under the v1.1 schema bundle."""
    from scpn_control.scpn.geometry_neutral_replay import (
        assert_v1_replay_loadable_under_v1_1_schema_bundle as _impl,
    )

    return _impl(report)


def build_geometry_neutral_aer_admission_metadata(
    *,
    admission_report: Mapping[str, Any],
    decode_strategy: str,
    decode_window_ns: int,
    n_features: int,
    feature_normalisation: str = "unit",
    require_monotonic: bool = False,
    feature_vector: object | None = None,
) -> dict[str, Any]:
    """Build AER-admission metadata for a geometry-neutral replay."""
    from scpn_control.scpn.geometry_neutral_replay import build_aer_admission_metadata as _impl

    return _impl(
        admission_report=admission_report,
        decode_strategy=decode_strategy,
        decode_window_ns=decode_window_ns,
        n_features=n_features,
        feature_normalisation=feature_normalisation,
        require_monotonic=require_monotonic,
        feature_vector=feature_vector,
    )


def attach_geometry_neutral_aer_admission_metadata(
    report: Mapping[str, Any], aer_admission: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach AER-admission metadata to a geometry-neutral replay report."""
    from scpn_control.scpn.geometry_neutral_replay import attach_aer_admission_metadata as _impl

    return _impl(report, aer_admission)


def save_geometry_neutral_replay_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Save a geometry-neutral replay report to disk."""
    from scpn_control.scpn.geometry_neutral_replay import save_geometry_neutral_replay_report as _impl

    return _impl(report, output_path)


def load_geometry_neutral_replay_report(path: str | Path) -> dict[str, Any]:
    """Load a geometry-neutral replay report from disk."""
    from scpn_control.scpn.geometry_neutral_replay import load_geometry_neutral_replay_report as _impl

    return _impl(path)


def render_geometry_neutral_markdown(report: Mapping[str, Any]) -> str:
    """Render a geometry-neutral replay report as Markdown."""
    from scpn_control.scpn.geometry_neutral_replay import render_geometry_neutral_markdown as _impl

    return _impl(report)


def geometry_neutral_replay_evidence(
    report: Mapping[str, Any],
    *,
    generated_utc: str | None = None,
    measured_or_benchmark_artefact_sha256: str | None = None,
    device_claim_allowed: bool = False,
) -> GeometryNeutralReplayEvidence:
    """Build admission evidence for a geometry-neutral replay report."""
    from scpn_control.scpn.geometry_neutral_replay import geometry_neutral_replay_evidence as _impl

    return _impl(
        report,
        generated_utc=generated_utc,
        measured_or_benchmark_artefact_sha256=measured_or_benchmark_artefact_sha256,
        device_claim_allowed=device_claim_allowed,
    )


def assert_geometry_neutral_replay_claim_admissible(
    evidence: GeometryNeutralReplayEvidence,
) -> GeometryNeutralReplayEvidence:
    """Assert geometry-neutral replay evidence is admissible, raising otherwise."""
    from scpn_control.scpn.geometry_neutral_replay import assert_geometry_neutral_replay_claim_admissible as _impl

    return _impl(evidence)


def save_geometry_neutral_replay_evidence(evidence: GeometryNeutralReplayEvidence, output_path: str | Path) -> None:
    """Save geometry-neutral replay evidence to disk."""
    from scpn_control.scpn.geometry_neutral_replay import save_geometry_neutral_replay_evidence as _impl

    _impl(evidence, output_path)


def load_geometry_neutral_replay_evidence(
    path: str | Path, *, require_device_claim: bool = False
) -> GeometryNeutralReplayEvidence:
    """Load geometry-neutral replay evidence from disk."""
    from scpn_control.scpn.geometry_neutral_replay import load_geometry_neutral_replay_evidence as _impl

    return _impl(path, require_device_claim=require_device_claim)


__all__ = [
    "StochasticPetriNet",
    "FusionCompiler",
    "CompiledNet",
    "ControlObservation",
    "ControlAction",
    "ControlTargets",
    "ControlScales",
    "FeatureAxisSpec",
    "extract_features",
    "feature_error_components",
    "decode_actions",
    "decode_action_vector",
    "AERControlObservation",
    "DecodeStrategy",
    "FeatureNormalisation",
    "SpikeBuffer",
    "SpikeEvent",
    "decode_rate",
    "decode_temporal",
    "decode_isi",
    "MagneticConfiguration",
    "ActuatorChannel",
    "ActuatorSet",
    "DiagnosticChannel",
    "DiagnosticFrame",
    "ControlObjective",
    "ReplayScenario",
    "GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION",
    "GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION_V1_1",
    "SUPPORTED_GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSIONS",
    "GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION",
    "GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION",
    "GEOMETRY_NEUTRAL_REPLAY_AER_ADMISSION_SCHEMA_VERSION",
    "generate_geometry_neutral_report",
    "validate_geometry_neutral_report",
    "load_geometry_neutral_replay_schema",
    "register_geometry_neutral_replay_v1_1_schema",
    "assert_geometry_neutral_v1_replay_loadable_under_v1_1_schema_bundle",
    "build_geometry_neutral_aer_admission_metadata",
    "attach_geometry_neutral_aer_admission_metadata",
    "save_geometry_neutral_replay_report",
    "load_geometry_neutral_replay_report",
    "render_geometry_neutral_markdown",
    "geometry_neutral_replay_evidence",
    "assert_geometry_neutral_replay_claim_admissible",
    "save_geometry_neutral_replay_evidence",
    "load_geometry_neutral_replay_evidence",
    "Artifact",
    "load_artifact",
    "save_artifact",
    "validate_artifact",
    "get_artifact_json_schema",
    "NeuroSymbolicController",
    "RuntimeTarget",
    "TimingEnvelope",
    "ControllerRuntimeBinding",
    "CertificateReplayResult",
    "compute_petri_topology_digest",
    "issue_runtime_safety_certificate",
    "validate_runtime_safety_certificate_payload",
    "replay_runtime_safety_certificate",
    "assert_runtime_certificate_admissible",
    "FormalPetriNetVerifier",
    "ReachableMarking",
    "FormalViolation",
    "ReachabilityReport",
    "FormalPropertyReport",
    "FormalVerificationReport",
    "LeanFormalVerificationError",
    "LeanFormalVerificationReport",
    "compute_assumption_sha256",
    "PlaceInvariant",
    "SafetyCertificatePolicy",
    "SafetyCertificateBundlePolicy",
    "admit_safety_certificate_bundle_artifact",
    "build_safety_certificate_bundle_artifact",
    "CTLFormula",
    "LTLFormula",
    "AlwaysBounded",
    "AlwaysEventuallyMarked",
    "EventuallyFires",
    "FireLeadsToMarking",
    "NeverCoMarked",
    "verify_formal_contracts",
    "build_safety_certificate_bundle_payload",
    "build_safety_certificate_payload",
    "generate_safety_certificate",
    "validate_safety_certificate_bundle_payload",
    "validate_safety_certificate_payload",
    "validate_safety_certificate_bundle_artifact",
    "write_safety_certificate_bundle",
    "write_safety_certificate",
    "build_lean_formal_report_payload",
    "load_lean_formal_report",
    "validate_bounded_proof_assumptions",
    "validate_lean_formal_report_payload",
    "validate_required_contract_theorem_coverage",
    "write_lean_formal_report",
    "FPGAConfig",
    "HDLExportEvidence",
    "compile_to_verilog",
    "compile_to_vhdl",
    "estimate_resources",
    "export_bitstream_project",
    "hdl_export_evidence",
    "assert_hdl_export_claim_admissible",
    "save_hdl_export_evidence",
    "load_hdl_export_evidence",
]
