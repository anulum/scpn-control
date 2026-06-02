# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: SCPN package public exports.
from __future__ import annotations

"""SCPN Petri net to SNN compilation pipeline."""

from scpn_control.scpn.artifact import (
    Artifact,
    get_artifact_json_schema,
    load_artifact,
    save_artifact,
)
from scpn_control.scpn.compiler import CompiledNet, FusionCompiler
from scpn_control.scpn.contracts import (
    ControlAction,
    ControlObservation,
    ControlScales,
    ControlTargets,
    decode_actions,
    extract_features,
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
from scpn_control.scpn.controller import NeuroSymbolicController
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
    SafetyCertificateBundlePolicy,
    SafetyCertificatePolicy,
    admit_safety_certificate_bundle_artifact,
    build_safety_certificate_bundle_artifact,
    build_safety_certificate_bundle_payload,
    build_safety_certificate_payload,
    generate_safety_certificate,
    validate_safety_certificate_bundle_payload,
    validate_safety_certificate_payload,
    validate_safety_certificate_bundle_artifact,
    verify_formal_contracts,
    write_safety_certificate_bundle,
    write_safety_certificate,
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
from scpn_control.scpn.structure import StochasticPetriNet

GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay.v1"
GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay-manifest.v1"
GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay-evidence.v1"


def generate_geometry_neutral_report(*, steps: int = 12, seed: int = 314159) -> dict[str, object]:
    from scpn_control.scpn.geometry_neutral_replay import generate_geometry_neutral_report as _impl

    return _impl(steps=steps, seed=seed)


def validate_geometry_neutral_report(report: object) -> None:
    from scpn_control.scpn.geometry_neutral_replay import validate_geometry_neutral_report as _impl

    _impl(report)  # type: ignore[arg-type]


def render_geometry_neutral_markdown(report: object) -> str:
    from scpn_control.scpn.geometry_neutral_replay import render_geometry_neutral_markdown as _impl

    return _impl(report)  # type: ignore[arg-type]


def geometry_neutral_replay_evidence(report: object, **kwargs: object) -> object:
    from scpn_control.scpn.geometry_neutral_replay import geometry_neutral_replay_evidence as _impl

    return _impl(report, **kwargs)  # type: ignore[arg-type]


def assert_geometry_neutral_replay_claim_admissible(evidence: object) -> object:
    from scpn_control.scpn.geometry_neutral_replay import assert_geometry_neutral_replay_claim_admissible as _impl

    return _impl(evidence)  # type: ignore[arg-type]


def save_geometry_neutral_replay_evidence(evidence: object, output_path: object) -> None:
    from scpn_control.scpn.geometry_neutral_replay import save_geometry_neutral_replay_evidence as _impl

    _impl(evidence, output_path)  # type: ignore[arg-type]


def load_geometry_neutral_replay_evidence(path: object, **kwargs: object) -> object:
    from scpn_control.scpn.geometry_neutral_replay import load_geometry_neutral_replay_evidence as _impl

    return _impl(path, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "StochasticPetriNet",
    "FusionCompiler",
    "CompiledNet",
    "ControlObservation",
    "ControlAction",
    "ControlTargets",
    "ControlScales",
    "extract_features",
    "decode_actions",
    "MagneticConfiguration",
    "ActuatorChannel",
    "ActuatorSet",
    "DiagnosticChannel",
    "DiagnosticFrame",
    "ControlObjective",
    "ReplayScenario",
    "GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION",
    "GEOMETRY_NEUTRAL_REPLAY_MANIFEST_SCHEMA_VERSION",
    "GEOMETRY_NEUTRAL_REPLAY_EVIDENCE_SCHEMA_VERSION",
    "generate_geometry_neutral_report",
    "validate_geometry_neutral_report",
    "render_geometry_neutral_markdown",
    "geometry_neutral_replay_evidence",
    "assert_geometry_neutral_replay_claim_admissible",
    "save_geometry_neutral_replay_evidence",
    "load_geometry_neutral_replay_evidence",
    "Artifact",
    "load_artifact",
    "save_artifact",
    "get_artifact_json_schema",
    "NeuroSymbolicController",
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
