# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Scpn Package
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Neuro-Symbolic Logic Compiler
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
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
from scpn_control.scpn.fpga_export import (
    FPGAConfig,
    compile_to_verilog,
    compile_to_vhdl,
    estimate_resources,
    export_bitstream_project,
)
from scpn_control.scpn.structure import StochasticPetriNet

GEOMETRY_NEUTRAL_REPLAY_SCHEMA_VERSION = "scpn-control.geometry-neutral-replay.v1"


def generate_geometry_neutral_report(*, steps: int = 12, seed: int = 314159) -> dict[str, object]:
    from scpn_control.scpn.geometry_neutral_replay import generate_geometry_neutral_report as _impl

    return _impl(steps=steps, seed=seed)


def validate_geometry_neutral_report(report: object) -> None:
    from scpn_control.scpn.geometry_neutral_replay import validate_geometry_neutral_report as _impl

    _impl(report)  # type: ignore[arg-type]


def render_geometry_neutral_markdown(report: object) -> str:
    from scpn_control.scpn.geometry_neutral_replay import render_geometry_neutral_markdown as _impl

    return _impl(report)  # type: ignore[arg-type]

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
    "generate_geometry_neutral_report",
    "validate_geometry_neutral_report",
    "render_geometry_neutral_markdown",
    "Artifact",
    "load_artifact",
    "save_artifact",
    "get_artifact_json_schema",
    "NeuroSymbolicController",
    "FPGAConfig",
    "compile_to_verilog",
    "compile_to_vhdl",
    "estimate_resources",
    "export_bitstream_project",
]
