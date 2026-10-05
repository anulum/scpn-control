# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Formal validator declaration and public command evidence

"""Authored report declarations for reader tests; no Lean proof execution witness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from scpn_control.scpn.artifact import compute_artifact_payload_sha256, save_artifact
from scpn_control.scpn.artifact_model import FormalVerificationEvidence
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.lean_verification import LeanFormalVerificationReport, write_lean_formal_report
from scpn_control.scpn.structure import StochasticPetriNet


def declared_lean_case(root: Path) -> tuple[Path, Path, Path]:
    """Compile a real artifact and write two distinct valid report declarations.

    Parameters
    ----------
    root : pathlib.Path
        Isolated caller-owned output root; parent directories are created.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        Named declaration, another valid declaration with different proof-source
        digest, and artifact bound to the named declaration's raw-byte digest.

    Notes
    -----
    The actual public compiler constructs the tiny artifact. Report proof/source
    hashes and Lean version are authored metadata, not independently compiled
    proof evidence. This fixture exercises reader contracts only.
    """
    net = StochasticPetriNet()
    net.add_place("P0")
    net.add_place("P1")
    net.add_transition("T0", threshold=0.5)
    net.add_arc("P0", "T0", 0.8)
    net.add_arc("T0", "P1", 0.7)
    compiled = FusionCompiler(bitstream_length=64, seed=0).compile(net)
    artifact = compiled.export_artifact(
        name="declaration-only-reader-case",
        readout_config={
            "actions": [{"name": "ctrl", "pos_place": 1, "neg_place": 0}],
            "gains": [1.0],
            "abs_max": [10.0],
            "slew_per_s": [100.0],
        },
        injection_config=[{"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True}],
    )
    report = LeanFormalVerificationReport(
        status="pass",
        solver="Lean 4.13.0",
        lean_version="4.13.0",
        checked_specs=["pid.actuator_saturation", "snn.marking_bounds"],
        artifact_sha256=compute_artifact_payload_sha256(artifact),
        proof_source_sha256="b" * 64,
        lakefile_sha256="c" * 64,
        theorem_names=["ScpnControl.PID.actuatorSaturationPreserved", "ScpnControl.SNN.markingBoundsPreserved"],
        theorem_modules=["ScpnControl.PID", "ScpnControl.SNN"],
        proved_contracts=["pid.actuator_saturation", "snn.marking_bounds"],
        module_paths=["scpn_control.control.pid_controller", "scpn_control.scpn.controller"],
        safety_case_ids=["SC-PID-ACTUATOR-SATURATION", "SC-SNN-MARKING-BOUNDS"],
        claim_boundary="bounded Lean proof over exported controller envelope",
        proof_assumptions=[
            "bounded actuator command interval from exported artifact readout limits",
            "bounded SNN marking interval [0, 1] from compiled artifact topology",
        ],
    )
    named = root / "reports/lean.json"
    payload = write_lean_formal_report(report, named)
    other = root / "reports/different-valid-lean.json"
    write_lean_formal_report(replace(report, proof_source_sha256="d" * 64), other)
    assert named.read_bytes() != other.read_bytes()
    artifact.formal_verification = FormalVerificationEvidence(
        required=True,
        status="pass",
        backend="lean4",
        solver=report.solver,
        max_depth=0,
        checked_specs=report.checked_specs,
        artifact_sha256=report.artifact_sha256,
        report_sha256=hashlib.sha256(named.read_bytes()).hexdigest(),
        claim_boundary=report.claim_boundary,
        report_uri="reports/lean.json",
        generated_utc="2026-10-02T00:00:00Z",
        lean_version=report.lean_version,
        lakefile_sha256=report.lakefile_sha256,
        proof_source_sha256=report.proof_source_sha256,
        theorem_names=report.theorem_names,
        theorem_modules=report.theorem_modules,
        proved_contracts=report.proved_contracts,
        module_paths=report.module_paths,
        safety_case_ids=report.safety_case_ids,
        proof_assumptions=report.proof_assumptions,
        assumption_sha256=str(payload["assumption_sha256"]),
    )
    artifact_path = root / "artifact.scpnctl.json"
    save_artifact(artifact, artifact_path)
    assert json.loads(artifact_path.read_text())["formal_verification"]["backend"] == "lean4"
    return named, other, artifact_path
