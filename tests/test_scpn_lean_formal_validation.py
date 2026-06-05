# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Lean formal-verification evidence validator tests.
"""Tests for the Lean formal-verification validation executable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scpn_control.scpn.artifact import compute_artifact_payload_sha256, load_artifact, save_artifact
from scpn_control.scpn.compiler import FusionCompiler
from scpn_control.scpn.lean_verification import LeanFormalVerificationReport, write_lean_formal_report
from scpn_control.scpn.structure import StochasticPetriNet
from validation.validate_scpn_lean_formal import main, validate_lean_formal_evidence


def _lean_report(*, artifact_sha256: str = "a" * 64) -> LeanFormalVerificationReport:
    return LeanFormalVerificationReport(
        status="pass",
        solver="Lean 4.13.0",
        lean_version="4.13.0",
        checked_specs=["pid.actuator_saturation", "snn.marking_bounds"],
        artifact_sha256=artifact_sha256,
        proof_source_sha256="b" * 64,
        lakefile_sha256="c" * 64,
        theorem_names=[
            "ScpnControl.PID.actuatorSaturationPreserved",
            "ScpnControl.SNN.markingBoundsPreserved",
        ],
        theorem_modules=["ScpnControl.PID", "ScpnControl.SNN"],
        proved_contracts=["pid.actuator_saturation", "snn.marking_bounds"],
        module_paths=[
            "src/scpn_control/control/pid_controller.py",
            "src/scpn_control/scpn/controller.py",
        ],
        safety_case_ids=["SC-PID-ACTUATOR-SATURATION", "SC-SNN-MARKING-BOUNDS"],
        claim_boundary="bounded Lean proof over exported controller envelope",
        proof_assumptions=[
            "bounded actuator command interval from exported artifact readout limits",
            "bounded SNN marking interval [0, 1] from compiled artifact topology",
        ],
    )


def _artifact_path(tmp_path: Path) -> Path:
    net = StochasticPetriNet()
    net.add_place("P0")
    net.add_place("P1")
    net.add_transition("T0", threshold=0.5)
    net.add_arc("P0", "T0", 0.8)
    net.add_arc("T0", "P1", 0.7)
    compiled = FusionCompiler(bitstream_length=64, seed=0).compile(net)
    artifact = compiled.export_artifact(
        name="lean-validation-artifact",
        readout_config={
            "actions": [{"name": "act0", "pos_place": 1, "neg_place": 0}],
            "gains": [1.0],
            "abs_max": [10.0],
            "slew_per_s": [100.0],
        },
        injection_config=[
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
    )
    path = tmp_path / "artifact.scpnctl.json"
    save_artifact(artifact, path)
    return path


def test_lean_formal_validator_admits_valid_report(tmp_path: Path) -> None:
    report_path = tmp_path / "lean-report.json"
    payload = write_lean_formal_report(_lean_report(), report_path)

    result = validate_lean_formal_evidence(report_path)

    assert result.status == "pass"
    assert result.errors == ()
    assert result.backend == "lean4"
    assert result.report_sha256 == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert result.theorem_names == tuple(payload["theorem_names"])


def test_lean_formal_validator_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    report_path = tmp_path / "duplicate.json"
    report_path.write_text('{"status": "pass", "status": "fail"}', encoding="utf-8")

    result = validate_lean_formal_evidence(report_path)

    assert result.status == "fail"
    assert result.errors == ("duplicate JSON key: status",)


def test_lean_formal_validator_rejects_missing_required_contract(tmp_path: Path) -> None:
    report_path = tmp_path / "missing-contract.json"
    payload = write_lean_formal_report(_lean_report(), report_path)
    payload.pop("payload_sha256")
    payload["proved_contracts"] = ["pid.actuator_saturation"]
    report_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    result = validate_lean_formal_evidence(report_path)

    assert result.status == "fail"
    assert any("missing required contracts" in error for error in result.errors)


def test_lean_formal_validator_rejects_solver_version_mismatch(tmp_path: Path) -> None:
    report_path = tmp_path / "solver-mismatch.json"
    payload = write_lean_formal_report(_lean_report(), report_path)
    payload.pop("payload_sha256")
    payload["solver"] = "Lean 4.12.0"
    report_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    result = validate_lean_formal_evidence(report_path)

    assert result.status == "fail"
    assert any("solver must include lean_version" in error for error in result.errors)


def test_lean_formal_validator_rejects_unrelated_module_path(tmp_path: Path) -> None:
    report_path = tmp_path / "unrelated-path.json"
    payload = write_lean_formal_report(_lean_report(), report_path)
    payload.pop("payload_sha256")
    payload["module_paths"] = [
        "src/scpn_control/control/pid_controller.py",
        "src/scpn_control/scpn/geometry_neutral_replay.py",
    ]
    report_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    result = validate_lean_formal_evidence(report_path)

    assert result.status == "fail"
    assert any("module_paths missing required paths" in error for error in result.errors)


def test_lean_formal_validator_rejects_unknown_report_field(tmp_path: Path) -> None:
    report_path = tmp_path / "unknown-field.json"
    payload = write_lean_formal_report(_lean_report(), report_path)
    payload.pop("payload_sha256")
    payload["certification_status"] = "certified"
    report_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    result = validate_lean_formal_evidence(report_path)

    assert result.status == "fail"
    assert any("unsupported fields" in error for error in result.errors)


def test_lean_formal_validator_admits_artifact_bound_to_report(tmp_path: Path) -> None:
    artifact_path = _artifact_path(tmp_path)
    artifact = load_artifact(artifact_path)
    report_root = tmp_path / "reports"
    report_path = report_root / "validation" / "reports" / "scpn_lean4_formal.json"
    report = _lean_report(artifact_sha256=compute_artifact_payload_sha256(artifact))
    write_lean_formal_report(report, report_path)
    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["formal_verification"] = {
        "required": True,
        "status": "pass",
        "backend": "lean4",
        "solver": report.solver,
        "max_depth": 0,
        "checked_specs": report.checked_specs,
        "artifact_sha256": report.artifact_sha256,
        "report_sha256": report_sha256,
        "claim_boundary": report.claim_boundary,
        "report_uri": "validation/reports/scpn_lean4_formal.json",
        "generated_utc": "2026-06-02T00:00:00Z",
        "lean_version": report.lean_version,
        "lakefile_sha256": report.lakefile_sha256,
        "proof_source_sha256": report.proof_source_sha256,
        "theorem_names": report.theorem_names,
        "theorem_modules": report.theorem_modules,
        "proved_contracts": report.proved_contracts,
        "module_paths": report.module_paths,
        "safety_case_ids": report.safety_case_ids,
        "proof_assumptions": report.proof_assumptions,
        "assumption_sha256": json.loads(report_path.read_text(encoding="utf-8"))["assumption_sha256"],
    }
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_lean_formal_evidence(
        report_path,
        artifact_path=artifact_path,
        formal_report_root=report_root,
    )

    assert result.status == "pass"
    assert result.artifact_admitted is True


def test_lean_formal_validator_cli_returns_nonzero_for_invalid_report(tmp_path: Path) -> None:
    report_path = tmp_path / "invalid.json"
    report_path.write_text('{"schema_version": "wrong"}', encoding="utf-8")

    assert main([str(report_path)]) == 1
