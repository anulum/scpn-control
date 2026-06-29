# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Evidence gap matrix tests

from __future__ import annotations

import json
from pathlib import Path

from pytest import CaptureFixture

from tools.evidence_gap_matrix import ROOT, build_evidence_gap_matrix, main


def test_evidence_gap_matrix_matches_repository_traceability_inventory() -> None:
    matrix = build_evidence_gap_matrix(ROOT / "validation" / "physics_traceability.json")

    assert len(matrix.entries) == 61
    assert matrix.public_claim_blocked == 60
    assert matrix.open_fidelity_gaps == 60
    assert len(matrix.trackers) == 8
    assert matrix.untracked_open_entries == 0
    assert matrix.status_counts == {
        "bounded_model": 34,
        "external_dependency_blocked": 4,
        "reference_validated": 1,
        "validation_gap": 22,
    }
    assert {package.tracker.issue for package in matrix.work_packages} == {47, 48, 49, 50, 51, 52, 53}


def test_evidence_gap_matrix_renders_tracker_work_package_details() -> None:
    matrix = build_evidence_gap_matrix(ROOT / "validation" / "physics_traceability.json")
    rendered = matrix.to_markdown()

    assert "# SCPN Control Evidence Gap Matrix" in rendered
    assert "Public full-fidelity claims blocked: `60`" in rendered
    assert "### Tracker #47: External gyrokinetic validation artefacts" in rendered
    assert "`src/scpn_control/core/gk_interface.py`" in rendered


def test_evidence_gap_matrix_cli_writes_json_and_markdown(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    output_json = tmp_path / "matrix.json"
    output_md = tmp_path / "matrix.md"

    assert main(["--output-json", str(output_json), "--output-md", str(output_md)]) == 0
    output = capsys.readouterr().out
    assert "Evidence gap matrix:" in output

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "scpn-control.evidence-gap-matrix.v1"
    assert payload["summary"]["public_claim_blocked"] == 60
    assert "Tracker #47" in output_md.read_text(encoding="utf-8")


def test_evidence_gap_matrix_cli_emits_json_stdout(capsys: CaptureFixture[str]) -> None:
    assert main(["--json-out"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["summary"]["open_fidelity_gaps"] == 60
    assert payload["summary"]["untracked_open_entries"] == 0


def test_evidence_gap_matrix_cli_reports_missing_registry(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    missing_registry = tmp_path / "missing.json"

    assert main(["--registry", str(missing_registry)]) == 1
    assert "Evidence gap matrix failed:" in capsys.readouterr().err


def test_evidence_gap_matrix_docs_include_entrypoint() -> None:
    validation_docs = (ROOT / "docs" / "validation.md").read_text(encoding="utf-8")

    assert "scpn-evidence-gap-matrix --output-json artifacts/evidence_gap_matrix.json" in validation_docs
    assert "scpn-control.evidence-gap-matrix.v1" in validation_docs
