# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Evidence gap matrix tests

"""Tests for the repository-wide external-evidence gap inventory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from pytest import CaptureFixture

from tools.evidence_gap_matrix import ROOT, build_evidence_gap_matrix, main

_OPEN_STATUSES = frozenset({"bounded_model", "external_dependency_blocked", "validation_gap"})


def _registry_summary() -> tuple[int, int, int, dict[str, int]]:
    """Count the registry directly, without the matrix builder.

    Returns the entry count, the entries without public-claim permission, the
    entries in an open fidelity status, and the count per status.
    """
    registry = json.loads((ROOT / "validation" / "physics_traceability.json").read_text(encoding="utf-8"))
    entries = registry["entries"]
    statuses = Counter(entry["fidelity_status"] for entry in entries)
    blocked = sum(1 for entry in entries if entry["public_claim_allowed"] is False)
    open_gaps = sum(count for status, count in statuses.items() if status in _OPEN_STATUSES)
    return len(entries), blocked, open_gaps, dict(statuses)


def test_evidence_gap_matrix_matches_repository_traceability_inventory() -> None:
    """The matrix summary must match the canonical traceability registry."""
    matrix = build_evidence_gap_matrix(ROOT / "validation" / "physics_traceability.json")

    total, blocked, open_gaps, statuses = _registry_summary()

    assert total > 0 and open_gaps > 0
    assert len(matrix.entries) == total
    assert matrix.public_claim_blocked == blocked
    assert matrix.open_fidelity_gaps == open_gaps
    assert len(matrix.trackers) == 8
    assert matrix.untracked_open_entries == 0
    assert matrix.status_counts == statuses
    assert {package.tracker.issue for package in matrix.work_packages} == {47, 48, 49, 50, 51, 52, 53}


def test_evidence_gap_matrix_renders_tracker_work_package_details() -> None:
    """Rendered work packages must retain tracker and source-path detail."""
    matrix = build_evidence_gap_matrix(ROOT / "validation" / "physics_traceability.json")
    rendered = matrix.to_markdown()

    assert "# SCPN Control Evidence Gap Matrix" in rendered
    assert f"Public full-fidelity claims blocked: `{_registry_summary()[1]}`" in rendered
    assert "### Tracker #47: External gyrokinetic validation artefacts" in rendered
    assert "`src/scpn_control/core/gk_interface.py`" in rendered


def test_evidence_gap_matrix_cli_writes_json_and_markdown(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The CLI must write equivalent machine- and human-readable matrices."""
    output_json = tmp_path / "matrix.json"
    output_md = tmp_path / "matrix.md"

    assert main(["--output-json", str(output_json), "--output-md", str(output_md)]) == 0
    output = capsys.readouterr().out
    assert "Evidence gap matrix:" in output

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "scpn-control.evidence-gap-matrix.v1"
    assert payload["summary"]["public_claim_blocked"] == _registry_summary()[1]
    assert "Tracker #47" in output_md.read_text(encoding="utf-8")


def test_evidence_gap_matrix_cli_emits_json_stdout(capsys: CaptureFixture[str]) -> None:
    """The stdout mode must expose the same canonical matrix summary."""
    assert main(["--json-out"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["summary"]["open_fidelity_gaps"] == _registry_summary()[2]
    assert payload["summary"]["untracked_open_entries"] == 0


def test_evidence_gap_matrix_cli_reports_missing_registry(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """A missing canonical registry must fail with an actionable message."""
    missing_registry = tmp_path / "missing.json"

    assert main(["--registry", str(missing_registry)]) == 1
    assert "Evidence gap matrix failed:" in capsys.readouterr().err


def test_evidence_gap_matrix_docs_include_entrypoint() -> None:
    """Validation documentation must retain the reproducible CLI entrypoint."""
    validation_docs = (ROOT / "docs" / "validation.md").read_text(encoding="utf-8")

    assert "python tools/evidence_gap_matrix.py --output-json artifacts/evidence_gap_matrix.json" in validation_docs
    assert "scpn-control.evidence-gap-matrix.v1" in validation_docs


@pytest.mark.parametrize(
    "kind",
    [
        "registry",
        "same",
        "hardlink-input",
        "hardlink-outputs",
        "directory",
        "blocked-parent",
        "selected-ledger-existing",
        "selected-ledger-missing",
        "reports",
        "refresh",
        "new-report",
    ],
)
@pytest.mark.parametrize("isolated", [True, False])
def test_cold_cli_refuses_outputs_and_preserves_inputs(tmp_path: Path, kind: str, isolated: bool) -> None:
    """A cold standalone command preserves actual declaration bytes before any replacement."""
    selected = tmp_path / "selected/validation"
    selected.mkdir(parents=True)
    registry = selected / "physics_traceability.json"
    registry.write_bytes((ROOT / "validation/physics_traceability.json").read_bytes())
    first, second = tmp_path / "matrix.json", tmp_path / "matrix.md"
    first.write_bytes(b"original first output")
    second.write_bytes(b"original second output")
    options = ["--output-json", str(first), "--output-md", str(second)]
    if kind == "registry":
        options[1] = str(registry)
    elif kind == "same":
        options[3] = str(first)
    elif kind == "hardlink-input":
        first.unlink()
        first.hardlink_to(registry)
    elif kind == "hardlink-outputs":
        second.unlink()
        second.hardlink_to(first)
    elif kind == "directory":
        second.unlink()
        second.mkdir()
    elif kind == "blocked-parent":
        options[3] = str(second / "matrix.md")
    elif kind.startswith("selected-ledger"):
        ledger = selected / "public_claim_ledger.json"
        if kind.endswith("existing"):
            ledger.write_bytes((ROOT / "validation/public_claim_ledger.json").read_bytes())
        options[1] = str(ledger)
    elif kind in {"reports", "refresh", "new-report"}:
        namespace = selected / ("report_refreshes" if kind == "refresh" else "reports")
        if kind != "new-report":
            namespace.mkdir()
            (namespace / "existing.json").write_bytes(b"preserved evidence")
        options[1] = str(namespace / "new.json")
    else:
        raise AssertionError(kind)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if isolated else []),
            str(ROOT / "tools/evidence_gap_matrix.py"),
            "--registry",
            str(registry),
            *options,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and not result.stdout
    assert "Traceback" not in result.stderr and str(tmp_path) not in result.stderr
    if kind == "blocked-parent":
        assert result.stderr == "Evidence gap matrix failed: inputs or outputs could not be inspected\n"
    else:
        assert result.stderr.startswith("Evidence gap matrix failed: Inventory outputs must")
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("isolated", [True, False])
def test_cold_cli_keeps_stdout_precedence_and_complete_file_pair(tmp_path: Path, isolated: bool) -> None:
    """Cold startup renders canonical declarations through both files and JSON stdout."""
    first, second = tmp_path / "matrix.json", tmp_path / "matrix.md"
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if isolated else []),
            str(ROOT / "tools/evidence_gap_matrix.py"),
            "--json-out",
            "--markdown-out",
            "--output-json",
            str(first),
            "--output-md",
            str(second),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    expected = build_evidence_gap_matrix(ROOT / "validation/physics_traceability.json")
    assert result.stdout.encode() == first.read_bytes()
    assert json.loads(result.stdout) == expected.to_dict()
    assert second.read_text() == expected.to_markdown()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["matrix.json", "matrix.md"]


def test_public_cli_markdown_stdout_matches_complete_registry(capsys: CaptureFixture[str]) -> None:
    """Markdown-only stdout exposes the same complete declared planning inventory."""
    expected = build_evidence_gap_matrix(ROOT / "validation/physics_traceability.json")
    assert main(["--markdown-out"]) == 0
    captured = capsys.readouterr()
    assert captured.out == expected.to_markdown()
    assert not captured.err
