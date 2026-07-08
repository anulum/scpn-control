# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native coverage matrix guard tests.

"""Tests for the native coverage matrix guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.native_coverage_matrix import (
    DEFAULT_DOCS,
    DEFAULT_PYPROJECT,
    DEFAULT_WORKFLOW,
    main,
    validate_native_coverage_matrix,
)


def _copy_matrix_inputs(tmp_path: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    """Copy the live matrix inputs into a temporary fixture set."""
    workflow = tmp_path / "ci.yml"
    pyproject = tmp_path / "pyproject.toml"
    docs = (tmp_path / "validation.md", tmp_path / "development.md")
    workflow.write_text(DEFAULT_WORKFLOW.read_text(encoding="utf-8"), encoding="utf-8")
    pyproject.write_text(DEFAULT_PYPROJECT.read_text(encoding="utf-8"), encoding="utf-8")
    for source, target in zip(DEFAULT_DOCS, docs, strict=True):
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return workflow, pyproject, docs


def test_live_native_coverage_matrix_is_wired() -> None:
    """The checked-out CI, threshold, and public docs expose the matrix contract."""
    matrix = validate_native_coverage_matrix()

    assert matrix.passed is True
    assert matrix.findings == ()
    assert matrix.to_jsonable()["schema_version"] == "scpn-control.native-coverage-matrix.v1"


def test_native_coverage_matrix_rejects_missing_rust_present_artifact(tmp_path: Path) -> None:
    """The guard fails when the Rust-present coverage data artifact is not uploaded."""
    workflow, pyproject, docs = _copy_matrix_inputs(tmp_path)
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace("coverage-data-rust", "coverage-data-native"),
        encoding="utf-8",
    )

    matrix = validate_native_coverage_matrix(workflow, pyproject, docs)

    assert matrix.passed is False
    assert [finding.check for finding in matrix.findings] == ["rust-present coverage data artifact"]


def test_native_coverage_matrix_rejects_missing_combine_gate(tmp_path: Path) -> None:
    """The guard fails when the merged report is no longer gated at 100 percent."""
    workflow, pyproject, docs = _copy_matrix_inputs(tmp_path)
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace("python -m coverage report --fail-under=100", ""),
        encoding="utf-8",
    )

    matrix = validate_native_coverage_matrix(workflow, pyproject, docs)

    assert matrix.passed is False
    assert [finding.check for finding in matrix.findings] == ["combined coverage job"]


def test_native_coverage_matrix_rejects_stale_public_docs(tmp_path: Path) -> None:
    """The guard fails when public docs stop naming the matrix schema."""
    workflow, pyproject, docs = _copy_matrix_inputs(tmp_path)
    for doc_path in docs:
        doc_path.write_text(
            doc_path.read_text(encoding="utf-8").replace("scpn-control.native-coverage-matrix.v1", "removed"),
            encoding="utf-8",
        )

    matrix = validate_native_coverage_matrix(workflow, pyproject, docs)

    assert matrix.passed is False
    assert [finding.check for finding in matrix.findings] == ["public docs"]


def test_native_coverage_matrix_cli_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI emits the stable JSON schema used by preflight and CI diagnostics."""
    assert main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "findings": [],
        "passed": True,
        "schema_version": "scpn-control.native-coverage-matrix.v1",
    }


def test_native_coverage_matrix_cli_text_success(capsys: pytest.CaptureFixture[str]) -> None:
    """The text CLI reports success for the live repository contract."""
    assert main([]) == 0

    assert "Native coverage matrix guard passed." in capsys.readouterr().out


def test_native_coverage_matrix_cli_text_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The text CLI reports each failed contract for diagnostic use."""
    workflow, pyproject, docs = _copy_matrix_inputs(tmp_path)
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace("coverage-report-combined", "coverage-report"),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--workflow",
                str(workflow),
                "--pyproject",
                str(pyproject),
                "--docs",
                *(str(doc_path) for doc_path in docs),
            ]
        )
        == 1
    )

    output = capsys.readouterr().out
    assert "Native coverage matrix guard FAILED" in output
    assert "combined coverage job" in output
