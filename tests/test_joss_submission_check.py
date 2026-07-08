# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JOSS submission guard tests.

"""Regression tests for the local JOSS submission guard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools import check_joss_submission


ROOT = Path(__file__).resolve().parents[1]


def _write_valid_joss_tree(root: Path) -> None:
    """Create a minimal JOSS paper tree that satisfies the submission guard."""

    (root / "docs").mkdir()
    (root / "paper.md").write_text(
        "\n".join(
            [
                "---",
                "title: 'SCPN Control: Test Paper'",
                "orcid: 0009-0009-3560-0851",
                "bibliography: paper.bib",
                "---",
                "# Summary",
                "quantitative external-code claims remain blocked until real artefacts are admitted [@dimits2000].",
                "# Statement of Need",
                "# Implementation",
                "production runtime claims remain subject to runtime-admission evidence",
                "# Validation",
                "# Acknowledgements",
                "# References",
            ]
        ),
        encoding="utf-8",
    )
    (root / "docs" / "joss_paper.md").write_text(
        "\n".join(
            [
                "# JOSS Paper: SCPN Control",
                "SCPN Control: Test Paper",
                "canonical JOSS-formatted source is [`paper.md`](../paper.md)",
                "with bibliography in [`paper.bib`](../paper.bib)",
                "Use this paper draft as the publication-grade summary",
                "Keep the manuscript aligned with reproducible code and benchmark evidence.",
                "Do not introduce new benchmark claims in this document without upstream evidence updates.",
                "Reference [@dimits2000].",
            ]
        ),
        encoding="utf-8",
    )
    (root / "paper.bib").write_text(
        "@article{dimits2000,\n  title = {Dimits test},\n  year = {2000}\n}\n",
        encoding="utf-8",
    )


def _point_guard_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Redirect the guard module constants to a temporary repository tree."""

    monkeypatch.setattr(check_joss_submission, "ROOT", root)
    monkeypatch.setattr(check_joss_submission, "PAPER_PATH", root / "paper.md")
    monkeypatch.setattr(check_joss_submission, "DOCS_PATH", root / "docs" / "joss_paper.md")
    monkeypatch.setattr(check_joss_submission, "BIB_PATH", root / "paper.bib")


def test_joss_submission_guard_accepts_current_repository() -> None:
    """Run the JOSS submission guard through its production CLI path."""

    assert check_joss_submission.main() == 0

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_joss_submission.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: JOSS paper, bibliography, and docs mirror" in result.stdout


def test_joss_submission_guard_reports_missing_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing JOSS paper files must fail closed with all file diagnostics."""

    _point_guard_at(monkeypatch, tmp_path)

    assert check_joss_submission.main() == 1
    output = capsys.readouterr().out
    assert "MISSING: paper.md" in output
    assert "MISSING: docs/joss_paper.md" in output
    assert "MISSING: paper.bib" in output


def test_joss_submission_guard_handles_non_repo_paths() -> None:
    """Path diagnostics must remain stable when a checked path is outside ROOT."""

    outside_path = Path("/tmp/scpn-control-outside-paper.md")

    assert check_joss_submission._relative(outside_path) == outside_path.as_posix()


def test_joss_submission_guard_reports_editorial_and_citation_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stale editorial markers and missing citation keys must be reported."""

    _write_valid_joss_tree(tmp_path)
    (tmp_path / "paper.md").write_text(
        "\n".join(
            [
                "---",
                "title: 'SCPN Control: Drifted Paper'",
                "bibliography: paper.bib",
                "---",
                "# Summary",
                "A stale claim [@missingkey].",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "docs" / "joss_paper.md").write_text(
        "# JOSS Paper: SCPN Control\nReference [@missingdoc].\n",
        encoding="utf-8",
    )
    (tmp_path / "paper.bib").write_text(
        "\n".join(
            [
                "@article{dimits2000,",
                "  title = {First}",
                "}",
                "@article{dimits2000,",
                "  title = {Duplicate}",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    _point_guard_at(monkeypatch, tmp_path)

    assert check_joss_submission.main() == 1
    output = capsys.readouterr().out
    assert "paper.md missing 'orcid: 0009-0009-3560-0851'" in output
    assert "docs/joss_paper.md missing 'canonical JOSS-formatted source" in output
    assert "docs/joss_paper.md missing paper title 'SCPN Control: Drifted Paper'" in output
    assert "paper.bib duplicate bibliography key 'dimits2000'" in output
    assert "paper.md cites missing bibliography keys: missingkey" in output
    assert "docs/joss_paper.md cites missing bibliography keys: missingdoc" in output


def test_joss_submission_guard_reports_missing_yaml_title(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A paper without a YAML title cannot enter the JOSS review workflow."""

    _write_valid_joss_tree(tmp_path)
    paper = tmp_path / "paper.md"
    paper.write_text(
        paper.read_text(encoding="utf-8").replace("title: 'SCPN Control: Test Paper'\n", ""), encoding="utf-8"
    )
    _point_guard_at(monkeypatch, tmp_path)

    assert "MISMATCH: paper.md missing YAML title" in check_joss_submission.check_repository()


def test_joss_submission_guard_skips_missing_markdown_during_citation_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Citation scanning should still run when one markdown mirror is absent."""

    _write_valid_joss_tree(tmp_path)
    (tmp_path / "docs" / "joss_paper.md").unlink()
    _point_guard_at(monkeypatch, tmp_path)

    errors = check_joss_submission.check_repository()

    assert "MISSING: docs/joss_paper.md" in errors
    assert not any("cites missing bibliography keys" in error for error in errors)


def test_joss_submission_guard_accepts_redirected_valid_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Temporary valid paper trees exercise the success path without repo files."""

    _write_valid_joss_tree(tmp_path)
    _point_guard_at(monkeypatch, tmp_path)

    assert check_joss_submission.check_repository() == []
