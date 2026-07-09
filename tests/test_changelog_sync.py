# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Changelog sync guard tests.
"""Regression tests for the changelog mirror sync guard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_changelog_sync import changelog_sync_errors, main

ROOT = Path(__file__).resolve().parents[1]


def _write_changelogs(repo: Path, root_text: str, docs_text: str | None = None) -> None:
    """Write a minimal root changelog and rendered docs mirror."""

    (repo / "docs").mkdir()
    (repo / "CHANGELOG.md").write_text(root_text, encoding="utf-8")
    (repo / "docs" / "changelog.md").write_text(
        root_text if docs_text is None else docs_text,
        encoding="utf-8",
    )


def test_repository_changelog_mirror_is_current() -> None:
    """The committed rendered changelog mirror must match the root changelog."""

    assert changelog_sync_errors(ROOT) == []


def test_changelog_sync_guard_passes_for_matching_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Matching changelog files pass through the production CLI path."""

    _write_changelogs(tmp_path, "# Changelog\n\n## Unreleased\n")

    assert main(["--repo", str(tmp_path)]) == 0
    assert "PASS: docs/changelog.md matches CHANGELOG.md" in capsys.readouterr().out


def test_changelog_sync_guard_rejects_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stale rendered changelog mirror fails closed with stable paths."""

    _write_changelogs(tmp_path, "# Changelog\n\n## Unreleased\n", "# Changelog\n")

    assert main(["--repo", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "FAIL: changelog mirror drift detected" in output
    assert "docs/changelog.md differs from CHANGELOG.md" in output


def test_changelog_sync_guard_reports_missing_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing changelog files fail closed before content comparison."""

    assert main(["--repo", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "missing CHANGELOG.md" in output
    assert "missing docs/changelog.md" in output


def test_changelog_sync_script_entrypoint() -> None:
    """The script entrypoint runs the same guard as direct imports."""

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_changelog_sync.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: docs/changelog.md matches CHANGELOG.md" in result.stdout
