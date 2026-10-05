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


@pytest.mark.parametrize("target", ["CHANGELOG.md", "docs/changelog.md"])
def test_script_refuses_unreadable_file_carrier(tmp_path: Path, target: str) -> None:
    """A directory at either actual file location is a handled CLI read refusal."""
    _write_changelogs(tmp_path, (ROOT / "CHANGELOG.md").read_text())
    carrier = tmp_path / target
    carrier.unlink()
    carrier.mkdir()
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_changelog_sync.py"), "--repo", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == "FAIL: could not read changelog files\n"
    assert result.stderr == ""


@pytest.mark.parametrize("missing", ["CHANGELOG.md", "docs/changelog.md"])
def test_api_names_only_the_missing_carrier(tmp_path: Path, missing: str) -> None:
    """One missing actual file is reported without inventing a second error."""
    _write_changelogs(tmp_path, (ROOT / "CHANGELOG.md").read_text())
    (tmp_path / missing).unlink()
    assert changelog_sync_errors(tmp_path) == [f"missing {missing}"]


def test_api_preserves_byte_and_read_error_contract(tmp_path: Path) -> None:
    """Equal binary content passes; newline changes fail and OS errors propagate."""
    _write_changelogs(tmp_path, (ROOT / "CHANGELOG.md").read_text())
    left, right = tmp_path / "CHANGELOG.md", tmp_path / "docs/changelog.md"
    original = left.read_bytes()
    right.write_bytes(original.replace(b"\n", b"\r\n"))
    assert changelog_sync_errors(tmp_path)
    for path in (left, right):
        path.write_bytes(original + b"\xff")
    assert changelog_sync_errors(tmp_path) == []
    right.unlink()
    right.mkdir()
    with pytest.raises(IsADirectoryError):
        changelog_sync_errors(tmp_path)


def test_script_root_selection_and_argv_ownership(tmp_path: Path) -> None:
    """The default script root works from another cwd; direct argv stays unchanged."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_changelog_sync.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == "PASS: docs/changelog.md matches CHANGELOG.md\n"
    _write_changelogs(tmp_path, (ROOT / "CHANGELOG.md").read_text())
    argv = ["--repo", str(tmp_path)]
    assert main(argv) == 0
    assert argv == ["--repo", str(tmp_path)]


def test_native_example_executes_actual_changelog_comparison() -> None:
    """The owning native example compares canonical bytes through its public API."""
    import doctest

    from tools import check_changelog_sync

    result = doctest.testmod(check_changelog_sync)
    assert result.failed == 0 and result.attempted == 1
