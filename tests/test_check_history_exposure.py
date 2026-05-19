#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Git History Exposure Guard Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.check_history_exposure import collect_exposures, main


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "History Test")
    _git(tmp_path, "config", "user.email", "history@example.invalid")
    _write(tmp_path / "src" / "public.py", "value = 1\n")
    _git(tmp_path, "add", "src/public.py")
    _git(tmp_path, "commit", "-m", "initial")
    return tmp_path


def test_current_tree_exposure_is_reported(repo: Path) -> None:
    _write(repo / ".coordination" / "sessions" / "scpn-control" / "SESSION_LOG.md", "internal\n")
    _git(repo, "add", ".coordination/sessions/scpn-control/SESSION_LOG.md")
    _git(repo, "commit", "-m", "track internal log")

    exposures = collect_exposures(repo, include_history=False)

    assert [exposure.path for exposure in exposures] == [
        ".coordination/sessions/scpn-control/SESSION_LOG.md",
    ]
    assert len(exposures[0].first_commit) == 40


def test_history_exposure_is_reported_after_current_tree_cleanup(repo: Path) -> None:
    _write(repo / "docs" / "internal" / "audit.md", "private\n")
    _git(repo, "add", "docs/internal/audit.md")
    _git(repo, "commit", "-m", "track internal audit")
    _git(repo, "rm", "docs/internal/audit.md")
    _git(repo, "commit", "-m", "remove internal audit")

    assert collect_exposures(repo, include_history=False) == []

    exposures = collect_exposures(repo, include_history=True)

    assert [exposure.path for exposure in exposures] == ["docs/internal/audit.md"]
    assert len(exposures[0].first_commit) == 40


def test_allowed_secret_sharing_path_is_not_reported(repo: Path) -> None:
    _write(repo / "src" / "scpn_control" / "secret_sharing.py", "value = 1\n")
    _git(repo, "add", "src/scpn_control/secret_sharing.py")
    _git(repo, "commit", "-m", "add allowed secret sharing module")

    assert collect_exposures(repo, include_history=True) == []


def test_main_returns_failure_for_exposure(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(repo / ".env", "TOKEN=placeholder\n")
    _git(repo, "add", "-f", ".env")
    _git(repo, "commit", "-m", "track env")

    result = main(["--repo", str(repo), "--current-only"])

    captured = capsys.readouterr()
    assert result == 1
    assert "FAIL: blocked internal or sensitive paths found in current tree" in captured.out
    assert "  - .env" in captured.out
