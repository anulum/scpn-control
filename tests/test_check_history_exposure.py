#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Git History Exposure Guard Tests.

"""Exercise actual Git index/history names, literal lookup and authored CLI failures."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_history_exposure import (
    collect_current_paths,
    collect_exposures,
    collect_history_paths,
    find_first_commit,
    main,
)

ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> None:
    """Run real Git operations only inside the isolated test worktree."""
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def _write(path: Path, content: str) -> None:
    """Persist explicit fixture content without real credentials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Create a genuine committed Git worktree with a public fixture file."""
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "History Test")
    _git(tmp_path, "config", "user.email", "history@example.invalid")
    _write(tmp_path / "src" / "public.py", "value = 1\n")
    _git(tmp_path, "add", "src/public.py")
    _git(tmp_path, "commit", "-m", "initial")
    return tmp_path


def test_current_tree_exposure_is_reported(repo: Path) -> None:
    """Actual committed internal names are blocked in the index-only inspection."""
    _write(repo / ".coordination" / "sessions" / "scpn-control" / "SESSION_LOG.md", "internal\n")
    _git(repo, "add", ".coordination/sessions/scpn-control/SESSION_LOG.md")
    _git(repo, "commit", "-m", "track internal log")

    exposures = collect_exposures(repo, include_history=False)

    assert [exposure.path for exposure in exposures] == [
        ".coordination/sessions/scpn-control/SESSION_LOG.md",
    ]
    assert len(exposures[0].first_commit) == 40


def test_history_exposure_is_reported_after_current_tree_cleanup(repo: Path) -> None:
    """Deleting a committed internal name does not remove its all-ref history finding."""
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
    """The unchanged explicit source-code exception remains allowed."""
    _write(repo / "src" / "scpn_control" / "secret_sharing.py", "value = 1\n")
    _git(repo, "add", "src/scpn_control/secret_sharing.py")
    _git(repo, "commit", "-m", "add allowed secret sharing module")

    assert collect_exposures(repo, include_history=True) == []


def test_main_returns_failure_for_exposure(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The real public command reports tracked environment names with failure1."""
    _write(repo / ".env", "TOKEN=placeholder\n")
    _git(repo, "add", "-f", ".env")
    _git(repo, "commit", "-m", "track env")

    result = main(["--repo", str(repo), "--current-only"])

    captured = capsys.readouterr()
    assert result == 1
    assert "FAIL: blocked internal or sensitive paths found in current tree" in captured.out
    assert "  - .env" in captured.out


def _git_bytes(repo: Path, *args: str, data: bytes | None = None) -> bytes:
    """Capture real Git plumbing output while preserving its binary filename framing."""
    return subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True, check=True).stdout


def _history_path(repo: Path, name: bytes, *, parent: str | None = None) -> str:
    """Store a genuine history tree without creating platform-restricted working filenames."""
    blob = _git_bytes(repo, "hash-object", "-w", "--stdin", data=b"explicit history fixture\n").strip()
    tree = _git_bytes(repo, "mktree", "-z", data=b"100644 blob " + blob + b"\t" + name + b"\0").strip()
    previous = parent or _git_bytes(repo, "rev-parse", "HEAD").decode().strip()
    commit = (
        _git_bytes(repo, "commit-tree", tree.decode(), "-p", previous, data=b"Real unusual-name history fixture\n")
        .decode()
        .strip()
    )
    _git(repo, "update-ref", "refs/heads/unusual-name-fixture", commit)
    return commit


def _source_cli(repo: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the cold standard-library command against an explicit real Git repository."""
    return subprocess.run(
        [sys.executable, "-S", str(ROOT / "tools/check_history_exposure.py"), "--repo", str(repo), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.parametrize(
    "name", [".env.fixture\ncontinued", ".env.fixture\ttab", '.env.fixture"quote', ".env.é_fixture"]
)
def test_all_ref_history_preserves_quoted_git_names(repo: Path, name: str) -> None:
    """Actual Git objects with unusual names remain literal findings rather than quoted-text false passes."""
    commit = _history_path(repo, name.encode())
    assert name in collect_history_paths(repo)
    findings = collect_exposures(repo, include_history=True)
    assert [(entry.path, entry.first_commit) for entry in findings] == [(name, commit)]
    assert collect_exposures(repo, include_history=False) == []
    result = _source_cli(repo, "--json")
    assert result.returncode == 1 and result.stderr == ""
    assert json.loads(result.stdout) == [{"path": name, "first_commit": commit}]


def test_index_names_preserve_unicode_and_leading_whitespace(repo: Path) -> None:
    """The index includes staged additions and retains whitespace that changes the pattern policy."""
    names = [".env.é_fixture", " .env.public-fixture"]
    for name in names:
        _write(repo / name, "explicit index fixture\n")
        _git(repo, "add", "--", name)
    assert set(names) <= collect_current_paths(repo)
    findings = collect_exposures(repo, include_history=False)
    assert [(entry.path, entry.first_commit) for entry in findings] == [(names[0], "")]
    result = _source_cli(repo, "--current-only", "--json")
    assert result.returncode == 1 and json.loads(result.stdout) == [{"path": names[0], "first_commit": ""}]
    text = _source_cli(repo, "--current-only")
    assert text.returncode == 1 and names[0] in text.stdout and "first_seen=" not in text.stdout


def test_first_commit_lookup_treats_pathspec_metacharacters_literally(repo: Path) -> None:
    """An older wildcard-compatible name cannot supply the literal bracketed path's commit ID."""
    first = _history_path(repo, b".env.fixturea")
    second = _history_path(repo, b".env.fixture[ab]", parent=first)
    assert find_first_commit(repo, ".env.fixture[ab]") == second
    assert find_first_commit(repo, ".env.fixturea") == first


def test_history_collector_preserves_non_utf8_git_bytes(repo: Path) -> None:
    """Surrogateescape preserves actual object-tree names that cannot decode as strict UTF-8."""
    name = b".env.fixture-\xff"
    _history_path(repo, name)
    assert name.decode("utf-8", errors="surrogateescape") in collect_history_paths(repo)


@pytest.mark.parametrize("current", [False, True])
def test_clean_public_command_has_explicit_text_and_json_success(
    repo: Path, current: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    """An actual clean index/history returns0 with the documented scope and complete empty JSON."""
    args = ["--repo", str(repo), *(["--current-only"] if current else [])]
    assert main(args) == 0
    assert "OK: no blocked" in capsys.readouterr().out
    assert main([*args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_text_history_findings_escape_control_characters_on_one_line(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A literal newline in a Git path cannot insert an additional human report line."""
    name = ".env.fixture\ncontinued"
    commit = _history_path(repo, name.encode())
    assert main(["--repo", str(repo)]) == 1
    result = capsys.readouterr()
    assert result.err == "" and len(result.out.splitlines()) == 2
    assert json.dumps(name) in result.out and commit in result.out


@pytest.mark.parametrize("failure", ["missing", "nongit", "git_absent"])
def test_cold_command_operational_refusal_never_emits_success_json(repo: Path, tmp_path: Path, failure: str) -> None:
    """Missing worktrees and a genuinely unavailable Git executable refuse with authored2 and no partial JSON."""
    selected = repo
    env = None
    if failure == "missing":
        selected = repo / "absent"
    elif failure == "nongit":
        selected = tmp_path.parent / (tmp_path.name + "-nongit")
        selected.mkdir()
    else:
        empty_bin = tmp_path / "empty-bin"
        empty_bin.mkdir()
        env = dict(os.environ, PATH=str(empty_bin))
    result = _source_cli(selected, "--json", env=env)
    assert result.returncode == 2 and result.stdout == ""
    assert result.stderr == "could not inspect Git paths for history exposure\n"


def test_public_collectors_propagate_git_errors_and_cli_refuses_invalid_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Typed collectors preserve Git failures while the command authors path refusals before output."""
    with pytest.raises(subprocess.CalledProcessError):
        collect_current_paths(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        collect_history_paths(tmp_path)
    assert main(["--repo", str(tmp_path) + "\0", "--json"]) == 2
    result = capsys.readouterr()
    assert result.out == "" and result.err == "could not inspect Git paths for history exposure\n"


def test_subdirectory_selection_refuses_prefix_limited_index_inspection(repo: Path) -> None:
    """A child Git context cannot silently narrow the root-level path policy to its own index prefix."""
    with pytest.raises(ValueError, match="Git worktree root"):
        collect_exposures(repo / "src", include_history=False)
    result = _source_cli(repo / "src", "--current-only", "--json")
    assert result.returncode == 2 and result.stdout == ""
    assert result.stderr == "could not inspect Git paths for history exposure\n"


@pytest.mark.parametrize(("option", "exit_code"), [("--help", 0), ("--unsupported", 2)])
def test_public_history_command_preserves_argparse_exits(option: str, exit_code: int) -> None:
    """Explicit help and invalid options retain parser0/2 without a Git inspection."""
    with pytest.raises(SystemExit) as error:
        main([option])
    assert error.value.code == exit_code
