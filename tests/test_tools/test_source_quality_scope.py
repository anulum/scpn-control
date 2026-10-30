# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real Git source enrollment and refusal contracts.
"""Exercise selection against actual Git objects without manufacturing commits."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.source_quality_scope import _paths, main, select_changed_files

ROOT = Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> bytes:
    """Execute native Git in the owned fixture and require successful setup."""
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[Path, str]:
    """Reuse existing project objects read-only in a minimal real Git fixture."""
    head = _git(ROOT, "rev-parse", "HEAD").decode().strip()
    objects = Path(_git(ROOT, "rev-parse", "--git-path", "objects").decode().strip())
    if not objects.is_absolute():
        objects = ROOT / objects
    _git(tmp_path, "init", "--quiet", "--initial-branch=main")
    # Bytes, not text: Windows text mode would end the path with CR and Git
    # would then find no alternate object store.
    (tmp_path / ".git/objects/info/alternates").write_bytes(objects.resolve().as_posix().encode("utf-8") + b"\n")
    _git(tmp_path, "update-ref", "refs/heads/main", head)
    _git(tmp_path, "read-tree", "HEAD")
    policy = tmp_path / "tools/source_header_policy.toml"
    policy.parent.mkdir()
    policy.write_bytes(_git(ROOT, "show", f"{head}:tools/source_header_policy.toml"))
    return tmp_path, head


def _write(repo: Path, name: str, content: bytes = b"value = 1\n") -> Path:
    """Create one physical source without changing production paths or index."""
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _cli(repo: Path, base: str) -> subprocess.CompletedProcess[str]:
    """Run the physical production CLI, optionally measuring its entire module."""
    command = [sys.executable, "-B"]
    coverage_rc = os.environ.get("SCPN_SOURCE_SCOPE_COVERAGE_RC")
    if coverage_rc:
        command += ["-m", "coverage", "run", "--parallel-mode", f"--rcfile={coverage_rc}"]
    return subprocess.run(
        [*command, str(ROOT / "tools/source_quality_scope.py"), "--repo", str(repo), "--base", base],
        capture_output=True,
        text=True,
        check=False,
    )


def test_git_enrolls_staged_unstaged_new_roots_and_unusual_names(git_repo: tuple[Path, str]) -> None:
    """Root selectors cannot omit new source, and NUL framing preserves names."""
    repo, base = git_repo
    staged = _write(repo, "new-root/staged.py")
    _git(repo, "add", "--", str(staged.relative_to(repo)))
    staged.write_bytes(b"value = 2\n")
    _write(repo, "new-root/space and\nž.py")
    _write(repo, "new-root/UPPER.PY")
    before = _git(repo, "ls-files", "--stage", "-z")
    selected = select_changed_files(repo, base)
    assert [item.path for item in selected] == [
        "new-root/UPPER.PY",
        "new-root/space and\nž.py",
        "new-root/staged.py",
    ]
    assert all(item.profile == "python" for item in selected)
    assert selected[-1].sha256 == hashlib.sha256(staged.read_bytes()).hexdigest()
    assert _git(repo, "ls-files", "--stage", "-z") == before
    result = _cli(repo, base)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["paths"][1]["path"] == "new-root/space and\nž.py"


@pytest.mark.parametrize(
    "name,profile",
    [
        ("new/main.rs", "rust"),
        ("new/main.ts", "typescript"),
        ("new/main.tsx", "typescript"),
        ("new/main.js", "javascript"),
        ("new/main.mjs", "javascript"),
        ("new/Main.lean", "lean"),
        ("new/main.sh", "shell"),
        (".githooks/pre-push", "shell"),
        ("new/hooks/pre-push", "shell"),
        ("new/main.cpp", "cpp"),
        ("new/main.h", "c-header"),
        ("new/main.hpp", "cpp-header"),
        ("new/main.html", "html"),
        ("new/main.ipynb", "notebook"),
        ("new/data.json", "carrier"),
        ("new/config.toml", "carrier"),
        ("new/notes.md", "carrier"),
    ],
)
def test_every_native_family_and_supporting_carrier_remains_visible(
    git_repo: tuple[Path, str], name: str, profile: str
) -> None:
    """Header exemptions do not remove notebooks, data or prose from selection."""
    repo, base = git_repo
    _write(repo, name)
    selected = select_changed_files(repo, base)
    assert len(selected) == 1 and selected[0].path == name and selected[0].profile == profile


def test_real_git_rename_enrolls_destination_and_deletion_is_not_current_source(git_repo: tuple[Path, str]) -> None:
    """The complete moved owner is selected rather than only its changed lines."""
    repo, base = git_repo
    # Choose a real tracked Python blob so Git can recognise an exact rename.
    original = next(
        name
        for name in _git(repo, "ls-files").decode().splitlines()
        if name.startswith("tools/") and name.endswith(".py")
    )
    path = _write(repo, original, _git(repo, "show", f"HEAD:{original}"))
    destination = path.with_name("renamed_owner.py")
    path.rename(destination)
    _git(repo, "add", "--", original, str(destination.relative_to(repo)))
    selected = select_changed_files(repo, base)
    assert [item.path for item in selected] == ["tools/renamed_owner.py"]
    destination.unlink()
    _git(repo, "add", "--", "tools/renamed_owner.py")
    assert select_changed_files(repo, base) == ()


def test_changes_already_in_head_are_selected_from_original_commit(git_repo: tuple[Path, str]) -> None:
    """A clean HEAD comparison cannot hide committed work after the anchor."""
    repo, head = git_repo
    parent = _git(repo, "rev-parse", "HEAD^").decode().strip()
    changed = (
        _git(repo, "diff", "--name-only", "--diff-filter=ACMR", "-z", parent, head).decode().strip("\0").split("\0")
    )
    for name in changed:
        _write(repo, name, _git(repo, "show", f"HEAD:{name}"))
    selected = select_changed_files(repo, parent)
    assert {item.path for item in selected} == set(changed)


@pytest.mark.parametrize("name", ["new/main.go", "new/unknown.extension", "new/without_suffix"])
def test_unknown_formats_refuse_instead_of_passing_empty_selection(git_repo: tuple[Path, str], name: str) -> None:
    """No unqualified language or extensionless script can disappear silently."""
    repo, base = git_repo
    _write(repo, name)
    with pytest.raises(RuntimeError, match="explicit format/profile"):
        select_changed_files(repo, base)
    result = _cli(repo, base)
    assert result.returncode == 2 and "refused" in result.stderr and not result.stdout


@pytest.mark.parametrize("base", ["HEAD", "", "a" * 39, "A" * 40, "0" * 40])
def test_missing_or_untrusted_history_never_falls_back(git_repo: tuple[Path, str], base: str) -> None:
    """An unavailable anchor cannot substitute HEAD or accept the current tree."""
    repo, _head = git_repo
    with pytest.raises(RuntimeError):
        select_changed_files(repo, base)


def test_real_nonancestor_and_blob_anchors_refuse(git_repo: tuple[Path, str]) -> None:
    """Existing objects must be ancestral commits, not arbitrary valid hashes."""
    repo, head = git_repo
    blob = _git(repo, "rev-parse", "HEAD:pyproject.toml").decode().strip()
    with pytest.raises(RuntimeError, match="commit object"):
        select_changed_files(repo, blob)
    parent = _git(repo, "rev-parse", "HEAD^").decode().strip()
    _git(repo, "update-ref", "refs/heads/main", parent)
    with pytest.raises(RuntimeError, match="merge-base"):
        select_changed_files(repo, head)


def test_real_unmerged_index_refuses(git_repo: tuple[Path, str]) -> None:
    """Actual conflict stages are not filtered out as ordinary deletions."""
    repo, base = git_repo
    blob = _git(repo, "rev-parse", "HEAD:pyproject.toml").decode().strip()
    subprocess.run(
        ["git", "-C", str(repo), "update-index", "--index-info"],
        input=f"100644 {blob} 1\tconflict.py\n100644 {blob} 2\tconflict.py\n".encode(),
        check=True,
        capture_output=True,
    )
    with pytest.raises(RuntimeError, match="Unmerged"):
        select_changed_files(repo, base)


def test_symlink_and_noncanonical_paths_refuse(git_repo: tuple[Path, str]) -> None:
    """Source aliases cannot claim the bytes of an unmeasured external owner."""
    repo, base = git_repo
    link = repo / "linked.py"
    link.symlink_to(ROOT / "tools/preflight.py")
    with pytest.raises(RuntimeError, match="symlinked"):
        select_changed_files(repo, base)
    link.unlink()
    _write(repo, "back\\slash.py")
    with pytest.raises(RuntimeError, match="canonical relative"):
        select_changed_files(repo, base)


def test_non_utf8_source_name_refuses(git_repo: tuple[Path, str]) -> None:
    """A source whose native name is not UTF-8 cannot be measured."""
    repo, base = git_repo
    native_name = os.fsencode(repo) + b"/invalid_\xff.py"
    try:
        descriptor = os.open(native_name, os.O_WRONLY | os.O_CREAT, 0o600)
    except (OSError, UnicodeError) as error:
        pytest.skip(f"this file system refuses non-UTF-8 names, so the case cannot arise: {error}")
    os.close(descriptor)
    with pytest.raises(RuntimeError, match="UTF-8"):
        select_changed_files(repo, base)


def test_missing_policy_unreadable_source_and_invalid_policy_refuse(git_repo: tuple[Path, str]) -> None:
    """Unavailable source bytes and declarations cannot become empty success."""
    repo, base = git_repo
    source = _write(repo, "unreadable.py")
    source.chmod(0)
    try:
        with pytest.raises(RuntimeError, match="Cannot read"):
            select_changed_files(repo, base)
    finally:
        source.chmod(0o600)
    source.unlink()
    policy = repo / "tools/source_header_policy.toml"
    policy.unlink()
    with pytest.raises(RuntimeError, match="Cannot read"):
        select_changed_files(repo, base)
    policy.write_text("invalid = [", encoding="utf-8")
    with pytest.raises(RuntimeError, match="explicitly reviewed"):
        select_changed_files(repo, base)


def test_new_policy_exemption_cannot_hide_an_unqualified_language(git_repo: tuple[Path, str]) -> None:
    """The original format policy prevents candidate code from exempting itself."""
    repo, base = git_repo
    policy = repo / "tools/source_header_policy.toml"
    policy.write_text(policy.read_text().replace('".py",', '".py", ".go",'), encoding="utf-8")
    _write(repo, "new/main.go")
    with pytest.raises(RuntimeError, match="explicitly reviewed"):
        select_changed_files(repo, base)


def test_real_gitlink_is_not_a_readable_source_file(git_repo: tuple[Path, str]) -> None:
    """A new submodule needs its own qualification rather than an empty directory."""
    repo, base = git_repo
    (repo / "nested").mkdir()
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{base},nested")
    with pytest.raises(RuntimeError, match="regular file"):
        select_changed_files(repo, base)


def test_nul_protocol_rejects_truncation_before_path_interpretation() -> None:
    """A truncated producer record cannot masquerade as a valid source name."""
    with pytest.raises(RuntimeError, match="NUL terminated"):
        _paths(b"new/source.py")


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_invalid_timeouts_refuse(git_repo: tuple[Path, str], timeout: float) -> None:
    """A missing bound or Boolean cannot silently disable the Git timeout."""
    repo, base = git_repo
    with pytest.raises(RuntimeError, match="finite and positive"):
        select_changed_files(repo, base, timeout_s=timeout)


def test_actual_git_timeout_and_absent_executable_refuse(
    git_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real process-launch timeout and missing-tool errors propagate as refusal."""
    repo, base = git_repo
    with pytest.raises(RuntimeError, match="successful Git"):
        select_changed_files(repo, base, timeout_s=1e-12)
    monkeypatch.setenv("PATH", str(repo / "absent-bin"))
    with pytest.raises(RuntimeError, match="successful Git"):
        select_changed_files(repo, base)


def test_subdirectory_and_nonrepository_refuse(git_repo: tuple[Path, str], tmp_path: Path) -> None:
    """The selected path namespace must be the exact real working-tree root."""
    repo, base = git_repo
    with pytest.raises(RuntimeError, match="working-tree root"):
        select_changed_files(repo / "tools", base)
    with pytest.raises(RuntimeError, match="successful Git"):
        select_changed_files(tmp_path / "absent-repository", base)


def test_cli_api_refusal_and_argument_parser(capsys: pytest.CaptureFixture[str], git_repo: tuple[Path, str]) -> None:
    """Public CLI status distinguishes refused measurement from a valid empty set."""
    repo, base = git_repo
    assert main(["--repo", str(repo), "--base", base]) == 0
    assert json.loads(capsys.readouterr().out)["paths"] == []
    assert main(["--repo", str(repo), "--base", "HEAD"]) == 2
    assert "refused" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
