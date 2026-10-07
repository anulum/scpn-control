# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Private documentation declarations and actual Git CLI contracts.
"""Exercise public inspectors and the unchanged CLI source in real repositories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_docs_internal_private import (
    check_gitignore_rules,
    check_mkdocs_excludes_internal,
    check_no_history_internal_paths,
    check_no_tracked_internal_paths,
    collect_errors,
    main,
)

REPOSITORY = Path(__file__).resolve().parents[1]
GUARD = REPOSITORY / "tools/check_docs_internal_private.py"


def _git(root: Path, *args: str) -> str:
    """Run an actual fixture Git command and reject native command failures.

    Parameters
    ----------
    root : pathlib.Path
        Temporary repository root.
    *args : str
        Literal Git arguments.

    Returns
    -------
    str
        UTF-8 standard output.
    """
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create a small real Git history with byte-identical production CLI source.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned temporary directory.

    Returns
    -------
    pathlib.Path
        Repository root containing accepted declarations and a public commit.
    """
    root = tmp_path / "repository"
    (root / "tools").mkdir(parents=True)
    copy = root / "tools/check_docs_internal_private.py"
    shutil.copyfile(GUARD, copy)
    assert hashlib.sha256(copy.read_bytes()).digest() == hashlib.sha256(GUARD.read_bytes()).digest()
    (root / ".gitignore").write_text("docs/internal/\ndocs/internal/**\n", encoding="utf-8")
    (root / "mkdocs.yml").write_text("site_name: Fixture\nexclude_docs: |\n  internal/**\n", encoding="utf-8")
    _git(root, "init", "--quiet", "--initial-branch=main")
    _git(root, "config", "user.name", "Documentation guard fixture")
    _git(root, "config", "user.email", "guard-fixture@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    (root / ".git/hooks-disabled").mkdir()
    _git(root, "config", "core.hooksPath", str(root / ".git/hooks-disabled"))
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "Create public fixture")
    return root


def _cli(root: Path, *args: str, search_path: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the byte-identical fixture CLI in an actual child interpreter.

    Parameters
    ----------
    root : pathlib.Path
        Real fixture repository containing the copied production script.
    *args : str
        Public CLI arguments.
    search_path : str or None, default None
        Executable-search path, optionally an actual empty directory.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Native exit status and captured UTF-8 output. If a qualification
        coverage configuration is supplied, the child measures the entire script.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    if search_path is not None:
        env["PATH"] = search_path
    command = [sys.executable]
    coverage_config = env.get("SCPN_GUARD_COVERAGE_RC")
    if coverage_config:
        command += ["-m", "coverage", "run", "--rcfile=" + coverage_config, "--parallel-mode"]
    command += [str(root / "tools/check_docs_internal_private.py"), *args]
    return subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("docs/internal/\n", True),
        ("docs/internal/**\n", True),
        ("# policy\n\n!docs/public.md\ndocs/internal/   \n", True),
        ("\n# docs/internal/\n", False),
        (" docs/internal/\n", False),
        ("docs/internal/ # inline comment\n", False),
        ("docs/internal/\n!docs/internal/private.md\n", False),
        ("!docs/internal/\ndocs/internal/\n", False),
        ("docs/internal/**\n!docs/**\n", False),
        ("docs/internal/**\n!other.md\n", False),
        ("!docs/**\ndocs/internal/**\n", True),
        ("!docs/**\n", False),
        ("\\!docs/internal/\ndocs/internal/\n", True),
    ],
)
def test_ignore_declaration_policy(text: str, accepted: bool) -> None:
    """Require an exact root rule after broad negations and reject literal leaks.

    Parameters
    ----------
    text : str
        Gitignore declaration, including significant leading whitespace.
    accepted : bool
        Expected policy disposition.
    """
    assert (check_gitignore_rules(text) == []) is accepted


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("exclude_docs: |\n  internal/**\n", True),
        ('"exclude_docs": "internal/**"\n', True),
        ("exclude_docs: &private |\n  !.assets\n  internal/**\nextra: *private\n", True),
        ("exclude_docs: |\n  internal/**\n  !internal/private.md\n  internal/**\n", True),
        ("exclude_docs: |\n  internal/**\n  # !internal/private.md\n  public/**\n", True),
        ("", False),
        ("[]\n", False),
        ("!custom {}\n", False),
        ("{}\n", False),
        ("extra: {exclude_docs: 'internal/**'}\n", False),
        ("exclude_docs: |\n  public/**\n# internal/**\n", False),
        ("exclude_docs: |\n  public/**\nextra_css: [docs/internal/style.css]\n", False),
        ("exclude_docs: |\n  internal/**\n  !internal/private.md\n", False),
        ("exclude_docs: |\n  internal/**\n  !**/*.md\n", False),
        ("exclude_docs: |\n  internal/**\n  !public.md\n", False),
        ("exclude_docs: |\n  docs/internal/**\n", False),
        ("exclude_docs: |\n  internal/** # comment\n", False),
        ("exclude_docs: |\n  /internal/**\n", False),
        ("exclude_docs: [internal/**]\n", False),
        ("exclude_docs: {}\n", False),
        ("exclude_docs: null\n", False),
        ("exclude_docs: true\n", False),
        ("exclude_docs: !environment internal/**\n", False),
        ("exclude_docs: 'internal/**'\nexclude_docs: 'public/**'\n", False),
        ("site_name: First\nsite_name: Second\nexclude_docs: 'internal/**'\n", False),
        ("1: other\nexclude_docs: 'internal/**'\n", False),
        ("? [exclude_docs]\n: internal/**\n", False),
        ("<<: {exclude_docs: 'internal/**'}\n", False),
        ("exclude_docs: *undefined\n", False),
        ("exclude_docs: [\n", False),
        ("exclude_docs: 'internal/**'\n---\nexclude_docs: 'public/**'\n", False),
    ],
)
def test_mkdocs_actual_scalar_policy(text: str, accepted: bool) -> None:
    """Inspect YAML structure, exact relative patterns, and exclusion order.

    Parameters
    ----------
    text : str
        Actual single- or malformed multi-document YAML.
    accepted : bool
        Expected declaration disposition.
    """
    assert (check_mkdocs_excludes_internal(text) == []) is accepted


def test_unrelated_yaml_tag_is_not_constructed(tmp_path: Path) -> None:
    """Accept ordinary exclusions without executing an unrelated Python tag.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for observing any attempted constructor side effect.
    """
    sentinel = tmp_path / "constructor-ran"
    expression = f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    text = (
        "exclude_docs: 'internal/**'\nextra: !!python/object/apply:builtins.eval\n  - " + json.dumps(expression) + "\n"
    )
    assert check_mkdocs_excludes_internal(text) == []
    assert not sentinel.exists()


def test_exact_path_prefix_sorting_and_history_limit() -> None:
    """Retain exact private names and duplicates, distinguishing similar prefixes."""
    public = ["docs/internal-extra/file.md", "other/docs/internal/file.md", "docs/internality", "README.md"]
    assert check_no_tracked_internal_paths(public) == []
    assert check_no_history_internal_paths(public) == []
    private = ["docs/internal/z.md", "docs/internal", "docs/internal/á\nname.md", "docs/internal/z.md"]
    original = private.copy()
    assert check_no_tracked_internal_paths(public + private)[1:] == ["  - " + path for path in sorted(private)]
    assert check_no_history_internal_paths(private)[1:] == ["  - " + path for path in sorted(private)]
    assert private == original
    many = [f"docs/internal/{index:03}.md" for index in range(51)]
    assert check_no_history_internal_paths(many)[1:] == ["  - " + path for path in many[:50]] + ["  - ..."]


def test_canonical_public_collect_and_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Exercise repository-bound public collection and an explicit CLI argument list.

    Parameters
    ----------
    capsys : pytest.CaptureFixture[str]
        Capture the real public CLI's output.
    """
    assert collect_errors() == []
    assert main(["--skip-history"]) == 0
    assert "history not checked" in capsys.readouterr().out


@pytest.mark.parametrize("skip_history", [False, True])
def test_real_clean_repository_cli(repository: Path, skip_history: bool) -> None:
    """Report only the actual history scope checked in a real committed repository.

    Parameters
    ----------
    repository : pathlib.Path
        Real temporary Git repository.
    skip_history : bool
        Whether to omit the full local-history scan.
    """
    result = _cli(repository, *(["--skip-history"] if skip_history else []))
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert ("history not checked" in result.stdout) is skip_history
    assert ("local history clean" in result.stdout) is not skip_history


@pytest.mark.parametrize("missing", [".gitignore", "mkdocs.yml"])
def test_real_missing_configuration_cli(repository: Path, missing: str) -> None:
    """Return authored findings for a missing root declaration.

    Parameters
    ----------
    repository : pathlib.Path
        Real temporary Git repository.
    missing : str
        Configuration file to remove from this fixture only.
    """
    (repository / missing).unlink()
    result = _cli(repository)
    assert result.returncode == 1
    assert "FAIL:" in result.stdout
    assert missing + " is missing" in result.stdout


def test_real_multiple_declaration_findings_cli(repository: Path) -> None:
    """Collect both failed declarations instead of stopping after the first.

    Parameters
    ----------
    repository : pathlib.Path
        Real temporary Git repository.
    """
    (repository / ".gitignore").write_text("# docs/internal/\n", encoding="utf-8")
    (repository / "mkdocs.yml").write_text("exclude_docs: public/**\n", encoding="utf-8")
    result = _cli(repository)
    assert result.returncode == 1
    assert result.stdout.index(".gitignore must contain") < result.stdout.index("mkdocs.yml exclude_docs must include")


@pytest.mark.parametrize(
    ("relative", "private"),
    [
        ("docs/internal/private.md", True),
        ("docs/internal/á.txt", True),
        pytest.param(
            "docs/internal/new\nline.md",
            True,
            marks=pytest.mark.skipif(
                os.name == "nt",
                reason="a file name cannot contain a line feed on Windows, so the index entry cannot be staged",
            ),
        ),
        ("\ndocs/internal/outside.md", False),
    ],
)
def test_real_index_and_noncurrent_history_paths_cli(repository: Path, relative: str, private: bool) -> None:
    """Detect exact private paths in the index and a noncurrent local reference.

    Parameters
    ----------
    repository : pathlib.Path
        Real temporary Git repository.
    relative : str
        Index path requiring literal NUL-delimited Git path handling.
    private : bool
        Whether the exact path belongs to the private tree.
    """
    blob = _git(repository, "hash-object", "-w", "mkdocs.yml").strip()
    insertion = subprocess.run(
        ["git", "-C", str(repository), "update-index", "-z", "--index-info"],
        input=f"100644 {blob}\t{relative}\0",
        encoding="utf-8",
        check=False,
        capture_output=True,
    )
    if insertion.returncode:
        assert os.name == "nt" and "\n" in relative
        assert "invalid path" in insertion.stderr.lower()
    else:
        indexed = _cli(repository, "--skip-history")
        assert indexed.returncode == (1 if private else 0)
        if private:
            assert relative in indexed.stdout
        assert ("tracked docs/internal" in indexed.stdout) is private
    components = relative.split("/")
    object_id = blob
    object_type = "blob"
    mode = "100644"
    for component in reversed(components):
        tree = subprocess.run(
            ["git", "-C", str(repository), "mktree", "-z"],
            input=f"{mode} {object_type} {object_id}\t{component}\0",
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        object_id = tree.stdout.strip()
        object_type = "tree"
        mode = "040000"
    commit = _git(repository, "commit-tree", object_id, "-p", "main", "-m", "Private stored-path fixture").strip()
    _git(repository, "branch", "private-history", commit)
    _git(repository, "reset", "--quiet", "--mixed", "HEAD")
    history = _cli(repository)
    assert history.returncode == (1 if private else 0)
    if private:
        assert relative in history.stdout
    assert ("found in git history" in history.stdout) is private
    assert "removed from the index" not in history.stdout
    unchecked = _cli(repository, "--skip-history")
    assert unchecked.returncode == 0
    assert "history not checked" in unchecked.stdout


@pytest.mark.parametrize("failure", ["invalid_utf8", "not_a_repository", "git_not_on_path"])
def test_native_read_and_git_errors_cannot_pass(repository: Path, failure: str) -> None:
    """Retain native decode, Git command, and executable lookup failures.

    Parameters
    ----------
    repository : pathlib.Path
        Real temporary Git repository.
    failure : str
        Actual filesystem or executable-search failure to induce.
    """
    search_path = None
    if failure == "invalid_utf8":
        (repository / ".gitignore").write_bytes(b"\xff")
        expected = "UnicodeDecodeError"
    elif failure == "not_a_repository":
        (repository / ".git").rename(repository / "retained-git-metadata")
        expected = "CalledProcessError"
    else:
        empty = repository / "empty-search-path"
        empty.mkdir()
        search_path = str(empty)
        expected = "FileNotFoundError"
    result = _cli(repository, search_path=search_path)
    assert result.returncode != 0
    assert "PASS:" not in result.stdout
    assert expected in result.stderr


@pytest.mark.parametrize(("argument", "status"), [("--help", 0), ("--unknown-option", 2)])
def test_real_argparse_cli(repository: Path, argument: str, status: int) -> None:
    """Preserve native argparse help and invalid-option exit semantics.

    Parameters
    ----------
    repository : pathlib.Path
        Real temporary Git repository.
    argument : str
        Public CLI option to exercise.
    status : int
        Expected argparse process exit code.
    """
    result = _cli(repository, argument)
    assert result.returncode == status
    assert "PASS:" not in result.stdout
