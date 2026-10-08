# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Local publication workflow tests.

"""Tests for exact-path release building, checking, and upload dispatch."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import publish


def test_distribution_paths_returns_only_exact_release_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shell metacharacters and unrelated files cannot enter publication argv."""
    monkeypatch.setattr(publish, "DIST", tmp_path)
    wheel = tmp_path / "package name.whl"
    sdist = tmp_path / "package-1.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    (tmp_path / "SHA256SUMS.txt").write_text("metadata", encoding="utf-8")

    assert publish.distribution_paths() == [str(wheel), str(sdist)]


def test_distribution_paths_rejects_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication cannot proceed without an exact artifact set."""
    monkeypatch.setattr(publish, "DIST", tmp_path)

    with pytest.raises(SystemExit, match="No distribution artifacts"):
        publish.distribution_paths()


def test_check_and_upload_pass_expanded_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Twine receives concrete argv entries rather than an inert glob string."""
    monkeypatch.setattr(publish, "DIST", tmp_path)
    artifact = tmp_path / "package.whl"
    artifact.write_bytes(b"wheel")
    calls: list[list[str]] = []

    def record(command: list[str], check: bool = True) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(publish, "_run", record)
    publish.check()
    publish.upload("testpypi")
    publish.upload("pypi")

    assert calls[0] == [sys.executable, "-m", "twine", "check", str(artifact)]
    assert calls[1] == [
        sys.executable,
        "-m",
        "twine",
        "upload",
        "--repository",
        "testpypi",
        str(artifact),
    ]
    assert calls[2] == [sys.executable, "-m", "twine", "upload", str(artifact)]


def test_build_dispatches_reproducible_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local workflow uses the same validated builder as hosted CI."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        publish,
        "_run",
        lambda command, check=True: calls.append(command),
    )

    publish.build()

    assert calls == [[sys.executable, "tools/build_release_artifacts.py"]]


def test_clean_dist_is_bounded_to_configured_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup replaces only the exact configured distribution directory."""
    distribution = tmp_path / "dist"
    distribution.mkdir()
    (distribution / "old.whl").write_bytes(b"old")
    sibling = tmp_path / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(publish, "DIST", distribution)

    publish.clean_dist()

    assert distribution.is_dir()
    assert not list(distribution.iterdir())
    assert sibling.read_text(encoding="utf-8") == "keep"


def test_clean_dist_creates_missing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup also initialises an absent exact distribution directory."""
    distribution = tmp_path / "dist"
    monkeypatch.setattr(publish, "DIST", distribution)

    publish.clean_dist()

    assert distribution.is_dir()


def test_run_wrapper_and_test_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Subprocess dispatch remains argv-based and the test gate is exact."""
    calls: list[list[str]] = []

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[list[str]]:
        calls.append(command)
        assert kwargs == {"cwd": publish.ROOT, "check": False}
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", execute)
    assert publish._run(["safe", "argument with spaces"], check=False).returncode == 0
    assert "safe argument with spaces" in capsys.readouterr().out

    monkeypatch.setattr(
        publish,
        "_run",
        lambda command, check=True: calls.append(command),
    )
    publish.run_tests()
    assert calls[-1][0:4] == [sys.executable, "-m", "pytest", "-p"]
    assert calls[-1][-4:] == ["tests/", "-x", "-q", "--tb=short"]


def test_read_and_bump_version_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic version bumps update only the canonical metadata field."""
    metadata = tmp_path / "pyproject.toml"
    monkeypatch.setattr(publish, "PYPROJECT", metadata)

    for part, expected in (("major", "2.0.0"), ("minor", "1.3.0"), ("patch", "1.2.4")):
        metadata.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
        assert publish.bump_version(part) == expected
        assert metadata.read_text(encoding="utf-8") == f'[project]\nversion = "{expected}"\n'

    metadata.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="Invalid bump part"):
        publish.bump_version("invalid")


def test_version_metadata_failures_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing and non-semantic package versions fail before publication."""
    metadata = tmp_path / "pyproject.toml"
    monkeypatch.setattr(publish, "PYPROJECT", metadata)
    metadata.write_text("[project]\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Cannot parse version"):
        publish.read_version()

    metadata.write_text('[project]\nversion = "1.2"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="not semver"):
        publish.bump_version("patch")


def _stub_main_operations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    calls: list[str],
) -> None:
    distribution = tmp_path / "dist"
    distribution.mkdir()
    artifact = distribution / "package.whl"
    artifact.write_bytes(b"wheel")
    monkeypatch.setattr(publish, "DIST", distribution)
    monkeypatch.setattr(publish, "read_version", lambda: "1.2.3")

    def record_bump(part: str) -> str:
        """Retain the original routing declaration without an untyped return expression."""
        calls.append(f"bump:{part}")
        return "2.0.0"

    monkeypatch.setattr(publish, "bump_version", record_bump)
    monkeypatch.setattr(publish, "run_tests", lambda: calls.append("tests"))
    monkeypatch.setattr(publish, "clean_dist", lambda: calls.append("clean"))
    monkeypatch.setattr(publish, "build", lambda: calls.append("build"))
    monkeypatch.setattr(publish, "check", lambda: calls.append("check"))
    monkeypatch.setattr(publish, "upload", lambda target: calls.append(f"upload:{target}"))


def test_main_dry_run_and_upload_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dry-run, TestPyPI, and confirmed PyPI routes exercise their exact gates."""
    calls: list[str] = []
    _stub_main_operations(monkeypatch, tmp_path, calls)

    monkeypatch.setattr(sys, "argv", ["publish.py", "--dry-run", "--skip-tests"])
    publish.main()
    assert calls == ["clean", "build", "check"]
    assert "Dry run" in capsys.readouterr().out

    calls.clear()
    monkeypatch.setattr(sys, "argv", ["publish.py", "--skip-tests"])
    publish.main()
    assert calls == ["clean", "build", "check", "upload:testpypi"]
    assert (
        "  Install: pip install -i https://test.pypi.org/simple/ scpn-control==1.2.3"
        in capsys.readouterr().out.splitlines()
    )

    calls.clear()
    monkeypatch.setattr(sys, "argv", ["publish.py", "--target", "pypi", "--confirm", "--bump", "major"])
    publish.main()
    assert calls == ["bump:major", "tests", "clean", "build", "check", "upload:pypi"]
    assert "pip install scpn-control==2.0.0" in capsys.readouterr().out


def test_main_rejects_unconfirmed_production_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A production PyPI upload always requires explicit confirmation."""
    monkeypatch.setattr(sys, "argv", ["publish.py", "--target", "pypi"])

    with pytest.raises(SystemExit, match="requires --confirm"):
        publish.main()


def test_native_run_observes_actual_child_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real subprocess wrapper retains native status and checked failure behavior."""
    monkeypatch.setattr(publish, "ROOT", tmp_path)
    command = [sys.executable, "-c", "import sys; sys.exit(23)"]
    assert publish._run(command, check=False).returncode == 23
    with pytest.raises(subprocess.CalledProcessError) as failure:
        publish._run(command)
    assert failure.value.returncode == 23


@pytest.mark.parametrize("kind", ["directory", "empty", "symlink"])
def test_distribution_selection_refuses_nonregular_or_empty_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A filename suffix alone cannot promote a directory/link/empty carrier to Twine."""
    monkeypatch.setattr(publish, "DIST", tmp_path)
    candidate = tmp_path / "release.whl"
    if kind == "directory":
        candidate.mkdir()
    elif kind == "empty":
        candidate.write_bytes(b"")
    else:
        target = tmp_path / "source.bin"
        target.write_bytes(b"owned content")
        candidate.symlink_to(target)
    with pytest.raises(SystemExit, match="nonempty regular files"):
        publish.distribution_paths()
    assert candidate.exists()


def test_distribution_selection_refuses_missing_and_linked_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a concrete configured distribution directory supplies artifact paths."""
    missing = tmp_path / "missing"
    monkeypatch.setattr(publish, "DIST", missing)
    with pytest.raises(SystemExit, match="real directory"):
        publish.distribution_paths()
    link = tmp_path / "linked"
    link.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(publish, "DIST", link)
    with pytest.raises(SystemExit, match="real directory"):
        publish.distribution_paths()


@pytest.mark.parametrize("kind", ["root", "source", "symlink", "file"])
def test_cleanup_refuses_selected_source_namespaces_and_invalid_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Actual source carriers and linked/non-directory roots survive cleanup refusal."""
    root = tmp_path / "repository"
    root.mkdir()
    source = root / "src"
    source.mkdir()
    retained = source / "owned.py"
    retained.write_text("retained source", encoding="utf-8")
    selected = root if kind == "root" else source
    if kind == "symlink":
        selected = tmp_path / "linked"
        selected.symlink_to(source, target_is_directory=True)
    elif kind == "file":
        selected = tmp_path / "file"
        selected.write_bytes(b"retained file")
    monkeypatch.setattr(publish, "ROOT", root)
    monkeypatch.setattr(publish, "PYPROJECT", root / "pyproject.toml")
    monkeypatch.setattr(publish, "DIST", selected)
    with pytest.raises(SystemExit, match="Distribution cleanup"):
        publish.clean_dist()
    assert retained.read_text(encoding="utf-8") == "retained source"


@pytest.mark.parametrize("aliased", [False, True], ids=["nested-package", "aliased-parent"])
def test_cleanup_preserves_full_source_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aliased: bool
) -> None:
    """Configured cleanup refuses a real package and its resolved parent alias."""
    repository = Path(publish.__file__).parents[1]
    root = tmp_path / "repository"
    source = root / "src" / "scpn_control"
    shutil.copytree(repository / "src" / "scpn_control", source, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(repository / "pyproject.toml", root / "pyproject.toml")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    selected = source
    if aliased:
        alias = tmp_path / "source-link"
        alias.symlink_to(root / "src", target_is_directory=True)
        selected = alias / "scpn_control"
    monkeypatch.setattr(publish, "ROOT", root)
    monkeypatch.setattr(publish, "PYPROJECT", root / "pyproject.toml")
    monkeypatch.setattr(publish, "DIST", selected)
    with pytest.raises(SystemExit, match="must not remove repository sources"):
        publish.clean_dist()
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_direct_upload_refuses_unknown_index_without_reading_artifacts() -> None:
    """Unknown targets do not fall through to default production PyPI dispatch."""
    with pytest.raises(SystemExit, match="Invalid upload target"):
        publish.upload("unrecognised")


def test_actual_cli_confirmation_and_help_never_start_a_release(tmp_path: Path) -> None:
    """Ordinary script entrypoints refuse unconfirmed PyPI and expose operator help."""
    tool = Path(publish.__file__)
    cwd = tmp_path / "caller"
    cwd.mkdir()
    environment = dict(os.environ)
    if "COVERAGE_PROCESS_START" in environment:
        support = tmp_path / "coverage-support"
        support.mkdir()
        (support / "sitecustomize.py").write_text("import coverage\ncoverage.process_startup()\n", encoding="utf-8")
        environment["PYTHONPATH"] = str(support)
    else:
        environment.pop("PYTHONPATH", None)
    for argv, expected in [(["--target", "pypi"], 1), (["--help"], 0)]:
        result = subprocess.run(
            [sys.executable, str(tool), *argv], cwd=cwd, env=environment, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == expected
        assert "Traceback" not in result.stderr
    assert not list(cwd.iterdir())


def test_actual_process_launch_refusal_stops_the_dry_run(tmp_path: Path) -> None:
    """A real audit policy blocks the first build launch before any backend or upload."""
    script = """
import sys
import subprocess
from pathlib import Path
from tools import publish
root = Path(sys.argv[1])
publish.ROOT = root
publish.PYPROJECT = root / "pyproject.toml"
publish.DIST = root / "dist"
launches = []
def deny(event, args):
    if event == "subprocess.Popen":
        launches.append(args[1])
        raise PermissionError("selected release launch refused")
sys.addaudithook(deny)
try:
    publish.main(["--dry-run", "--skip-tests"])
except SystemExit as error:
    assert str(error) == "Publish pipeline refused: a local workflow operation failed"
else:
    raise AssertionError("release launch refusal was ignored")
assert len(launches) == 1
expected = [sys.executable, "tools/build_release_artifacts.py"]
assert launches[0] == (subprocess.list2cmdline(expected) if sys.platform == "win32" else expected)
assert not list((root / "dist").iterdir())
"""
    metadata = tmp_path / "pyproject.toml"
    metadata.write_bytes((Path(publish.__file__).parents[1] / "pyproject.toml").read_bytes())
    before = metadata.read_bytes()
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert metadata.read_bytes() == before
