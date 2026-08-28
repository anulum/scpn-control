# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Local publication workflow tests.

"""Tests for exact-path release building, checking, and upload dispatch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

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

    def record(command: list[str], check: bool = True) -> subprocess.CompletedProcess[Any]:
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

    monkeypatch.setattr(publish.subprocess, "run", execute)
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
        metadata.write_text('version = "1.2.3"\n', encoding="utf-8")
        assert publish.bump_version(part) == expected
        assert metadata.read_text(encoding="utf-8") == f'version = "{expected}"\n'

    metadata.write_text('version = "1.2.3"\n', encoding="utf-8")
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

    metadata.write_text('version = "1.2"\n', encoding="utf-8")
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
    monkeypatch.setattr(publish, "bump_version", lambda part: calls.append(f"bump:{part}") or "2.0.0")
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
    assert "test.pypi.org" in capsys.readouterr().out

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
