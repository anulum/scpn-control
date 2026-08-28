# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reproducible release-artifact builder tests.

"""Tests for deterministic and private-safe Python distribution artifacts."""

from __future__ import annotations

import io
import subprocess
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import build_release_artifacts

EPOCH = 1_787_886_522


def _write_sdist(path: Path, *, mtime: int, member_name: str = "demo/module.py") -> None:
    with tarfile.open(path, "w:gz") as archive:
        directory = tarfile.TarInfo("demo")
        directory.type = tarfile.DIRTYPE
        directory.mtime = mtime
        archive.addfile(directory)
        payload = b"VALUE = 1\n"
        module = tarfile.TarInfo(member_name)
        module.size = len(payload)
        module.mtime = mtime
        archive.addfile(module, io.BytesIO(payload))


def _write_wheel(
    path: Path,
    *,
    target: str = "demo.cli:main",
    license_expression: str = "AGPL-3.0-or-later",
    entry_points: str | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("demo/__init__.py", "")
        archive.writestr("demo/cli.py", "def main():\n    return 0\n")
        archive.writestr(
            "demo-1.0.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: demo\nVersion: 1.0\nLicense-Expression: {license_expression}\n",
        )
        archive.writestr(
            "demo-1.0.dist-info/entry_points.txt",
            entry_points if entry_points is not None else f"[console_scripts]\ndemo = {target}\n",
        )


def test_normalise_sdist_is_byte_reproducible(tmp_path: Path) -> None:
    """Different source mtimes normalise to one byte-identical sdist."""
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_sdist(first, mtime=1)
    _write_sdist(second, mtime=2)

    build_release_artifacts.normalise_sdist(first, EPOCH)
    build_release_artifacts.normalise_sdist(second, EPOCH)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        assert [member.name for member in archive] == ["demo", "demo/module.py"]
        assert all(member.mtime == EPOCH for member in archive.getmembers())
        assert all(member.uid == member.gid == 0 for member in archive.getmembers())


@pytest.mark.parametrize("name", ("/absolute.py", "demo/../private.py", "demo/docs/internal/TODO.md"))
def test_normalise_sdist_rejects_unsafe_or_private_members(tmp_path: Path, name: str) -> None:
    """Archive traversal and private repository surfaces fail closed."""
    artifact = tmp_path / "unsafe.tar.gz"
    _write_sdist(artifact, mtime=1, member_name=name)

    with pytest.raises(ValueError, match="archive member"):
        build_release_artifacts.normalise_sdist(artifact, EPOCH)


def test_normalise_sdist_rejects_link_members(tmp_path: Path) -> None:
    """An sdist cannot smuggle symlink or hardlink indirection."""
    artifact = tmp_path / "link.tar.gz"
    with tarfile.open(artifact, "w:gz") as archive:
        link = tarfile.TarInfo("demo/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../private"
        archive.addfile(link)

    with pytest.raises(ValueError, match="unsupported sdist member type"):
        build_release_artifacts.normalise_sdist(artifact, EPOCH)


def test_validate_wheel_accepts_shipped_console_target(tmp_path: Path) -> None:
    """A console target shipped in the wheel passes artifact validation."""
    artifact = tmp_path / "demo-1.0-py3-none-any.whl"
    _write_wheel(artifact)

    summary = build_release_artifacts.validate_artifact(artifact)

    assert summary.path == artifact
    assert summary.entries == 4
    assert len(summary.sha256) == 64


def test_validate_wheel_accepts_package_target_and_no_console_section(tmp_path: Path) -> None:
    """Package targets and wheels without console scripts are both valid."""
    package_target = tmp_path / "package-1.0-py3-none-any.whl"
    _write_wheel(package_target, target="demo:main")
    assert build_release_artifacts.validate_artifact(package_target).entries == 4

    no_console = tmp_path / "library-1.0-py3-none-any.whl"
    _write_wheel(no_console, entry_points="[plugins]\ndemo = demo.cli:main\n")
    assert build_release_artifacts.validate_artifact(no_console).entries == 4


def test_validate_wheel_rejects_missing_console_target(tmp_path: Path) -> None:
    """A declared console target absent from the wheel fails validation."""
    artifact = tmp_path / "demo-1.0-py3-none-any.whl"
    _write_wheel(artifact, target="missing.module:main")

    with pytest.raises(ValueError, match="targets missing wheel module"):
        build_release_artifacts.validate_artifact(artifact)


def test_validate_wheel_rejects_bad_metadata_and_entry_point_cardinality(tmp_path: Path) -> None:
    """License and entry-point metadata must be singular and canonical."""
    bad_license = tmp_path / "bad-license.whl"
    _write_wheel(bad_license, license_expression="MIT")
    with pytest.raises(ValueError, match="lacks the canonical SPDX"):
        build_release_artifacts.validate_artifact(bad_license)

    missing_metadata = tmp_path / "missing-metadata.whl"
    with zipfile.ZipFile(missing_metadata, "w") as archive:
        archive.writestr("demo/__init__.py", "")
        archive.writestr("demo-1.0.dist-info/entry_points.txt", "[console_scripts]\n")
    with pytest.raises(ValueError, match="exactly one METADATA"):
        build_release_artifacts.validate_artifact(missing_metadata)

    missing_entry_points = tmp_path / "missing-entry-points.whl"
    with zipfile.ZipFile(missing_entry_points, "w") as archive:
        archive.writestr("demo/__init__.py", "")
        archive.writestr(
            "demo-1.0.dist-info/METADATA",
            "License-Expression: AGPL-3.0-or-later\n",
        )
    with pytest.raises(ValueError, match="exactly one entry_points"):
        build_release_artifacts.validate_artifact(missing_entry_points)


def test_validate_sdist_and_reject_unsupported_artifact(tmp_path: Path) -> None:
    """Validated sdists report identity while unknown formats fail closed."""
    artifact = tmp_path / "demo.tar.gz"
    _write_sdist(artifact, mtime=EPOCH)
    assert build_release_artifacts.validate_artifact(artifact).entries == 2

    with pytest.raises(ValueError, match="unsupported release artifact"):
        build_release_artifacts.validate_artifact(tmp_path / "demo.zip")


def test_source_date_epoch_rejects_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-integer reproducibility epochs fail before a build starts."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not-an-integer")

    with pytest.raises(ValueError, match="must be an integer"):
        build_release_artifacts._source_date_epoch(None)


def test_source_date_epoch_precedence_and_git_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit, environment, and exact-head epochs have deterministic precedence."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "42")
    assert build_release_artifacts._source_date_epoch(7) == 7
    assert build_release_artifacts._source_date_epoch(None) == 42

    monkeypatch.delenv("SOURCE_DATE_EPOCH")
    monkeypatch.setattr(
        build_release_artifacts.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="99\n"),
    )
    assert build_release_artifacts._source_date_epoch(None) == 99


def test_source_date_epoch_requires_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """The implicit epoch route fails explicitly when Git is unavailable."""
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    monkeypatch.setattr(build_release_artifacts.shutil, "which", lambda command: None)

    with pytest.raises(RuntimeError, match="git is required"):
        build_release_artifacts._source_date_epoch(None)


def test_build_release_artifacts_rejects_stale_and_missing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build never mixes stale artifacts or silently accepts missing outputs."""
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "old.whl").write_bytes(b"old")
    with pytest.raises(ValueError, match="stale distributions"):
        build_release_artifacts.build_release_artifacts(stale, epoch=EPOCH)

    monkeypatch.setattr(
        build_release_artifacts.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    with pytest.raises(ValueError, match="expected 2 distribution"):
        build_release_artifacts.build_release_artifacts(tmp_path / "empty", epoch=EPOCH)


@pytest.mark.parametrize("sdist_only", (False, True))
def test_build_release_artifacts_constructs_and_validates_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sdist_only: bool,
) -> None:
    """The orchestrator builds the requested set and normalises every sdist."""
    outdir = tmp_path / ("sdist" if sdist_only else "both")

    def fake_build(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[list[str]]:
        destination = Path(command[command.index("--outdir") + 1])
        _write_sdist(destination / "demo.tar.gz", mtime=1)
        if "--sdist" not in command:
            _write_wheel(destination / "demo-1.0-py3-none-any.whl")
        assert kwargs["env"]  # PATH and the pinned epoch are forwarded.
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build_release_artifacts.subprocess, "run", fake_build)
    summaries = build_release_artifacts.build_release_artifacts(
        outdir,
        epoch=EPOCH,
        sdist_only=sdist_only,
    )

    assert len(summaries) == (1 if sdist_only else 2)
    with tarfile.open(outdir / "demo.tar.gz", "r:gz") as archive:
        assert all(member.mtime == EPOCH for member in archive.getmembers())


def test_main_prints_artifact_identity_and_rejects_negative_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI output is evidence-bearing and negative epochs fail parsing."""
    artifact = tmp_path / "demo.whl"
    monkeypatch.setattr(
        build_release_artifacts,
        "build_release_artifacts",
        lambda *args, **kwargs: [build_release_artifacts.ArtifactSummary(artifact, 4, "a" * 64)],
    )
    assert build_release_artifacts.main(["--outdir", str(tmp_path), "--source-date-epoch", "7"]) == 0
    assert "demo.whl\tentries=4\tsha256=" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        build_release_artifacts.main(["--source-date-epoch", "-1"])
