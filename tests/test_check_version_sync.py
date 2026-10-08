# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Version guard runtime contracts.
"""Exercise the real release-text guard on copies of tracked metadata."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("shape", ["loop", "missing", "file"])
def test_repository_root_fault_is_an_authored_api_and_cli_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], shape: str
) -> None:
    """Refuse actual invalid repository carriers through both maintained commands."""
    from tools import check_version_sync

    repository = tmp_path / "repository"
    if shape == "loop":
        repository.symlink_to(repository.name, target_is_directory=True)
    elif shape == "file":
        repository.write_bytes(b"not a repository directory\n")
    assert check_version_sync.main(["--repo", str(repository)]) == 1
    direct = capsys.readouterr()
    assert not direct.err and "FAIL:" in direct.out
    if shape == "loop":
        assert direct.out == "FAIL: repository root could not be resolved\n"
    else:
        assert "repository root is not a directory" in direct.out
    process = subprocess.run(
        [sys.executable, str(ROOT / "tools/check_version_sync.py"), "--repo", str(repository)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 1
    assert process.stdout == direct.out and not process.stderr


def test_public_command_refuses_invalid_root_value(capsys: pytest.CaptureFixture[str]) -> None:
    """Refuse an embedded NUL through the real API without exposing a traceback."""
    from tools import check_version_sync

    assert check_version_sync.main(["--repo", "\0"]) == 1
    assert capsys.readouterr().out == "FAIL: repository root could not be resolved\n"


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """Copy the actual guard and its declared text inputs into an isolated checkout."""
    paths = [
        "tools/check_version_sync.py",
        "pyproject.toml",
        "CITATION.cff",
        ".zenodo.json",
        "README.md",
        "docs/api.md",
    ]
    paths.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / "docs").glob("release_notes_v*.md"))
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def _cli(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Execute the maintained CLI against the explicitly selected checkout."""
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/check_version_sync.py"), "--repo", str(checkout), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("name", ["CITATION.cff", ".zenodo.json"])
def test_secondary_version_is_required(checkout: Path, name: str) -> None:
    """Losing a required secondary version cannot produce an all-versions success."""
    (checkout / name).rename(checkout / (name + ".saved"))
    result = _cli(checkout)
    assert result.returncode == 1, result.stdout + result.stderr
    assert name in result.stdout and "OK: all versions" not in result.stdout


def test_tracked_release_text_passes_through_cli(checkout: Path) -> None:
    """Unchanged tracked carriers pass the bounded comparison without being rewritten."""
    before = {path: path.read_bytes() for path in checkout.rglob("*") if path.is_file()}
    result = _cli(checkout)
    assert result.returncode == 0 and "OK: all versions and release metadata" in result.stdout
    assert all(path.read_bytes() == data for path, data in before.items())


def test_default_root_is_the_script_checkout(checkout: Path) -> None:
    """The actual no-argument hook finds its owning root independently of the caller's cwd."""
    result = subprocess.run(
        [sys.executable, str(checkout / "tools/check_version_sync.py")],
        cwd=checkout / "docs",
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["pyproject.toml", "CITATION.cff", ".zenodo.json"])
def test_absent_version_field_is_refused(checkout: Path, name: str) -> None:
    """A present carrier without an extractable field cannot become version agreement."""
    (checkout / name).write_text("no version field\n", encoding="utf-8")
    result = _cli(checkout)
    assert result.returncode == 1 and f"could not extract version from {name}" in result.stdout


@pytest.mark.parametrize("name", ["CITATION.cff", ".zenodo.json"])
def test_secondary_version_drift_is_refused(checkout: Path, name: str) -> None:
    """Actual metadata version drift reports its owning file and canonical comparison."""
    target = checkout / name
    target.write_text(
        'version: "different"\n' if name.endswith("cff") else '{"version": "different"}', encoding="utf-8"
    )
    result = _cli(checkout)
    assert result.returncode == 1 and f"MISMATCH: {name}" in result.stdout


def test_missing_canonical_carrier_is_refused(checkout: Path) -> None:
    """A missing canonical assignment prevents checking incidental matching secondary text."""
    (checkout / "pyproject.toml").rename(checkout / "pyproject.saved")
    assert _cli(checkout).returncode == 1


@pytest.mark.parametrize("version", [".", "..", "../outside", "nested/version", "nested\\version"])
def test_version_cannot_select_another_release_note_path(checkout: Path, version: str) -> None:
    """Canonical filename components must not redirect the release-note read outside its convention."""
    (checkout / "pyproject.toml").write_text(f'version = "{version}"\n', encoding="utf-8")
    result = _cli(checkout)
    assert result.returncode == 1 and "safe filename component" in result.stdout


@pytest.mark.parametrize("name", ["README.md", "docs/api.md"])
def test_missing_required_text_is_refused(checkout: Path, name: str) -> None:
    """Removing required API/README text records the missing carrier without a crash."""
    (checkout / name).rename(checkout / (Path(name).name + ".saved"))
    result = _cli(checkout)
    assert result.returncode == 1 and name in result.stdout and "MISSING:" in result.stdout


def test_mismatched_required_literals_are_all_reported(checkout: Path) -> None:
    """Changed API, README and release notes produce independent literal findings."""
    for target in [
        checkout / "README.md",
        checkout / "docs/api.md",
        *list((checkout / "docs").glob("release_notes_v*.md")),
    ]:
        target.write_text("unrelated text\n", encoding="utf-8")
    result = _cli(checkout)
    assert result.returncode == 1
    assert "11 check(s) out of sync" in result.stdout
    assert "docs/api.md version marker" in result.stdout and "release-note heading" in result.stdout


@pytest.mark.parametrize("name", ["pyproject.toml", "README.md"])
def test_invalid_utf8_is_refused(checkout: Path, name: str) -> None:
    """Actual decoding failure becomes a CLI refusal instead of a traceback or success."""
    (checkout / name).write_bytes(b"\xff\xfe")
    result = _cli(checkout)
    assert result.returncode == 1 and "could not be read" in result.stdout and "Traceback" not in result.stderr


def test_nonfile_canonical_input_is_refused(checkout: Path) -> None:
    """An actual directory in place of required metadata is reported at the command boundary."""
    target = checkout / "pyproject.toml"
    target.rename(checkout / "pyproject.saved")
    target.mkdir()
    result = _cli(checkout)
    assert result.returncode == 1 and "could not be read" in result.stdout


@pytest.mark.parametrize("is_file", [False, True])
def test_invalid_selected_root_is_refused(tmp_path: Path, is_file: bool) -> None:
    """Missing and file roots cannot resolve to a successful carrier scope."""
    root = tmp_path / "not_a_directory"
    if is_file:
        root.write_text("existing file", encoding="utf-8")
    result = _cli(root)
    assert result.returncode == 1 and "not a directory" in result.stdout


def test_unknown_cli_argument_is_refused(checkout: Path) -> None:
    """Argument mistakes are rejected by the real parser rather than ignored by main."""
    result = _cli(checkout, "--unknown")
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr


def test_native_example_executes_the_owning_checkout() -> None:
    """Execute the public main example against real tracked release text."""
    import doctest

    from tools import check_version_sync

    result = doctest.testmod(check_version_sync, raise_on_error=True)
    assert result.failed == 0 and result.attempted == 3
