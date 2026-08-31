# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Project Metadata Tests
"""Regression tests for repository metadata and packaging configuration."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from tools import check_version_sync

ROOT = Path(__file__).resolve().parents[1]


def _load_pyproject() -> dict[str, Any]:
    """Return the parsed project metadata from the repository pyproject."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_python_and_spo_dependency_contract_is_bounded_and_locked() -> None:
    """Bind supported Python versions to the immutable public SPO release."""
    project = cast("dict[str, Any]", _load_pyproject()["project"])
    classifiers = cast("list[str]", project["classifiers"])
    dependencies = [Requirement(item) for item in cast("list[str]", project["dependencies"])]
    spo = next(item for item in dependencies if item.name == "scpn-phase-orchestrator")

    assert project["requires-python"] == ">=3.11,<3.14"
    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert {
        item.rsplit(" :: ", 1)[-1] for item in classifiers if item.startswith("Programming Language :: Python :: 3.")
    } == {
        "3.11",
        "3.12",
        "3.13",
    }
    assert spo.url is None
    assert Version("1.3.1") in spo.specifier
    assert Version("1.3.0") not in spo.specifier
    assert Version("1.4.0") not in spo.specifier

    lock_input = (ROOT / "requirements/ci-deps.in").read_text(encoding="utf-8")
    lock = (ROOT / "requirements/ci-deps.txt").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "scpn-phase-orchestrator==1.3.1" in lock_input
    assert "scpn-phase-orchestrator==1.3.1" in lock
    assert "c2d7c0a5c0ad47f420fee02e54ccc28122bf8d128eb3b80ca51ba5f034320274" in lock
    assert "c0318a85931eef3fba6615bb5ff587c749c5a83c766504d10cdf7f2ac94e6fe3" in lock
    assert 'python-version: ["3.11", "3.12", "3.13"]' in workflow


def _write_release_metadata(root: Path, version: str) -> None:
    """Create a minimal metadata tree that satisfies the version-sync guard."""
    (root / "docs").mkdir()
    (root / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    (root / "CITATION.cff").write_text(f'version: "{version}"\n', encoding="utf-8")
    (root / ".zenodo.json").write_text(f'{{"version": "{version}"}}\n', encoding="utf-8")
    (root / "docs" / "api.md").write_text(version, encoding="utf-8")
    (root / "README.md").write_text(
        "\n".join(
            [
                "https://img.shields.io/pypi/v/scpn-control",
                "https://img.shields.io/pypi/pyversions/scpn-control",
                "https://pepy.tech/project/scpn-control",
                "https://static.pepy.tech/badge/scpn-control",
                f"| Package version | {version} |",
                f"git tag v{version}",
            ]
        ),
        encoding="utf-8",
    )
    (root / "docs" / f"release_notes_v{version}.md").write_text(
        "\n".join(
            [
                f"# SCPN Control v{version} Release Notes",
                "## Publication boundary",
                "This source-level release history treats hosted status as external mutable state.",
            ]
        ),
        encoding="utf-8",
    )


def test_facility_optional_extra_declares_mdsplus_thin_client() -> None:
    """Require the facility extra to expose the MDSplus thin client."""
    pyproject = _load_pyproject()

    project = cast("dict[str, Any]", pyproject["project"])
    extras = cast("dict[str, list[str]]", project["optional-dependencies"])
    assert "facility" in extras
    assert "mdsthin>=1.6.3" in extras["facility"]
    assert "mdsthin>=1.6.3" in extras["all"]


def test_fusion_optional_extra_pins_the_ida_solver_major() -> None:
    """Keep the CONTROL IDA facade on the compatible FUSION 4.x contract."""
    pyproject = _load_pyproject()

    project = cast("dict[str, Any]", pyproject["project"])
    extras = cast("dict[str, list[str]]", project["optional-dependencies"])
    requirement = "scpn-fusion>=4.0,<5.0"
    assert requirement in extras["fusion"]
    assert requirement in extras["all"]


def test_mypy_optional_overrides_do_not_hide_removed_first_party_modules() -> None:
    """Keep optional mypy imports aligned with live repository modules."""
    pyproject = _load_pyproject()
    tool_config = cast("dict[str, Any]", pyproject["tool"])
    mypy_config = cast("dict[str, Any]", tool_config["mypy"])
    overrides = cast("list[dict[str, object]]", mypy_config["overrides"])

    override_modules: set[str] = set()
    for override in overrides:
        configured_modules = override.get("module")
        if isinstance(configured_modules, str):
            override_modules.add(configured_modules)
        elif isinstance(configured_modules, list):
            override_modules.update(module for module in configured_modules if isinstance(module, str))
        else:
            msg = "mypy override entries must declare module as a string or string list"
            raise AssertionError(msg)

    assert "director_module" not in override_modules


def test_version_sync_guard_covers_release_badges_and_metadata() -> None:
    """Run the release metadata guard through its production CLI path."""
    assert check_version_sync.main() == 0

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_version_sync.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: all versions and release metadata = 0.23.0" in result.stdout


def test_version_sync_guard_fails_without_canonical_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing canonical metadata must fail closed."""
    monkeypatch.setattr(check_version_sync, "ROOT", tmp_path)

    assert check_version_sync.main() == 1
    assert "could not extract version from pyproject.toml" in capsys.readouterr().out


def test_version_sync_guard_reports_metadata_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Version, docs, README badge, and release-note drift must fail together."""
    _write_release_metadata(tmp_path, "1.2.3")
    (tmp_path / "CITATION.cff").write_text('version: "1.2.2"\n', encoding="utf-8")
    (tmp_path / ".zenodo.json").write_text('{"version": "1.2.1"}\n', encoding="utf-8")
    (tmp_path / "docs" / "api.md").write_text("old-version", encoding="utf-8")
    (tmp_path / "README.md").write_text("https://img.shields.io/pypi/v/scpn-control\n", encoding="utf-8")
    (tmp_path / "docs" / "release_notes_v1.2.3.md").unlink()
    monkeypatch.setattr(check_version_sync, "ROOT", tmp_path)

    assert check_version_sync.main() == 1
    output = capsys.readouterr().out
    assert "CITATION.cff has '1.2.2'" in output
    assert ".zenodo.json has '1.2.1'" in output
    assert "docs/api.md version marker missing '1.2.3'" in output
    assert "README Python-version badge" in output
    assert "release-note heading file docs/release_notes_v1.2.3.md does not exist" in output
    assert "file(s) out of sync" in output


def test_version_sync_guard_warns_on_missing_secondary_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing optional metadata warns but missing badges still fail the guard."""
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
    monkeypatch.setattr(check_version_sync, "ROOT", tmp_path)

    assert check_version_sync.main() == 1
    output = capsys.readouterr().out
    assert "WARN: could not extract version from CITATION.cff" in output
    assert "WARN: could not extract version from .zenodo.json" in output
    assert "README PyPI version badge file README.md does not exist" in output
