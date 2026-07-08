# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Project Metadata Tests
"""Regression tests for repository metadata and packaging configuration."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol, cast

import pytest

from tools import check_version_sync


class _TomlModule(Protocol):
    def loads(self, data: str, /) -> dict[str, Any]:
        """Parse TOML text into a metadata dictionary."""


def _toml_module() -> _TomlModule:
    """Return the stdlib TOML parser, or the Python 3.10 backport."""
    module_name = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    return cast("_TomlModule", importlib.import_module(module_name))


ROOT = Path(__file__).resolve().parents[1]
TOML = _toml_module()


def _load_pyproject() -> dict[str, Any]:
    """Return the parsed project metadata from the repository pyproject."""
    return TOML.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


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
                f"`v{version}` commit/tag",
                f"`scpn-control=={version}`",
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
    assert "OK: all versions and release metadata = 0.22.1" in result.stdout


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
