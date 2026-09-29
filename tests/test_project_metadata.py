# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Project Metadata Tests
"""Regression tests for repository metadata and packaging configuration."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

from scpn_control.control.quantum_disruption_bridge import QUANTUM_BACKEND_OWNER
from tools import check_version_sync
from tools.ci_workflow_inventory import read_ci_workflow_source

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
    assert Version("1.4.3") in spo.specifier
    assert Version("1.4.0") not in spo.specifier
    assert Version("1.5.0") not in spo.specifier

    lock_input = (ROOT / "requirements/ci-deps.in").read_text(encoding="utf-8")
    lock = (ROOT / "requirements/ci-deps.txt").read_text(encoding="utf-8")
    workflow = read_ci_workflow_source()
    assert "scpn-phase-orchestrator==1.4.3" in lock_input
    assert "scpn-phase-orchestrator==1.4.3" in lock
    assert "5da94500760f9394a637f7edec044a844c12230d8d16c25a573b6e67a1ddb409" in lock
    assert "36495e013c6f436438437fab3a77611d401d4f0ef8b6838647fd33e19fc27979" in lock
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


# Zenodo relation-type vocabulary, read from
# https://zenodo.org/api/vocabularies/relationtypes on 2026-09-29 (34 entries).
# Zenodo does not accept the DataCite relations ``isRelatedTo`` and
# ``isAlternateIdentifier``; a release whose metadata names one is not archived.
ZENODO_RELATION_TYPES = frozenset(
    {
        "cites",
        "compiles",
        "continues",
        "describes",
        "documents",
        "hasmetadata",
        "haspart",
        "hasversion",
        "iscitedby",
        "iscompiledby",
        "iscontinuedby",
        "isderivedfrom",
        "isdescribedby",
        "isdocumentedby",
        "isidenticalto",
        "ismetadatafor",
        "isnewversionof",
        "isobsoletedby",
        "isoriginalformof",
        "ispartof",
        "ispreviousversionof",
        "ispublishedin",
        "isreferencedby",
        "isrequiredby",
        "isreviewedby",
        "issourceof",
        "issupplementedby",
        "issupplementto",
        "isvariantformof",
        "isversionof",
        "obsoletes",
        "references",
        "requires",
        "reviews",
    }
)
ZENODO_CONCEPT_DOI = "10.5281/zenodo.18804939"


def _zenodo_relations() -> dict[str, str]:
    """Return the ``.zenodo.json`` related identifiers as ``identifier -> relation``."""
    metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    related = cast("list[dict[str, str]]", metadata["related_identifiers"])
    return {item["identifier"]: item["relation"] for item in related}


def test_zenodo_relations_are_in_the_zenodo_relation_vocabulary() -> None:
    """Every archive relation must be one Zenodo accepts, compared case-insensitively."""
    relations = _zenodo_relations()

    assert len(relations) >= 1
    rejected = sorted(r for r in relations.values() if r.lower() not in ZENODO_RELATION_TYPES)
    assert rejected == []
    assert len(ZENODO_RELATION_TYPES) == 34


def test_zenodo_relations_match_the_package_dependency_contract() -> None:
    """Sibling-project relations must describe links the package actually has."""
    relations = _zenodo_relations()
    pyproject = _load_pyproject()
    project = cast("dict[str, Any]", pyproject["project"])
    extras = cast("dict[str, list[str]]", project["optional-dependencies"])

    assert relations["https://pypi.org/project/scpn-control/"] == "isVariantFormOf"
    assert relations["https://github.com/anulum/scpn-fusion-core"] == "references"
    assert "scpn-fusion>=4.0,<5.0" in extras["fusion"]
    assert relations["https://github.com/anulum/scpn-quantum-control"] == "references"
    assert QUANTUM_BACKEND_OWNER == "scpn-quantum-control"


def test_citation_doi_is_the_zenodo_concept_doi_shown_in_the_readme() -> None:
    """Cite the concept DOI while the current version has no Zenodo archive."""
    citation = cast("dict[str, Any]", yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8")))
    identifiers = cast("list[dict[str, str]]", citation["identifiers"])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert citation["doi"] == ZENODO_CONCEPT_DOI
    concept = [item for item in identifiers if item["description"] == "Zenodo concept DOI (all versions)"]
    assert [item["value"] for item in concept] == [ZENODO_CONCEPT_DOI]
    assert f"https://doi.org/{ZENODO_CONCEPT_DOI}" in readme
    labels = {item["value"]: item["description"] for item in identifiers}
    assert labels["10.5281/zenodo.18821816"] == "Zenodo archive (v0.4.0)"


def test_zenodo_text_and_date_describe_the_recorded_version() -> None:
    """Archive text names only the recorded version and its changelog release date."""
    metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    version = cast("str", metadata["version"])
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", changelog, re.MULTILINE)

    assert heading is not None
    assert metadata["publication_date"] == heading.group(1)
    assert f"v{version}" in metadata["notes"]
    for field in ("description", "notes"):
        assert set(re.findall(r"\bv(\d+\.\d+\.\d+)\b", metadata[field])) <= {version}
