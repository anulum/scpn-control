# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capability manifest tests

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "capability_manifest.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("capability_manifest", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def capability_tool():
    return _load_tool()


def test_manifest_scans_control_specific_surfaces(capability_tool) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)

    assert manifest["project"]["name"] == "scpn-control"
    assert manifest["project"]["package"] == "scpn_control"
    assert "dashboard" in manifest["project"]["optional_extras"]
    assert "facility" in manifest["project"]["optional_extras"]
    assert manifest["project"]["scripts"]["scpn-control"] == "scpn_control.cli:main"
    assert (
        manifest["project"]["scripts"]["scpn-check-generated-traceability"] == "tools.check_generated_traceability:main"
    )

    assert "FusionKernel" in manifest["python"]["public_api_exports"]
    assert "NeuroSymbolicController" in manifest["python"]["public_api_exports"]
    assert "src/scpn_control/core/fusion_kernel.py" in manifest["python"]["source_modules"]
    assert "src/scpn_control/control/gym_tokamak_env.py" in manifest["python"]["source_modules"]
    assert "src/scpn_control/phase/kuramoto.py" in manifest["python"]["source_modules"]
    assert "src/scpn_control/scpn/compiler.py" in manifest["python"]["source_modules"]

    assert "control-python/src/lib.rs" in "\n".join(manifest["rust"]["source_files"])
    assert manifest["rust"]["pyo3_exports"]

    assert "validation/validate_real_shots.py" in manifest["validation"]["scripts"]
    assert ".github/workflows/ci.yml" in manifest["ci"]["workflows"]
    assert "tests/test_cli.py" in manifest["tests"]["python_files"]
    assert "docs/validation.md" in manifest["docs"]["public_markdown"]
    assert all("/internal/" not in f"/{path}/" for path in manifest["docs"]["public_markdown"])

    for section, count_name, values_name in (
        ("project", "project_script_count", "scripts"),
        ("python", "source_module_count", "source_modules"),
        ("python", "public_class_count", "public_classes"),
        ("python", "public_api_export_count", "public_api_exports"),
        ("rust", "rust_source_file_count", "source_files"),
        ("rust", "pyo3_export_count", "pyo3_exports"),
        ("validation", "validation_script_count", "scripts"),
        ("tests", "python_test_file_count", "python_files"),
        ("docs", "public_markdown_count", "public_markdown"),
        ("ci", "workflow_count", "workflows"),
    ):
        assert manifest["counts"][count_name] == len(manifest[section][values_name])


def test_manifest_validation_rejects_count_drift(capability_tool) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)
    broken = copy.deepcopy(manifest)
    broken["counts"]["source_module_count"] += 1

    with pytest.raises(capability_tool.ManifestError, match="source_module_count"):
        capability_tool.validate_manifest(broken)


def test_manifest_validation_rejects_missing_project_scripts(capability_tool) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)
    broken = copy.deepcopy(manifest)
    broken["project"]["scripts"] = {}
    broken["counts"]["project_script_count"] = 0

    with pytest.raises(capability_tool.ManifestError, match="project scripts must not be empty"):
        capability_tool.validate_manifest(broken)


def test_manifest_validation_rejects_invalid_project_script_targets(capability_tool) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)
    broken = copy.deepcopy(manifest)
    broken["project"]["scripts"]["scpn-control"] = "scpn_control.cli"

    with pytest.raises(capability_tool.ManifestError, match="project scripts must map to import targets"):
        capability_tool.validate_manifest(broken)


def test_generated_outputs_are_current(capability_tool) -> None:
    expected_manifest = capability_tool.build_manifest(REPO_ROOT)
    expected_markdown = capability_tool.render_markdown(expected_manifest)
    manifest_path = REPO_ROOT / "docs" / "_generated" / "capability_manifest.json"
    markdown_path = REPO_ROOT / "docs" / "_generated" / "capability_snapshot.md"

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == expected_manifest
    assert markdown_path.read_text(encoding="utf-8") == expected_markdown


def test_readme_snapshot_matches_generated_markdown(capability_tool) -> None:
    generated = (REPO_ROOT / "docs" / "_generated" / "capability_snapshot.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert capability_tool.extract_readme_block(readme) == generated


def test_markdown_snapshot_is_readme_safe(capability_tool) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)
    markdown = capability_tool.render_markdown(manifest)

    assert markdown.startswith("<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->\n")
    assert "<h1" not in markdown.lower()
    assert "\n# " not in markdown
    assert "| Project scripts | 2 |" in markdown
    assert "".join(("Co", "dex")) not in markdown
    assert "".join(("Open", "AI")) not in markdown


def test_capability_manifest_docs_cover_project_scripts() -> None:
    docs = (REPO_ROOT / "docs" / "capability_manifest.md").read_text(encoding="utf-8")

    assert "packaged project script entry points" in docs
    assert "`[project.scripts]` from `pyproject.toml`" in docs
