# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capability manifest tests.

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from _pytest.capture import CaptureFixture


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool() -> ModuleType:
    return importlib.import_module("tools.capability_manifest")


@pytest.fixture()
def capability_tool() -> ModuleType:
    return _load_tool()


def _write_fixture_repo(repo: Path) -> None:
    (repo / "tools").mkdir()
    (repo / "src" / "fixture_pkg").mkdir(parents=True)
    (repo / "rust" / "src").mkdir(parents=True)
    (repo / "validation").mkdir()
    (repo / "tests").mkdir()
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / "docs" / "_generated").mkdir(parents=True)
    (repo / "docs" / "internal").mkdir(parents=True)

    (repo / "tools" / "capability_manifest.toml").write_text(
        """
schema_version = "fixture.schema"
project_label = "Fixture Control"
package_name = "fixture_pkg"
exclude_doc_parts = ["internal"]

[paths]
pyproject = "pyproject.toml"
package_root = "src/fixture_pkg"
source_roots = ["src/fixture_pkg", "missing_src"]
rust_root = "rust"
rust_wrappers = "rust/src/lib.rs"
validation_root = "validation"
tests_root = "tests"
workflows_root = ".github/workflows"
docs_root = "docs"
json_output = "docs/_generated/capability_manifest.json"
markdown_output = "docs/_generated/capability_snapshot.md"
readme = "README.md"
""".lstrip(),
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        """
[project]
name = "fixture-control"
version = "1.2.3"
requires-python = ">=3.10"

[project.optional-dependencies]
docs = []

[project.scripts]
fixture-control = "fixture_pkg.cli:main"
""".lstrip(),
        encoding="utf-8",
    )
    (repo / "src" / "fixture_pkg" / "__init__.py").write_text('__all__ = ["PublicClass"]\n', encoding="utf-8")
    (repo / "src" / "fixture_pkg" / "module.py").write_text("class PublicClass:\n    pass\n", encoding="utf-8")
    (repo / "src" / "fixture_pkg" / "broken.py").write_text("class Broken(:\n", encoding="utf-8")
    (repo / "rust" / "src" / "lib.rs").write_text(
        """
#[pyfunction]
fn fixture_tick() {}

#[pyclass]
pub struct PyFixture {}

fn register() {
    wrap_pyfunction!(fixture_tick, module);
    module.add_class::<PyFixture>();
}
""".lstrip(),
        encoding="utf-8",
    )
    (repo / "validation" / "check_fixture.py").write_text("print('check')\n", encoding="utf-8")
    (repo / "tests" / "test_fixture.py").write_text("def test_fixture() -> None:\n    assert True\n", encoding="utf-8")
    (repo / ".github" / "workflows" / "ci.yml").write_text("name: CI\n", encoding="utf-8")
    (repo / "docs" / "_generated" / "capability_snapshot.md").write_text("stale\n", encoding="utf-8")
    (repo / "docs" / "index.md").write_text("# Fixture docs\n", encoding="utf-8")
    (repo / "docs" / "internal" / "plan.md").write_text("# Internal\n", encoding="utf-8")
    (repo / "README.md").write_text(
        "Intro\n<!-- capability-snapshot:start -->\nstale\n<!-- capability-snapshot:end -->\n",
        encoding="utf-8",
    )


def test_manifest_scans_control_specific_surfaces(capability_tool: Any) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)

    assert manifest["project"]["name"] == "scpn-control"
    assert manifest["project"]["package"] == "scpn_control"
    assert "dashboard" in manifest["project"]["optional_extras"]
    assert "facility" in manifest["project"]["optional_extras"]
    assert manifest["project"]["scripts"]["scpn-control"] == "scpn_control.cli:main"
    assert (
        manifest["project"]["scripts"]["scpn-check-generated-traceability"] == "tools.check_generated_traceability:main"
    )
    assert manifest["project"]["scripts"]["scpn-evidence-gap-matrix"] == "tools.evidence_gap_matrix:main"
    assert (
        manifest["project"]["scripts"]["scpn-validation-report-freshness"] == "tools.validation_report_freshness:main"
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


def test_manifest_validation_rejects_count_drift(capability_tool: Any) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)
    broken = copy.deepcopy(manifest)
    broken["counts"]["source_module_count"] += 1

    with pytest.raises(capability_tool.ManifestError, match="source_module_count"):
        capability_tool.validate_manifest(broken)


def test_manifest_validation_rejects_missing_project_scripts(capability_tool: Any) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)
    broken = copy.deepcopy(manifest)
    broken["project"]["scripts"] = {}
    broken["counts"]["project_script_count"] = 0

    with pytest.raises(capability_tool.ManifestError, match="project scripts must not be empty"):
        capability_tool.validate_manifest(broken)


def test_manifest_validation_rejects_invalid_project_script_targets(capability_tool: Any) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)
    broken = copy.deepcopy(manifest)
    broken["project"]["scripts"]["scpn-control"] = "scpn_control.cli"

    with pytest.raises(capability_tool.ManifestError, match="project scripts must map to import targets"):
        capability_tool.validate_manifest(broken)


def test_manifest_validation_rejects_missing_public_api_exports(capability_tool: Any, tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    (tmp_path / "src" / "fixture_pkg" / "__init__.py").write_text("# no public exports\n", encoding="utf-8")

    with pytest.raises(capability_tool.ManifestError, match="public_api_exports"):
        capability_tool.build_manifest(tmp_path)


def test_manifest_validation_rejects_missing_pyo3_exports(capability_tool: Any, tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    (tmp_path / "rust" / "src" / "lib.rs").unlink()

    with pytest.raises(capability_tool.ManifestError, match="pyo3_exports"):
        capability_tool.build_manifest(tmp_path)


def test_manifest_validation_rejects_internal_public_markdown(capability_tool: Any) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)
    broken = copy.deepcopy(manifest)
    broken["docs"]["public_markdown"].append("docs/internal/plan.md")
    broken["counts"]["public_markdown_count"] += 1

    with pytest.raises(capability_tool.ManifestError, match="public markdown inventory"):
        capability_tool.validate_manifest(broken)


def test_fixture_repo_outputs_and_cli_are_generated(
    capability_tool: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture_repo(tmp_path)

    manifest = capability_tool.build_manifest(tmp_path)
    assert manifest["project"]["name"] == "fixture-control"
    assert manifest["python"]["public_classes"] == ["PublicClass"]
    assert "docs/internal/plan.md" not in manifest["docs"]["public_markdown"]

    capability_tool.write_outputs(tmp_path)
    assert capability_tool.check_outputs(tmp_path) == []
    assert capability_tool.extract_readme_block((tmp_path / "README.md").read_text(encoding="utf-8")).startswith(
        "**Capability Inventory**"
    )

    monkeypatch.chdir(tmp_path)
    assert capability_tool.main(["--check"]) == 0
    assert capability_tool.main([]) == 0


def test_check_outputs_reports_missing_stale_and_marker_failures(capability_tool: Any, tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    capability_tool.write_outputs(tmp_path)

    (tmp_path / "docs" / "_generated" / "capability_manifest.json").unlink()
    (tmp_path / "docs" / "_generated" / "capability_snapshot.md").write_text("stale\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("no markers\n", encoding="utf-8")

    assert capability_tool.check_outputs(tmp_path) == [
        "docs/_generated/capability_manifest.json is missing",
        "docs/_generated/capability_snapshot.md is stale",
        "README.md is missing capability snapshot markers",
    ]


def test_check_outputs_reports_readme_snapshot_drift(capability_tool: Any, tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    capability_tool.write_outputs(tmp_path)
    (tmp_path / "README.md").write_text(
        "Intro\n<!-- capability-snapshot:start -->\ndrift\n<!-- capability-snapshot:end -->\n",
        encoding="utf-8",
    )

    assert "README.md capability snapshot is stale" in capability_tool.check_outputs(tmp_path)


def test_write_outputs_rejects_readme_without_markers(capability_tool: Any, tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    (tmp_path / "README.md").write_text("no markers\n", encoding="utf-8")

    with pytest.raises(capability_tool.ManifestError, match="README.md is missing"):
        capability_tool.write_outputs(tmp_path)


def test_main_reports_check_failures(
    capability_tool: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    _write_fixture_repo(tmp_path)
    capability_tool.write_outputs(tmp_path)
    (tmp_path / "docs" / "_generated" / "capability_snapshot.md").write_text("stale\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    assert capability_tool.main(["--check"]) == 1
    assert "docs/_generated/capability_snapshot.md is stale" in capsys.readouterr().err


def test_generated_outputs_are_current(capability_tool: Any) -> None:
    expected_manifest = capability_tool.build_manifest(REPO_ROOT)
    expected_markdown = capability_tool.render_markdown(expected_manifest)
    manifest_path = REPO_ROOT / "docs" / "_generated" / "capability_manifest.json"
    markdown_path = REPO_ROOT / "docs" / "_generated" / "capability_snapshot.md"

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == expected_manifest
    assert markdown_path.read_text(encoding="utf-8") == expected_markdown


def test_readme_snapshot_matches_generated_markdown(capability_tool: Any) -> None:
    generated = (REPO_ROOT / "docs" / "_generated" / "capability_snapshot.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert capability_tool.extract_readme_block(readme) == generated


def test_markdown_snapshot_is_readme_safe(capability_tool: Any) -> None:
    manifest = capability_tool.build_manifest(REPO_ROOT)
    markdown = capability_tool.render_markdown(manifest)

    assert markdown.startswith("**Capability Inventory**\n")
    first_lines = "\n".join(markdown.splitlines()[:8])
    assert "SPDX-License-Identifier" not in first_lines
    assert "Commercial license available" not in first_lines
    assert "ORCID:" not in first_lines
    assert "<h1" not in markdown.lower()
    assert "\n# " not in markdown
    assert f"| Project scripts | {manifest['counts']['project_script_count']} |" in markdown
    assert "".join(("Co", "dex")) not in markdown
    assert "".join(("Open", "AI")) not in markdown


def test_capability_manifest_docs_cover_project_scripts() -> None:
    docs = (REPO_ROOT / "docs" / "capability_manifest.md").read_text(encoding="utf-8")

    assert "packaged project script entry points" in docs
    assert "`[project.scripts]` from `pyproject.toml`" in docs
