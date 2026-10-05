# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capability manifest tests.
"""Exercise static inventory and output contracts with real copied source carriers."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _pytest.capture import CaptureFixture

from tools import capability_manifest as capability_tool

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def actual_manifest() -> dict[str, Any]:
    """Inspect the unchanged canonical source once for consumer corruption cases."""
    return capability_tool.build_manifest(REPO_ROOT)


PRIVATE_NOTE = "docs/internal/working_note.md"


def _write_fixture_repo(repo: Path) -> None:
    """Copy actual repository carriers into a declared partial inventory scope.

    The original configuration and metadata remain authoritative. Copy one
    maintained file per configured source category and real wrapper/workflow/
    validation/test/docs carriers; source text is inventoried without importing
    it. Positive cases claim catalog mechanics, not executable model readiness.
    """
    names = [
        "tools/capability_manifest.toml",
        "pyproject.toml",
        "README.md",
        "src/scpn_control/__init__.py",
        "src/scpn_control/core/eqdsk.py",
        "src/scpn_control/control/director_interface.py",
        "src/scpn_control/phase/kuramoto.py",
        "src/scpn_control/scpn/compiler.py",
        "src/scpn_control/reactor_semantic_admission/admission.py",
        "scpn-control-rs/crates/control-python/src/lib.rs",
        "validation/validate_real_shots.py",
        "tests/test_check_commit_authorship.py",
        "docs/capability_manifest.md",
        ".github/workflows/ci-static-governance.yml",
    ]
    for name in names:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, target)
    # A Markdown file under the ignored private tree verifies the configured
    # exclusion. The private tree is absent from a clean checkout, so the
    # fixture writes its own file instead of copying one.
    internal = repo / PRIVATE_NOTE
    internal.parent.mkdir(parents=True, exist_ok=True)
    internal.write_text("# Working note\n\nNot part of the public inventory.\n", encoding="utf-8")


def test_manifest_scans_control_specific_surfaces(actual_manifest: dict[str, Any]) -> None:
    """Inspect actual repository declarations and configured counts through the public builder."""
    manifest = actual_manifest

    assert manifest["project"]["name"] == "scpn-control"
    assert manifest["project"]["package"] == "scpn_control"
    assert "dashboard" in manifest["project"]["optional_extras"]
    assert "facility" in manifest["project"]["optional_extras"]
    assert manifest["project"]["scripts"]["scpn-control"] == "scpn_control.cli:main"
    assert manifest["project"]["scripts"] == {"scpn-control": "scpn_control.cli:main"}

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


def test_manifest_validation_rejects_count_drift(actual_manifest: dict[str, Any]) -> None:
    """Reject a count mutation in a manifest derived from actual repository source."""
    manifest = actual_manifest
    broken = copy.deepcopy(manifest)
    broken["counts"]["source_module_count"] += 1

    with pytest.raises(capability_tool.ManifestError, match="source_module_count"):
        capability_tool.validate_manifest(broken)


def test_manifest_validation_rejects_missing_project_scripts(actual_manifest: dict[str, Any]) -> None:
    """An empty declared script mapping cannot retire the required command surface."""
    manifest = actual_manifest
    broken = copy.deepcopy(manifest)
    broken["project"]["scripts"] = {}
    broken["counts"]["project_script_count"] = 0

    with pytest.raises(capability_tool.ManifestError, match="project scripts must not be empty"):
        capability_tool.validate_manifest(broken)


def test_manifest_validation_rejects_invalid_project_script_targets(actual_manifest: dict[str, Any]) -> None:
    """Reject a declared entry point without its module/callable separator."""
    manifest = actual_manifest
    broken = copy.deepcopy(manifest)
    broken["project"]["scripts"]["scpn-control"] = "scpn_control.cli"

    with pytest.raises(capability_tool.ManifestError, match="project scripts must map to import targets"):
        capability_tool.validate_manifest(broken)


def test_manifest_validation_rejects_missing_public_api_exports(tmp_path: Path) -> None:
    """Removing the copied package export declaration refuses the inventory."""
    _write_fixture_repo(tmp_path)
    (tmp_path / "src" / "scpn_control" / "__init__.py").write_text("# no public exports\n", encoding="utf-8")

    with pytest.raises(capability_tool.ManifestError, match="public_api_exports"):
        capability_tool.build_manifest(tmp_path)


def test_manifest_validation_rejects_missing_pyo3_exports(tmp_path: Path) -> None:
    """A missing actual wrapper cannot be certified as a populated export surface."""
    _write_fixture_repo(tmp_path)
    (tmp_path / "scpn-control-rs/crates/control-python/src/lib.rs").unlink()

    with pytest.raises(capability_tool.ManifestError, match="pyo3_exports"):
        capability_tool.build_manifest(tmp_path)


def test_manifest_validation_rejects_internal_public_markdown(actual_manifest: dict[str, Any]) -> None:
    """Reject an internal-state path inserted into the public document inventory."""
    manifest = actual_manifest
    broken = copy.deepcopy(manifest)
    broken["docs"]["public_markdown"].append("docs/internal/plan.md")
    broken["counts"]["public_markdown_count"] += 1

    with pytest.raises(capability_tool.ManifestError, match="public markdown inventory"):
        capability_tool.validate_manifest(broken)


def test_fixture_repo_outputs_and_cli_are_generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Generate and check the actual copied corpus through public API and command functions."""
    _write_fixture_repo(tmp_path)

    manifest = capability_tool.build_manifest(tmp_path)
    assert manifest["project"]["name"] == "scpn-control"
    assert "GEqdsk" in manifest["python"]["public_classes"]
    assert (tmp_path / PRIVATE_NOTE).is_file()
    assert PRIVATE_NOTE not in manifest["docs"]["public_markdown"]

    capability_tool.write_outputs(tmp_path)
    assert capability_tool.check_outputs(tmp_path) == []
    assert capability_tool.extract_readme_block((tmp_path / "README.md").read_text(encoding="utf-8")).startswith(
        "**Capability Inventory**"
    )

    monkeypatch.chdir(tmp_path)
    assert capability_tool.main(["--check"]) == 0
    assert capability_tool.main([]) == 0


def test_check_outputs_reports_missing_stale_and_marker_failures(tmp_path: Path) -> None:
    """Report independent missing/stale output and invalid README errors together."""
    _write_fixture_repo(tmp_path)
    capability_tool.write_outputs(tmp_path)

    (tmp_path / "docs" / "_generated" / "capability_manifest.json").unlink()
    (tmp_path / "docs" / "_generated" / "capability_snapshot.md").write_text("stale\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("no markers\n", encoding="utf-8")

    assert capability_tool.check_outputs(tmp_path) == [
        "docs/_generated/capability_manifest.json is missing",
        "docs/_generated/capability_snapshot.md is stale",
        "README.md is missing or has invalid capability snapshot markers",
    ]


def test_check_outputs_reports_readme_snapshot_drift(tmp_path: Path) -> None:
    """A changed marker fragment is stale even while generated carriers remain intact."""
    _write_fixture_repo(tmp_path)
    capability_tool.write_outputs(tmp_path)
    (tmp_path / "README.md").write_text(
        "Intro\n<!-- capability-snapshot:start -->\ndrift\n<!-- capability-snapshot:end -->\n",
        encoding="utf-8",
    )

    assert "README.md capability snapshot is stale" in capability_tool.check_outputs(tmp_path)


def test_write_outputs_rejects_readme_without_markers(tmp_path: Path) -> None:
    """Refuse a README without a replaceable marker pair."""
    _write_fixture_repo(tmp_path)
    (tmp_path / "README.md").write_text("no markers\n", encoding="utf-8")

    with pytest.raises(capability_tool.ManifestError, match="README.md is missing"):
        capability_tool.write_outputs(tmp_path)


def test_main_reports_check_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """The actual check command reports stale copied output through stderr."""
    _write_fixture_repo(tmp_path)
    capability_tool.write_outputs(tmp_path)
    (tmp_path / "docs" / "_generated" / "capability_snapshot.md").write_text("stale\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    assert capability_tool.main(["--check"]) == 1
    assert "docs/_generated/capability_snapshot.md is stale" in capsys.readouterr().err


def test_generated_outputs_are_current(actual_manifest: dict[str, Any]) -> None:
    """The canonical generated snapshot must equal the current source inventory at task closeout."""
    expected_manifest = actual_manifest
    expected_markdown = capability_tool.render_markdown(expected_manifest)
    manifest_path = REPO_ROOT / "docs" / "_generated" / "capability_manifest.json"
    markdown_path = REPO_ROOT / "docs" / "_generated" / "capability_snapshot.md"

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == expected_manifest
    assert markdown_path.read_text(encoding="utf-8") == expected_markdown


def test_readme_snapshot_matches_generated_markdown() -> None:
    """Keep the canonical embedded fragment aligned with its actual generated carrier."""
    generated = (REPO_ROOT / "docs" / "_generated" / "capability_snapshot.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert capability_tool.extract_readme_block(readme) == generated


def test_markdown_snapshot_is_readme_safe(actual_manifest: dict[str, Any]) -> None:
    """Render a real inventory into the intended table/command fragment without a document heading."""
    manifest = actual_manifest
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
    """The owning guide documents the declared script field used by discovery."""
    docs = (REPO_ROOT / "docs" / "capability_manifest.md").read_text(encoding="utf-8")

    assert "packaged project script entry points" in docs
    assert "`[project.scripts]` from `pyproject.toml`" in docs


def _cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the real generator from a selected copied checkout, preserving argv transport."""
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/capability_manifest.py"), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_real_cli_generates_and_checks_actual_carriers(tmp_path: Path) -> None:
    """The registered command publishes and checks real copied Python/Rust/configuration text."""
    _write_fixture_repo(tmp_path)
    assert _cli(tmp_path).returncode == 0
    result = _cli(tmp_path, "--check")
    assert result.returncode == 0 and result.stdout == "" and result.stderr == ""
    payload = json.loads((tmp_path / "docs/_generated/capability_manifest.json").read_text())
    assert payload["project"]["scripts"] == {"scpn-control": "scpn_control.cli:main"}
    assert "GEqdsk" in payload["python"]["public_classes"] and payload["rust"]["pyo3_exports"]


def test_invalid_actual_source_cannot_be_an_empty_class_inventory(tmp_path: Path) -> None:
    """Corrupt a copied maintained source and refuse the public build and command."""
    _write_fixture_repo(tmp_path)
    source = tmp_path / "src/scpn_control/core/eqdsk.py"
    source.write_bytes(source.read_bytes() + b"\nclass Broken(:\n")
    with pytest.raises(capability_tool.ManifestError, match="could not inspect Python source"):
        capability_tool.build_manifest(tmp_path)
    result = _cli(tmp_path)
    assert result.returncode == 1 and "eqdsk.py" in result.stderr and "Traceback" not in result.stderr
    assert not (tmp_path / "docs/_generated/capability_manifest.json").exists()


@pytest.mark.parametrize(
    "markers",
    [
        "no markers",
        capability_tool.README_START,
        capability_tool.README_START + capability_tool.README_START + capability_tool.README_END,
        capability_tool.README_START + capability_tool.README_END + capability_tool.README_END,
        capability_tool.README_END + capability_tool.README_START,
    ],
)
def test_bad_readme_preserves_existing_generated_bytes(tmp_path: Path, markers: str) -> None:
    """README refusal happens before any prepared output can replace existing files."""
    _write_fixture_repo(tmp_path)
    capability_tool.write_outputs(tmp_path)
    paths = [tmp_path / "docs/_generated/capability_manifest.json", tmp_path / "docs/_generated/capability_snapshot.md"]
    before = {path: path.read_bytes() for path in paths}
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace('version = "0.23.0"', 'version = "0.23.0+probe"', 1))
    readme = tmp_path / "README.md"
    readme.write_text(markers)
    with pytest.raises(capability_tool.ManifestError, match="capability snapshot markers"):
        capability_tool.write_outputs(tmp_path)
    assert all(path.read_bytes() == original for path, original in before.items())
    assert readme.read_text() == markers
    assert _cli(tmp_path).returncode == 1
    assert _cli(tmp_path, "--check").returncode == 1


@pytest.mark.parametrize("value", [True, 1.0, "1", -1])
def test_declared_counts_require_nonboolean_integers(actual_manifest: dict[str, Any], value: object) -> None:
    """Reject type coercion/equality shortcuts in a real manifest's one-script count."""
    changed = copy.deepcopy(actual_manifest)
    changed["counts"]["project_script_count"] = value
    with pytest.raises(capability_tool.ManifestError, match="project_script_count"):
        capability_tool.validate_manifest(changed)


@pytest.mark.parametrize("target", ["module:", ":main", "module:main:extra", "module.1bad:main", 7])
def test_script_targets_require_declared_module_callable_shape(actual_manifest: dict[str, Any], target: object) -> None:
    """Validate target spelling without claiming that the declared command was imported."""
    changed = copy.deepcopy(actual_manifest)
    changed["project"]["scripts"]["scpn-control"] = target
    with pytest.raises(capability_tool.ManifestError, match="project scripts must map"):
        capability_tool.validate_manifest(changed)


@pytest.mark.parametrize("replacement", ["dict()", "[7]", "'export-as-string'"])
def test_exports_require_a_literal_string_sequence(tmp_path: Path, replacement: str) -> None:
    """Corrupt the copied package's actual export carrier rather than using a fake module."""
    import ast

    _write_fixture_repo(tmp_path)
    source = tmp_path / "src/scpn_control/__init__.py"
    original = source.read_text()
    assignment = next(
        node
        for node in ast.parse(original).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    assert assignment.end_lineno is not None
    lines = original.splitlines(keepends=True)
    lines[assignment.lineno - 1 : assignment.end_lineno] = [f"__all__ = {replacement}\n"]
    source.write_text("".join(lines))
    with pytest.raises(capability_tool.ManifestError, match="literal string sequence"):
        capability_tool.build_manifest(tmp_path)


def test_missing_optional_scan_root_is_only_presence_discovery(tmp_path: Path) -> None:
    """Absent scan roots are omitted rather than treated as available implementations."""
    _write_fixture_repo(tmp_path)
    phase = tmp_path / "src/scpn_control/phase"
    phase.rename(tmp_path / "phase.saved")
    report = capability_tool.build_manifest(tmp_path)
    assert "src/scpn_control/phase/kuramoto.py" not in report["python"]["source_modules"]
    assert "NeuroSymbolicController" in report["python"]["public_api_exports"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_json_renderer_refuses_nonfinite_metadata(actual_manifest: dict[str, Any], value: float) -> None:
    """A mutated supplied carrier cannot be serialized as nonstandard JSON numbers."""
    changed = copy.deepcopy(actual_manifest)
    changed["project"]["version"] = value
    with pytest.raises(ValueError, match="Out of range float values"):
        capability_tool.render_json(changed)


def test_check_reports_literal_drift_but_permits_platform_newlines(tmp_path: Path) -> None:
    """The documented normalized-text contract accepts CRLF and refuses semantic text drift."""
    _write_fixture_repo(tmp_path)
    capability_tool.write_outputs(tmp_path)
    output = tmp_path / "docs/_generated/capability_manifest.json"
    text = output.read_text()
    output.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert capability_tool.check_outputs(tmp_path) == []
    output.write_text(text.replace('"scpn-control"', '"different-control"', 1))
    assert "docs/_generated/capability_manifest.json is stale" in capability_tool.check_outputs(tmp_path)


def test_cli_configuration_and_output_errors_are_refusals(tmp_path: Path) -> None:
    """Real read/write failures reach the command boundary without a success or traceback."""
    assert _cli(tmp_path, "--check").returncode == 1
    _write_fixture_repo(tmp_path)
    output = tmp_path / "docs/_generated/capability_manifest.json"
    output.mkdir(parents=True)
    result = _cli(tmp_path)
    assert result.returncode == 1 and "refused" in result.stderr and "Traceback" not in result.stderr


def test_actual_inventory_native_example_executes() -> None:
    """Execute the discovery example on the real checkout without importing scanned implementations."""
    import doctest

    from tools import capability_manifest_inventory

    result = doctest.testmod(capability_manifest_inventory, raise_on_error=True)
    assert result.failed == 0 and result.attempted == 2


def test_chained_actual_export_declaration_keeps_its_labels(tmp_path: Path) -> None:
    """An ordinary assignment alias preserves the real package's declared export sequence."""
    _write_fixture_repo(tmp_path)
    expected = capability_tool.build_manifest(tmp_path)["python"]["public_api_exports"]
    package = tmp_path / "src/scpn_control/__init__.py"
    package.write_text(package.read_text().replace("__all__ = [", "export_alias = __all__ = [", 1))
    assert capability_tool.build_manifest(tmp_path)["python"]["public_api_exports"] == expected


def test_unpacking_cannot_substitute_for_a_package_export_sequence(tmp_path: Path) -> None:
    """Destructuring the real declaration cannot be interpreted as its original list."""
    _write_fixture_repo(tmp_path)
    package = tmp_path / "src/scpn_control/__init__.py"
    package.write_text(package.read_text().replace("__all__ = [", "(__all__,) = [", 1))
    with pytest.raises(capability_tool.ManifestError, match="public_api_exports"):
        capability_tool.build_manifest(tmp_path)


def test_local_export_assignment_cannot_admit_module_exports(tmp_path: Path) -> None:
    """Moving the actual export list into an existing function removes the package declaration."""
    import ast

    _write_fixture_repo(tmp_path)
    package = tmp_path / "src/scpn_control/__init__.py"
    original = package.read_text()
    tree = ast.parse(original)
    export = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    assert export.end_lineno is not None
    lines = original.splitlines(keepends=True)
    export_text = lines[export.lineno - 1 : export.end_lineno]
    del lines[export.lineno - 1 : export.end_lineno]
    moved = ast.parse("".join(lines))
    function = next(node for node in moved.body if isinstance(node, ast.FunctionDef))
    lines[function.body[0].lineno - 1 : function.body[0].lineno - 1] = ["    " + line for line in export_text]
    package.write_text("".join(lines))
    with pytest.raises(capability_tool.ManifestError, match="public_api_exports"):
        capability_tool.build_manifest(tmp_path)


def test_real_readme_fragment_accepts_no_leading_separator(tmp_path: Path) -> None:
    """The public extractor/checker preserves the actual generated fragment without a leading LF."""
    _write_fixture_repo(tmp_path)
    capability_tool.write_outputs(tmp_path)
    readme = tmp_path / "README.md"
    original = readme.read_text()
    changed = original.replace(capability_tool.README_START + "\n", capability_tool.README_START, 1)
    assert capability_tool.extract_readme_block(changed) == capability_tool.extract_readme_block(original)
    readme.write_text(changed)
    assert capability_tool.check_outputs(tmp_path) == []


def test_facade_native_renderer_exposes_the_owning_contracts() -> None:
    """Render the real public facade and retain the split owners' API docs in native output."""
    import html
    import pydoc
    import re

    rendered = pydoc.HTMLDoc().document(capability_tool)
    visible = " ".join(html.unescape(re.sub("<[^>]+>", " ", rendered)).split())
    for name in ["build_manifest", "validate_manifest", "render_json", "write_outputs", "check_outputs"]:
        assert name in visible
    assert "unqualified" in visible and "sequential" in visible and "LF/CRLF" in visible
