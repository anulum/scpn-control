# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public report writer input custody tests

"""Exercise real path aliases and public report commands over copied existing inputs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_control.cli import main as root_cli
from validation.generate_physics_traceability_report import main as markdown_main
from validation.report_output_paths import checked_report_destination, manifest_report_inputs
from validation.validate_data_manifests import ROOT
from validation.validate_data_manifests import main as manifest_main
from validation.validate_physics_traceability import main as physics_main

_ALIASES = ["direct", "normalised", "symlink", "hardlink", "parent_symlink"]


@pytest.fixture
def copied_registry(tmp_path: Path) -> Path:
    """Copy the actual canonical metadata registry without changing its declarations."""
    source = tmp_path / "registry.json"
    shutil.copy2(ROOT / "validation/physics_traceability.json", source)
    return source


@pytest.fixture
def copied_reference_tree(tmp_path: Path) -> Path:
    """Copy existing synthetic DIII-D data and manifests; no new scientific reference is manufactured."""
    root = tmp_path / "reference_data"
    shutil.copytree(ROOT / "validation/reference_data/diiid", root / "diiid")
    return root


def _alias(source: Path, kind: str) -> Path:
    """Return a real spelling, symbolic link, hard link or linked-parent alias of the copied input."""
    if kind == "direct":
        return source
    if kind == "normalised":
        return source.parent / ".." / source.parent.name / source.name
    if kind == "parent_symlink":
        parent = source.parent.with_name(source.parent.name + "_linked_parent")
        parent.symlink_to(source.parent, target_is_directory=True)
        return parent / source.name
    target = source.parent / "destination"
    if kind == "symlink":
        target.symlink_to(source)
    else:
        os.link(source, target)
    return target


@pytest.mark.parametrize("kind", _ALIASES)
def test_public_path_check_refuses_real_aliases(copied_registry: Path, kind: str) -> None:
    """The public predicate refuses each actual filesystem alias before altering copied input bytes."""
    before = copied_registry.read_bytes()
    destination = _alias(copied_registry, kind)
    with pytest.raises(ValueError, match="report output aliases a selected input"):
        checked_report_destination(destination, inputs=[copied_registry])
    assert copied_registry.read_bytes() == before


def test_public_path_check_preserves_unrelated_destinations(tmp_path: Path) -> None:
    """The predicate writes nothing and admits unrelated existing and absent outputs and inputs."""
    source = tmp_path / "input.json"
    source.write_text("input")
    target = tmp_path / "report.json"
    target.write_text("previous output")
    missing = tmp_path / "missing.json"
    assert checked_report_destination(target, inputs=[missing, source]) == target
    assert target.read_text() == "previous output"
    assert checked_report_destination(missing, inputs=[source]) == missing
    assert checked_report_destination(target, inputs=[]) == target
    with pytest.raises(ValueError, match="aliases"):
        checked_report_destination(missing, inputs=[missing])
    with pytest.raises(ValueError, match="aliases"):
        checked_report_destination(tmp_path, inputs=[tmp_path])


def test_public_path_check_refuses_unresolvable_paths(tmp_path: Path) -> None:
    """Actual NUL and self-looping output paths propagate resolution refusal without any write."""
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    with pytest.raises((RuntimeError, OSError)):
        checked_report_destination(loop, inputs=[])
    with pytest.raises(ValueError):
        checked_report_destination("bad\0path", inputs=[])


def test_manifest_input_discovery_protects_local_artifacts_and_invalid_metadata(
    copied_reference_tree: Path,
) -> None:
    """Discover actual local artifact inputs, retaining malformed manifest paths without remote retrieval."""
    root = copied_reference_tree
    coilset = root / "diiid/freegs/diiid_freegs_1p5MA_coilset.json"
    invalid = root / "diiid/manifests/mock_diiid_ci.manifest.json"
    invalid.write_text("{")
    remote = root / "diiid/freegs/manifests/diiid_freegs_1p5MA.manifest.json"
    payload = json.loads(remote.read_text())
    payload["artifacts"].append({"uri": "https://example.invalid/no-retrieval", "checksum_sha256": "d" * 64})
    remote.write_text(json.dumps(payload))
    inputs = manifest_report_inputs(root)
    assert root in inputs and invalid in inputs and remote in inputs
    assert coilset in inputs
    assert all(path.resolve().is_relative_to(root.resolve()) for path in inputs)
    with pytest.raises(ValueError, match="aliases"):
        checked_report_destination(coilset, inputs=inputs)


@pytest.mark.parametrize("module", ["validate_physics_traceability.py", "generate_physics_traceability_report.py"])
@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("kind", _ALIASES)
def test_actual_registry_scripts_refuse_input_aliases(
    copied_registry: Path, module: str, cold: bool, kind: str
) -> None:
    """Actual normal and stdlib-only scripts preserve the copied registry for every output alias."""
    source = copied_registry
    before = source.read_bytes()
    destination = _alias(source, kind)
    option = "--output-md" if module.startswith("generate") else "--output-json"
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if cold else []),
            str(ROOT / "validation" / module),
            "--registry",
            str(source),
            option,
            str(destination),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "report output aliases a selected input" in result.stderr
    assert source.read_bytes() == before
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("kind", _ALIASES)
def test_actual_valid_registry_scripts_and_root_callback_refuse_alias(
    copied_registry: Path, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Valid-registry public APIs and the registered command refuse aliases without input loss."""
    source = copied_registry
    before = source.read_bytes()
    destination = _alias(source, kind)
    assert physics_main(["--registry", str(source), "--output-json", str(destination), "--json-out"]) == 1
    assert json.loads(capsys.readouterr().out)["errors"][-1]["field"] == "output_json"
    assert markdown_main(["--registry", str(source), "--output-md", str(destination)]) == 1
    assert "report output aliases a selected input" in capsys.readouterr().err
    result = CliRunner().invoke(
        root_cli,
        ["validate-physics-traceability", "--registry", str(source), "--output-json", str(destination), "--json-out"],
    )
    assert result.exit_code == 1
    assert result.stderr == "Error: could not write physics traceability report\n"
    assert source.read_bytes() == before


@pytest.mark.parametrize("kind", _ALIASES)
def test_manifest_api_and_registered_command_refuse_selected_input_aliases(
    copied_reference_tree: Path, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Even FAIL reports cannot overwrite selected original manifest bytes through either public entry."""
    root = copied_reference_tree
    source = root / "diiid/manifests/mock_diiid_ci.manifest.json"
    before = source.read_bytes()
    destination = _alias(source, kind)
    args = ["--root", str(root), "--output-json", str(destination), "--json-out"]
    assert manifest_main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert "report output aliases a selected input" in report["errors"][-1]["error"]
    result = CliRunner().invoke(root_cli, ["validate-data-manifests", *args])
    assert result.exit_code == 1
    assert "report output aliases a selected input" in json.loads(result.stdout)["errors"][-1]["error"]
    assert source.read_bytes() == before


def test_manifest_command_refuses_declared_non_pattern_artifact_alias(
    copied_reference_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Protect the actual FreeGS coilset JSON referenced by a manifest, beyond required GEQDSK/NPZ glob inputs."""
    root = copied_reference_tree
    source = root / "diiid/freegs/diiid_freegs_1p5MA_coilset.json"
    before = source.read_bytes()
    assert manifest_main(["--root", str(root), "--output-json", str(source), "--json-out"]) == 1
    assert "aliases a selected input" in json.loads(capsys.readouterr().out)["errors"][-1]["error"]
    assert source.read_bytes() == before


def test_public_writers_preserve_unrelated_output_replacement(
    copied_registry: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Normal writers still emit actual sorted UTF-8 JSON or generated Markdown at unrelated existing targets."""
    output = tmp_path / "report.json"
    output.write_text("previous output")
    assert physics_main(["--registry", str(copied_registry), "--output-json", str(output)]) == 0
    capsys.readouterr()
    report = json.loads(output.read_text())
    assert report["status"] == "pass" and output.read_bytes().endswith(b"\n")
    assert markdown_main(["--registry", str(copied_registry), "--output-md", str(output)]) == 0
    assert "# Physics" in output.read_text()
    result = CliRunner().invoke(
        root_cli,
        [
            "validate-physics-traceability",
            "--registry",
            str(copied_registry),
            "--output-json",
            str(output),
            "--json-out",
        ],
    )
    assert result.exit_code == 0 and json.loads(output.read_text()) == json.loads(result.stdout)
