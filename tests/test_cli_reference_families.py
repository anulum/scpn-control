# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Reference CLI family integration

"""Exercise actual root CSV, compatibility and missing-evidence behavior after moving callbacks."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from scpn_control import cli_reference_engineering as engineering
from scpn_control import cli_reference_equilibrium as equilibrium
from scpn_control import cli_reference_instabilities as instabilities
from scpn_control import cli_reference_paths as paths
from scpn_control import cli_reference_static_mu as static_mu
from scpn_control import cli_reference_tracking as tracking
from scpn_control import cli_reference_transport as transport
from scpn_control.cli import main as root_cli


@pytest.mark.parametrize(
    ("cases", "backends", "expected_cases", "expected_backends"),
    [
        (" , ", " , ", [], []),
        (" stable_mode , , cyclone_base_case ", " gpu, cpu ", ["cyclone_base_case", "stable_mode"], ["cpu", "gpu"]),
    ],
)
def test_actual_jax_cli_csv_requirements(
    tmp_path: Path, cases: str, backends: str, expected_cases: list[str], expected_backends: list[str]
) -> None:
    """The moved root command trims real CSV options and keeps missing evidence fail-closed."""
    result = CliRunner().invoke(
        root_cli,
        [
            "validate-jax-gk-parity",
            "--artifact-root",
            str(tmp_path / "absent"),
            "--require-parity-artifacts",
            "--require-cases",
            cases,
            "--require-backends",
            backends,
            "--json-out",
        ],
    )
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["status"] == "fail" and report["parity_artifacts"] == 0
    assert report["required_cases"] == expected_cases
    assert report["required_backends"] == expected_backends
    assert report["errors"]


@pytest.mark.parametrize("command", ["validate-static-mu-analysis-reference", "validate-mu-synthesis-reference"])
def test_actual_static_canonical_and_hidden_alias(tmp_path: Path, command: str) -> None:
    """Canonical and hidden legacy commands retain real optional/required report and help behavior."""
    output = tmp_path / "reports" / "report.json"
    argv = [command, "--artifact-root", str(tmp_path / "absent"), "--output-json", str(output), "--json-out"]
    result = CliRunner().invoke(root_cli, argv)
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["status"] == "pass" and report["reference_artifacts"] == 0
    assert json.loads(output.read_text()) == report
    result = CliRunner().invoke(root_cli, [*argv, "--require-reference-artifacts"])
    assert result.exit_code == 1 and json.loads(result.stdout)["status"] == "fail"
    result = CliRunner().invoke(root_cli, [command, "--help"])
    assert result.exit_code == 0 and "--require-reference-artifacts" in result.stdout
    root_help = CliRunner().invoke(root_cli, ["--help"])
    assert "validate-static-mu-analysis-reference" in root_help.stdout
    assert "validate-mu-synthesis-reference" not in root_help.stdout


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("json_out", [False, True])
@pytest.mark.parametrize("persist", [False, True])
def test_current_drive_root_report_modes(tmp_path: Path, required: bool, json_out: bool, persist: bool) -> None:
    """Optional/required real root inspection preserves stdout, findings and persisted diagnostics."""
    absent = tmp_path / "absent"
    output = tmp_path / "reports" / "inspection.json"
    argv = ["validate-current-drive-reference", "--artifact-root", str(absent)]
    if required:
        argv.append("--require-reference-artifacts")
    if json_out:
        argv.append("--json-out")
    if persist:
        argv.extend(["--output-json", str(output)])
    result = CliRunner().invoke(root_cli, argv)
    status = "fail" if required else "pass"
    assert result.exit_code == int(required)
    if json_out:
        report = json.loads(result.stdout)
        assert report["status"] == status and report["reference_artifacts"] == 0
        assert result.stderr == ""
        if persist:
            assert json.loads(output.read_text()) == report
    else:
        assert result.stdout == f"Current-drive reference: {status} reference_artifacts=0\n"
        assert bool(result.stderr) is required
        if required:
            assert "no current-drive reference artifacts found" in result.stderr
    assert output.exists() is persist and not absent.exists()
    if persist:
        saved = json.loads(output.read_text())
        assert saved["status"] == status and saved["reference_artifacts"] == 0
        assert saved["require_reference_artifacts"] is required


def test_current_drive_root_fixed_write_refusal(tmp_path: Path) -> None:
    """A real blocked output parent preserves bytes and returns the original fixed Click write refusal."""
    blocker = tmp_path / "PRIVATE_OUTPUT_SENTINEL"
    blocker.write_bytes(b"preserved")
    result = CliRunner().invoke(
        root_cli,
        [
            "validate-current-drive-reference",
            "--artifact-root",
            str(tmp_path / "absent"),
            "--output-json",
            str(blocker / "report.json"),
            "--json-out",
        ],
    )
    assert result.exit_code == 1 and result.stdout == ""
    assert result.stderr == "Error: could not write current-drive reference report\n"
    assert "PRIVATE_OUTPUT_SENTINEL" not in result.stderr and blocker.read_bytes() == b"preserved"


def _assert_exported_and_root_report(tmp_path: Path, command: click.Command, required: bool) -> None:
    """Invoke the exported command and its real root registration with the same persisted inspection."""
    assert command.name is not None
    absent = tmp_path / "absent"
    output = tmp_path / "reports" / "inspection.json"
    argv = ["--artifact-root", str(absent), "--output-json", str(output), "--json-out"]
    if required:
        argv.append("--require-reference-artifacts")
    direct = CliRunner().invoke(command, argv)
    root = CliRunner().invoke(root_cli, [command.name, *argv])
    assert direct.exit_code == root.exit_code == int(required)
    assert direct.stdout == root.stdout
    assert direct.stderr == root.stderr == ""
    report = json.loads(root.stdout)
    assert report["status"] == ("fail" if required else "pass")
    assert report["reference_artifacts"] == 0
    assert bool(report["errors"]) is required
    assert json.loads(output.read_text()) == report
    assert not absent.exists()


@pytest.mark.parametrize("required", [False, True])
def test_engineering_export_is_the_exercised_root_command(tmp_path: Path, required: bool) -> None:
    """The registered engineering object preserves direct/root inspection and persistence behavior."""
    assert root_cli.commands["validate-current-drive-reference"] is engineering.validate_current_drive_reference_command
    _assert_exported_and_root_report(tmp_path, engineering.validate_current_drive_reference_command, required)


@pytest.mark.parametrize("required", [False, True])
def test_equilibrium_export_is_the_exercised_root_command(tmp_path: Path, required: bool) -> None:
    """The registered equilibrium object preserves direct/root inspection and persistence behavior."""
    assert (
        root_cli.commands["validate-neural-equilibrium-reference"]
        is equilibrium.validate_neural_equilibrium_reference_command
    )
    _assert_exported_and_root_report(tmp_path, equilibrium.validate_neural_equilibrium_reference_command, required)


@pytest.mark.parametrize("required", [False, True])
def test_instabilities_export_is_the_exercised_root_command(tmp_path: Path, required: bool) -> None:
    """The registered instability object preserves direct/root inspection and persistence behavior."""
    assert root_cli.commands["validate-elm-reference"] is instabilities.validate_elm_reference_command
    _assert_exported_and_root_report(tmp_path, instabilities.validate_elm_reference_command, required)


@pytest.mark.parametrize("required", [False, True])
def test_static_mu_export_is_the_exercised_root_command(tmp_path: Path, required: bool) -> None:
    """The registered static-analysis object preserves direct/root inspection and persistence behavior."""
    assert (
        root_cli.commands["validate-static-mu-analysis-reference"]
        is static_mu.validate_static_mu_analysis_reference_command
    )
    _assert_exported_and_root_report(tmp_path, static_mu.validate_static_mu_analysis_reference_command, required)


@pytest.mark.parametrize("required", [False, True])
def test_tracking_export_is_the_exercised_root_command(tmp_path: Path, required: bool) -> None:
    """The registered tracking object preserves direct/root inspection and persistence behavior."""
    assert root_cli.commands["validate-digital-twin-reference"] is tracking.validate_digital_twin_reference_command
    _assert_exported_and_root_report(tmp_path, tracking.validate_digital_twin_reference_command, required)


@pytest.mark.parametrize("required", [False, True])
def test_transport_export_is_the_exercised_root_command(tmp_path: Path, required: bool) -> None:
    """The registered transport object preserves direct/root inspection and persistence behavior."""
    assert root_cli.commands["validate-blob-transport-reference"] is transport.validate_blob_transport_reference_command
    _assert_exported_and_root_report(tmp_path, transport.validate_blob_transport_reference_command, required)


@pytest.mark.parametrize(
    "name",
    ["validate-current-drive-reference", "validate-density-reference", "validate-neural-equilibrium-reference"],
)
@pytest.mark.parametrize("invalid_kind", ["nul", "directory"])
def test_registered_shared_output_converter_refuses_before_callback(
    tmp_path: Path, name: str, invalid_kind: str
) -> None:
    """Registered public options use the shared path converter and preserve files on actual usage refusals."""
    command = root_cli.commands[name]
    option = next(parameter for parameter in command.params if parameter.name == "output_json")
    assert type(option.type).__module__ == paths.__name__
    assert isinstance(option.type, click.Path)
    marker = tmp_path / "preserved.data"
    marker.write_bytes(b"preserved")
    output = str(tmp_path / "report\0.json") if invalid_kind == "nul" else str(tmp_path)
    result = CliRunner().invoke(
        root_cli,
        [name, "--artifact-root", str(tmp_path / "absent"), "--output-json", output, "--json-out"],
    )
    assert result.exit_code == 2 and result.stdout == ""
    assert "Invalid value for '--output-json'" in result.stderr
    assert ("report output path must not contain NUL" if invalid_kind == "nul" else "is a directory") in result.stderr
    assert marker.read_bytes() == b"preserved"
    assert list(tmp_path.iterdir()) == [marker]
