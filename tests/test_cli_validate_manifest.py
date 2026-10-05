# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI validate manifest / data-manifest / traceability tests
"""Tests for the ``validate`` manifest, data-manifest, and physics-traceability report gates."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    """Provide an actual Click runner for registered command execution."""
    return CliRunner()


def test_validate_manifest_json_out(runner: CliRunner, tmp_path: Path) -> None:
    """Validate manifest json out."""
    manifest_path = tmp_path / "real_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "diii-d-163303-control-replay",
                "machine": "DIII-D",
                "shot": "163303",
                "synthetic": False,
                "source": {
                    "kind": "mdsplus",
                    "uri": "mdsplus://DIII-D/163303",
                    "access": "facility-approved",
                },
                "retrieved_at": "2026-05-18T01:20:00Z",
                "checksum_sha256": "b" * 64,
                "licence": "facility data policy",
                "signals": [
                    {
                        "name": "plasma_current",
                        "path": "\\\\IP",
                        "units": "A",
                        "timebase": "s",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["validate-manifest", str(manifest_path), "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {
        "dataset_id": "diii-d-163303-control-replay",
        "kind": "real",
        "machine": "DIII-D",
        "shot": "163303",
        "signals": 1,
        "source_kind": "mdsplus",
        "status": "pass",
    }


def test_validate_manifest_rejects_mock_as_real(runner: CliRunner, tmp_path: Path) -> None:
    """Validate manifest rejects mock as real."""
    manifest_path = tmp_path / "invalid_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "bad-real-claim",
                "machine": "DIII-D",
                "shot": "999999",
                "synthetic": False,
                "source": {
                    "kind": "mock",
                    "uri": "tests/mock_diiid.py",
                    "access": "repository fixture",
                },
                "retrieved_at": "2026-05-18T01:20:00Z",
                "checksum_sha256": "c" * 64,
                "licence": "repository fixture",
                "signals": [
                    {
                        "name": "normalised_beta",
                        "path": "beta_N",
                        "units": "1",
                        "timebase": "s",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["validate-manifest", str(manifest_path), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert "synthetic or mock source" in data["error"]


def test_validate_manifest_verifies_repository_artifact(runner: CliRunner) -> None:
    """Validate manifest verifies repository artifact."""
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "reference_data"
        / "diiid"
        / "manifests"
        / "diiid_hmode_1p5MA.geqdsk.manifest.json"
    )

    result = runner.invoke(main, ["validate-manifest", str(manifest_path), "--verify-artifact", "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["artifact_verified"] is True


def test_validate_manifest_text_success(runner: CliRunner, tmp_path: Path) -> None:
    """Validate manifest text success."""
    manifest_path = tmp_path / "real_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "diii-d-163303-text",
                "machine": "DIII-D",
                "shot": "163303",
                "synthetic": False,
                "source": {
                    "kind": "mdsplus",
                    "uri": "mdsplus://DIII-D/163303",
                    "access": "facility-approved",
                },
                "retrieved_at": "2026-05-18T01:20:00Z",
                "checksum_sha256": "d" * 64,
                "licence": "facility data policy",
                "signals": [
                    {
                        "name": "plasma_current",
                        "path": "\\\\IP",
                        "units": "A",
                        "timebase": "s",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(main, ["validate-manifest", str(manifest_path)])

    assert result.exit_code == 0
    assert "Dataset: diii-d-163303-text" in result.output
    assert "Kind: real" in result.output
    assert "Source: mdsplus" in result.output
    assert "Status: pass" in result.output


def test_validate_manifest_text_failure(runner: CliRunner, tmp_path: Path) -> None:
    """Validate manifest text failure."""
    manifest_path = tmp_path / "invalid_manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    result = runner.invoke(main, ["validate-manifest", str(manifest_path)])

    assert result.exit_code == 1
    assert "Status: fail" in result.output
    assert "manifest missing required key" in result.output


def test_validate_data_manifests_json_out(runner: CliRunner) -> None:
    """Validate data manifests json out."""
    root = Path(__file__).resolve().parents[1] / "validation" / "reference_data"

    result = runner.invoke(main, ["validate-data-manifests", "--root", str(root), "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["total"] >= 3
    assert data["artifact_coverage"]["covered"] == 21
    assert data["acquisition_specs"]["total"] >= 1


def test_validate_data_manifests_text_and_output_file(runner: CliRunner, tmp_path: Path) -> None:
    """Validate data manifests text and output file."""
    root = Path(__file__).resolve().parents[1] / "validation" / "reference_data"
    output_json = tmp_path / "reports" / "data_manifests.json"

    result = runner.invoke(
        main,
        [
            "validate-data-manifests",
            "--root",
            str(root),
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 0
    assert "Data manifests: pass" in result.output
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["status"] == "pass"


def test_validate_data_manifests_text_reports_errors(runner: CliRunner, tmp_path: Path) -> None:
    """Validate data manifests text reports errors."""
    result = runner.invoke(main, ["validate-data-manifests", "--root", str(tmp_path)])

    assert result.exit_code == 1
    assert "Data manifests: fail" in result.output
    assert "ERROR" in result.output
    assert "no data manifests found" in result.output


def test_validate_data_manifests_reports_failures(runner: CliRunner, tmp_path: Path) -> None:
    """Validate data manifests reports failures."""
    result = runner.invoke(main, ["validate-data-manifests", "--root", str(tmp_path), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no data manifests found"


def test_validate_data_manifests_can_require_real_acquisition(runner: CliRunner) -> None:
    """Validate data manifests can require real acquisition."""
    root = Path(__file__).resolve().parents[1] / "validation" / "reference_data"

    result = runner.invoke(
        main,
        [
            "validate-data-manifests",
            "--root",
            str(root),
            "--require-real-acquisition",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["acquisition_specs"]["pending"] >= 1
    assert any(error["error"] == "missing acquired MDSplus manifest" for error in data["errors"])


def test_validate_physics_traceability_json_out(runner: CliRunner) -> None:
    """Validate physics traceability json out."""
    registry = Path(__file__).resolve().parents[1] / "validation" / "physics_traceability.json"

    result = runner.invoke(main, ["validate-physics-traceability", "--registry", str(registry), "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["open_fidelity_gaps"] >= 5
    assert data["public_claim_blocked"] >= 5


def test_validate_physics_traceability_reports_failures(runner: CliRunner, tmp_path: Path) -> None:
    """Validate physics traceability reports failures."""
    registry = tmp_path / "physics_traceability.json"
    registry.write_text(json.dumps({"schema_version": "1.0", "entries": []}), encoding="utf-8")

    result = runner.invoke(main, ["validate-physics-traceability", "--registry", str(registry), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    fields = {error["field"] for error in data["errors"]}
    assert "entries" in fields
    assert "spdx_license_id" in fields


def test_validate_physics_traceability_text_and_output_file(runner: CliRunner, tmp_path: Path) -> None:
    """Validate physics traceability text and output file."""
    registry = Path(__file__).resolve().parents[1] / "validation" / "physics_traceability.json"
    output_json = tmp_path / "reports" / "physics_traceability.json"

    result = runner.invoke(
        main,
        [
            "validate-physics-traceability",
            "--registry",
            str(registry),
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 0
    assert "Physics traceability: pass" in result.output
    assert "external_validation_trackers=8" in result.output
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["status"] == "pass"


def test_validate_physics_traceability_text_reports_errors(runner: CliRunner, tmp_path: Path) -> None:
    """Validate physics traceability text reports errors."""
    registry = tmp_path / "physics_traceability.json"
    registry.write_text(json.dumps({"schema_version": "1.0", "entries": []}), encoding="utf-8")

    result = runner.invoke(main, ["validate-physics-traceability", "--registry", str(registry)])

    assert result.exit_code == 1
    assert "Physics traceability: fail" in result.output
    assert "ERROR" in result.output
    assert ".entries:" in result.output


@pytest.mark.parametrize("kind", ["json", "utf8", "decoder-depth"])
def test_actual_registered_manifest_cli_reports_decode_refusal(runner: CliRunner, tmp_path: Path, kind: str) -> None:
    """The registered command translates actual JSON/UTF-8/depth errors into FAIL."""
    depth = 10_000
    contents = b"\xff" if kind == "utf8" else b"{"
    if kind == "decoder-depth":
        contents = b"[" * depth + b"]" * depth
    path = tmp_path / "m.json"
    path.write_bytes(contents)
    result = runner.invoke(main, ["validate-manifest", str(path), "--json-out"])
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["status"] == "fail" and "cannot load manifest" in report["error"]


def test_actual_registered_manifest_cli_does_not_claim_remote_bytes_verified(runner: CliRunner, tmp_path: Path) -> None:
    """Selecting verification does not authenticate remote provenance without local artifacts."""
    path = tmp_path / "m.json"
    payload = {
        "schema_version": "1.0",
        "dataset_id": "test-remote",
        "machine": "DIII-D",
        "shot": "1",
        "synthetic": False,
        "source": {"kind": "mdsplus", "uri": "mdsplus://DIII-D/1", "access": "test"},
        "retrieved_at": "declared only",
        "licence": "test declaration",
        "checksum_sha256": "a" * 64,
        "signals": [{"name": "ip", "path": "\\IP", "units": "A", "timebase": "s"}],
    }
    path.write_text(json.dumps(payload))
    result = runner.invoke(main, ["validate-manifest", str(path), "--verify-artifact", "--json-out"])
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["status"] == "pass" and report["artifact_verified"] is False


def test_actual_registered_directory_cli_reports_unwritable_output(runner: CliRunner, tmp_path: Path) -> None:
    """An actual parent-file obstruction produces JSON FAIL through the registered command."""
    parent = tmp_path / "parent"
    parent.write_bytes(b"blocked parent")
    result = runner.invoke(
        main, ["validate-data-manifests", "--output-json", str(parent / "report.json"), "--json-out"]
    )
    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["status"] == "fail" and "cannot write report" in report["errors"][-1]["error"]


@pytest.mark.parametrize("synthetic", [True, False])
def test_actual_registered_cli_verifies_single_local_source(runner: CliRunner, tmp_path: Path, synthetic: bool) -> None:
    """The verification observation is true when actual selected local bytes are checked."""
    artifact = tmp_path / "archive.bin"
    content = b"CLI custody test, no facility measurement"
    artifact.write_bytes(content)
    payload = {
        "schema_version": "1.0",
        "dataset_id": "test-local",
        "machine": "DIII-D",
        "shot": "1",
        "synthetic": synthetic,
        "source": {"kind": "synthetic" if synthetic else "local_archive", "uri": artifact.name, "access": "test"},
        "retrieved_at": "declared only",
        "licence": "test declaration",
        "checksum_sha256": sha256(content).hexdigest(),
        "synthetic_generator": "test",
        "synthetic_seed": 1,
        "signals": [{"name": "ip", "path": "ip", "units": "A", "timebase": "s"}],
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload))
    result = runner.invoke(main, ["validate-manifest", str(path), "--verify-artifact", "--json-out"])
    assert result.exit_code == 0
    assert json.loads(result.output)["artifact_verified"] is True
