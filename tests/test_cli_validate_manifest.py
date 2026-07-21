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
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_validate_manifest_json_out(runner, tmp_path):
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


def test_validate_manifest_rejects_mock_as_real(runner, tmp_path):
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


def test_validate_manifest_verifies_repository_artifact(runner):
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


def test_validate_manifest_text_success(runner, tmp_path):
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


def test_validate_manifest_text_failure(runner, tmp_path):
    manifest_path = tmp_path / "invalid_manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    result = runner.invoke(main, ["validate-manifest", str(manifest_path)])

    assert result.exit_code == 1
    assert "Status: fail" in result.output
    assert "manifest missing required key" in result.output


def test_validate_data_manifests_json_out(runner):
    root = Path(__file__).resolve().parents[1] / "validation" / "reference_data"

    result = runner.invoke(main, ["validate-data-manifests", "--root", str(root), "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["total"] >= 3
    assert data["artifact_coverage"]["covered"] == 21
    assert data["acquisition_specs"]["total"] >= 1


def test_validate_data_manifests_text_and_output_file(runner, tmp_path):
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


def test_validate_data_manifests_text_reports_errors(runner, tmp_path):
    result = runner.invoke(main, ["validate-data-manifests", "--root", str(tmp_path)])

    assert result.exit_code == 1
    assert "Data manifests: fail" in result.output
    assert "ERROR" in result.output
    assert "no data manifests found" in result.output


def test_validate_data_manifests_reports_failures(runner, tmp_path):
    result = runner.invoke(main, ["validate-data-manifests", "--root", str(tmp_path), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no data manifests found"


def test_validate_data_manifests_can_require_real_acquisition(runner):
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


def test_validate_physics_traceability_json_out(runner):
    registry = Path(__file__).resolve().parents[1] / "validation" / "physics_traceability.json"

    result = runner.invoke(main, ["validate-physics-traceability", "--registry", str(registry), "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["open_fidelity_gaps"] >= 5
    assert data["public_claim_blocked"] >= 5


def test_validate_physics_traceability_reports_failures(runner, tmp_path):
    registry = tmp_path / "physics_traceability.json"
    registry.write_text(json.dumps({"schema_version": "1.0", "entries": []}), encoding="utf-8")

    result = runner.invoke(main, ["validate-physics-traceability", "--registry", str(registry), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    fields = {error["field"] for error in data["errors"]}
    assert "entries" in fields
    assert "spdx_license_id" in fields


def test_validate_physics_traceability_text_and_output_file(runner, tmp_path):
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


def test_validate_physics_traceability_text_reports_errors(runner, tmp_path):
    registry = tmp_path / "physics_traceability.json"
    registry.write_text(json.dumps({"schema_version": "1.0", "entries": []}), encoding="utf-8")

    result = runner.invoke(main, ["validate-physics-traceability", "--registry", str(registry)])

    assert result.exit_code == 1
    assert "Physics traceability: fail" in result.output
    assert "ERROR" in result.output
    assert ".entries:" in result.output
