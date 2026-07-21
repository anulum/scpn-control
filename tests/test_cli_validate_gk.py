# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI validate gyrokinetic-reference tests
"""Tests for the ``validate`` gyrokinetic reference validators (geometry, species, parity, OOD)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_validate_gk_crosscode_requires_external_runs(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-gk-crosscode",
            "--evidence-root",
            str(tmp_path),
            "--require-external-runs",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no real external GK evidence reports found"


def test_validate_gk_geometry_reference_json_out(runner):
    reference_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "reference_data"
        / "gk_geometry"
        / "miller_reference_cases.json"
    )

    result = runner.invoke(
        main, ["validate-gk-geometry-reference", "--reference-path", str(reference_path), "--json-out"]
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["cases"] == 3


def test_validate_gk_species_reference_json_out(runner):
    reference_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "reference_data"
        / "gk_species"
        / "species_collision_reference_cases.json"
    )

    result = runner.invoke(
        main, ["validate-gk-species-reference", "--reference-path", str(reference_path), "--json-out"]
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["cases"] == 4


def test_validate_gk_geometry_reference_text_and_output_file(runner, tmp_path):
    reference_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "reference_data"
        / "gk_geometry"
        / "miller_reference_cases.json"
    )
    output_json = tmp_path / "reports" / "gk_geometry.json"

    result = runner.invoke(
        main,
        [
            "validate-gk-geometry-reference",
            "--reference-path",
            str(reference_path),
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 0
    assert "GK geometry reference: pass cases=3" in result.output
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "pass"


def test_validate_gk_geometry_reference_text_failure(runner, tmp_path):
    reference_path = tmp_path / "bad_geometry.json"
    reference_path.write_text(json.dumps({"schema_version": "1.0", "cases": []}), encoding="utf-8")

    result = runner.invoke(main, ["validate-gk-geometry-reference", "--reference-path", str(reference_path)])

    assert result.exit_code == 1
    assert "GK geometry reference: fail" in result.output
    assert "ERROR" in result.output


def test_validate_gk_species_reference_text_and_output_file(runner, tmp_path):
    reference_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "reference_data"
        / "gk_species"
        / "species_collision_reference_cases.json"
    )
    output_json = tmp_path / "reports" / "gk_species.json"

    result = runner.invoke(
        main,
        [
            "validate-gk-species-reference",
            "--reference-path",
            str(reference_path),
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 0
    assert "GK species reference: pass cases=4" in result.output
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "pass"


def test_validate_gk_species_reference_text_failure(runner, tmp_path):
    reference_path = tmp_path / "bad_species.json"
    reference_path.write_text(json.dumps({"schema_version": "1.0", "cases": []}), encoding="utf-8")

    result = runner.invoke(main, ["validate-gk-species-reference", "--reference-path", str(reference_path)])

    assert result.exit_code == 1
    assert "GK species reference: fail" in result.output
    assert "ERROR" in result.output


def test_validate_jax_gk_parity_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-jax-gk-parity",
            "--artifact-root",
            str(tmp_path),
            "--require-parity-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no JAX GK parity artifacts found"


def test_validate_jax_gk_parity_cli_requires_case_backend_pairs(runner):
    result = runner.invoke(
        main,
        [
            "validate-jax-gk-parity",
            "--require-parity-artifacts",
            "--require-cases",
            "cyclone_base_case,tem_kinetic_electron,stable_mode",
            "--require-backends",
            "cpu,gpu",
            "--json-out",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "pass"
    assert data["required_cases"] == ["cyclone_base_case", "stable_mode", "tem_kinetic_electron"]
    assert data["required_backends"] == ["cpu", "gpu"]


def test_validate_gk_ood_calibration_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-gk-ood-calibration",
            "--artifact-root",
            str(tmp_path),
            "--require-campaign-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no GK OOD calibration artifacts found"


def test_validate_gk_interface_artifacts_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-gk-interface-artifacts",
            "--artifact-root",
            str(tmp_path),
            "--require-interface-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no external GK interface artefacts found"
