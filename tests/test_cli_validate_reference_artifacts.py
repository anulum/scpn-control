# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI validate-*-reference artifact-requirement tests
"""Tests that each per-domain ``validate-*-reference`` command fails closed without its artifacts."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_validate_blob_transport_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-blob-transport-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no blob transport reference artifacts found"


def test_validate_elm_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-elm-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no ELM reference artifacts found"


def test_validate_eped_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-eped-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no EPED reference artifacts found"


def test_validate_marfe_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-marfe-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no MARFE reference artifacts found"


def test_validate_ntm_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-ntm-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no NTM reference artifacts found"


def test_validate_neural_equilibrium_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-neural-equilibrium-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no neural equilibrium reference artefacts found"


def test_validate_neural_transport_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-neural-transport-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no neural transport reference artifacts found"


def test_validate_neural_turbulence_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-neural-turbulence-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no neural turbulence reference artifacts found"


def test_validate_orbit_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-orbit-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no orbit reference artifacts found"


def test_validate_uncertainty_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-uncertainty-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no uncertainty reference artifacts found"


def test_validate_vmec_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-vmec-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no VMEC reference artifacts found"


def test_validate_rzip_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-rzip-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no RZIP reference artifacts found"


def test_validate_density_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-density-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no density reference artifacts found"


def test_validate_disruption_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-disruption-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no disruption reference artifacts found"


def test_validate_digital_twin_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-digital-twin-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no digital twin reference artifacts found"


def test_validate_soc_reference_requires_artifacts(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate-soc-reference",
            "--artifact-root",
            str(tmp_path),
            "--require-reference-artifacts",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["errors"][0]["error"] == "no SOC reference artifacts found"
