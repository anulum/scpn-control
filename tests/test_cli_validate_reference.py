# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI validate-*-reference command tests
"""Parametrized tests for the validate-*-reference / validate-gk CLI commands."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize(
    ("command", "root_option", "require_flag", "summary", "count_key", "expected_error"),
    [
        (
            "validate-gk-crosscode",
            "--evidence-root",
            "--require-external-runs",
            "GK cross-code evidence: fail",
            "external_runs",
            "no real external GK evidence reports found",
        ),
        (
            "validate-jax-gk-parity",
            "--artifact-root",
            "--require-parity-artifacts",
            "JAX GK parity: fail",
            "parity_artifacts",
            "no JAX GK parity artifacts found",
        ),
        (
            "validate-gk-ood-calibration",
            "--artifact-root",
            "--require-campaign-artifacts",
            "GK OOD calibration: fail",
            "campaign_artifacts",
            "no GK OOD calibration artifacts found",
        ),
        (
            "validate-gk-interface-artifacts",
            "--artifact-root",
            "--require-interface-artifacts",
            "GK interface artifacts: fail",
            "interface_artifacts",
            "no external GK interface artefacts found",
        ),
        (
            "validate-blob-transport-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "Blob transport reference: fail",
            "reference_artifacts",
            "no blob transport reference artifacts found",
        ),
        (
            "validate-elm-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "ELM reference: fail",
            "reference_artifacts",
            "no ELM reference artifacts found",
        ),
        (
            "validate-eped-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "EPED reference: fail",
            "reference_artifacts",
            "no EPED reference artifacts found",
        ),
        (
            "validate-marfe-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "MARFE reference: fail",
            "reference_artifacts",
            "no MARFE reference artifacts found",
        ),
        (
            "validate-ntm-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "NTM reference: fail",
            "reference_artifacts",
            "no NTM reference artifacts found",
        ),
        (
            "validate-neural-equilibrium-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "Neural equilibrium reference: fail",
            "reference_artifacts",
            "no neural equilibrium reference artefacts found",
        ),
        (
            "validate-neural-transport-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "Neural transport reference: fail",
            "reference_artifacts",
            "no neural transport reference artifacts found",
        ),
        (
            "validate-neural-turbulence-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "Neural turbulence reference: fail",
            "reference_artifacts",
            "no neural turbulence reference artifacts found",
        ),
        (
            "validate-orbit-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "Orbit reference: fail",
            "reference_artifacts",
            "no orbit reference artifacts found",
        ),
        (
            "validate-uncertainty-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "Uncertainty reference: fail",
            "reference_artifacts",
            "no uncertainty reference artifacts found",
        ),
        (
            "validate-vmec-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "VMEC reference: fail",
            "reference_artifacts",
            "no VMEC reference artifacts found",
        ),
        (
            "validate-rzip-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "RZIP reference: fail",
            "reference_artifacts",
            "no RZIP reference artifacts found",
        ),
        (
            "validate-density-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "Density reference: fail",
            "reference_artifacts",
            "no density reference artifacts found",
        ),
        (
            "validate-disruption-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "Disruption reference: fail",
            "reference_artifacts",
            "no disruption reference artifacts found",
        ),
        (
            "validate-digital-twin-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "Digital twin reference: fail",
            "reference_artifacts",
            "no digital twin reference artifacts found",
        ),
        (
            "validate-soc-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "SOC reference: fail",
            "reference_artifacts",
            "no SOC reference artifacts found",
        ),
    ],
)
def test_gk_validation_text_error_paths(
    runner,
    tmp_path,
    command,
    root_option,
    require_flag,
    summary,
    count_key,
    expected_error,
):
    result = runner.invoke(main, [command, root_option, str(tmp_path), require_flag])

    assert result.exit_code == 1
    assert summary in result.output
    assert f"{count_key}=0" in result.output
    assert expected_error in result.output


@pytest.mark.parametrize(
    ("command", "root_option", "require_flag", "output_name"),
    [
        ("validate-gk-crosscode", "--evidence-root", "--require-external-runs", "crosscode.json"),
        ("validate-jax-gk-parity", "--artifact-root", "--require-parity-artifacts", "jax_parity.json"),
        ("validate-gk-ood-calibration", "--artifact-root", "--require-campaign-artifacts", "ood.json"),
        ("validate-gk-interface-artifacts", "--artifact-root", "--require-interface-artifacts", "interface.json"),
        ("validate-blob-transport-reference", "--artifact-root", "--require-reference-artifacts", "blob.json"),
        ("validate-elm-reference", "--artifact-root", "--require-reference-artifacts", "elm.json"),
        ("validate-eped-reference", "--artifact-root", "--require-reference-artifacts", "eped.json"),
        ("validate-marfe-reference", "--artifact-root", "--require-reference-artifacts", "marfe.json"),
        ("validate-ntm-reference", "--artifact-root", "--require-reference-artifacts", "ntm.json"),
        (
            "validate-neural-equilibrium-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "neural_equilibrium.json",
        ),
        (
            "validate-neural-transport-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "neural_transport.json",
        ),
        (
            "validate-neural-turbulence-reference",
            "--artifact-root",
            "--require-reference-artifacts",
            "neural_turbulence.json",
        ),
        ("validate-orbit-reference", "--artifact-root", "--require-reference-artifacts", "orbit.json"),
        ("validate-uncertainty-reference", "--artifact-root", "--require-reference-artifacts", "uncertainty.json"),
        ("validate-vmec-reference", "--artifact-root", "--require-reference-artifacts", "vmec.json"),
        ("validate-rzip-reference", "--artifact-root", "--require-reference-artifacts", "rzip.json"),
        ("validate-density-reference", "--artifact-root", "--require-reference-artifacts", "density.json"),
        ("validate-disruption-reference", "--artifact-root", "--require-reference-artifacts", "disruption.json"),
        ("validate-digital-twin-reference", "--artifact-root", "--require-reference-artifacts", "digital_twin.json"),
        ("validate-soc-reference", "--artifact-root", "--require-reference-artifacts", "soc.json"),
    ],
)
def test_gk_validation_output_json_files_on_failures(runner, tmp_path, command, root_option, require_flag, output_name):
    output_json = tmp_path / "reports" / output_name

    result = runner.invoke(
        main,
        [
            command,
            root_option,
            str(tmp_path / "missing-artifacts"),
            require_flag,
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 1
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["errors"]


@pytest.mark.parametrize(
    ("command", "root_option"),
    [
        ("validate-gk-crosscode", "--evidence-root"),
        ("validate-gk-ood-calibration", "--artifact-root"),
        ("validate-gk-interface-artifacts", "--artifact-root"),
        ("validate-blob-transport-reference", "--artifact-root"),
        ("validate-elm-reference", "--artifact-root"),
        ("validate-eped-reference", "--artifact-root"),
        ("validate-marfe-reference", "--artifact-root"),
        ("validate-ntm-reference", "--artifact-root"),
        ("validate-neural-equilibrium-reference", "--artifact-root"),
        ("validate-neural-transport-reference", "--artifact-root"),
        ("validate-neural-turbulence-reference", "--artifact-root"),
        ("validate-orbit-reference", "--artifact-root"),
        ("validate-uncertainty-reference", "--artifact-root"),
        ("validate-vmec-reference", "--artifact-root"),
        ("validate-rzip-reference", "--artifact-root"),
        ("validate-current-drive-reference", "--artifact-root"),
        ("validate-mu-synthesis-reference", "--artifact-root"),
        ("validate-volt-second-reference", "--artifact-root"),
        ("validate-burn-reference", "--artifact-root"),
        ("validate-density-reference", "--artifact-root"),
        ("validate-free-boundary-reference", "--artifact-root"),
        ("validate-disruption-reference", "--artifact-root"),
        ("validate-digital-twin-reference", "--artifact-root"),
        ("validate-soc-reference", "--artifact-root"),
    ],
)
def test_gk_validation_passes_when_evidence_not_required(runner, tmp_path, command, root_option):
    """Without the ``--require-*`` flag an empty artifact root is a clean pass, not a failure.

    The failure suites above cover the ``status != "pass"`` exit; this covers the complementary
    fall-through where no evidence is required, so the command returns without raising.
    """
    result = runner.invoke(main, [command, root_option, str(tmp_path), "--json-out"])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["status"] == "pass"
