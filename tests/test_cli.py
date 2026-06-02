# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI Tests

"""Tests for the scpn-control Click CLI entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_demo_json_out(runner):
    result = runner.invoke(main, ["demo", "--steps", "10", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in ("scenario", "steps", "final_state", "final_error", "converged"):
        assert key in data
    assert data["steps"] == 10
    assert isinstance(data["final_state"], float)
    assert isinstance(data["converged"], bool)


def test_demo_pid_text(runner):
    result = runner.invoke(main, ["demo", "--scenario", "pid", "--steps", "5"])
    assert result.exit_code == 0
    assert "Scenario: pid" in result.output
    assert "Steps: 5" in result.output


def test_benchmark_json_out(runner):
    result = runner.invoke(main, ["benchmark", "--n-bench", "100", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in ("n_bench", "pid_us_per_step", "snn_us_per_step", "speedup_ratio"):
        assert key in data
    assert data["n_bench"] == 100
    assert data["pid_us_per_step"] > 0


def test_validate_json_out(runner):
    result = runner.invoke(main, ["validate", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "status" in data
    assert data["status"] in ("pass", "fail")
    assert data["data_manifests"]["status"] == "pass"
    assert data["data_manifests"]["total"] >= 3
    assert data["jax_gk_parity"]["status"] == "pass"
    assert data["jax_gk_parity"]["parity_artifacts"] >= 6
    assert data["physics_traceability"]["status"] == "pass"
    assert data["physics_traceability"]["public_claim_blocked"] >= 1


def test_validate_reports_manifest_gate_failures(runner, tmp_path):
    result = runner.invoke(main, ["validate", "--data-manifest-root", str(tmp_path), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["data_manifests"]["status"] == "fail"
    assert data["data_manifests"]["errors"][0]["error"] == "no data manifests found"
    assert data["jax_gk_parity"]["status"] == "pass"
    assert data["physics_traceability"]["status"] == "pass"


def test_validate_reports_jax_gk_parity_gate_failures(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "validate",
            "--no-data-manifests",
            "--jax-gk-parity-root",
            str(tmp_path),
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["data_manifests"]["status"] == "skipped"
    assert data["jax_gk_parity"]["status"] == "fail"
    assert data["jax_gk_parity"]["errors"][0]["error"] == "no JAX GK parity artifacts found"
    assert data["physics_traceability"]["status"] == "pass"


def test_validate_reports_physics_traceability_gate_failures(runner, tmp_path):
    registry = tmp_path / "physics_traceability.json"
    registry.write_text('{"schema_version":"1.0","entries":[]}', encoding="utf-8")
    result = runner.invoke(
        main,
        [
            "validate",
            "--no-data-manifests",
            "--no-jax-gk-parity",
            "--physics-traceability-registry",
            str(registry),
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["data_manifests"]["status"] == "skipped"
    assert data["jax_gk_parity"]["status"] == "skipped"
    assert data["physics_traceability"]["status"] == "fail"
    assert any(error["field"] == "entries" for error in data["physics_traceability"]["errors"])


def _release_evidence_report() -> dict[str, object]:
    cases = ("cyclone_base_case", "tem_kinetic_electron", "stable_mode")
    backends = ("cpu", "gpu")
    return {
        "transport_solver_available": True,
        "import_clean": True,
        "status": "pass",
        "data_manifests": {
            "status": "pass",
            "total": 5,
            "real": 4,
            "synthetic": 1,
            "artifact_coverage": {"expected": 21, "covered": 21, "missing": []},
        },
        "jax_gk_parity": {
            "status": "pass",
            "parity_artifacts": 6,
            "required_cases": list(cases),
            "required_backends": list(backends),
            "entries": [{"case": case, "backend": backend} for case in cases for backend in backends],
        },
        "physics_traceability": {
            "status": "pass",
            "total": 54,
            "open_fidelity_gaps": 53,
            "public_claim_blocked": 53,
        },
    }


def test_validate_release_evidence_json_out(runner, tmp_path):
    report_path = tmp_path / "release_evidence_report.json"
    report_path.write_text(json.dumps(_release_evidence_report()), encoding="utf-8")

    result = runner.invoke(main, ["validate-release-evidence", str(report_path), "--json-out"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    assert payload["schema_version"] == "scpn-control.release-evidence-admission.v1"
    assert payload["admitted_gates"] == ["data_manifests", "jax_gk_parity", "physics_traceability"]
    assert len(payload["report_sha256"]) == 64


def test_validate_release_evidence_reports_failures(runner, tmp_path):
    report = _release_evidence_report()
    report["jax_gk_parity"] = {"status": "skipped"}
    report_path = tmp_path / "release_evidence_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = runner.invoke(main, ["validate-release-evidence", str(report_path), "--json-out"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert "jax_gk_parity.status must be 'pass', got 'skipped'" in payload["errors"]


def test_validate_can_skip_data_manifest_gate(runner):
    result = runner.invoke(main, ["validate", "--no-data-manifests"])

    assert result.exit_code == 0
    assert "Data manifests: SKIPPED" in result.output
    assert "JAX GK parity: pass" in result.output
    assert "Physics traceability: pass" in result.output
    assert "Status:" in result.output


def test_validate_can_skip_jax_gk_parity_gate(runner):
    result = runner.invoke(main, ["validate", "--no-data-manifests", "--no-jax-gk-parity"])

    assert result.exit_code == 0
    assert "Data manifests: SKIPPED" in result.output
    assert "JAX GK parity: SKIPPED" in result.output
    assert "Physics traceability: pass" in result.output


def test_validate_can_skip_physics_traceability_gate(runner):
    result = runner.invoke(
        main,
        ["validate", "--no-data-manifests", "--no-jax-gk-parity", "--no-physics-traceability"],
    )

    assert result.exit_code == 0
    assert "Data manifests: SKIPPED" in result.output
    assert "JAX GK parity: SKIPPED" in result.output
    assert "Physics traceability: SKIPPED" in result.output


def test_validate_text_reports_manifest_gate_errors(runner, tmp_path):
    result = runner.invoke(main, ["validate", "--data-manifest-root", str(tmp_path)])

    assert result.exit_code == 1
    assert "Data manifests: fail" in result.output
    assert "ERROR" in result.output
    assert "no data manifests found" in result.output


def test_validate_command_is_import_clean_in_fresh_process():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    completed = subprocess.run(
        [sys.executable, "-m", "scpn_control.cli", "validate", "--json-out"],
        check=True,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    data = json.loads(completed.stdout)
    assert data["import_clean"] is True
    assert "contaminated_module" not in data


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


@dataclass(frozen=True)
class _AcquisitionResult:
    dataset_id: str
    output_npz: Path
    manifest_json: Path
    checksum_sha256: str


def test_acquire_mdsplus_shot_manual_text_success(runner, tmp_path, monkeypatch):
    import scpn_control.core.mdsplus_acquisition as acquisition

    calls = {}

    def fake_acquire_mdsplus_shot(**kwargs):
        calls.update(kwargs)
        return _AcquisitionResult(
            dataset_id="diii-d-163303-manual",
            output_npz=Path(kwargs["output_npz"]),
            manifest_json=Path(kwargs["manifest_json"]),
            checksum_sha256="e" * 64,
        )

    monkeypatch.setattr(acquisition, "acquire_mdsplus_shot", fake_acquire_mdsplus_shot)
    output_npz = tmp_path / "shot.npz"
    manifest_json = tmp_path / "manifest.json"
    signal = json.dumps({"name": "ip", "node": "\\\\IP", "units": "A", "timebase": "s"})

    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--tree",
            "DIII-D",
            "--shot",
            "163303",
            "--signal",
            signal,
            "--output-npz",
            str(output_npz),
            "--manifest-json",
            str(manifest_json),
        ],
    )

    assert result.exit_code == 0
    assert calls["tree"] == "DIII-D"
    assert calls["shot"] == 163303
    assert calls["source_uri"] == "mdsplus://DIII-D/163303"
    assert calls["signals"][0].name == "ip"
    assert "Dataset: diii-d-163303-manual" in result.output
    assert "Status: pass" in result.output


def test_acquire_mdsplus_shot_spec_json_success(runner, tmp_path, monkeypatch):
    import scpn_control.core.mdsplus_acquisition as acquisition

    signal = acquisition.MDSplusSignalSpec(name="ip", node="\\IP", units="A", timebase="s")
    request = acquisition.MDSplusAcquisitionRequest(
        tree="DIII-D",
        shot=163303,
        source_uri="mdsplus://DIII-D/163303",
        access_policy="facility-approved",
        licence="facility data policy",
        signals=[signal],
    )
    calls = {}

    def fake_load_mdsplus_acquisition_request(path):
        calls["spec_path"] = path
        return request

    def fake_acquire_mdsplus_shot(**kwargs):
        calls.update(kwargs)
        return _AcquisitionResult(
            dataset_id="diii-d-163303-spec",
            output_npz=Path(kwargs["output_npz"]),
            manifest_json=Path(kwargs["manifest_json"]),
            checksum_sha256="f" * 64,
        )

    monkeypatch.setattr(acquisition, "load_mdsplus_acquisition_request", fake_load_mdsplus_acquisition_request)
    monkeypatch.setattr(acquisition, "acquire_mdsplus_shot", fake_acquire_mdsplus_shot)
    spec_path = tmp_path / "request.json"
    spec_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--spec-json",
            str(spec_path),
            "--source-uri",
            "mdsplus://override/163303",
            "--output-npz",
            str(tmp_path / "shot.npz"),
            "--manifest-json",
            str(tmp_path / "manifest.json"),
            "--json-out",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dataset_id"] == "diii-d-163303-spec"
    assert calls["spec_path"] == str(spec_path)
    assert calls["tree"] == "DIII-D"
    assert calls["shot"] == 163303
    assert calls["source_uri"] == "mdsplus://override/163303"
    assert calls["signals"] == [signal]


def test_acquire_mdsplus_shot_manual_missing_inputs_text_failure(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--output-npz",
            str(tmp_path / "shot.npz"),
            "--manifest-json",
            str(tmp_path / "manifest.json"),
        ],
    )

    assert result.exit_code == 1
    assert "Status: fail" in result.output
    assert "--tree is required" in result.output


def test_acquire_mdsplus_shot_missing_shot_and_signal_failures(runner, tmp_path):
    base_args = [
        "acquire-mdsplus-shot",
        "--tree",
        "DIII-D",
        "--output-npz",
        str(tmp_path / "shot.npz"),
        "--manifest-json",
        str(tmp_path / "manifest.json"),
        "--json-out",
    ]

    missing_shot = runner.invoke(main, base_args)
    assert missing_shot.exit_code == 1
    assert "--shot is required" in json.loads(missing_shot.output)["error"]

    missing_signal = runner.invoke(main, [*base_args, "--shot", "163303"])
    assert missing_signal.exit_code == 1
    assert "at least one --signal" in json.loads(missing_signal.output)["error"]


def test_acquire_mdsplus_shot_rejects_non_object_signal_json(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--tree",
            "DIII-D",
            "--shot",
            "163303",
            "--signal",
            "[1, 2, 3]",
            "--output-npz",
            str(tmp_path / "shot.npz"),
            "--manifest-json",
            str(tmp_path / "manifest.json"),
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert "signal specification must be a JSON object" in payload["error"]


def test_live_command_wires_monitor_and_server(runner, monkeypatch):
    import scpn_control.phase.realtime_monitor as realtime_monitor
    import scpn_control.phase.ws_phase_stream as ws_phase_stream

    calls = {}

    class FakeMonitor:
        @classmethod
        def from_paper27(cls, *, L, N_per, zeta_uniform, psi_driver):
            calls["monitor"] = {
                "L": L,
                "N_per": N_per,
                "zeta_uniform": zeta_uniform,
                "psi_driver": psi_driver,
            }
            return "monitor"

    class FakeServer:
        def __init__(
            self,
            *,
            monitor,
            tick_interval_s,
            api_key,
            command_rate_limit,
            command_rate_window_s,
            max_payload_bytes,
            max_client_write_buffer_bytes,
            require_client_auth,
            allow_query_token_auth,
            require_tls,
            allow_insecure_remote,
            allowed_origins,
            allowed_actions,
        ):
            calls["server_init"] = {
                "monitor": monitor,
                "tick_interval_s": tick_interval_s,
                "api_key": api_key,
                "command_rate_limit": command_rate_limit,
                "command_rate_window_s": command_rate_window_s,
                "max_payload_bytes": max_payload_bytes,
                "max_client_write_buffer_bytes": max_client_write_buffer_bytes,
                "require_client_auth": require_client_auth,
                "allow_query_token_auth": allow_query_token_auth,
                "require_tls": require_tls,
                "allow_insecure_remote": allow_insecure_remote,
                "allowed_origins": allowed_origins,
                "allowed_actions": allowed_actions,
            }

        def serve_sync(self, *, host, port, ssl_context):
            calls["serve"] = {"host": host, "port": port, "ssl_context": ssl_context}

    monkeypatch.setattr(realtime_monitor, "RealtimeMonitor", FakeMonitor)
    monkeypatch.setattr(ws_phase_stream, "PhaseStreamServer", FakeServer)

    result = runner.invoke(
        main,
        [
            "live",
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--layers",
            "3",
            "--n-per",
            "4",
            "--zeta",
            "0.7",
            "--psi",
            "0.2",
            "--tick-interval",
            "0.005",
            "--api-key",
            "secret-token-123456",
            "--max-payload-bytes",
            "1234",
            "--max-client-write-buffer-bytes",
            "8192",
            "--allow-query-token-auth",
            "--require-tls",
            "--allow-insecure-remote",
            "--allowed-origin",
            "https://ops.example",
            "--allowed-action",
            "set_psi",
            "--allowed-action",
            "stop",
        ],
    )

    assert result.exit_code == 0
    assert calls["monitor"] == {"L": 3, "N_per": 4, "zeta_uniform": 0.7, "psi_driver": 0.2}
    assert calls["server_init"] == {
        "monitor": "monitor",
        "tick_interval_s": 0.005,
        "api_key": "secret-token-123456",
        "command_rate_limit": 20,
        "command_rate_window_s": 1.0,
        "max_payload_bytes": 1234,
        "max_client_write_buffer_bytes": 8192,
        "require_client_auth": True,
        "allow_query_token_auth": True,
        "require_tls": True,
        "allow_insecure_remote": True,
        "allowed_origins": ("https://ops.example",),
        "allowed_actions": ("set_psi", "stop"),
    }
    assert calls["serve"] == {"host": "127.0.0.1", "port": 9001, "ssl_context": None}
    assert "Starting phase sync server on ws://127.0.0.1:9001" in result.output


def test_hil_test_nonexistent_dir(runner):
    result = runner.invoke(main, ["hil-test", "--shots-dir", "nonexistent_dir_12345", "--json-out"])
    assert result.exit_code != 0


def test_live_help(runner):
    result = runner.invoke(main, ["live", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--zeta" in result.output


def test_benchmark_text_output(runner):
    result = runner.invoke(main, ["benchmark", "--n-bench", "50"])
    assert result.exit_code == 0
    assert "PID:" in result.output
    assert "SNN:" in result.output
    assert "Ratio:" in result.output


def test_validate_text_output(runner):
    result = runner.invoke(main, ["validate"])
    assert result.exit_code == 0
    assert "Transport solver:" in result.output
    assert "Import clean:" in result.output
    assert "Status:" in result.output


def test_info_json_out(runner):
    result = runner.invoke(main, ["info", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "version" in data
    assert "python" in data
    assert "numpy" in data
    assert "rust_backend" in data


def test_info_text_output(runner):
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "scpn-control" in result.output
    assert "Rust backend:" in result.output
    assert "Python:" in result.output
    assert "NumPy:" in result.output
    assert "neural_equilibrium_sparc.npz" in result.output


def test_hil_test_with_mock_shots(runner, tmp_path):
    """Exercise hil-test loading NPZ files (lines 222-243)."""
    rng = np.random.default_rng(0)
    for name in ("shot_001", "shot_002"):
        np.savez(
            tmp_path / f"{name}.npz",
            psi=rng.standard_normal((10, 10)),
            ip=np.array([15e6]),
        )
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path), "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["n_shots"] == 2
    assert len(data["shots"]) == 2
    assert data["shots"][0]["status"] == "loaded"
    assert "psi" in data["shots"][0]["keys"]


def test_hil_test_text_output(runner, tmp_path):
    np.savez(tmp_path / "s42.npz", plasma=np.zeros(5))
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "1 shots" in result.output
    assert "s42" in result.output


def test_hil_test_empty_dir(runner, tmp_path):
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path), "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["n_shots"] == 0


def test_info_json_includes_weights_list(runner):
    result = runner.invoke(main, ["info", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data["weights"], list)


def test_demo_combined_json(runner):
    result = runner.invoke(main, ["demo", "--scenario", "combined", "--steps", "5", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["scenario"] == "combined"


def test_demo_snn_text(runner):
    result = runner.invoke(main, ["demo", "--scenario", "snn", "--steps", "3"])
    assert result.exit_code == 0
    assert "Scenario: snn" in result.output


def test_validate_rmse_command(runner, tmp_path, monkeypatch):
    """Exercise cli.py lines 238-248: validate-rmse imports rmse_dashboard."""
    called = {}

    def fake_rmse_main():
        import json as _json
        from pathlib import Path as _P

        out = _P(tmp_path) / "rmse_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps({"status": "pass"}))
        called["invoked"] = True
        return 0

    monkeypatch.setitem(
        __import__("sys").modules,
        "validation.rmse_dashboard",
        type("M", (), {"main": staticmethod(fake_rmse_main)})(),
    )

    result = runner.invoke(
        main,
        [
            "validate-rmse",
            "--json-out",
            "--output-json",
            str(tmp_path / "rmse_report.json"),
            "--output-md",
            str(tmp_path / "rmse_report.md"),
        ],
    )
    assert called.get("invoked")
    assert "pass" in result.output


def test_acquire_mdsplus_shot_fails_closed_before_external_access(runner, tmp_path):
    result = runner.invoke(
        main,
        [
            "acquire-mdsplus-shot",
            "--shot",
            "163303",
            "--signal",
            json.dumps({"name": "ip", "node": "\\IP", "units": "A", "timebase": "s"}),
            "--output-npz",
            str(tmp_path / "shot.npz"),
            "--manifest-json",
            str(tmp_path / "manifest.json"),
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert "--tree is required" in payload["error"]
    assert not (tmp_path / "shot.npz").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_parse_mdsplus_signal_spec_rejects_non_object_payload() -> None:
    from scpn_control.cli import _parse_mdsplus_signal_spec

    with pytest.raises(ValueError, match="JSON object"):
        _parse_mdsplus_signal_spec(json.dumps(["not", "a", "signal"]))
