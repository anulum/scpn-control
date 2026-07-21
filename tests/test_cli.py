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
    assert data["multi_shot_campaign"]["status"] == "pass"
    assert set(data["multi_shot_campaign"]["admitted_surfaces"]) == {"python", "pyo3", "rust"}
    assert data["runtime_admission"]["status"] == "pass"
    assert data["runtime_admission"]["production_claim_allowed"] is False
    assert data["native_formal_certificate"]["status"] == "pass"
    assert data["native_formal_certificate"]["admitted_cases"]
    assert len(data["native_formal_certificate"]["certificate_assumption_sha256"]) == 64


def test_validate_reports_manifest_gate_failures(runner, tmp_path):
    result = runner.invoke(main, ["validate", "--data-manifest-root", str(tmp_path), "--json-out"])

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["data_manifests"]["status"] == "fail"
    assert data["data_manifests"]["errors"][0]["error"] == "no data manifests found"
    assert data["jax_gk_parity"]["status"] == "pass"
    assert data["physics_traceability"]["status"] == "pass"
    assert data["multi_shot_campaign"]["status"] == "pass"
    assert data["runtime_admission"]["status"] == "pass"
    assert data["native_formal_certificate"]["status"] == "pass"


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
    assert data["multi_shot_campaign"]["status"] == "pass"
    assert data["runtime_admission"]["status"] == "pass"
    assert data["native_formal_certificate"]["status"] == "pass"


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
    assert data["multi_shot_campaign"]["status"] == "pass"
    assert data["runtime_admission"]["status"] == "pass"
    assert data["native_formal_certificate"]["status"] == "pass"


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
        "multi_shot_campaign": {
            "status": "pass",
            "errors": [],
            "admitted_surfaces": ["python", "pyo3", "rust"],
            "pyo3_status": "ok",
            "python_report_sha256": "c" * 64,
            "rust_report_sha256": "d" * 64,
            "python_payload_sha256": "e" * 64,
            "rust_payload_sha256": "f" * 64,
            "production_claim_allowed": False,
            "minimum_digest_count": 2,
        },
        "runtime_admission": {
            "status": "pass",
            "errors": [],
            "report_sha256": "1" * 64,
            "payload_sha256": "2" * 64,
            "benchmark_evidence_class": "local_regression",
            "production_claim_allowed": False,
            "admission_status": "fail",
            "admission_error_count": 3,
            "samples": 500,
        },
        "native_formal_certificate": {
            "status": "pass",
            "admitted_cases": ["std:spin:aot_certificate:stride_1"],
            "certificate_assumption_sha256": "a" * 64,
            "benchmark_evidence_class": "local_regression",
            "production_claim_allowed": False,
            "errors": [],
            "report_sha256": "b" * 64,
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
    assert payload["admitted_gates"] == [
        "data_manifests",
        "jax_gk_parity",
        "physics_traceability",
        "multi_shot_campaign",
        "runtime_admission",
        "native_formal_certificate",
    ]
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
    assert "Multi-shot campaign evidence: pass" in result.output
    assert "Runtime admission evidence: pass" in result.output
    assert "Native formal certificate: pass" in result.output
    assert "Status:" in result.output


def test_validate_can_skip_jax_gk_parity_gate(runner):
    result = runner.invoke(main, ["validate", "--no-data-manifests", "--no-jax-gk-parity"])

    assert result.exit_code == 0
    assert "Data manifests: SKIPPED" in result.output
    assert "JAX GK parity: SKIPPED" in result.output
    assert "Physics traceability: pass" in result.output
    assert "Multi-shot campaign evidence: pass" in result.output
    assert "Runtime admission evidence: pass" in result.output
    assert "Native formal certificate: pass" in result.output


def test_validate_can_skip_physics_traceability_gate(runner):
    result = runner.invoke(
        main,
        ["validate", "--no-data-manifests", "--no-jax-gk-parity", "--no-physics-traceability"],
    )

    assert result.exit_code == 0
    assert "Data manifests: SKIPPED" in result.output
    assert "JAX GK parity: SKIPPED" in result.output
    assert "Physics traceability: SKIPPED" in result.output
    assert "Multi-shot campaign evidence: pass" in result.output
    assert "Runtime admission evidence: pass" in result.output
    assert "Native formal certificate: pass" in result.output


def test_validate_can_skip_native_formal_certificate_gate(runner):
    result = runner.invoke(
        main,
        [
            "validate",
            "--no-data-manifests",
            "--no-jax-gk-parity",
            "--no-physics-traceability",
            "--no-multi-shot-campaign-evidence",
            "--no-runtime-admission-evidence",
            "--no-native-formal-certificate",
        ],
    )

    assert result.exit_code == 0
    assert "Multi-shot campaign evidence: SKIPPED" in result.output
    assert "Runtime admission evidence: SKIPPED" in result.output
    assert "Native formal certificate: SKIPPED" in result.output


def test_validate_reports_native_formal_certificate_gate_failures(runner, tmp_path):
    report = tmp_path / "native_formal_report.json"
    report.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "validate",
            "--no-data-manifests",
            "--no-jax-gk-parity",
            "--no-physics-traceability",
            "--no-multi-shot-campaign-evidence",
            "--no-runtime-admission-evidence",
            "--native-formal-certificate-report",
            str(report),
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "fail"
    assert data["native_formal_certificate"]["status"] == "fail"
    assert "at least one AOT certificate case must be admitted" in data["native_formal_certificate"]["errors"]


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
        [
            sys.executable,
            "-m",
            "scpn_control.cli",
            "validate",
            "--no-data-manifests",
            "--no-jax-gk-parity",
            "--no-physics-traceability",
            "--no-multi-shot-campaign-evidence",
            "--no-runtime-admission-evidence",
            "--json-out",
        ],
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


def test_validate_import_clean_loop_exhausts_when_no_viz_modules_present(runner, monkeypatch):
    """In-process: with matplotlib/torch/streamlit absent, the hygiene loop runs to completion.

    Exercises the loop-exhaust and back-edge arcs (92->99, 93->92) that the subprocess
    variant above cannot cover (subprocess execution is not traced in-process).
    """
    for module_name in ("matplotlib", "torch", "streamlit"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    result = runner.invoke(
        main,
        [
            "validate",
            "--no-data-manifests",
            "--no-jax-gk-parity",
            "--no-physics-traceability",
            "--no-multi-shot-campaign-evidence",
            "--no-runtime-admission-evidence",
            "--no-native-formal-certificate",
            "--json-out",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
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


def test_validate_text_output(runner):
    result = runner.invoke(main, ["validate"])
    assert result.exit_code == 0
    assert "Transport solver:" in result.output
    assert "Import clean:" in result.output
    assert "Status:" in result.output


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


def test_validate_rmse_without_json_out_skips_report_echo(runner, tmp_path, monkeypatch):
    """Without ``--json-out`` the report file is never echoed (arc 452->458)."""

    def fake_rmse_main() -> int:
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
            "--output-json",
            str(tmp_path / "rmse_report.json"),
            "--output-md",
            str(tmp_path / "rmse_report.md"),
        ],
    )

    assert result.exit_code == 0
    assert "Report SHA-256" not in result.output


def test_validate_rmse_json_out_skips_echo_when_report_absent(runner, tmp_path, monkeypatch):
    """With ``--json-out`` but no written report, the missing-file guard skips echo (arc 456->458)."""

    def fake_rmse_main() -> int:
        return 0  # deliberately does not write the output JSON

    monkeypatch.setitem(
        __import__("sys").modules,
        "validation.rmse_dashboard",
        type("M", (), {"main": staticmethod(fake_rmse_main)})(),
    )

    missing_report = tmp_path / "never_written.json"
    result = runner.invoke(
        main,
        [
            "validate-rmse",
            "--json-out",
            "--output-json",
            str(missing_report),
            "--output-md",
            str(tmp_path / "rmse_report.md"),
        ],
    )

    assert result.exit_code == 0
    assert not missing_report.exists()
    assert result.output.strip() == ""
