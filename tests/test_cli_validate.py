# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI validate command tests
"""Tests for the top-level ``validate`` command: gates, gate-skips, release evidence, text output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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


def test_validate_text_output(runner):
    result = runner.invoke(main, ["validate"])
    assert result.exit_code == 0
    assert "Transport solver:" in result.output
    assert "Import clean:" in result.output
    assert "Status:" in result.output


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
