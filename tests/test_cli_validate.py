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
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _fresh_cli(
    arguments: list[str], *, viz_import: bool = False, source_root: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Exercise the real root CLI in a fresh process, optionally with actual visualization imports or a copied installation."""
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"}
    env["PYTHONPATH"] = str(source_root or repo / "src") + os.pathsep + str(repo)
    if viz_import:
        code = "import matplotlib; from scpn_control.cli import main; main(" + repr(arguments) + ")"
        argv = [sys.executable, "-c", code]
    else:
        argv = [sys.executable, "-m", "scpn_control.cli", *arguments]
    return subprocess.run(argv, cwd=repo, env=env, capture_output=True, text=True, check=False, timeout=30)


def test_validate_json_out() -> None:
    """The actual default CLI inspects all real repository reports and emits complete enabled-gate summaries."""
    result = _fresh_cli(["validate", "--json-out"])
    assert result.returncode == 0
    data = json.loads((result.stdout + result.stderr))
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


def test_validate_reports_manifest_gate_failures(tmp_path: Path) -> None:
    """An actual empty manifest root fails while other repository readers preserve their results."""
    result = _fresh_cli(["validate", "--data-manifest-root", str(tmp_path), "--json-out"])

    assert result.returncode == 1
    data = json.loads((result.stdout + result.stderr))
    assert data["status"] == "fail"
    assert data["data_manifests"]["status"] == "fail"
    assert data["data_manifests"]["errors"][0]["error"] == "no data manifests found"
    assert data["jax_gk_parity"]["status"] == "pass"
    assert data["physics_traceability"]["status"] == "pass"
    assert data["multi_shot_campaign"]["status"] == "pass"
    assert data["runtime_admission"]["status"] == "pass"
    assert data["native_formal_certificate"]["status"] == "pass"


def test_validate_reports_jax_gk_parity_gate_failures(tmp_path: Path) -> None:
    """An empty actual parity directory refuses required campaign evidence through the registered command."""
    result = _fresh_cli(
        [
            "validate",
            "--no-data-manifests",
            "--jax-gk-parity-root",
            str(tmp_path),
            "--json-out",
        ],
    )

    assert result.returncode == 1
    data = json.loads((result.stdout + result.stderr))
    assert data["status"] == "fail"
    assert data["data_manifests"]["status"] == "skipped"
    assert data["jax_gk_parity"]["status"] == "fail"
    assert data["jax_gk_parity"]["errors"][0]["error"] == "no JAX GK parity artifacts found"
    assert data["physics_traceability"]["status"] == "pass"
    assert data["multi_shot_campaign"]["status"] == "pass"
    assert data["runtime_admission"]["status"] == "pass"
    assert data["native_formal_certificate"]["status"] == "pass"


def test_validate_reports_physics_traceability_gate_failures(tmp_path: Path) -> None:
    """A malformed copied claim registry produces structured registered-CLI findings."""
    registry = tmp_path / "physics_traceability.json"
    registry.write_text('{"schema_version":"1.0","entries":[]}', encoding="utf-8")
    result = _fresh_cli(
        [
            "validate",
            "--no-data-manifests",
            "--no-jax-gk-parity",
            "--physics-traceability-registry",
            str(registry),
            "--json-out",
        ],
    )

    assert result.returncode == 1
    data = json.loads((result.stdout + result.stderr))
    assert data["status"] == "fail"
    assert data["data_manifests"]["status"] == "skipped"
    assert data["jax_gk_parity"]["status"] == "skipped"
    assert data["physics_traceability"]["status"] == "fail"
    assert any(error["field"] == "entries" for error in data["physics_traceability"]["errors"])
    assert data["multi_shot_campaign"]["status"] == "pass"
    assert data["runtime_admission"]["status"] == "pass"
    assert data["native_formal_certificate"]["status"] == "pass"


def test_validate_release_evidence_json_out(tmp_path: Path) -> None:
    """Illustrative complete summary declarations pass the actual release reader without claiming authenticated evidence."""
    report_path = tmp_path / "release_evidence_report.json"
    report_path.write_text(json.dumps(_release_evidence_report()), encoding="utf-8")

    result = _fresh_cli(["validate-release-evidence", str(report_path), "--json-out"])

    assert result.returncode == 0
    payload = json.loads((result.stdout + result.stderr))
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


def test_validate_release_evidence_reports_failures(tmp_path: Path) -> None:
    """A skipped mandatory summary gate fails the actual release reader CLI."""
    report = _release_evidence_report()
    report["jax_gk_parity"] = {"status": "skipped"}
    report_path = tmp_path / "release_evidence_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _fresh_cli(["validate-release-evidence", str(report_path), "--json-out"])

    assert result.returncode == 1
    payload = json.loads((result.stdout + result.stderr))
    assert payload["status"] == "fail"
    assert "jax_gk_parity.status must be 'pass', got 'skipped'" in payload["errors"]


def test_validate_can_skip_data_manifest_gate() -> None:
    """An explicit scoped manifest skip preserves the remaining actual repository reader summaries."""
    result = _fresh_cli(["validate", "--no-data-manifests"])

    assert result.returncode == 0
    assert "Data manifests: SKIPPED" in (result.stdout + result.stderr)
    assert "JAX GK parity: pass" in (result.stdout + result.stderr)
    assert "Physics traceability: pass" in (result.stdout + result.stderr)
    assert "Multi-shot campaign evidence: pass" in (result.stdout + result.stderr)
    assert "Runtime admission evidence: pass" in (result.stdout + result.stderr)
    assert "Native formal certificate: pass" in (result.stdout + result.stderr)
    assert "Status:" in (result.stdout + result.stderr)


def test_validate_can_skip_jax_gk_parity_gate() -> None:
    """Explicit scoped parity skips remain visible while remaining repository gates run."""
    result = _fresh_cli(["validate", "--no-data-manifests", "--no-jax-gk-parity"])

    assert result.returncode == 0
    assert "Data manifests: SKIPPED" in (result.stdout + result.stderr)
    assert "JAX GK parity: SKIPPED" in (result.stdout + result.stderr)
    assert "Physics traceability: pass" in (result.stdout + result.stderr)
    assert "Multi-shot campaign evidence: pass" in (result.stdout + result.stderr)
    assert "Runtime admission evidence: pass" in (result.stdout + result.stderr)
    assert "Native formal certificate: pass" in (result.stdout + result.stderr)


def test_validate_can_skip_physics_traceability_gate() -> None:
    """An explicit scoped registry skip remains visible in the real text command."""
    result = _fresh_cli(
        ["validate", "--no-data-manifests", "--no-jax-gk-parity", "--no-physics-traceability"],
    )

    assert result.returncode == 0
    assert "Data manifests: SKIPPED" in (result.stdout + result.stderr)
    assert "JAX GK parity: SKIPPED" in (result.stdout + result.stderr)
    assert "Physics traceability: SKIPPED" in (result.stdout + result.stderr)
    assert "Multi-shot campaign evidence: pass" in (result.stdout + result.stderr)
    assert "Runtime admission evidence: pass" in (result.stdout + result.stderr)
    assert "Native formal certificate: pass" in (result.stdout + result.stderr)


def test_validate_can_skip_native_formal_certificate_gate() -> None:
    """All explicit gate skips still exercise actual transport and import hygiene."""
    result = _fresh_cli(
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

    assert result.returncode == 0
    assert "Multi-shot campaign evidence: SKIPPED" in (result.stdout + result.stderr)
    assert "Runtime admission evidence: SKIPPED" in (result.stdout + result.stderr)
    assert "Native formal certificate: SKIPPED" in (result.stdout + result.stderr)


def test_validate_reports_native_formal_certificate_gate_failures(tmp_path: Path) -> None:
    """An invalid certificate report fails the enabled actual reader and exit code."""
    report = tmp_path / "native_formal_report.json"
    report.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")

    result = _fresh_cli(
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

    assert result.returncode == 1
    data = json.loads((result.stdout + result.stderr))
    assert data["status"] == "fail"
    assert data["native_formal_certificate"]["status"] == "fail"
    assert "at least one AOT certificate case must be admitted" in data["native_formal_certificate"]["errors"]


def test_validate_text_reports_manifest_gate_errors(tmp_path: Path) -> None:
    """The real CLI emits empty-manifest findings to its text/error streams and exits one."""
    result = _fresh_cli(["validate", "--data-manifest-root", str(tmp_path)])

    assert result.returncode == 1
    assert "Data manifests: fail" in (result.stdout + result.stderr)
    assert "ERROR" in (result.stdout + result.stderr)
    assert "no data manifests found" in (result.stdout + result.stderr)


def test_validate_command_is_import_clean_in_fresh_process() -> None:
    """A fresh actual CLI retains import hygiene while inspecting the enabled repository formal report."""
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


def test_validate_import_clean_loop_exhausts_when_no_viz_modules_present() -> None:
    """A fresh real CLI exhausts the hygiene loop without deleting or mocking module state."""
    result = _fresh_cli(
        [
            "validate",
            "--no-data-manifests",
            "--no-jax-gk-parity",
            "--no-physics-traceability",
            "--no-multi-shot-campaign-evidence",
            "--no-runtime-admission-evidence",
            "--no-native-formal-certificate",
            "--json-out",
        ]
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["import_clean"] is True and data["status"] == "pass"
    assert "contaminated_module" not in data


def test_validate_text_output() -> None:
    """The complete real repository report readers produce text summaries through the registered CLI."""
    result = _fresh_cli(["validate"])
    assert result.returncode == 0
    assert "Transport solver:" in (result.stdout + result.stderr)
    assert "Import clean:" in (result.stdout + result.stderr)
    assert "Status:" in (result.stdout + result.stderr)


def _release_evidence_report() -> dict[str, object]:
    """Construct inherited illustrative declaration metadata; labels and digest spellings do not authenticate artifacts."""
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


@pytest.mark.parametrize("json_out", [False, True])
def test_actual_visualization_import_refuses_with_nonzero_exit(json_out: bool) -> None:
    """A real Matplotlib import makes the root CLI fail both its JSON/text status and process exit code."""
    args = [
        "validate",
        "--no-data-manifests",
        "--no-jax-gk-parity",
        "--no-physics-traceability",
        "--no-multi-shot-campaign-evidence",
        "--no-runtime-admission-evidence",
        "--no-native-formal-certificate",
    ]
    result = _fresh_cli(args + (["--json-out"] if json_out else []), viz_import=True)
    assert result.returncode == 1, result.stdout + result.stderr
    if json_out:
        payload = json.loads(result.stdout)
        assert payload["status"] == "fail" and payload["import_clean"] is False
        assert payload["contaminated_module"] == "matplotlib"
    else:
        assert "Import clean: FAIL" in result.stdout and "Status: fail" in result.stdout


@pytest.mark.parametrize("json_out", [False, True])
def test_actual_copied_installation_missing_transport_refuses(tmp_path: Path, json_out: bool) -> None:
    """A real copied package without its transport owner refuses import admission instead of reporting PASS/exit0."""
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "src"
    shutil.copytree(root / "src/scpn_control", source / "scpn_control", ignore=shutil.ignore_patterns("__pycache__"))
    transport = source / "scpn_control/core/integrated_transport_solver.py"
    transport.rename(tmp_path / "withheld_integrated_transport_solver.py")
    args = [
        "validate",
        "--no-data-manifests",
        "--no-jax-gk-parity",
        "--no-physics-traceability",
        "--no-multi-shot-campaign-evidence",
        "--no-runtime-admission-evidence",
        "--no-native-formal-certificate",
    ]
    result = _fresh_cli(args + (["--json-out"] if json_out else []), source_root=source)
    assert result.returncode == 1, result.stdout + result.stderr
    if json_out:
        payload = json.loads(result.stdout)
        assert payload["status"] == "fail" and payload["transport_solver_available"] is False
    else:
        assert "Transport solver: MISSING" in result.stdout and "Status: fail" in result.stdout


@pytest.mark.parametrize("gate", ["jax", "traceability", "campaign", "formal"])
def test_root_text_preserves_enabled_reader_findings(tmp_path: Path, gate: str) -> None:
    """Real empty directories or invalid report files reach each enabled reader's error stream and exit one."""
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version":"1.0","entries":[]}', encoding="utf-8")
    base = ["validate", "--no-data-manifests", "--no-runtime-admission-evidence"]
    if gate == "jax":
        args = base + [
            "--jax-gk-parity-root",
            str(tmp_path / "empty"),
            "--no-physics-traceability",
            "--no-multi-shot-campaign-evidence",
            "--no-native-formal-certificate",
        ]
        expected = "ERROR "
    elif gate == "traceability":
        args = base + [
            "--no-jax-gk-parity",
            "--physics-traceability-registry",
            str(path),
            "--no-multi-shot-campaign-evidence",
            "--no-native-formal-certificate",
        ]
        expected = "ERROR "
    elif gate == "campaign":
        args = base + [
            "--no-jax-gk-parity",
            "--no-physics-traceability",
            "--multi-shot-campaign-python-report",
            str(path),
            "--multi-shot-campaign-rust-report",
            str(path),
            "--no-native-formal-certificate",
        ]
        expected = "ERROR multi_shot_campaign:"
    else:
        args = base + [
            "--no-jax-gk-parity",
            "--no-physics-traceability",
            "--no-multi-shot-campaign-evidence",
            "--native-formal-certificate-report",
            str(path),
        ]
        expected = "ERROR native_formal_certificate:"
    result = _fresh_cli(args)
    assert result.returncode == 1, result.stdout + result.stderr
    assert expected in result.stderr and "Status: fail" in result.stdout


def test_release_cli_text_unparsable_report_has_no_digest(tmp_path: Path) -> None:
    """An actual unparsable release report fails text admission without a report SHA carrier."""
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    result = _fresh_cli(["validate-release-evidence", str(path)])
    assert result.returncode == 1 and "Release evidence: fail" in result.stdout
    assert "Report SHA-256:" not in result.stdout and "ERROR" in result.stderr
