# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Release Evidence Validation Tests

"""Exercise release declaration admission through real API and CLI entry points.

The inherited complete report is an illustrative schema fixture. Its invented
counts, status labels and digest spellings are not authenticated artifacts,
reference measurements, runtime admission or production evidence.
"""

from __future__ import annotations

import doctest
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

import validation.validate_release_evidence as release_module
from scpn_control.cli import main as control_cli
from validation.validate_release_evidence import REQUIRED_GATES, main, validate_release_evidence


def _valid_report() -> dict[str, object]:
    """Build the inherited schema fixture without asserting authentic release evidence."""
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


def test_release_evidence_admits_complete_passing_report(tmp_path: Path) -> None:
    """A complete passing top-level validation report is admitted."""
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(_valid_report()), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "pass"
    assert result.errors == ()
    assert result.report_sha256 is not None
    assert result.admitted_gates == (
        "data_manifests",
        "jax_gk_parity",
        "physics_traceability",
        "multi_shot_campaign",
        "runtime_admission",
        "native_formal_certificate",
    )


def test_release_evidence_rejects_skipped_required_gate(tmp_path: Path) -> None:
    """Release evidence cannot skip mandatory provenance, parity, traceability, or formal gates."""
    report = _valid_report()
    report["jax_gk_parity"] = {"status": "skipped"}
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "jax_gk_parity.status must be 'pass', got 'skipped'" in result.errors


def test_release_evidence_rejects_invalid_native_certificate_digest(tmp_path: Path) -> None:
    """Native formal certificate evidence must bind a SHA-256 assumption digest."""
    report = _valid_report()
    native_formal = report["native_formal_certificate"]
    assert isinstance(native_formal, dict)
    native_formal["certificate_assumption_sha256"] = "not-a-digest"
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "native_formal_certificate.certificate_assumption_sha256 must be a SHA-256 hex digest" in result.errors


def test_release_evidence_rejects_native_formal_production_overclaim(tmp_path: Path) -> None:
    """Release evidence cannot promote local native-formal evidence to production timing claims."""
    report = _valid_report()
    native_formal = report["native_formal_certificate"]
    assert isinstance(native_formal, dict)
    native_formal["production_claim_allowed"] = True
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "local native formal evidence must not allow production benchmark claims" in result.errors


def test_release_evidence_rejects_native_formal_production_class_without_claim_boundary(tmp_path: Path) -> None:
    """Production native-formal evidence must carry a production claim boundary."""
    report = _valid_report()
    native_formal = report["native_formal_certificate"]
    assert isinstance(native_formal, dict)
    native_formal["benchmark_evidence_class"] = "production_benchmark"
    native_formal["production_claim_allowed"] = False
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "production native formal evidence must allow production benchmark claims" in result.errors


def test_release_evidence_rejects_native_formal_errors(tmp_path: Path) -> None:
    """Native formal certificate admission must not carry lower-level validator errors."""
    report = _valid_report()
    native_formal = report["native_formal_certificate"]
    assert isinstance(native_formal, dict)
    native_formal["errors"] = ["production benchmark evidence must not come from a dirty workspace"]
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "native_formal_certificate.errors must be empty" in result.errors


def test_release_evidence_rejects_incomplete_multi_shot_campaign_evidence(tmp_path: Path) -> None:
    """Release evidence cannot admit multi-shot campaigns without Python, PyO3, and Rust surfaces."""
    report = _valid_report()
    multi_shot = report["multi_shot_campaign"]
    assert isinstance(multi_shot, dict)
    multi_shot["admitted_surfaces"] = ["python", "rust"]
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "multi_shot_campaign.admitted_surfaces must include python, pyo3, and rust" in result.errors


def test_release_evidence_rejects_runtime_admission_production_overclaim(tmp_path: Path) -> None:
    """Release evidence cannot promote local runtime-admission evidence to production timing claims."""
    report = _valid_report()
    runtime_admission = report["runtime_admission"]
    assert isinstance(runtime_admission, dict)
    runtime_admission["production_claim_allowed"] = True
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "local runtime admission evidence must not allow production benchmark claims" in result.errors


def test_release_evidence_rejects_runtime_admission_production_class_without_claim_boundary(tmp_path: Path) -> None:
    """Production runtime-admission evidence must carry production claim admission."""
    report = _valid_report()
    runtime_admission = report["runtime_admission"]
    assert isinstance(runtime_admission, dict)
    runtime_admission["benchmark_evidence_class"] = "production_benchmark"
    runtime_admission["production_claim_allowed"] = False
    runtime_admission["admission_status"] = "pass"
    runtime_admission["admission_error_count"] = 0
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "production runtime admission evidence must allow production benchmark claims" in result.errors


def test_release_evidence_rejects_runtime_admission_production_class_failed_admission(tmp_path: Path) -> None:
    """Production runtime-admission evidence cannot carry fail-closed scheduler errors."""
    report = _valid_report()
    runtime_admission = report["runtime_admission"]
    assert isinstance(runtime_admission, dict)
    runtime_admission["benchmark_evidence_class"] = "production_benchmark"
    runtime_admission["production_claim_allowed"] = True
    runtime_admission["admission_status"] = "fail"
    runtime_admission["admission_error_count"] = 3
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "production runtime admission evidence must pass strict runtime admission" in result.errors
    assert "production runtime admission evidence must not carry admission errors" in result.errors


def test_release_evidence_rejects_incomplete_jax_case_backend_pairs(tmp_path: Path) -> None:
    """Every required JAX GK case/backend pair must be present in the report."""
    report = _valid_report()
    parity = report["jax_gk_parity"]
    assert isinstance(parity, dict)
    entries = parity["entries"]
    assert isinstance(entries, list)
    parity["entries"] = entries[:-1]
    path = tmp_path / "release_evidence_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert "jax_gk_parity.entries must include every required case/backend pair" in result.errors


def test_release_evidence_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    """Duplicate JSON keys are rejected so an attacker cannot shadow status fields."""
    path = tmp_path / "release_evidence_report.json"
    path.write_text('{"status": "pass", "status": "fail"}', encoding="utf-8")

    result = validate_release_evidence(path)

    assert result.status == "fail"
    assert result.errors == ("duplicate JSON key: status",)


def _write_report(tmp_path: Path, report: object) -> Path:
    """Persist an explicitly illustrative or malformed declaration for the actual reader."""
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("gate", REQUIRED_GATES)
@pytest.mark.parametrize("section", [None, [], {}, {"status": "fail"}])
def test_required_gate_shape_and_status_refusal(tmp_path: Path, gate: str, section: object) -> None:
    """Missing objects, empty objects and fail declarations cannot admit a required gate."""
    report = _valid_report()
    report[gate] = section
    result = validate_release_evidence(_write_report(tmp_path, report))
    assert result.status == "fail" and result.report_sha256 is not None
    assert gate not in result.admitted_gates
    assert any(gate in error for error in result.errors)


@pytest.mark.parametrize(
    "gate,key,value,error_fragment",
    [
        ("", "status", [], "status must"),
        ("", "transport_solver_available", 1, "transport_solver_available"),
        ("", "import_clean", "true", "import_clean"),
        ("data_manifests", "total", True, "total"),
        ("data_manifests", "total", 0, "total"),
        ("data_manifests", "total", 2.0, "total"),
        ("data_manifests", "real", True, "real"),
        ("data_manifests", "real", -1, "real"),
        ("data_manifests", "synthetic", "1", "synthetic"),
        ("data_manifests", "artifact_coverage", None, "artifact_coverage"),
        ("data_manifests", "artifact_coverage", {}, "counts"),
        ("data_manifests", "artifact_coverage", {"expected": True, "covered": 1, "missing": []}, "counts"),
        ("data_manifests", "artifact_coverage", {"expected": 1, "covered": False, "missing": []}, "counts"),
        ("data_manifests", "artifact_coverage", {"expected": 2, "covered": 1, "missing": []}, "cover every"),
        ("data_manifests", "artifact_coverage", {"expected": 1, "covered": 1, "missing": ["a"]}, "missing"),
        ("jax_gk_parity", "parity_artifacts", False, "parity_artifacts"),
        ("jax_gk_parity", "required_cases", [{}], "required_cases"),
        ("jax_gk_parity", "required_cases", [], "required_cases"),
        ("jax_gk_parity", "required_cases", "cyclone_base_case", "required_cases"),
        ("jax_gk_parity", "required_backends", None, "required_backends"),
        ("jax_gk_parity", "required_backends", ["cpu"], "required_backends"),
        ("jax_gk_parity", "entries", None, "entries must be a list"),
        ("jax_gk_parity", "entries", [None], "string case and backend"),
        ("jax_gk_parity", "entries", [{"case": [], "backend": "cpu"}], "string case and backend"),
        ("jax_gk_parity", "entries", [{"case": "stable_mode", "backend": {}}], "string case and backend"),
        ("physics_traceability", "total", 0, "total"),
        ("physics_traceability", "open_fidelity_gaps", [], "open_fidelity_gaps"),
        ("physics_traceability", "public_claim_blocked", None, "public_claim_blocked"),
        ("physics_traceability", "public_claim_blocked", 52, "block every"),
        ("multi_shot_campaign", "admitted_surfaces", None, "list of strings"),
        ("multi_shot_campaign", "admitted_surfaces", [{}], "list of strings"),
        ("multi_shot_campaign", "admitted_surfaces", [], "include python"),
        ("multi_shot_campaign", "pyo3_status", "unavailable", "pyo3_status"),
        ("multi_shot_campaign", "python_report_sha256", "A" * 64, "python_report_sha256"),
        ("multi_shot_campaign", "rust_report_sha256", None, "rust_report_sha256"),
        ("multi_shot_campaign", "python_payload_sha256", "x", "python_payload_sha256"),
        ("multi_shot_campaign", "rust_payload_sha256", "G" * 64, "rust_payload_sha256"),
        ("multi_shot_campaign", "minimum_digest_count", 0, "minimum_digest_count"),
        ("multi_shot_campaign", "production_claim_allowed", 0, "boolean"),
        ("multi_shot_campaign", "errors", ["bad"], "errors must be empty"),
        ("runtime_admission", "report_sha256", "x", "report_sha256"),
        ("runtime_admission", "payload_sha256", False, "payload_sha256"),
        ("runtime_admission", "benchmark_evidence_class", {}, "recognised"),
        ("runtime_admission", "benchmark_evidence_class", "unknown", "recognised"),
        ("runtime_admission", "production_claim_allowed", 1, "boolean"),
        ("runtime_admission", "admission_status", [], "admission_status"),
        ("runtime_admission", "admission_status", "unknown", "admission_status"),
        ("runtime_admission", "admission_error_count", -1, "admission_error_count"),
        ("runtime_admission", "samples", 0, "samples"),
        ("runtime_admission", "errors", None, "errors must be empty"),
        ("native_formal_certificate", "admitted_cases", None, "non-empty list"),
        ("native_formal_certificate", "admitted_cases", [], "non-empty list"),
        ("native_formal_certificate", "admitted_cases", [1], "AOT certificate case labels"),
        ("native_formal_certificate", "admitted_cases", ["wrong"], "AOT certificate case labels"),
        ("native_formal_certificate", "report_sha256", "a" * 63, "report_sha256"),
        ("native_formal_certificate", "benchmark_evidence_class", {}, "recognised"),
        ("native_formal_certificate", "benchmark_evidence_class", "unknown", "recognised"),
        ("native_formal_certificate", "production_claim_allowed", None, "boolean"),
    ],
)
def test_public_reader_refuses_malformed_declarations(
    tmp_path: Path, gate: str, key: str, value: object, error_fragment: str
) -> None:
    """Malformed JSON field domains become deterministic findings instead of runtime type errors or false PASS."""
    report = _valid_report()
    target = cast(dict[str, Any], report[gate]) if gate else report
    target[key] = value
    result = validate_release_evidence(_write_report(tmp_path, report))
    assert result.status == "fail"
    assert result.report_sha256 is not None
    assert any(error_fragment in error for error in result.errors), result.errors


def test_declared_production_and_zero_coverage_domains(tmp_path: Path) -> None:
    """Well-typed production flags and zero expected artifacts pass declarations without attesting timing or artifacts."""
    report = _valid_report()
    coverage = cast(dict[str, Any], report["data_manifests"])
    coverage["artifact_coverage"] = {"expected": 0, "covered": 0, "missing": []}
    for gate in ["runtime_admission", "native_formal_certificate"]:
        section = cast(dict[str, Any], report[gate])
        section["benchmark_evidence_class"] = "production_benchmark"
        section["production_claim_allowed"] = True
    runtime = cast(dict[str, Any], report["runtime_admission"])
    runtime["admission_status"] = "pass"
    runtime["admission_error_count"] = 0
    report["extra"] = {"finite_float": 1.25, "unvalidated_metadata": ["accepted"]}
    result = validate_release_evidence(_write_report(tmp_path, report))
    assert result.status == "pass" and not result.errors


@pytest.mark.parametrize(
    "blob,fragment",
    [
        (b"[]", "root must be"),
        (b"{", "Expecting"),
        (b"\xff", "utf-8"),
        (b'{"extra":{"x":1,"x":2}}', "duplicate JSON key"),
        (b'{"extra":NaN}', "nonfinite"),
        (b'{"extra":Infinity}', "nonfinite"),
        (b'{"extra":-Infinity}', "nonfinite"),
        (b'{"extra":[1e999]}', "nonfinite"),
        pytest.param(None, "root must be", id="deep-nonobject-root"),
    ],
)
def test_report_decoder_failures_are_structured(tmp_path: Path, blob: bytes | None, fragment: str) -> None:
    """Invalid encoding, shadowed keys, nonfinite tokens and deeply nested non-object roots fail before digest admission."""
    path = tmp_path / "raw.json"
    deep_root = blob is None
    if blob is None:
        depth = sys.getrecursionlimit() + 100
        blob = b"[" * depth + b"0" + b"]" * depth
    path.write_bytes(blob)
    result = validate_release_evidence(path)
    assert result.status == "fail" and result.report_sha256 is None and result.admitted_gates == ()
    allowed = (fragment, "recursion") if deep_root else (fragment,)
    assert any(item.lower() in result.errors[0].lower() for item in allowed)


@pytest.mark.parametrize("directory", [False, True])
def test_unreadable_report_refuses_without_digest(tmp_path: Path, directory: bool) -> None:
    """Missing files and actual directory reads report IO failure without a byte digest."""
    path = tmp_path / "report.json"
    if directory:
        path.mkdir()
    result = validate_release_evidence(path)
    assert result.status == "fail" and result.report_sha256 is None and result.errors


def test_digest_tracks_exact_bytes_and_gate_labels_are_declarations(tmp_path: Path) -> None:
    """Different whitespace changes the input hash; declared pass gates remain labelled when their fields fail."""
    import hashlib

    report = _valid_report()
    section = cast(dict[str, Any], report["runtime_admission"])
    section["samples"] = 0
    path = _write_report(tmp_path, report)
    result = validate_release_evidence(path)
    assert result.status == "fail" and result.admitted_gates == REQUIRED_GATES
    assert result.report_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    changed = validate_release_evidence(path)
    assert changed.report_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert changed.report_sha256 != result.report_sha256 and changed.errors == result.errors


@pytest.mark.parametrize("passing", [False, True])
@pytest.mark.parametrize("json_out", [False, True])
def test_native_and_registered_commands_execute_reader(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], passing: bool, json_out: bool
) -> None:
    """Both actual CLI entry points produce matching refusal/pass declarations and exit codes without invoking producers."""
    report = _valid_report()
    if not passing:
        report["import_clean"] = False
    path = _write_report(tmp_path, report)
    args = [str(path)] + (["--json-out"] if json_out else [])
    assert main(args) == (0 if passing else 1)
    native = capsys.readouterr()
    result = CliRunner().invoke(control_cli, ["validate-release-evidence", *args])
    assert result.exit_code == (0 if passing else 1), result.output
    assert path.read_text() == json.dumps(report) + "\n"
    if json_out:
        assert json.loads(native.out) == json.loads(result.output)
        assert json.loads(native.out)["schema_version"] == release_module.RELEASE_EVIDENCE_SCHEMA_VERSION
    else:
        assert f"Release evidence: {'pass' if passing else 'fail'}" in native.out
        assert "Report SHA-256:" in native.out and "Report SHA-256:" in result.output
        if not passing:
            assert "ERROR import_clean" in native.out and "ERROR import_clean" in result.output


def test_native_text_decoder_failure_and_parser_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Unparsable native text has no digest; missing required parser argument preserves exit2."""
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    assert main([str(path)]) == 1
    output = capsys.readouterr().out
    assert "Release evidence: fail" in output and "ERROR" in output and "Report SHA-256:" not in output
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


@pytest.mark.parametrize("passing", [False, True])
def test_standalone_stdlib_cli_from_other_directory(tmp_path: Path, passing: bool) -> None:
    """The actual -S script needs no installed project or PYTHONPATH and exits according to declaration findings."""
    report = _valid_report()
    if not passing:
        section = cast(dict[str, Any], report["jax_gk_parity"])
        section["required_cases"] = [{}]
    path = _write_report(tmp_path, report)
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    result = subprocess.run(
        [sys.executable, "-S", str(Path(release_module.__file__)), str(path), "--json-out"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == (0 if passing else 1), result.stderr
    assert not result.stderr and json.loads(result.stdout)["status"] == ("pass" if passing else "fail")


def test_native_release_example_runs_actual_reader() -> None:
    """Execute the owning source's negative example against an actual temporary invalid file."""
    result = doctest.testmod(release_module, raise_on_error=True)
    assert result.failed == 0 and result.attempted == 3
