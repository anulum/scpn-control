# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tracker #53 evidence-gate tests
"""Tests for the tracker #53 hardware/runtime evidence manifest gate."""

from __future__ import annotations

import doctest
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

import validation.validate_tracker53_evidence as tracker_module
from validation.validate_tracker53_evidence import (
    DEFAULT_NATIVE_FORMAL_REPORT,
    DEFAULT_REGISTRY,
    DEFAULT_RUNTIME_REPORT,
    DEFAULT_Z3_REPORT,
    TRACKER53_MODULE_ORDER,
    TRACKER53_SCHEMA_VERSION,
    build_tracker53_manifest,
    main,
    validate_tracker53_evidence,
)


def test_repository_tracker53_manifest_is_bounded_and_blocked() -> None:
    """Actual historical defaults admit only bounded local declarations."""
    result = validate_tracker53_evidence()

    assert result.status == "pass"
    assert result.production_claim_allowed is False
    assert result.errors == ()
    assert result.tracker_issue == 53
    assert {entry["module_path"] for entry in result.entries} == {
        "src/scpn_control/core/checkpoint.py",
        "src/scpn_control/phase/kuramoto.py",
        "src/scpn_control/scpn/formal_verification.py",
        "src/scpn_control/scpn/fpga_export.py",
        "src/scpn_control",
        "src/scpn_control/core/runtime_admission.py",
    }
    assert result.evidence_classes["src/scpn_control/core/runtime_admission.py"] == "runtime_local_regression"
    assert result.evidence_classes["src/scpn_control/scpn/fpga_export.py"] == "generated_hdl"


def test_tracker53_production_claim_requirement_fails_closed() -> None:
    """Fixed unqualified surfaces block aggregate production regardless of local PASS."""
    result = validate_tracker53_evidence(require_production_claim=True)

    assert result.status == "fail"
    assert result.production_claim_allowed is False
    assert any("production tracker #53 claim requires qualified hardware evidence" in error for error in result.errors)


def test_tracker53_manifest_is_digest_bound() -> None:
    """The public builder assigns the declared v1 consistency digest."""
    result = validate_tracker53_evidence()
    manifest = build_tracker53_manifest(result)

    assert manifest["schema_version"] == TRACKER53_SCHEMA_VERSION
    assert manifest["status"] == "pass"
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["production_claim_allowed"] is False


def test_tracker53_manifest_json_output(tmp_path: Path) -> None:
    """An explicit scratch output path produces the actual manifest bytes."""
    output = tmp_path / "tracker53.json"

    result = validate_tracker53_evidence(output_json=output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.status == "pass"
    assert payload["schema_version"] == TRACKER53_SCHEMA_VERSION
    assert payload["manifest_sha256"] == build_tracker53_manifest(result)["manifest_sha256"]
    assert payload["tracker_issue"] == 53


def test_tracker53_rejects_missing_runtime_report(tmp_path: Path) -> None:
    """Actual missing runtime input propagates the public lower-reader finding."""
    missing = tmp_path / "missing_runtime_report.json"

    result = validate_tracker53_evidence(runtime_report=missing)

    assert result.status == "fail"
    assert any("runtime_admission.report" in error for error in result.errors)


def test_tracker53_cli_reports_fail_closed_production_requirement(capsys: pytest.CaptureFixture[str]) -> None:
    """The real public CLI retains the production refusal and JSON declaration boundary."""
    import validation.validate_tracker53_evidence as mod

    assert mod.main(["--require-production-claim", "--json-out"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert payload["production_claim_allowed"] is False
    assert any(
        "production tracker #53 claim requires qualified hardware evidence" in error for error in payload["errors"]
    )


def _object(path: Path) -> dict[str, object]:
    """Load actual copied JSON objects with typed test metadata."""
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _write_object(path: Path, payload: dict[str, object]) -> Path:
    """Write only test-owned declarations, never a canonical artifact."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("source", ["registry", "z3"])
@pytest.mark.parametrize(
    "blob",
    [
        b"{",
        b"\xff",
        b"[]",
        b'{"x":1,"x":2}',
        b'{"extra":NaN}',
        b'{"extra":Infinity}',
        b'{"extra":-Infinity}',
        b'{"extra":1e999}',
        b"[" * 2048 + b"0" + b"]" * 2048,
    ],
)
def test_aggregate_actual_decoder_refusals(tmp_path: Path, source: str, blob: bytes) -> None:
    """Real registry/Z3 decoding refuses ambiguous or nonfinite metadata."""
    path = tmp_path / "bad.json"
    path.write_bytes(blob)
    result = validate_tracker53_evidence(
        registry=path if source == "registry" else DEFAULT_REGISTRY,
        z3_report=path if source == "z3" else DEFAULT_Z3_REPORT,
    )
    assert result.status == "fail" and result.production_claim_allowed is False


@pytest.mark.parametrize("source", ["registry", "z3", "formal", "runtime"])
@pytest.mark.parametrize("missing", [False, True])
def test_aggregate_actual_missing_or_directory_inputs(tmp_path: Path, source: str, missing: bool) -> None:
    """Actual file absence and directory reads propagate lower-reader findings."""
    path = tmp_path / "absent" if missing else tmp_path
    result = validate_tracker53_evidence(
        registry=path if source == "registry" else DEFAULT_REGISTRY,
        z3_report=path if source == "z3" else DEFAULT_Z3_REPORT,
        native_formal_report=path if source == "formal" else DEFAULT_NATIVE_FORMAL_REPORT,
        runtime_report=path if source == "runtime" else DEFAULT_RUNTIME_REPORT,
    )
    assert result.status == "fail" and result.errors and result.production_claim_allowed is False


@pytest.mark.parametrize("status", [None, {}, "fail", "unknown"])
def test_z3_status_must_declare_pass(tmp_path: Path, status: object) -> None:
    """A copied Z3 object cannot keep aggregate PASS after its declared status is invalid or failed."""
    payload = _object(DEFAULT_Z3_REPORT)
    payload["status"] = status
    path = _write_object(tmp_path / "z3.json", payload)
    result = validate_tracker53_evidence(z3_report=path)
    assert result.status == "fail" and "formal_verification.z3_report.status must be 'pass'" in result.errors
    formal = next(e for e in result.entries if e["module_path"].endswith("formal_verification.py"))
    assert formal["qualified_for_production_claim"] is False
    assert formal["z3_report"]["report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_external_z3_copy_admits_bounded_metadata(tmp_path: Path) -> None:
    """A valid actual Z3 copy outside the checkout preserves exact bytes without relative-path failure."""
    path = tmp_path / "external-z3.json"
    shutil.copyfile(DEFAULT_Z3_REPORT, path)
    result = validate_tracker53_evidence(z3_report=path)
    assert result.status == "pass" and result.production_claim_allowed is False
    formal = next(e for e in result.entries if e["module_path"].endswith("formal_verification.py"))
    assert formal["z3_report"]["path"] == str(path)
    assert formal["z3_report"]["report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("shape", [{}, {"entries": None}, {"entries": []}])
def test_empty_registry_cannot_establish_six_surfaces(tmp_path: Path, shape: dict[str, object]) -> None:
    """Decoded empty/missing registry entries become findings rather than invented surfaces."""
    path = _write_object(tmp_path / "registry.json", shape)
    result = validate_tracker53_evidence(registry=path)
    assert result.status == "fail" and result.entries == ()


@pytest.mark.parametrize("mode", ["duplicate", "invalid-path", "float-issue", "extra-unrelated"])
def test_real_registry_selection_and_refusal(tmp_path: Path, mode: str) -> None:
    """Actual registry copies enforce integer issue selection, unique paths and six defining surfaces."""
    payload = _object(DEFAULT_REGISTRY)
    entries = payload["entries"]
    assert isinstance(entries, list)
    selected = [e for e in entries if isinstance(e, dict) and e.get("external_validation_tracker_issue") == 53]
    assert len(selected) >= 6
    if mode == "duplicate":
        entries.append(dict(selected[0]))
    elif mode == "invalid-path":
        selected[0]["module_path"] = []
    elif mode == "float-issue":
        for entry in selected:
            entry["external_validation_tracker_issue"] = 53.0
    else:
        entries.extend(
            [
                0,
                {},
                {"external_validation_tracker_issue": 52},
                {"external_validation_tracker_issue": 53, "module_path": "unselected/module.py"},
            ]
        )
    path = _write_object(tmp_path / "registry.json", payload)
    result = validate_tracker53_evidence(registry=path)
    assert result.status == ("pass" if mode == "extra-unrelated" else "fail")
    assert result.production_claim_allowed is False


@pytest.mark.parametrize("require", ["false", 0, None, {}])
def test_require_production_runtime_argument_refused(require: object) -> None:
    """Non-boolean runtime arguments become typed FAIL instead of truthy/falsey policy coercion."""
    result = validate_tracker53_evidence(require_production_claim=cast(bool, require))
    assert result.status == "fail" and result.require_production_claim is False
    assert "require_production_claim must be a boolean" in result.errors


@pytest.mark.parametrize("parent_file", [False, True])
def test_actual_manifest_output_failure_is_structured(tmp_path: Path, parent_file: bool) -> None:
    """Real directory or file-parent output failures return FAIL and cannot publish a passing receipt."""
    output = tmp_path
    if parent_file:
        parent = tmp_path / "parent"
        parent.write_text("file", encoding="utf-8")
        output = parent / "result.json"
    result = validate_tracker53_evidence(output_json=output)
    assert result.status == "fail" and result.production_claim_allowed is False
    assert any("tracker53.output_json" in e for e in result.errors)


def test_actual_invalid_output_path_returns_findings() -> None:
    """A real invalid OS path through the public string API becomes a refusal, not a ValueError traceback."""
    result = validate_tracker53_evidence(output_json="bad\0name.json")
    assert result.status == "fail" and result.production_claim_allowed is False
    assert any("tracker53.output_json" in e for e in result.errors)


def _production_shaped_copies(tmp_path: Path) -> tuple[Path, Path]:
    """Relabel actual historical metadata for classification tests, never measured or certified production."""
    runtime = _object(DEFAULT_RUNTIME_REPORT)
    runtime.update(
        evidence_class="production_benchmark",
        production_claim_allowed=True,
        last_admission_status="pass",
        last_admission_errors=[],
    )
    unsigned = dict(runtime)
    unsigned["payload_sha256"] = ""
    runtime["payload_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    formal = _object(DEFAULT_NATIVE_FORMAL_REPORT)
    context = formal["benchmark_context"]
    assert isinstance(context, dict)
    context.update(
        evidence_class="production_benchmark",
        production_claim_allowed=True,
        isolation_method="declared-isolation",
        other_heavy_jobs_running=False,
    )
    formal["workspace_dirty"] = False
    return _write_object(tmp_path / "runtime.json", runtime), _write_object(tmp_path / "formal.json", formal)


@pytest.mark.parametrize("invalid", ["none", "runtime", "formal", "z3"])
def test_only_passing_reader_metadata_assigns_qualified_child_class(tmp_path: Path, invalid: str) -> None:
    """Actual production-shaped copies expose per-entry classification while aggregate operation stays blocked."""
    runtime, formal = _production_shaped_copies(tmp_path)
    z3 = DEFAULT_Z3_REPORT
    if invalid == "runtime":
        runtime.write_bytes(b"{}")
    elif invalid == "formal":
        formal.write_bytes(b"{}")
    elif invalid == "z3":
        payload = _object(DEFAULT_Z3_REPORT)
        payload["status"] = "fail"
        z3 = _write_object(tmp_path / "z3.json", payload)
    result = validate_tracker53_evidence(runtime_report=runtime, native_formal_report=formal, z3_report=z3)
    assert result.status == ("pass" if invalid == "none" else "fail")
    assert result.production_claim_allowed is False
    runtime_entry = next(e for e in result.entries if e["module_path"].endswith("runtime_admission.py"))
    formal_entry = next(e for e in result.entries if e["module_path"].endswith("formal_verification.py"))
    assert runtime_entry["qualified_for_production_claim"] is (invalid != "runtime")
    assert formal_entry["qualified_for_production_claim"] is (invalid not in ("formal", "z3"))
    blocked = validate_tracker53_evidence(
        runtime_report=runtime, native_formal_report=formal, z3_report=z3, require_production_claim=True
    )
    assert blocked.status == "fail" and blocked.production_claim_allowed is False


def test_actual_manifest_consistency_and_nested_alias_boundary(tmp_path: Path) -> None:
    """The public builder binds current metadata; nested aliases are explicit, not claimed deep immutability."""
    result = validate_tracker53_evidence()
    manifest = build_tracker53_manifest(result)
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    assert (
        manifest["manifest_sha256"]
        == hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    assert tuple(e["module_path"] for e in result.entries) == TRACKER53_MODULE_ORDER
    assert manifest["evidence_classes"] is result.evidence_classes
    assert manifest["entries"][0] is result.entries[0]
    output = tmp_path / "nested" / "manifest.json"
    written = validate_tracker53_evidence(output_json=output, require_production_claim=True)
    assert written.status == "fail"
    assert json.loads(output.read_text(encoding="utf-8")) == build_tracker53_manifest(written)


def test_tracker_example_and_public_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute the actual documented refusal and ordinary text/JSON CLI paths."""
    examples = doctest.testmod(tracker_module)
    assert examples.attempted == 2 and examples.failed == 0
    assert main([]) == 0
    assert "Tracker #53 evidence gate: pass" in capsys.readouterr().out
    assert main(["--require-production-claim"]) == 1
    assert "ERROR production tracker #53" in capsys.readouterr().out
    assert main(["--json-out"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    with pytest.raises(SystemExit) as error:
        main(["--unknown"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("require", [False, True])
@pytest.mark.parametrize("site_disabled", [False, True])
def test_real_tracker_standalone_other_cwd(tmp_path: Path, require: bool, site_disabled: bool) -> None:
    """Actual -S standalone bootstrap reaches real lower readers without site packages or cwd assumptions."""
    args = [sys.executable] + (["-S"] if site_disabled else [])
    args += [str(Path(tracker_module.__file__).resolve()), "--json-out"]
    if require:
        args += ["--require-production-claim"]
    process = subprocess.run(
        args,
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == int(require) and process.stderr == ""
    payload = json.loads(process.stdout)
    assert payload["status"] == ("fail" if require else "pass") and payload["production_claim_allowed"] is False
