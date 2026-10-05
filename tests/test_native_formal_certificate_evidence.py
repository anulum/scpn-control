# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native Formal Certificate Evidence Tests

"""Exercise real persisted reports and legacy schema fixtures through public readers.

Legacy test CPU names, digest strings and summary metrics are invented schema
examples, not measured or authenticated proof/control evidence. New negatives
copy the actual historical report; no solver, native extension or production
qualification is mocked.
"""

from __future__ import annotations

import doctest
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

import validation.validate_native_formal_certificate_evidence as native_module
from validation.validate_native_formal_certificate_evidence import (
    BENCHMARK_CONTEXT_SCHEMA_VERSION,
    CERTIFICATE_ID,
    CERTIFICATE_SCHEMA_VERSION,
    DEFAULT_REPORT,
    main,
    validate_native_formal_certificate_evidence,
)


def _summary(*, digest: str = "a" * 64, dropped: int = 0, failures: int = 0, p99: float = 2.0) -> dict[str, object]:
    """Build legacy schema-only summary declarations without a claimed measurement."""
    return {
        "runs": 2,
        "avg_cycle_us": {"min": 1.0, "p50": 1.5, "p95": 1.8, "p99": p99, "max": p99, "mean": 1.6},
        "effective_step_us": {"min": 100.0, "p50": 100.0, "p95": 100.0, "p99": 100.0, "max": 100.0, "mean": 100.0},
        "formal_generated_total": 20,
        "formal_submitted_total": 20,
        "formal_checked_total": 20,
        "formal_dropped_total": dropped,
        "formal_failures_total": failures,
        "certificate_admitted_total": 2,
        "certificate_schema_versions": [CERTIFICATE_SCHEMA_VERSION],
        "certificate_ids": [CERTIFICATE_ID],
        "certificate_assumption_sha256_values": [digest],
        "sync_wait_count_total": 0,
        "sync_wait_p99_ns_max": 0,
        "drops_total": 0,
        "publish_failures_total": 0,
        "udp_sink_packets_total": 20,
        "safety_headroom_pct_p99_cycle": 98.0,
    }


def _payload(**summary_overrides: object) -> dict[str, object]:
    """Build a legacy schema fixture; labels/metrics are not authenticated native evidence."""
    summary = _summary()
    summary.update(summary_overrides)
    return {
        "schema": "scpn-control.native_formal_modes.v1",
        "workspace_dirty": True,
        "benchmark_context": {
            "schema_version": BENCHMARK_CONTEXT_SCHEMA_VERSION,
            "evidence_class": "local_regression",
            "production_claim_allowed": False,
            "command": ["python", "scripts/benchmark_native_formal_modes.py"],
            "affinity_cpus": [0, 1],
            "reserved_core_set": [0, 1],
            "isolation_method": "none",
            "host_load_before": "1.00 1.00 1.00 1/100 1",
            "host_load_after": "1.01 1.00 1.00 1/100 2",
            "cpu_governor": "performance",
            "cpu_frequency_context": "min_khz=1000000 max_khz=1000000",
            "hardware_model": "test cpu",
            "os": "test os",
            "python": "3.12",
            "runtime_versions": {"python": "3.12"},
            "other_heavy_jobs_running": "unknown",
            "claim_boundary": "local regression evidence only",
        },
        "summaries": {
            "std:spin:aot_certificate:stride_1": summary,
            "std:spin:disabled:stride_30": {
                **_summary(digest=""),
                "formal_generated_total": 0,
                "formal_submitted_total": 0,
                "formal_checked_total": 0,
                "certificate_admitted_total": 0,
                "certificate_schema_versions": [],
                "certificate_ids": [],
                "certificate_assumption_sha256_values": [],
            },
        },
    }


def _write_report(path: Path, payload: dict[str, object]) -> Path:
    """Write test-owned JSON declaration bytes, never canonical evidence."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_native_formal_certificate_evidence_admits_complete_report(tmp_path: Path) -> None:
    """Legacy schema-only declarations admit the expected AOT label without physical proof."""
    report = _write_report(tmp_path / "report.json", _payload())

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "pass"
    assert result.admitted_cases == ("std:spin:aot_certificate:stride_1",)
    assert result.certificate_assumption_sha256 == "a" * 64
    assert result.benchmark_evidence_class == "local_regression"
    assert result.production_claim_allowed is False
    assert result.errors == ()


def test_native_formal_certificate_evidence_rejects_drops(tmp_path: Path) -> None:
    """Declared dropped checks refuse otherwise complete schema metadata."""
    report = _write_report(tmp_path / "report.json", _payload(formal_dropped_total=1))

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "fail"
    assert any("dropped checks must be zero" in error for error in result.errors)


def test_native_formal_certificate_evidence_rejects_threshold_regression(tmp_path: Path) -> None:
    """Declared p99 above the public bound returns a threshold finding."""
    report = _write_report(tmp_path / "report.json", _payload(avg_cycle_us={"p99": 20.0}))

    result = validate_native_formal_certificate_evidence(report, max_aot_p99_cycle_us=10.0)

    assert result.status == "fail"
    assert any("exceeds" in error for error in result.errors)


def test_native_formal_certificate_evidence_rejects_digest_instability(tmp_path: Path) -> None:
    """Conflicting declared certificate digest spellings fail aggregate stability."""
    payload = _payload()
    summaries = payload["summaries"]
    assert isinstance(summaries, dict)
    summaries["std:sleep:aot_certificate:stride_1"] = _summary(digest="b" * 64)
    report = _write_report(tmp_path / "report.json", payload)

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "fail"
    assert any("stable across admitted cases" in error for error in result.errors)


def test_native_formal_certificate_evidence_rejects_missing_benchmark_context(tmp_path: Path) -> None:
    """An absent context fails required metadata admission."""
    payload = _payload()
    del payload["benchmark_context"]
    report = _write_report(tmp_path / "report.json", payload)

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "fail"
    assert "benchmark_context must be an object" in result.errors


def test_native_formal_certificate_evidence_rejects_unidentified_command(tmp_path: Path) -> None:
    """The declared command must identify the actual producer script."""
    payload = _payload()
    context = payload["benchmark_context"]
    assert isinstance(context, dict)
    context["command"] = ["python", "other_benchmark.py"]
    report = _write_report(tmp_path / "report.json", payload)

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "fail"
    assert "benchmark_context.command must identify the native formal benchmark" in result.errors


def test_native_formal_certificate_evidence_rejects_unisolated_production_claim(tmp_path: Path) -> None:
    """Production-labelled schema metadata with missing isolation and dirty workspace fails."""
    payload = _payload()
    context = payload["benchmark_context"]
    assert isinstance(context, dict)
    context["evidence_class"] = "production_benchmark"
    context["production_claim_allowed"] = True
    context["isolation_method"] = "none"
    context["other_heavy_jobs_running"] = "unknown"
    report = _write_report(tmp_path / "report.json", payload)

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "fail"
    assert "production benchmark evidence requires an explicit CPU/core isolation method" in result.errors
    assert "production benchmark evidence must declare whether other heavy jobs were running" in result.errors
    assert "production benchmark evidence must not come from a dirty workspace" in result.errors


def test_repository_native_formal_certificate_evidence_is_admitted() -> None:
    """Actual historical local evidence admits its recorded stable certificate digest."""
    result = validate_native_formal_certificate_evidence()

    assert result.status == "pass"
    assert result.benchmark_evidence_class == "local_regression"
    assert result.production_claim_allowed is False
    assert result.certificate_assumption_sha256 == ("ee058c7c918ce8eb800c03e0c6e5ae979ba01f95dc48c6da3dc3c1f63391fdfd")


def _real_payload() -> dict[str, object]:
    """Read the actual historical report as mutable copied declarations."""
    payload: object = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _mutate_real(tmp_path: Path, keys: tuple[str, ...], value: object) -> Path:
    """Write a single schema-negative mutation of the historical report to scratch."""
    payload = _real_payload()
    parent = payload
    for key in keys[:-1]:
        child = parent[key]
        assert isinstance(child, dict)
        parent = cast(dict[str, object], child)
    parent[keys[-1]] = value
    return _write_report(tmp_path / "real-copy.json", payload)


@pytest.mark.parametrize(
    "keys,value,fragment",
    [
        (("schema",), "wrong", "schema must"),
        (("benchmark_context",), None, "benchmark_context must"),
        (("benchmark_context", "schema_version"), None, "schema_version"),
        (("benchmark_context", "evidence_class"), {}, "evidence_class"),
        (("benchmark_context", "evidence_class"), "unknown", "evidence_class"),
        (("benchmark_context", "production_claim_allowed"), {}, "must be a boolean"),
        (("benchmark_context", "production_claim_allowed"), True, "local regression"),
        (("benchmark_context", "command"), [], "command"),
        (("benchmark_context", "command"), [None], "command"),
        (("benchmark_context", "command"), ["other"], "identify"),
        (("benchmark_context", "affinity_cpus"), [], "affinity_cpus"),
        (("benchmark_context", "affinity_cpus"), [True], "affinity_cpus"),
        (("benchmark_context", "reserved_core_set"), [-1], "reserved_core_set"),
        (("benchmark_context", "isolation_method"), "", "isolation_method"),
        (("benchmark_context", "hardware_model"), None, "hardware_model"),
        (("benchmark_context", "runtime_versions"), {}, "runtime_versions"),
        (("benchmark_context", "runtime_versions"), None, "runtime_versions"),
        (("benchmark_context", "other_heavy_jobs_running"), {}, "heavy_jobs"),
        (("summaries",), None, "summaries"),
    ],
)
def test_real_native_context_refusals(tmp_path: Path, keys: tuple[str, ...], value: object, fragment: str) -> None:
    """Actual copied metadata refuses required shape/type/context domains without coercion."""
    report = _mutate_real(tmp_path, keys, value)
    result = validate_native_formal_certificate_evidence(report)
    assert result.status == "fail" and any(fragment in e for e in result.errors)
    assert result.report_sha256 == hashlib.sha256(report.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "field,value,fragment",
    [
        ("certificate_admitted_total", True, "certificate_admitted_total"),
        ("certificate_admitted_total", 1, "every AOT run"),
        ("runs", 0, "runs"),
        ("runs", True, "runs"),
        ("formal_generated_total", 0, "generate"),
        ("formal_generated_total", None, "non-negative integer"),
        ("formal_submitted_total", 1, "submitted checks"),
        ("formal_submitted_total", True, "non-negative integer"),
        ("formal_checked_total", 1, "checked proofs"),
        ("formal_checked_total", None, "non-negative integer"),
        ("formal_dropped_total", -1, "dropped"),
        ("formal_failures_total", True, "failures"),
        ("certificate_schema_versions", [], "schema version"),
        ("certificate_ids", [], "id mismatch"),
        ("certificate_assumption_sha256_values", [], "exactly one"),
        ("certificate_assumption_sha256_values", [None], "exactly one"),
        ("certificate_assumption_sha256_values", ["g" * 64], "exactly one"),
        ("certificate_assumption_sha256_values", ["a" * 63], "exactly one"),
        ("avg_cycle_us", None, "avg_cycle_us"),
        ("avg_cycle_us", {"p99": 0}, "positive and finite"),
        ("avg_cycle_us", {"p99": True}, "positive and finite"),
        ("avg_cycle_us", {"p99": 10**1000}, "positive and finite"),
        ("avg_cycle_us", {"p99": 11.0}, "exceeds"),
    ],
)
def test_real_native_case_refusals(tmp_path: Path, field: str, value: object, fragment: str) -> None:
    """Real AOT summary mutations reach count/certificate/threshold refusals without a fake proof engine."""
    payload = _real_payload()
    summaries = payload["summaries"]
    assert isinstance(summaries, dict)
    label = next(k for k in summaries if ":aot_certificate:" in k)
    summary = summaries[label]
    assert isinstance(summary, dict)
    summary[field] = value
    report = _write_report(tmp_path / "case.json", payload)
    result = validate_native_formal_certificate_evidence(report)
    assert result.status == "fail" and any(fragment in e for e in result.errors)


@pytest.mark.parametrize("threshold", [0.0, -1.0, True, None, "bad", {}, float("nan"), float("inf"), 10**1000])
def test_invalid_actual_threshold_returns_findings(threshold: object) -> None:
    """Runtime API misuse cannot crash or apply an undefined AOT latency bound."""
    result = validate_native_formal_certificate_evidence(max_aot_p99_cycle_us=cast(float, threshold))
    assert result.status == "fail" and result.admitted_cases == ()
    assert result.certificate_assumption_sha256 is None
    assert "max_aot_p99_cycle_us must be positive and finite" in result.errors


@pytest.mark.parametrize(
    "blob",
    [
        b"{",
        b"\xff",
        b"[]",
        b"null",
        b'{"x":1,"x":2}',
        b'{"extra":NaN}',
        b'{"extra":Infinity}',
        b'{"extra":-Infinity}',
        b'{"extra":1e999}',
        b"[" * 2048 + b"0" + b"]" * 2048,
    ],
)
def test_native_actual_decoder_refusals(tmp_path: Path, blob: bytes) -> None:
    """Actual byte decoding refuses ambiguous/non-object/nonfinite carriers without tracebacks."""
    report = tmp_path / "bad.json"
    report.write_bytes(blob)
    result = validate_native_formal_certificate_evidence(report)
    assert result.status == "fail" and result.report_sha256 is None


@pytest.mark.parametrize("missing", [False, True])
def test_native_actual_read_errors(tmp_path: Path, missing: bool) -> None:
    """Actual absent files and directory inputs become read findings."""
    result = validate_native_formal_certificate_evidence(tmp_path / "absent" if missing else tmp_path)
    assert result.status == "fail" and result.report_sha256 is None


@pytest.mark.parametrize("dirty", [None, "false", True])
def test_production_context_requires_literal_clean_declaration(tmp_path: Path, dirty: object) -> None:
    """Copied production labels do not manufacture a clean-workspace admission from missing/ill-typed flags."""
    payload = _real_payload()
    context = payload["benchmark_context"]
    assert isinstance(context, dict)
    context.update(
        evidence_class="production_benchmark",
        production_claim_allowed=True,
        isolation_method="declared-isolation",
        other_heavy_jobs_running=False,
    )
    payload["workspace_dirty"] = dirty
    report = _write_report(tmp_path / "production-copy.json", payload)
    result = validate_native_formal_certificate_evidence(report)
    assert result.status == "fail" and result.production_claim_allowed is True
    assert any("workspace" in e for e in result.errors)


def test_copied_labels_expose_metadata_and_nonchecks(tmp_path: Path) -> None:
    """A resealed production-labelled copy tests declaration semantics, not measured hardware qualification."""
    payload = _real_payload()
    context = payload["benchmark_context"]
    assert isinstance(context, dict)
    context.update(
        evidence_class="production_benchmark",
        production_claim_allowed=True,
        isolation_method="declared-isolation",
        other_heavy_jobs_running=True,
    )
    payload["workspace_dirty"] = False
    payload["extra"] = {"finite": 0.25}
    report = _write_report(tmp_path / "declared.json", payload)
    result = validate_native_formal_certificate_evidence(report)
    assert result.status == "pass" and result.production_claim_allowed is True
    assert result.report_sha256 == hashlib.sha256(report.read_bytes()).hexdigest()
    mapping = result.as_dict()
    errors = mapping["errors"]
    cases = mapping["admitted_cases"]
    assert isinstance(errors, list) and isinstance(cases, list)
    errors.append("caller")
    cases.clear()
    assert result.errors == () and result.admitted_cases


def test_production_class_requires_true_declaration_flag(tmp_path: Path) -> None:
    """Actual local report relabelled as production cannot admit while retaining a false declaration flag."""
    report = _mutate_real(tmp_path, ("benchmark_context", "evidence_class"), "production_benchmark")
    result = validate_native_formal_certificate_evidence(report)
    assert result.status == "fail" and result.production_claim_allowed is False
    assert "production benchmark evidence must set production_claim_allowed=true" in result.errors


def test_non_aot_and_missing_summary_shapes(tmp_path: Path) -> None:
    """Non-AOT objects do not admit a certificate; malformed summary values still refuse."""
    shapes: list[object] = [{}, None]
    for value in shapes:
        report = _mutate_real(tmp_path, ("summaries",), {"non-aot": value})
        result = validate_native_formal_certificate_evidence(report)
        assert result.status == "fail" and result.admitted_cases == ()
        assert "at least one AOT certificate case must be admitted" in result.errors


def test_native_example_and_public_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Execute the real documented example and public JSON CLI, including parsing refusal."""
    examples = doctest.testmod(native_module)
    assert examples.attempted == 2 and examples.failed == 0
    assert main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"{}")
    assert main([str(empty)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "fail"
    with pytest.raises(SystemExit) as error:
        main(["--unknown"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_native_standalone_other_cwd_without_site_packages(tmp_path: Path) -> None:
    """Real standard-library standalone invocation reads default evidence independently of cwd/PYTHONPATH."""
    process = subprocess.run(
        [sys.executable, "-S", str(Path(native_module.__file__).resolve())],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0 and process.stderr == ""
    assert json.loads(process.stdout)["status"] == "pass"
