# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native Formal Certificate Evidence Tests

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_native_formal_certificate_evidence import (
    BENCHMARK_CONTEXT_SCHEMA_VERSION,
    CERTIFICATE_ID,
    CERTIFICATE_SCHEMA_VERSION,
    validate_native_formal_certificate_evidence,
)


def _summary(*, digest: str = "a" * 64, dropped: int = 0, failures: int = 0, p99: float = 2.0) -> dict[str, object]:
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_native_formal_certificate_evidence_admits_complete_report(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "report.json", _payload())

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "pass"
    assert result.admitted_cases == ("std:spin:aot_certificate:stride_1",)
    assert result.certificate_assumption_sha256 == "a" * 64
    assert result.benchmark_evidence_class == "local_regression"
    assert result.production_claim_allowed is False
    assert result.errors == ()


def test_native_formal_certificate_evidence_rejects_drops(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "report.json", _payload(formal_dropped_total=1))

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "fail"
    assert any("dropped checks must be zero" in error for error in result.errors)


def test_native_formal_certificate_evidence_rejects_threshold_regression(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "report.json", _payload(avg_cycle_us={"p99": 20.0}))

    result = validate_native_formal_certificate_evidence(report, max_aot_p99_cycle_us=10.0)

    assert result.status == "fail"
    assert any("exceeds" in error for error in result.errors)


def test_native_formal_certificate_evidence_rejects_digest_instability(tmp_path: Path) -> None:
    payload = _payload()
    summaries = payload["summaries"]
    assert isinstance(summaries, dict)
    summaries["std:sleep:aot_certificate:stride_1"] = _summary(digest="b" * 64)
    report = _write_report(tmp_path / "report.json", payload)

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "fail"
    assert any("stable across admitted cases" in error for error in result.errors)


def test_native_formal_certificate_evidence_rejects_missing_benchmark_context(tmp_path: Path) -> None:
    payload = _payload()
    del payload["benchmark_context"]
    report = _write_report(tmp_path / "report.json", payload)

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "fail"
    assert "benchmark_context must be an object" in result.errors


def test_native_formal_certificate_evidence_rejects_unidentified_command(tmp_path: Path) -> None:
    payload = _payload()
    context = payload["benchmark_context"]
    assert isinstance(context, dict)
    context["command"] = ["python", "other_benchmark.py"]
    report = _write_report(tmp_path / "report.json", payload)

    result = validate_native_formal_certificate_evidence(report)

    assert result.status == "fail"
    assert "benchmark_context.command must identify the native formal benchmark" in result.errors


def test_native_formal_certificate_evidence_rejects_unisolated_production_claim(tmp_path: Path) -> None:
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
    result = validate_native_formal_certificate_evidence()

    assert result.status == "pass"
    assert result.benchmark_evidence_class == "local_regression"
    assert result.production_claim_allowed is False
    assert result.certificate_assumption_sha256 == ("ee058c7c918ce8eb800c03e0c6e5ae979ba01f95dc48c6da3dc3c1f63391fdfd")
