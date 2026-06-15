# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — benchmark regression gate tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmark_regression_gate import (
    BASELINE_SCHEMA,
    REPORT_SCHEMA,
    VERDICT_SCHEMA,
    _payload_digest,
    canonical_metrics_digest,
    compare,
    gate,
    hardware_mismatch,
    main,
    metric_direction,
    parse_thresholds,
    resolve_threshold,
    validate_report,
    verify_baseline_integrity,
)

THRESHOLDS = {
    "default": {"p50_us": 1.5, "p95_us": 1.75, "p99_us": 2.0, "throughput_ops_s": 0.6},
}


def _benchmarks(p50: float = 100.0, throughput: float = 10_000.0) -> dict:
    return {
        "capacitor_bank_discharge": {
            "languages": {
                "python": {"p50_us": p50, "p95_us": p50 * 1.2, "p99_us": p50 * 1.4, "throughput_ops_s": throughput},
                "rust": {
                    "p50_us": p50 / 80.0,
                    "p95_us": p50 / 60.0,
                    "p99_us": p50 / 40.0,
                    "throughput_ops_s": throughput * 80.0,
                },
            },
            "cross_language_parity": {"max_relative_difference": 1.4e-16},
        }
    }


def _report(benchmarks: dict | None = None) -> dict:
    payload = {
        "schema_version": REPORT_SCHEMA,
        "generated_utc": "2026-06-15T21:00:00Z",
        "evidence_class": "local_regression",
        "production_claim_allowed": False,
        "provenance": {"commit": "abc123", "cpu_model": "test-cpu"},
        "benchmarks": benchmarks if benchmarks is not None else _benchmarks(),
    }
    payload["payload_sha256"] = _payload_digest(payload)
    return payload


def _baseline(benchmarks: dict | None = None) -> dict:
    bm = benchmarks if benchmarks is not None else _benchmarks()
    payload = {
        "schema_version": BASELINE_SCHEMA,
        "suite": "capacitor_bank",
        "baseline_commit": "def456",
        "measured_utc": "2026-06-15T20:00:00Z",
        "evidence_class": "local_regression",
        "production_claim_allowed": False,
        "provenance": {"commit": "def456"},
        "benchmarks": bm,
    }
    payload["baseline_sha256"] = canonical_metrics_digest(bm)
    return payload


# ── threshold parsing ─────────────────────────────────────────────────


def test_parse_thresholds_accepts_default_and_overrides() -> None:
    parsed = parse_thresholds({"default": {"p50_us": 1.5}, "capacitor_bank_discharge": {"p50_us": 1.2}})
    assert parsed["default"]["p50_us"] == 1.5
    assert parsed["capacitor_bank_discharge"]["p50_us"] == 1.2


def test_parse_thresholds_requires_default_table() -> None:
    with pytest.raises(ValueError, match="default"):
        parse_thresholds({"capacitor_bank_discharge": {"p50_us": 1.2}})


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
def test_parse_thresholds_rejects_non_positive_or_infinite_ratio(bad: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        parse_thresholds({"default": {"p50_us": bad}})


def test_parse_thresholds_rejects_non_numeric_ratio() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        parse_thresholds({"default": {"p50_us": "fast"}})


def test_parse_thresholds_rejects_bool_disguised_as_number() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        parse_thresholds({"default": {"p50_us": True}})


def test_resolve_threshold_prefers_benchmark_override() -> None:
    thresholds = {"default": {"p50_us": 1.5}, "bench": {"p50_us": 1.2}}
    assert resolve_threshold(thresholds, "bench", "p50_us") == 1.2
    assert resolve_threshold(thresholds, "other", "p50_us") == 1.5
    assert resolve_threshold(thresholds, "bench", "unknown_metric") is None


# ── metric direction ──────────────────────────────────────────────────


@pytest.mark.parametrize("metric", ["p50_us", "p95_us", "peak_rss_mb", "latency_us"])
def test_latency_and_memory_metrics_are_upper_bounded(metric: str) -> None:
    assert metric_direction(metric) == "upper"


@pytest.mark.parametrize("metric", ["throughput_ops_s", "rust_speedup_vs_python"])
def test_throughput_metrics_are_lower_bounded(metric: str) -> None:
    assert metric_direction(metric) == "lower"


# ── report validation ─────────────────────────────────────────────────


def test_validate_report_accepts_well_formed_report() -> None:
    assert validate_report(_report()) == []


def test_validate_report_rejects_wrong_schema() -> None:
    report = _report()
    report["schema_version"] = "something-else"
    errors = validate_report(report)
    assert any("schema_version" in e for e in errors)


def test_validate_report_rejects_empty_benchmarks() -> None:
    report = _report(benchmarks={})
    assert any("empty" in e for e in validate_report(report))


def test_validate_report_detects_tampered_payload_digest() -> None:
    report = _report()
    # Mutate a metric without recomputing payload_sha256.
    report["benchmarks"]["capacitor_bank_discharge"]["languages"]["python"]["p50_us"] = 1.0
    errors = validate_report(report)
    assert any("payload_sha256" in e for e in errors)


def test_validate_report_flags_missing_payload_digest() -> None:
    report = _report()
    del report["payload_sha256"]
    assert any("payload_sha256" in e for e in validate_report(report))


# ── baseline integrity ────────────────────────────────────────────────


def test_verify_baseline_integrity_accepts_consistent_baseline() -> None:
    assert verify_baseline_integrity(_baseline()) == []


def test_verify_baseline_integrity_rejects_checksum_mismatch() -> None:
    baseline = _baseline()
    # Tamper with a metric but keep the stale digest -> mismatch.
    baseline["benchmarks"]["capacitor_bank_discharge"]["languages"]["rust"]["p50_us"] = 0.001
    errors = verify_baseline_integrity(baseline)
    assert any("baseline_sha256" in e and "tampered" in e for e in errors)


def test_verify_baseline_integrity_flags_missing_digest() -> None:
    baseline = _baseline()
    del baseline["baseline_sha256"]
    assert any("missing baseline_sha256" in e for e in verify_baseline_integrity(baseline))


# ── comparison ────────────────────────────────────────────────────────


def test_compare_passes_when_report_matches_baseline() -> None:
    assert compare(_report(), _baseline(), THRESHOLDS) == []


def test_compare_flags_latency_regression() -> None:
    report = _report(_benchmarks(p50=400.0))  # 4x slower than baseline p50=100
    findings = compare(report, _baseline(), THRESHOLDS)
    assert any(f.kind == "regression" and f.metric == "p50_us" and f.direction == "upper" for f in findings)


def test_compare_flags_throughput_regression() -> None:
    # Drop throughput far below the 0.6 lower bound while keeping latency fine.
    bench = _benchmarks()
    bench["capacitor_bank_discharge"]["languages"]["python"]["throughput_ops_s"] = 100.0
    report = _report(bench)
    # Re-stamp not needed for compare (compare ignores payload digest).
    findings = compare(report, _baseline(), THRESHOLDS)
    assert any(f.kind == "regression" and f.metric == "throughput_ops_s" and f.direction == "lower" for f in findings)


def test_compare_fails_closed_on_missing_metric() -> None:
    bench = _benchmarks()
    del bench["capacitor_bank_discharge"]["languages"]["rust"]["p99_us"]
    report = _report(bench)
    findings = compare(report, _baseline(), THRESHOLDS)
    assert any(f.kind == "missing_metric" and f.metric == "p99_us" and f.language == "rust" for f in findings)


def test_compare_fails_closed_on_missing_benchmark() -> None:
    report = _report(benchmarks={})
    findings = compare(report, _baseline(), THRESHOLDS)
    assert findings and all(f.kind == "missing_metric" for f in findings)


def test_compare_flags_policy_gap_when_threshold_absent() -> None:
    thresholds = {"default": {"p50_us": 1.5}}  # no policy for p95/p99/throughput
    findings = compare(_report(), _baseline(), thresholds)
    assert any(f.kind == "policy_gap" and f.metric == "p95_us" for f in findings)


# ── gate verdict ──────────────────────────────────────────────────────


def test_gate_admits_clean_run() -> None:
    verdict = gate(_report(), _baseline(), THRESHOLDS, generated_utc="2026-06-15T21:30:00Z")
    assert verdict["schema_version"] == VERDICT_SCHEMA
    assert verdict["passed"] is True
    assert verdict["findings"] == []
    assert verdict["baseline_commit"] == "def456"
    assert verdict["report_commit"] == "abc123"


def test_gate_fails_on_invalid_report_before_comparing() -> None:
    report = _report()
    report["schema_version"] = "bad"
    verdict = gate(report, _baseline(), THRESHOLDS, generated_utc="t")
    assert verdict["passed"] is False
    assert any(f["kind"] == "report_invalid" for f in verdict["findings"])


def test_gate_fails_on_tampered_baseline() -> None:
    baseline = _baseline()
    baseline["baseline_sha256"] = "0" * 64
    verdict = gate(_report(), baseline, THRESHOLDS, generated_utc="t")
    assert verdict["passed"] is False
    assert any(f["kind"] == "baseline_invalid" for f in verdict["findings"])


def test_gate_verdict_digest_is_deterministic() -> None:
    verdict = gate(_report(), _baseline(), THRESHOLDS, generated_utc="fixed")
    digest = verdict.pop("payload_sha256")
    assert digest == _payload_digest(verdict)


# ── CLI / fail-closed on missing files ────────────────────────────────


def test_main_fails_closed_when_report_file_missing(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    rc = main(["--report", str(tmp_path / "nope.json"), "--baseline", str(baseline)])
    assert rc == 1


def test_main_fails_closed_when_baseline_file_missing(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    rc = main(["--report", str(report), "--baseline", str(tmp_path / "nope.json")])
    assert rc == 1


def test_main_passes_on_clean_run(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    thresholds = tmp_path / "thresholds.toml"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    thresholds.write_text(
        "[default]\np50_us = 1.5\np95_us = 1.75\np99_us = 2.0\nthroughput_ops_s = 0.6\n",
        encoding="utf-8",
    )
    verdict_out = tmp_path / "verdict.json"
    rc = main(
        [
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--thresholds",
            str(thresholds),
            "--json-out",
            str(verdict_out),
        ]
    )
    assert rc == 0
    written = json.loads(verdict_out.read_text(encoding="utf-8"))
    assert written["passed"] is True


# ── hardware-mismatch guard ───────────────────────────────────────────


def test_hardware_mismatch_flags_different_cpus() -> None:
    report = _report()
    report["provenance"]["cpu_model"] = "AMD EPYC 7763"
    baseline = _baseline()
    baseline["provenance"]["cpu_model"] = "11th Gen Intel Core i5-11600K"
    finding = hardware_mismatch(report, baseline)
    assert finding is not None and finding.kind == "hardware_mismatch"


def test_hardware_mismatch_allows_same_cpu() -> None:
    report = _report()
    report["provenance"]["cpu_model"] = "same-cpu"
    baseline = _baseline()
    baseline["provenance"]["cpu_model"] = "same-cpu"
    assert hardware_mismatch(report, baseline) is None


def test_hardware_mismatch_skips_when_cpu_unknown() -> None:
    # Missing CPU provenance must not fabricate a mismatch.
    assert hardware_mismatch(_report(), _baseline()) is None


def test_gate_fails_on_hardware_mismatch() -> None:
    report = _report()
    report["provenance"]["cpu_model"] = "cpu-a"
    report["payload_sha256"] = _payload_digest({k: v for k, v in report.items() if k != "payload_sha256"})
    baseline = _baseline()
    baseline["provenance"]["cpu_model"] = "cpu-b"
    verdict = gate(report, baseline, THRESHOLDS, generated_utc="t")
    assert verdict["passed"] is False
    assert any(f["kind"] == "hardware_mismatch" for f in verdict["findings"])


def test_main_evidence_only_never_fails(tmp_path: Path) -> None:
    # A real regression that would fail the gate must be reported but not block
    # when running in evidence-only mode on a generic runner.
    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    thresholds = tmp_path / "thresholds.toml"
    regressed = _report(_benchmarks(p50=10_000.0))  # far above baseline
    report.write_text(json.dumps(regressed), encoding="utf-8")
    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    thresholds.write_text(
        "[default]\np50_us = 1.5\np95_us = 1.75\np99_us = 2.0\nthroughput_ops_s = 0.6\n", encoding="utf-8"
    )
    verdict_out = tmp_path / "verdict.json"
    rc = main(
        [
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--thresholds",
            str(thresholds),
            "--json-out",
            str(verdict_out),
            "--evidence-only",
        ]
    )
    assert rc == 0
    written = json.loads(verdict_out.read_text(encoding="utf-8"))
    assert written["passed"] is False  # the verdict still records the regression


def test_main_fails_closed_on_invalid_threshold_file(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    thresholds = tmp_path / "thresholds.toml"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    # No [default] table -> parse_thresholds raises -> gate fails closed.
    thresholds.write_text("[capacitor_bank_discharge]\np50_us = 1.2\n", encoding="utf-8")
    rc = main(["--report", str(report), "--baseline", str(baseline), "--thresholds", str(thresholds)])
    assert rc == 1
