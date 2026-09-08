# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — benchmark regression gate tests

"""Exercise benchmark admission, ratio policies and real command-line verdicts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
    """Build synthetic Python/Rust metric fixtures in microseconds and operations/s."""
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
    """Stamp supplied test metrics as a digest-consistent, non-production report."""
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
    """Stamp a synthetic baseline while preserving deliberately invalid test metrics."""
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
    """Accept a valid default table and a benchmark-specific override."""
    parsed = parse_thresholds({"default": {"p50_us": 1.5}, "capacitor_bank_discharge": {"p50_us": 1.2}})
    assert parsed["default"]["p50_us"] == 1.5
    assert parsed["capacitor_bank_discharge"]["p50_us"] == 1.2


def test_parse_thresholds_requires_default_table() -> None:
    """Refuse a policy that supplies overrides without a default table."""
    with pytest.raises(ValueError, match="default"):
        parse_thresholds({"capacitor_bank_discharge": {"p50_us": 1.2}})


@pytest.mark.parametrize(
    "bad", [0.0, -1.0, float("inf"), float("nan"), 10**400], ids=["zero", "negative", "infinity", "nan", "overflow"]
)
def test_parse_thresholds_rejects_invalid_numeric_ratio(bad: float) -> None:
    """Refuse unusable ratio bounds before comparing metrics."""
    with pytest.raises(ValueError, match="positive and finite"):
        parse_thresholds({"default": {"p50_us": bad}})


def test_parse_thresholds_rejects_non_numeric_ratio() -> None:
    """Refuse text values instead of coercing them to numeric ratios."""
    with pytest.raises(ValueError, match="must be a number"):
        parse_thresholds({"default": {"p50_us": "fast"}})


def test_parse_thresholds_rejects_bool_disguised_as_number() -> None:
    """Distinguish booleans from numeric threshold ratios."""
    with pytest.raises(ValueError, match="must be a number"):
        parse_thresholds({"default": {"p50_us": True}})


def test_resolve_threshold_prefers_benchmark_override() -> None:
    """Use the named benchmark override instead of the default bound."""
    thresholds = {"default": {"p50_us": 1.5}, "bench": {"p50_us": 1.2}}
    assert resolve_threshold(thresholds, "bench", "p50_us") == 1.2
    assert resolve_threshold(thresholds, "other", "p50_us") == 1.5
    assert resolve_threshold(thresholds, "bench", "unknown_metric") is None


# ── metric direction ──────────────────────────────────────────────────


@pytest.mark.parametrize("metric", ["p50_us", "p95_us", "peak_rss_mb", "latency_us"])
def test_latency_and_memory_metrics_are_upper_bounded(metric: str) -> None:
    """Apply upper ratio bounds to latency and memory metric names."""
    assert metric_direction(metric) == "upper"


@pytest.mark.parametrize("metric", ["throughput_ops_s", "rust_speedup_vs_python"])
def test_throughput_metrics_are_lower_bounded(metric: str) -> None:
    """Apply lower ratio bounds to throughput and speedup metric names."""
    assert metric_direction(metric) == "lower"


# ── report validation ─────────────────────────────────────────────────


def test_validate_report_accepts_well_formed_report() -> None:
    """Accept a nonempty report whose payload digest matches its contents."""
    assert validate_report(_report()) == []


def test_validate_report_rejects_wrong_schema() -> None:
    """Refuse a report schema outside the supported version."""
    report = _report()
    report["schema_version"] = "something-else"
    errors = validate_report(report)
    assert any("schema_version" in e for e in errors)


def test_validate_report_rejects_empty_benchmarks() -> None:
    """Refuse an empty report even when its checksum is consistent."""
    report = _report(benchmarks={})
    assert any("empty" in e for e in validate_report(report))


def test_validate_report_detects_tampered_payload_digest() -> None:
    """Detect a changed metric whose original payload checksum was retained."""
    report = _report()
    # Mutate a metric without recomputing payload_sha256.
    report["benchmarks"]["capacitor_bank_discharge"]["languages"]["python"]["p50_us"] = 1.0
    errors = validate_report(report)
    assert any("payload_sha256" in e for e in errors)


def test_validate_report_flags_missing_payload_digest() -> None:
    """Require the report payload checksum to be present."""
    report = _report()
    del report["payload_sha256"]
    assert any("payload_sha256" in e for e in validate_report(report))


# ── baseline integrity ────────────────────────────────────────────────


def test_verify_baseline_integrity_accepts_consistent_baseline() -> None:
    """Accept a populated baseline with a matching metric-block checksum."""
    assert verify_baseline_integrity(_baseline()) == []


def test_verify_baseline_integrity_rejects_checksum_mismatch() -> None:
    """Detect metric changes beneath an unchanged baseline checksum."""
    baseline = _baseline()
    # Tamper with a metric but keep the stale digest -> mismatch.
    baseline["benchmarks"]["capacitor_bank_discharge"]["languages"]["rust"]["p50_us"] = 0.001
    errors = verify_baseline_integrity(baseline)
    assert any("baseline_sha256" in e and "tampered" in e for e in errors)


def test_verify_baseline_integrity_flags_missing_digest() -> None:
    """Require an explicit checksum for baseline metrics."""
    baseline = _baseline()
    del baseline["baseline_sha256"]
    assert any("missing baseline_sha256" in e for e in verify_baseline_integrity(baseline))


# ── comparison ────────────────────────────────────────────────────────


def test_compare_passes_when_report_matches_baseline() -> None:
    """Produce no ratio findings for matching validated metric values."""
    assert compare(_report(), _baseline(), THRESHOLDS) == []


def test_compare_flags_latency_regression() -> None:
    """Report a fourfold latency increase as an upper-bound regression."""
    report = _report(_benchmarks(p50=400.0))  # 4x slower than baseline p50=100
    findings = compare(report, _baseline(), THRESHOLDS)
    assert any(f.kind == "regression" and f.metric == "p50_us" and f.direction == "upper" for f in findings)


def test_compare_flags_throughput_regression() -> None:
    # Drop throughput far below the 0.6 lower bound while keeping latency fine.
    """Report falling throughput as a lower-bound regression."""
    bench = _benchmarks()
    bench["capacitor_bank_discharge"]["languages"]["python"]["throughput_ops_s"] = 100.0
    report = _report(bench)
    # Re-stamp not needed for compare (compare ignores payload digest).
    findings = compare(report, _baseline(), THRESHOLDS)
    assert any(f.kind == "regression" and f.metric == "throughput_ops_s" and f.direction == "lower" for f in findings)


def test_compare_fails_closed_on_missing_metric() -> None:
    """Retain benchmark language and metric identity for missing Rust p99 data."""
    bench = _benchmarks()
    del bench["capacitor_bank_discharge"]["languages"]["rust"]["p99_us"]
    report = _report(bench)
    findings = compare(report, _baseline(), THRESHOLDS)
    assert any(f.kind == "missing_metric" and f.metric == "p99_us" and f.language == "rust" for f in findings)


def test_compare_fails_closed_on_missing_benchmark() -> None:
    """Report required metrics when the comparison report has no benchmark."""
    report = _report(benchmarks={})
    findings = compare(report, _baseline(), THRESHOLDS)
    assert findings and all(f.kind == "missing_metric" for f in findings)


def test_compare_flags_policy_gap_when_threshold_absent() -> None:
    """Report a missing bound rather than inventing a threshold."""
    thresholds = {"default": {"p50_us": 1.5}}  # no policy for p95/p99/throughput
    findings = compare(_report(), _baseline(), thresholds)
    assert any(f.kind == "policy_gap" and f.metric == "p95_us" for f in findings)


# ── gate verdict ──────────────────────────────────────────────────────


def test_gate_admits_clean_run() -> None:
    """Retain source identities and an empty finding list on admission."""
    verdict = gate(_report(), _baseline(), THRESHOLDS, generated_utc="2026-06-15T21:30:00Z")
    assert verdict["schema_version"] == VERDICT_SCHEMA
    assert verdict["passed"] is True
    assert verdict["findings"] == []
    assert verdict["baseline_commit"] == "def456"
    assert verdict["report_commit"] == "abc123"


def test_gate_fails_on_invalid_report_before_comparing() -> None:
    """Classify an invalid report before attempting numeric comparison."""
    report = _report()
    report["schema_version"] = "bad"
    verdict = gate(report, _baseline(), THRESHOLDS, generated_utc="t")
    assert verdict["passed"] is False
    assert any(f["kind"] == "report_invalid" for f in verdict["findings"])


def test_gate_fails_on_tampered_baseline() -> None:
    """Classify a mismatched baseline digest as rejected evidence."""
    baseline = _baseline()
    baseline["baseline_sha256"] = "0" * 64
    verdict = gate(_report(), baseline, THRESHOLDS, generated_utc="t")
    assert verdict["passed"] is False
    assert any(f["kind"] == "baseline_invalid" for f in verdict["findings"])


def test_gate_verdict_digest_is_deterministic() -> None:
    """Bind the verdict checksum to deterministic verdict content."""
    verdict = gate(_report(), _baseline(), THRESHOLDS, generated_utc="fixed")
    digest = verdict.pop("payload_sha256")
    assert digest == _payload_digest(verdict)


# ── CLI / fail-closed on missing files ────────────────────────────────


def test_main_fails_closed_when_report_file_missing(tmp_path: Path) -> None:
    """Return failure when the report path does not exist."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    rc = main(["--report", str(tmp_path / "nope.json"), "--baseline", str(baseline)])
    assert rc == 1


def test_main_fails_closed_when_baseline_file_missing(tmp_path: Path) -> None:
    """Return failure when the baseline path does not exist."""
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    rc = main(["--report", str(report), "--baseline", str(tmp_path / "nope.json")])
    assert rc == 1


def test_main_passes_on_clean_run(tmp_path: Path) -> None:
    """Write a successful JSON verdict for a valid file-based comparison."""
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
    """Refuse comparison when both declared CPU models differ."""
    report = _report()
    report["provenance"]["cpu_model"] = "AMD EPYC 7763"
    baseline = _baseline()
    baseline["provenance"]["cpu_model"] = "11th Gen Intel Core i5-11600K"
    finding = hardware_mismatch(report, baseline)
    assert finding is not None and finding.kind == "hardware_mismatch"


def test_hardware_mismatch_allows_same_cpu() -> None:
    """Allow the hardware check when declared CPU models agree."""
    report = _report()
    report["provenance"]["cpu_model"] = "same-cpu"
    baseline = _baseline()
    baseline["provenance"]["cpu_model"] = "same-cpu"
    assert hardware_mismatch(report, baseline) is None


def test_hardware_mismatch_skips_when_cpu_unknown() -> None:
    # Missing CPU provenance must not fabricate a mismatch.
    """Avoid fabricating a CPU mismatch when one declaration is absent."""
    assert hardware_mismatch(_report(), _baseline()) is None


def test_gate_fails_on_hardware_mismatch() -> None:
    """Reject a digest-consistent report with a declared CPU mismatch."""
    report = _report()
    report["provenance"]["cpu_model"] = "cpu-a"
    report["payload_sha256"] = _payload_digest({k: v for k, v in report.items() if k != "payload_sha256"})
    baseline = _baseline()
    baseline["provenance"]["cpu_model"] = "cpu-b"
    verdict = gate(report, baseline, THRESHOLDS, generated_utc="t")
    assert verdict["passed"] is False
    assert any(f["kind"] == "hardware_mismatch" for f in verdict["findings"])


def test_main_evidence_only_records_rejection_with_zero_exit(tmp_path: Path) -> None:
    # A real regression that would fail the gate must be reported but not block
    # when running in evidence-only mode on a generic runner.
    """Keep a regression verdict false even when evidence-only mode exits zero."""
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
    """Return failure for a threshold file without the required default table."""
    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    thresholds = tmp_path / "thresholds.toml"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    # No [default] table -> parse_thresholds raises -> gate fails closed.
    thresholds.write_text("[capacitor_bank_discharge]\np50_us = 1.2\n", encoding="utf-8")
    rc = main(["--report", str(report), "--baseline", str(baseline), "--thresholds", str(thresholds)])
    assert rc == 1


# ── remaining guard branches ──────────────────────────────────────────


def test_parse_thresholds_rejects_non_table_section() -> None:
    """Refuse a scalar where the policy requires a benchmark table."""
    with pytest.raises(ValueError, match="must be a table"):
        parse_thresholds({"default": {"p50_us": 1.5}, "capacitor_bank_discharge": "not-a-table"})


def test_validate_report_flags_absent_benchmarks_block() -> None:
    """Require the report benchmarks container to exist."""
    report = _report()
    del report["benchmarks"]
    assert any("no benchmarks block" in e for e in validate_report(report))


def test_verify_baseline_integrity_rejects_wrong_schema() -> None:
    """Refuse a baseline schema outside the supported version."""
    baseline = _baseline()
    baseline["schema_version"] = "other"
    assert any("schema_version" in e for e in verify_baseline_integrity(baseline))


def test_verify_baseline_integrity_flags_absent_benchmarks_block() -> None:
    """Require the baseline benchmarks container to exist."""
    baseline = {"schema_version": BASELINE_SCHEMA, "baseline_sha256": "0" * 64}
    errors = verify_baseline_integrity(baseline)
    assert any("no benchmarks block" in e for e in errors)


def test_compare_skips_non_numeric_baseline_metric() -> None:
    # A non-numeric baseline metric is skipped, not compared.
    """Keep the raw comparison helper separate from full numeric admission."""
    bench = _benchmarks()
    bench["capacitor_bank_discharge"]["languages"]["python"]["p50_us"] = "fast"
    findings = compare(_report(), _baseline(bench), THRESHOLDS)
    assert all(f.metric != "p50_us" or f.language != "python" for f in findings)


def test_main_returns_failure_on_real_regression(tmp_path: Path) -> None:
    """Return a failing strict exit code for a real file-based regression."""
    report = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    thresholds = tmp_path / "thresholds.toml"
    report.write_text(json.dumps(_report(_benchmarks(p50=10_000.0))), encoding="utf-8")
    baseline.write_text(json.dumps(_baseline()), encoding="utf-8")
    thresholds.write_text(
        "[default]\np50_us = 1.5\np95_us = 1.75\np99_us = 2.0\nthroughput_ops_s = 0.6\n", encoding="utf-8"
    )
    rc = main(["--report", str(report), "--baseline", str(baseline), "--thresholds", str(thresholds)])
    assert rc == 1


@pytest.mark.parametrize(
    "benchmarks",
    [
        {},
        {"case": {}},
        {"case": {"languages": {}}},
        {"case": {"languages": {"python": {}}}},
        {"case": {"languages": {"python": {"p95_us": float("inf")}}}},
        {"case": {"languages": {"python": {"p95_us": float("nan")}}}},
        {"case": {"languages": {"python": {"p95_us": 0.0}}}},
    ],
)
def test_gate_refuses_empty_or_nonfinite_baseline_metrics(benchmarks: dict) -> None:
    """A consistent checksum cannot qualify an unusable metric baseline."""
    verdict = gate(_report(), _baseline(benchmarks), THRESHOLDS, generated_utc="fixed")
    assert verdict["passed"] is False
    assert any(f["kind"] == "baseline_invalid" for f in verdict["findings"])


@pytest.mark.parametrize("document", ["report", "baseline"])
@pytest.mark.parametrize(
    "bad",
    [None, True, "fast", -1.0, float("inf"), float("nan"), 10**400],
    ids=["null", "boolean", "string", "negative", "infinity", "nan", "overflow"],
)
def test_gate_rejects_invalid_numeric_domains(document: str, bad: object) -> None:
    """Checksummed invalid numbers cannot become successful ratio evidence."""
    metrics = _benchmarks()
    metrics["capacitor_bank_discharge"]["languages"]["python"]["p95_us"] = bad
    report = _report(metrics) if document == "report" else _report()
    baseline = _baseline(metrics) if document == "baseline" else _baseline()
    verdict = gate(report, baseline, THRESHOLDS, generated_utc="fixed")
    assert verdict["passed"] is False
    assert any(f["kind"] == document + "_invalid" for f in verdict["findings"])


@pytest.mark.parametrize("document", ["report", "baseline"])
@pytest.mark.parametrize("benchmark", [None, [], {"languages": []}, {"languages": {"python": []}}])
def test_gate_refuses_malformed_language_and_metric_maps(document: str, benchmark: object) -> None:
    """Structural failures are verdict findings, not skipped comparisons."""
    metrics = {"case": benchmark}
    report = _report(metrics) if document == "report" else _report()
    baseline = _baseline(metrics) if document == "baseline" else _baseline()
    verdict = gate(report, baseline, THRESHOLDS, generated_utc="fixed")
    assert verdict["passed"] is False
    assert any(f["kind"] == document + "_invalid" for f in verdict["findings"])


@pytest.mark.parametrize("evidence_only", [False, True])
def test_real_cli_reports_invalid_baseline_without_admission(tmp_path: Path, evidence_only: bool) -> None:
    """Strict exit failure and explicit evidence-only reporting keep the same verdict."""
    report, baseline = tmp_path / "report.json", tmp_path / "baseline.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    baseline.write_text(json.dumps(_baseline({})), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "tools/benchmark_regression_gate.py"),
        "--report",
        str(report),
        "--baseline",
        str(baseline),
    ]
    if evidence_only:
        command.append("--evidence-only")
    result = subprocess.run(
        command, env=dict(os.environ, PYTHONPATH=str(root / "src")), capture_output=True, text=True, timeout=30
    )
    assert result.returncode == (0 if evidence_only else 1), result.stderr
    assert "[baseline_invalid]" in result.stdout + result.stderr
    assert "gate passed" not in result.stdout


def test_gate_allows_zero_report_latency_against_positive_baseline() -> None:
    """A finite zero observation is valid while ratio denominators stay positive."""
    metrics = _benchmarks()
    metrics["capacitor_bank_discharge"]["languages"]["python"]["p95_us"] = 0.0
    assert gate(_report(metrics), _baseline(), THRESHOLDS, generated_utc="fixed")["passed"] is True


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), 0.0, True, 10**400], ids=["nan", "infinity", "zero", "boolean", "overflow"]
)
def test_public_gate_rejects_unvalidated_threshold_values(bad: object) -> None:
    """Direct Python admission applies the same numeric policy rules as the CLI."""
    policy = {"default": dict(THRESHOLDS["default"], p95_us=bad)}
    verdict = gate(_report(), _baseline(), policy, generated_utc="fixed")
    assert verdict["passed"] is False
    assert any(f["kind"] == "policy_invalid" for f in verdict["findings"])


def test_public_gate_requires_default_policy_table() -> None:
    """A complete-looking override cannot bypass the required default policy."""
    verdict = gate(_report(), _baseline(), {"capacitor_bank_discharge": THRESHOLDS["default"]}, generated_utc="fixed")
    assert verdict["passed"] is False
    assert any(f["kind"] == "policy_invalid" for f in verdict["findings"])
