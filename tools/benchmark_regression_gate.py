#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — benchmark regression gate for Python/Rust latency parity paths.
"""Fail-closed gate comparing a fresh benchmark report against a tracked baseline.

A benchmark suite report (`scpn-control.benchmark-regression.v1`, produced by
`tools/run_benchmark_suite.py`) carries per-benchmark, per-language latency
percentiles (p50/p95/p99) and throughput plus run provenance. This gate loads
that report, the admitted baseline (`scpn-control.benchmark-baseline.v1`), and an
explicit threshold policy, then admits the run only when every baseline metric is
present in the report and within its allowed regression margin.

It fails closed: a missing report, a missing baseline, a tampered baseline
(checksum mismatch), a benchmark/language/metric present in the baseline but
absent from the report, or a metric with no threshold policy are all failures —
never silent passes. Latency and memory metrics are upper-bounded
(current <= baseline * ratio); throughput is lower-bounded
(current >= baseline * ratio).

The pure-logic helpers carry no IO so they can be unit-tested directly; only
`main` and the loaders touch the filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI.
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

REPORT_SCHEMA = "scpn-control.benchmark-regression.v1"
BASELINE_SCHEMA = "scpn-control.benchmark-baseline.v1"
VERDICT_SCHEMA = "scpn-control.benchmark-gate-verdict.v1"

# Metrics whose value should fall (latency, memory) versus rise (throughput).
LOWER_IS_BETTER_TOKENS = ("throughput", "ops_s", "speedup")


@dataclass(frozen=True)
class Finding:
    """A single gate failure: a regression, missing evidence, or policy gap."""

    kind: str  # "regression" | "missing_metric" | "policy_gap"
    benchmark: str
    language: str
    metric: str
    baseline_value: float | None
    current_value: float | None
    ratio: float | None
    threshold: float | None
    direction: str
    detail: str


def metric_direction(metric: str) -> str:
    """Return "lower" when higher is better (throughput), else "upper"."""
    name = metric.lower()
    if any(token in name for token in LOWER_IS_BETTER_TOKENS):
        return "lower"
    return "upper"


def canonical_metrics_digest(benchmarks: dict[str, Any]) -> str:
    """Stable SHA-256 of the benchmarks metric block for tamper detection."""
    serialised = json.dumps(benchmarks, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


def _payload_digest(payload: dict[str, Any]) -> str:
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


def parse_thresholds(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Validate and normalise a threshold policy mapping.

    Shape: ``{"default": {metric: ratio, ...}, "<benchmark>": {metric: ratio}}``.
    Every ratio must be a positive finite number. A missing ``default`` table or
    a non-positive ratio is a policy error (raised), not a silent default.
    """
    if "default" not in raw or not isinstance(raw["default"], dict) or not raw["default"]:
        raise ValueError("threshold policy must define a non-empty [default] table")
    normalised: dict[str, dict[str, float]] = {}
    for section, table in raw.items():
        if not isinstance(table, dict):
            raise ValueError(f"threshold section '{section}' must be a table")
        metrics: dict[str, float] = {}
        for metric, ratio in table.items():
            if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
                raise ValueError(f"threshold '{section}.{metric}' must be a number")
            value = float(ratio)
            if not (value > 0.0) or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"threshold '{section}.{metric}' must be positive and finite")
            metrics[metric] = value
        normalised[section] = metrics
    return normalised


def resolve_threshold(thresholds: dict[str, dict[str, float]], benchmark: str, metric: str) -> float | None:
    """Per-benchmark override wins over the default; None means no policy."""
    bench_table = thresholds.get(benchmark, {})
    if metric in bench_table:
        return bench_table[metric]
    return thresholds.get("default", {}).get(metric)


def validate_report(report: dict[str, Any]) -> list[str]:
    """Structural + integrity validation of a benchmark suite report."""
    errors: list[str] = []
    if report.get("schema_version") != REPORT_SCHEMA:
        errors.append(f"report schema_version must be {REPORT_SCHEMA!r}")
    if "benchmarks" not in report or not isinstance(report["benchmarks"], dict):
        errors.append("report has no benchmarks block")
    elif not report["benchmarks"]:
        errors.append("report benchmarks block is empty")
    if "payload_sha256" not in report:
        errors.append("report is missing payload_sha256")
    else:
        stamped = report["payload_sha256"]
        recomputed = _payload_digest({k: v for k, v in report.items() if k != "payload_sha256"})
        if stamped != recomputed:
            errors.append("report payload_sha256 does not match its contents (tampered or stale)")
    return errors


def verify_baseline_integrity(baseline: dict[str, Any]) -> list[str]:
    """Reject a baseline whose recorded digest does not match its metrics."""
    errors: list[str] = []
    if baseline.get("schema_version") != BASELINE_SCHEMA:
        errors.append(f"baseline schema_version must be {BASELINE_SCHEMA!r}")
    if "benchmarks" not in baseline or not isinstance(baseline["benchmarks"], dict):
        errors.append("baseline has no benchmarks block")
        return errors
    stamped = baseline.get("baseline_sha256")
    if not stamped:
        errors.append("baseline is missing baseline_sha256")
    else:
        recomputed = canonical_metrics_digest(baseline["benchmarks"])
        if stamped != recomputed:
            errors.append("baseline_sha256 does not match the baseline metrics (tampered)")
    return errors


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if out == out else None


def compare(
    report: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
) -> list[Finding]:
    """Compare every baseline metric against the report, fail-closed on gaps."""
    findings: list[Finding] = []
    report_benches = report.get("benchmarks", {})
    for bench_name, base_bench in baseline.get("benchmarks", {}).items():
        report_bench = report_benches.get(bench_name)
        base_langs = base_bench.get("languages", {})
        for language, base_metrics in base_langs.items():
            report_metrics = None
            if isinstance(report_bench, dict):
                report_metrics = report_bench.get("languages", {}).get(language)
            for metric, base_raw in base_metrics.items():
                base_value = _coerce_float(base_raw)
                if base_value is None:
                    continue
                direction = metric_direction(metric)
                threshold = resolve_threshold(thresholds, bench_name, metric)
                if threshold is None:
                    findings.append(
                        Finding(
                            "policy_gap",
                            bench_name,
                            language,
                            metric,
                            base_value,
                            None,
                            None,
                            None,
                            direction,
                            "no threshold policy for this metric",
                        )
                    )
                    continue
                current_value = None
                if isinstance(report_metrics, dict):
                    current_value = _coerce_float(report_metrics.get(metric))
                if current_value is None:
                    findings.append(
                        Finding(
                            "missing_metric",
                            bench_name,
                            language,
                            metric,
                            base_value,
                            None,
                            None,
                            threshold,
                            direction,
                            "metric present in baseline but absent from report",
                        )
                    )
                    continue
                ratio = current_value / base_value if base_value != 0.0 else float("inf")
                regressed = ratio > threshold if direction == "upper" else ratio < threshold
                if regressed:
                    bound = "<=" if direction == "upper" else ">="
                    findings.append(
                        Finding(
                            "regression",
                            bench_name,
                            language,
                            metric,
                            base_value,
                            current_value,
                            ratio,
                            threshold,
                            direction,
                            f"ratio {ratio:.3f} violates {bound} {threshold:.3f}",
                        )
                    )
    return findings


def hardware_mismatch(report: dict[str, Any], baseline: dict[str, Any]) -> Finding | None:
    """Flag a latency comparison across different CPUs as invalid.

    Absolute-latency regression gating is only meaningful when the report and the
    baseline were measured on the same declared hardware. When both records carry
    a CPU model and they differ, the comparison cannot be trusted.
    """
    report_cpu = report.get("provenance", {}).get("cpu_model")
    baseline_cpu = baseline.get("provenance", {}).get("cpu_model")
    if report_cpu and baseline_cpu and report_cpu != baseline_cpu:
        return Finding(
            "hardware_mismatch",
            "",
            "",
            "",
            None,
            None,
            None,
            None,
            "",
            f"report CPU {report_cpu!r} != baseline CPU {baseline_cpu!r}; "
            "latency comparison is invalid off declared hardware",
        )
    return None


def gate(
    report: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
    *,
    generated_utc: str,
) -> dict[str, Any]:
    """Run validation + comparison and assemble a fail-closed verdict report."""
    findings: list[Finding] = []
    for error in validate_report(report):
        findings.append(Finding("report_invalid", "", "", "", None, None, None, None, "", error))
    for error in verify_baseline_integrity(baseline):
        findings.append(Finding("baseline_invalid", "", "", "", None, None, None, None, "", error))
    # Only compare metrics when both documents are structurally sound, otherwise
    # the comparison itself is unreliable; the structural failures above already
    # fail the gate.
    if not findings:
        mismatch = hardware_mismatch(report, baseline)
        if mismatch is not None:
            findings.append(mismatch)
        findings.extend(compare(report, baseline, thresholds))
    passed = not findings
    verdict = {
        "schema_version": VERDICT_SCHEMA,
        "generated_utc": generated_utc,
        "passed": passed,
        "baseline_commit": baseline.get("baseline_commit"),
        "report_commit": report.get("provenance", {}).get("commit"),
        "findings": [asdict(f) for f in findings],
    }
    verdict["payload_sha256"] = _payload_digest(verdict)
    return verdict


# ── filesystem / CLI ──────────────────────────────────────────────────


def _load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_thresholds_file(path: Path) -> dict[str, dict[str, float]]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return parse_thresholds(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark regression gate.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "regression_thresholds.toml",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--generated-utc", default="")
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Collect and report the verdict but never fail (for generic CI "
        "runners whose CPU differs from the declared-hardware baseline).",
    )
    args = parser.parse_args(argv)

    # Missing-evidence fail-closed: an absent report or baseline is a failure,
    # not a skip.
    for label, path in (("report", args.report), ("baseline", args.baseline)):
        if not path.is_file():
            print(f"benchmark gate FAILED: {label} not found at {path}", file=sys.stderr)
            return 1

    report = _load_json(args.report)
    baseline = _load_json(args.baseline)
    try:
        thresholds = load_thresholds_file(args.thresholds)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"benchmark gate FAILED: threshold policy error: {exc}", file=sys.stderr)
        return 1

    verdict = gate(report, baseline, thresholds, generated_utc=args.generated_utc)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not verdict["passed"]:
        stream = sys.stdout if args.evidence_only else sys.stderr
        header = "benchmark gate findings (evidence-only):" if args.evidence_only else "benchmark gate FAILED:"
        print(header, file=stream)
        for finding in verdict["findings"]:
            scope = "/".join(p for p in (finding["benchmark"], finding["language"], finding["metric"]) if p)
            print(f"  - [{finding['kind']}] {scope}: {finding['detail']}", file=stream)
        if not args.evidence_only:
            return 1
        return 0
    compared = sum(len(m) for b in baseline.get("benchmarks", {}).values() for m in b.get("languages", {}).values())
    print(f"benchmark gate passed: {compared} baseline metric(s) within policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
