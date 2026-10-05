# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — benchmark regression gate for Python/Rust latency parity paths.

"""Pure comparison findings and verdict assembly for declared benchmark metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tools.benchmark_gate_policy import (
    _coerce_float,
    _payload_digest,
    metric_direction,
    parse_thresholds,
    resolve_threshold,
    validate_report,
    verify_baseline_integrity,
)

VERDICT_SCHEMA = "scpn-control.benchmark-gate-verdict.v1"


@dataclass(frozen=True)
class Finding:
    """Immutable declared-metric comparison finding.

    Attributes
    ----------
    kind : str
        Failure classification, including input, policy and hardware failures.
    benchmark, language, metric : str
        Source mapping keys; empty for document-wide findings.
    baseline_value, current_value, ratio, threshold : float or None
        Declared values and dimensionless ratios; no timing is measured.
    direction, detail : str
        Bound direction and diagnostic text.

    Notes
    -----
    Floating division can overflow a ratio. This record is not authenticated
    evidence, and independently changing a digest does not make it so.
    """

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


def compare(
    report: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
) -> list[Finding]:
    """Compare prevalidated declared metrics against required baseline entries.

    Parameters
    ----------
    report, baseline : dict
        Validated maps; use gate for complete schema/domain/policy checks.
    thresholds : dict
        Validated dimensionless ratios, with benchmark override precedence.

    Returns
    -------
    list of Finding
        Missing-policy, missing-metric or strict ratio violations in baseline
        iteration order. Equal bounds pass; extra report metrics are ignored.

    Notes
    -----
    No IO or shared state is used. Lower bounds apply to throughput/ops_s/speedup
    names; all others use upper bounds. Caller units must already agree. Raw
    invalid maps may raise native errors; this helper is not an admission API.
    """
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
    """Compare two present, truthy declared CPU model values.

    Parameters
    ----------
    report, baseline : dict
        Decoded records with optional provenance mappings.

    Returns
    -------
    Finding or None
        Mismatch finding when both CPU declarations differ; otherwise None.

    Notes
    -----
    Missing identity supplies no comparability proof. This reads metadata,
    not actual host hardware; malformed provenance can raise AttributeError.
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
    """Assemble a diagnostic verdict after local record and policy validation.

    Parameters
    ----------
    report, baseline : dict
        Decoded metric records. Provenance, if present, must be a mapping.
    thresholds : dict
        Dimensionless ratios; parse_thresholds applies before comparison.
    generated_utc : str
        Caller label copied verbatim, without calendar or clock validation.

    Returns
    -------
    dict
        VERDICT_SCHEMA, passed flag, source labels, findings and payload digest.
        Valid-input CPU mismatch is retained alongside numeric comparisons.

    Notes
    -----
    No measurement, source authentication or physical admission occurs.
    Native errors from malformed provenance or unserialisable caller values
    propagate. Instances and findings are local, without locks/shared state.
    """
    findings: list[Finding] = []
    for error in validate_report(report):
        findings.append(Finding("report_invalid", "", "", "", None, None, None, None, "", error))
    for error in verify_baseline_integrity(baseline):
        findings.append(Finding("baseline_invalid", "", "", "", None, None, None, None, "", error))
    try:
        thresholds = parse_thresholds(thresholds)
    except ValueError as error:
        findings.append(Finding("policy_invalid", "", "", "", None, None, None, None, "", str(error)))
    # Only compare metrics when both documents and policy are valid, otherwise
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
