# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — benchmark regression gate for Python/Rust latency parity paths.

"""Numeric-domain and ratio-policy checks for declared benchmark records."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

REPORT_SCHEMA = "scpn-control.benchmark-regression.v1"

BASELINE_SCHEMA = "scpn-control.benchmark-baseline.v1"

LOWER_IS_BETTER_TOKENS = ("throughput", "ops_s", "speedup")


def metric_direction(metric: str) -> str:
    """Select the dimensionless ratio bound from a metric name.

    Parameters
    ----------
    metric : str
        Case-insensitive name; throughput, ops_s and speedup substrings select
        a lower bound. No unit conversion or metric registration is performed.

    Returns
    -------
    str
        "lower" for those substrings, otherwise "upper".
    """
    name = metric.lower()
    if any(token in name for token in LOWER_IS_BETTER_TOKENS):
        return "lower"
    return "upper"


def canonical_metrics_digest(benchmarks: dict[str, Any]) -> str:
    """Hash the sorted compact JSON metric mapping.

    Parameters
    ----------
    benchmarks : dict
        JSON-serialisable mapping, including any embedded descriptive fields.

    Returns
    -------
    str
        SHA-256 of default json.dumps UTF-8 bytes; insertion order is ignored.

    Raises
    ------
    TypeError, ValueError
        Values cannot be serialised by the standard JSON encoder.

    Notes
    -----
    This digest is a consistency check, not origin authentication. Numeric
    domains are checked separately; this encoder alone admits nonfinite floats.
    """
    serialised = json.dumps(benchmarks, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


def _payload_digest(payload: dict[str, Any]) -> str:
    """Hash the entire supplied mapping with the original compact JSON convention.

    Standard encoder errors propagate. This does not remove a stamp field,
    authenticate origin, or validate numeric domains.
    """
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


def parse_thresholds(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Validate dimensionless ratio tables without substituting policy.

    Parameters
    ----------
    raw : dict
        Nonempty default metric table plus optional benchmark overrides.

    Returns
    -------
    dict
        Fresh tables of positive finite float ratios; bool is refused.

    Raises
    ------
    ValueError
        Missing/empty default, nontable section or unusable ratio.

    Notes
    -----
    Metric names carry units externally. No default bound is invented.
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
            value = _coerce_float(ratio)
            if value is None or value <= 0.0:
                raise ValueError(f"threshold '{section}.{metric}' must be positive and finite")
            metrics[metric] = value
        normalised[section] = metrics
    return normalised


def resolve_threshold(thresholds: dict[str, dict[str, float]], benchmark: str, metric: str) -> float | None:
    """Look up a benchmark-specific ratio before the default table.

    Parameters
    ----------
    thresholds : dict
        Previously validated ratio tables.
    benchmark, metric : str
        Literal mapping keys; no unit conversion or case folding is applied.

    Returns
    -------
    float or None
        Selected ratio, or None when no policy exists.
    """
    bench_table = thresholds.get(benchmark, {})
    if metric in bench_table:
        return bench_table[metric]
    return thresholds.get("default", {}).get(metric)


def _metric_block_errors(benchmarks: dict[str, Any], label: str) -> list[str]:
    """List malformed language maps or unusable numeric metrics.

    Baseline denominators must be positive; report observations may be zero.
    All metric values must be finite numbers, excluding bool. This inspects
    declarations and performs no benchmark measurement.
    """
    errors: list[str] = []
    if not benchmarks:
        return [f"{label} benchmarks block is empty"]
    for name, benchmark in benchmarks.items():
        languages = benchmark.get("languages") if isinstance(benchmark, dict) else None
        if not isinstance(languages, dict) or not languages:
            errors.append(f"{label} benchmark {name!r} requires a nonempty languages map")
            continue
        for language, metrics in languages.items():
            if not isinstance(metrics, dict) or not metrics:
                errors.append(f"{label} {name}/{language} requires a nonempty metric map")
                continue
            for metric, raw in metrics.items():
                value = _coerce_float(raw)
                if value is None or value < 0.0 or (label == "baseline" and value == 0.0):
                    domain = "positive" if label == "baseline" else "nonnegative"
                    errors.append(f"{label} {name}/{language}/{metric} must be a finite {domain} number")
    return errors


def validate_report(report: dict[str, Any]) -> list[str]:
    """Inspect report schema, metric domains and whole-payload consistency.

    Parameters
    ----------
    report : dict
        Decoded report using REPORT_SCHEMA and a payload_sha256 stamp.

    Returns
    -------
    list of str
        Ordered findings; empty means these local checks pass.

    Raises
    ------
    TypeError, ValueError
        An otherwise supplied payload cannot be JSON-serialised.

    Notes
    -----
    The supplied stamp is compared to all fields except itself. This neither
    authenticates the producer/CPU nor evaluates cross-language parity metadata.
    """
    errors: list[str] = []
    if report.get("schema_version") != REPORT_SCHEMA:
        errors.append(f"report schema_version must be {REPORT_SCHEMA!r}")
    if "benchmarks" not in report or not isinstance(report["benchmarks"], dict):
        errors.append("report has no benchmarks block")
    else:
        errors.extend(_metric_block_errors(report["benchmarks"], "report"))
    if "payload_sha256" not in report:
        errors.append("report is missing payload_sha256")
    else:
        stamped = report["payload_sha256"]
        recomputed = _payload_digest({k: v for k, v in report.items() if k != "payload_sha256"})
        if stamped != recomputed:
            errors.append("report payload_sha256 does not match its contents (tampered or stale)")
    return errors


def verify_baseline_integrity(baseline: dict[str, Any]) -> list[str]:
    """Inspect baseline schema, denominators and metric-block consistency.

    Parameters
    ----------
    baseline : dict
        Decoded baseline using BASELINE_SCHEMA and baseline_sha256.

    Returns
    -------
    list of str
        Ordered findings; empty means these local checks pass.

    Notes
    -----
    Only the benchmarks block is hashed. Other provenance is not authenticated.
    Serialisation errors propagate; these checks perform no measurement.
    """
    errors: list[str] = []
    if baseline.get("schema_version") != BASELINE_SCHEMA:
        errors.append(f"baseline schema_version must be {BASELINE_SCHEMA!r}")
    if "benchmarks" not in baseline or not isinstance(baseline["benchmarks"], dict):
        errors.append("baseline has no benchmarks block")
        return errors
    errors.extend(_metric_block_errors(baseline["benchmarks"], "baseline"))
    stamped = baseline.get("baseline_sha256")
    if not stamped:
        errors.append("baseline is missing baseline_sha256")
    else:
        recomputed = canonical_metrics_digest(baseline["benchmarks"])
        if stamped != recomputed:
            errors.append("baseline_sha256 does not match the baseline metrics (tampered)")
    return errors


def _coerce_float(value: Any) -> float | None:
    """Return a finite float for int/float input, or None.

    Reject bool, nonnumeric values, overflow and nonfinite conversions without
    raising. No sign constraint or unit conversion is applied here.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        out = float(value)
    except OverflowError:
        return None
    return out if math.isfinite(out) else None
