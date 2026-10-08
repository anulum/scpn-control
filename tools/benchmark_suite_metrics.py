# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark suite input and consumed result contracts.
"""Validate selected settings and consumed benchmark declarations before publication."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TypedDict


class BenchmarkBlock(TypedDict):
    """Hold one checked language map and optional parity observation."""

    languages: dict[str, dict[str, float]]
    cross_language_parity: dict[str, float] | None


class SuiteReportBody(TypedDict):
    """Describe the V1 report fields hashed before adding the payload digest."""

    schema_version: str
    campaign_id: str | None
    generated_utc: str
    evidence_class: str
    production_claim_allowed: bool
    provenance: dict[str, object]
    settings: dict[str, int]
    benchmarks: dict[str, BenchmarkBlock]


class SuiteReport(SuiteReportBody):
    """Add the finite canonical-payload SHA256 stamp to the V1 body."""

    payload_sha256: str


class AdapterResult(TypedDict):
    """Hold checked per-language gate metrics and consistent backend declarations."""

    languages: dict[str, dict[str, float]]
    cross_language_parity: dict[str, float] | None
    rust_available: bool


class BenchmarkSuiteRefusal(ValueError):
    """Refuse invalid suite settings or consumed measurement declarations.

    Parameters
    ----------
    message : str
        Deliberately authored refusal, without native exception text.
    """


def check_suite_settings(
    names: list[str],
    steps: int,
    warmup: int,
    evidence_class: str,
    generated_utc: str,
    *,
    registered: Mapping[str, object],
) -> None:
    """Refuse invalid selections, counts and declaration labels before timing.

    Parameters
    ----------
    names : list of str
        Requested distinct nonempty registered names.
    steps : int
        Positive measured sample count.
    warmup : int
        Nonnegative unrecorded sample count.
    evidence_class : str
        Nonblank caller-supplied class label.
    generated_utc : str
        Timezone-aware UTC receipt.
    registered : mapping of str to object
        Actual owning registry used to resolve selected names.

    Raises
    ------
    BenchmarkSuiteRefusal
        Selection, count, label or UTC receipt is refused. No adapter runs here.
    """
    if not names or len(set(names)) != len(names) or any(name not in registered for name in names):
        raise BenchmarkSuiteRefusal("select a nonempty set of distinct registered benchmarks")
    if type(steps) is not int or steps <= 0 or type(warmup) is not int or warmup < 0:
        raise BenchmarkSuiteRefusal("steps must be positive and warmup nonnegative integers")
    if not isinstance(evidence_class, str) or not evidence_class.strip():
        raise BenchmarkSuiteRefusal("evidence class must be nonblank")
    try:
        stamp = datetime.fromisoformat(generated_utc.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise BenchmarkSuiteRefusal("generated timestamp must describe UTC") from exc
    if stamp.utcoffset() != UTC.utcoffset(stamp):
        raise BenchmarkSuiteRefusal("generated timestamp must describe UTC")


def _language_metrics(stats: Mapping[str, object]) -> dict[str, float]:
    """Normalize finite nonnegative microseconds without changing throughput arithmetic.

    Parameters
    ----------
    stats : dict of str to object
        Consumed mean_us, median_us, p95_us and p99_us statistics.

    Returns
    -------
    dict of str to float
        p50/p95/p99 microseconds and reciprocal-mean operations per second. Zero
        mean retains the original zero-throughput convention.

    Raises
    ------
    BenchmarkSuiteRefusal
        Numeric types, finite/nonnegative domains, percentile order or derived
        throughput are invalid. Unused harness fields are not qualified here.
    """
    values: dict[str, float] = {}
    for key in ("mean_us", "median_us", "p95_us", "p99_us"):
        raw = stats.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise BenchmarkSuiteRefusal("latency statistics must be finite nonnegative numbers")
        try:
            value = float(raw)
        except OverflowError as exc:
            raise BenchmarkSuiteRefusal("latency statistics must be finite nonnegative numbers") from exc
        if not math.isfinite(value) or value < 0:
            raise BenchmarkSuiteRefusal("latency statistics must be finite nonnegative numbers")
        values[key] = value
    if not values["median_us"] <= values["p95_us"] <= values["p99_us"]:
        raise BenchmarkSuiteRefusal("latency percentiles must be ordered")
    mean_us = values["mean_us"]
    throughput = 1.0e6 / mean_us if mean_us > 0.0 else 0.0
    if not math.isfinite(throughput):
        raise BenchmarkSuiteRefusal("derived throughput must be finite")
    return {
        "p50_us": values["median_us"],
        "p95_us": values["p95_us"],
        "p99_us": values["p99_us"],
        "throughput_ops_s": throughput,
    }


def _payload_digest(payload: Mapping[str, object]) -> str:
    """Hash finite canonical JSON without a digest field added by this function.

    Parameters
    ----------
    payload : dict of str to object
        Exact declarations to encode; callers exclude any existing digest field.

    Returns
    -------
    str
        SHA256 of sorted compact UTF8 JSON.

    Raises
    ------
    TypeError, ValueError
        A value is not JSON-serializable or is nonfinite.
    """
    serialised = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(serialised).hexdigest()


def checked_mapping(value: object, label: str) -> dict[str, object]:
    """Require a string-keyed object without coercing its keys or values.

    Parameters
    ----------
    value : object
        Consumed mapping candidate.
    label : str
        Authored context label for a shape refusal.

    Returns
    -------
    dict of str to object
        Shallow copy with checked keys; nested values still need owning validation.

    Raises
    ------
    BenchmarkSuiteRefusal
        Object shape or key type is invalid.
    """
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BenchmarkSuiteRefusal(f"{label} must be a string-keyed object")
    return {key: item for key, item in value.items()}


def checked_adapter_result(value: object) -> AdapterResult:
    """Validate consumed gate metrics and coherent Rust/parity declarations.

    Parameters
    ----------
    value : object
        Adapter record with languages, rust_available and cross_language_parity.

    Returns
    -------
    AdapterResult
        Finite nonnegative metrics with ordered percentiles and exact boolean
        availability. A measured Rust row requires finite nonnegative parity.

    Raises
    ------
    BenchmarkSuiteRefusal
        Consumed keys/domains or backend/parity consistency are refused.

    Notes
    -----
    A coherent failing parity value is preserved. No new physical acceptance
    threshold, producer authentication or timing measurement runs here.
    """
    result = checked_mapping(value, "benchmark adapter result")
    raw_languages = checked_mapping(result.get("languages"), "adapter languages")
    if not raw_languages:
        raise BenchmarkSuiteRefusal("adapter languages must be nonempty")
    languages: dict[str, dict[str, float]] = {}
    for name, raw_metrics in raw_languages.items():
        if not name.strip():
            raise BenchmarkSuiteRefusal("language names must be nonblank")
        metrics = checked_mapping(raw_metrics, "language metrics")
        if not {"p50_us", "p95_us", "p99_us", "throughput_ops_s"}.issubset(metrics):
            raise BenchmarkSuiteRefusal("language metrics require latency percentiles and throughput")
        checked: dict[str, float] = {}
        for metric, raw in metrics.items():
            if not metric.strip() or isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise BenchmarkSuiteRefusal("gate metrics must be named finite nonnegative numbers")
            try:
                number = float(raw)
            except OverflowError as exc:
                raise BenchmarkSuiteRefusal("gate metrics must be named finite nonnegative numbers") from exc
            if not math.isfinite(number) or number < 0:
                raise BenchmarkSuiteRefusal("gate metrics must be named finite nonnegative numbers")
            checked[metric] = number
        if not checked["p50_us"] <= checked["p95_us"] <= checked["p99_us"]:
            raise BenchmarkSuiteRefusal("latency percentiles must be ordered")
        languages[name] = checked
    available = result.get("rust_available")
    if type(available) is not bool or available != ("rust" in languages):
        raise BenchmarkSuiteRefusal("Rust availability must agree with the measured language map")
    parity_value = result.get("cross_language_parity")
    parity = None
    if available:
        raw_parity = checked_mapping(parity_value, "cross-language parity")
        difference = raw_parity.get("max_relative_difference")
        if isinstance(difference, bool) or not isinstance(difference, (int, float)):
            raise BenchmarkSuiteRefusal("cross-language difference must be finite and nonnegative")
        try:
            difference = float(difference)
        except OverflowError as exc:
            raise BenchmarkSuiteRefusal("cross-language difference must be finite and nonnegative") from exc
        if not math.isfinite(difference) or difference < 0:
            raise BenchmarkSuiteRefusal("cross-language difference must be finite and nonnegative")
        parity = {"max_relative_difference": difference}
    elif parity_value is not None:
        raise BenchmarkSuiteRefusal("cross-language parity requires a measured Rust backend")
    return {"languages": languages, "cross_language_parity": parity, "rust_available": available}
