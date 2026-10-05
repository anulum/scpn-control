# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime admission evidence validation.
"""Inspect a persisted admission-probe report and its canonical self-digest.

This standard-library reader checks declarations and exact bytes, not the
current host, scheduler, kernel or authenticity of a producer. A self-digest
detects inconsistent report edits; anyone can edit and reseal a report. Local
regression may declare failed runtime admission with explanatory errors while
production claims remain false. Production-class PASS is a consistent report
declaration, not independent realtime qualification or control readiness.

Use ``python validation/validate_runtime_admission_evidence.py --report FILE``
with optional ``--json-out``. The root ``scpn-control validate`` consumer uses
this reader by default. The reader neither runs the benchmark nor writes an
artifact. Caller-relative paths and symlinks are followed; no filesystem
containment or input size/depth budget is supplied. Extra fields remain
unvalidated except duplicate/nonfinite floating JSON checks and inclusion in
the self-digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ADMISSION_EVIDENCE_SCHEMA_VERSION = "scpn-control.runtime-admission-evidence-admission.v1"
RUNTIME_ADMISSION_BENCHMARK_SCHEMA_VERSION = "scpn-control.runtime-admission-benchmark.v1"
DEFAULT_REPORT = ROOT / "validation" / "reports" / "runtime_admission_release_20260605T000000Z.json"
BENCHMARK_EVIDENCE_CLASSES = frozenset({"local_regression", "production_benchmark"})


@dataclass(frozen=True)
class RuntimeAdmissionEvidenceAdmission:
    """Frozen findings and typed declarations from one persisted report.

    Attributes
    ----------
    status : str
        ``pass`` when all reader checks succeed, else ``fail``.
    errors : tuple[str, ...]
        Ordered read, schema, context, metric and self-digest findings.
    report_sha256 : str or None
        Digest of exact successfully decoded JSON-object bytes; read/decode
        failure yields ``None``. This does not authenticate the report.
    payload_sha256 : str or None
        Declared string self-digest, including malformed spelling on FAIL;
        non-string declarations yield ``None``.
    benchmark_evidence_class : str or None
        Declared string class, including unknown strings on FAIL.
    production_claim_allowed : bool or None
        Declared boolean, with other types represented as ``None``.
    admission_status : str or None
        Declared string probe status, including unknown strings on FAIL.
    admission_error_count : int or None
        Length of the declared error list, or ``None`` for a non-list field.
    samples : int or None
        Positive integer declared sample count, otherwise ``None``.
    """

    status: str
    errors: tuple[str, ...]
    report_sha256: str | None
    payload_sha256: str | None
    benchmark_evidence_class: str | None
    production_claim_allowed: bool | None
    admission_status: str | None
    admission_error_count: int | None
    samples: int | None

    def as_dict(self) -> dict[str, Any]:
        """Return a fresh admission mapping with schema and a mutable copy of findings.

        Returns
        -------
        dict[str, Any]
            All nine result fields plus schema version. Mutation does not
            change this frozen result; typed declarations retain FAIL values.
        """
        return {
            "schema_version": RUNTIME_ADMISSION_EVIDENCE_SCHEMA_VERSION,
            "status": self.status,
            "errors": list(self.errors),
            "report_sha256": self.report_sha256,
            "payload_sha256": self.payload_sha256,
            "benchmark_evidence_class": self.benchmark_evidence_class,
            "production_claim_allowed": self.production_claim_allowed,
            "admission_status": self.admission_status,
            "admission_error_count": self.admission_error_count,
            "samples": self.samples,
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse key shadowing in every JSON object, including unrelated metadata."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_float(token: str) -> float:
    """Refuse nonfinite constants or overflowing exponents at any JSON depth."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite JSON number: {token}")
    return value


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    """Decode UTF-8 objects once and hash the same exact successfully decoded bytes."""
    blob = path.read_bytes()
    payload = json.loads(
        blob.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_float=_parse_json_float,
        parse_constant=_parse_json_float,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return payload, hashlib.sha256(blob).hexdigest()


def _sha256_hex(value: object) -> bool:
    """Recognise lowercase SHA-256 spelling without coercing other JSON types."""
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _positive_int(value: object) -> TypeGuard[int]:
    """Recognise a positive JSON integer while refusing booleans and floats."""
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _finite_non_negative(value: object) -> TypeGuard[float]:
    """Recognise nonnegative finite numeric declarations, including convertible integers."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and value >= 0.0
    except OverflowError:
        return False


def _non_empty_sequence(value: object) -> TypeGuard[list[object] | tuple[object, ...]]:
    """Recognise a nonempty list/tuple before inspecting context elements."""
    return isinstance(value, list | tuple) and bool(value)


def _validate_payload_hash(payload: dict[str, Any], errors: list[str]) -> None:
    """Check canonical self-consistency with the digest field blank and all extra fields included."""
    supplied = payload.get("payload_sha256")
    if not _sha256_hex(supplied):
        errors.append("runtime_admission.payload_sha256 must be a SHA-256 hex digest")
        return
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if supplied != digest:
        errors.append("runtime_admission.payload_sha256 does not match canonical payload")


def _validate_context(payload: dict[str, Any], errors: list[str]) -> None:
    """Require numeric CPU/load declarations and recorded context strings, not live host equivalence."""
    context = payload.get("context")
    if not isinstance(context, dict):
        errors.append("runtime_admission.context must be an object")
        return
    affinity = context.get("cpu_affinity")
    if not _non_empty_sequence(affinity):
        errors.append("runtime_admission.context.cpu_affinity must be a non-empty sequence")
    elif not all(isinstance(core, int) and not isinstance(core, bool) and core >= 0 for core in affinity):
        errors.append("runtime_admission.context.cpu_affinity must contain non-negative integer CPU IDs")
    if not isinstance(context.get("platform"), str) or not context.get("platform"):
        errors.append("runtime_admission.context.platform must be recorded")
    if not isinstance(context.get("python"), str) or not context.get("python"):
        errors.append("runtime_admission.context.python must be recorded")
    if not isinstance(context.get("isolation_method"), str) or not context.get("isolation_method"):
        errors.append("runtime_admission.context.isolation_method must be recorded")
    if not _non_empty_sequence(context.get("loadavg_start")) or not _non_empty_sequence(context.get("loadavg_end")):
        errors.append("runtime_admission.context must record loadavg_start and loadavg_end")
    for field in ("loadavg_start", "loadavg_end"):
        values = context.get(field)
        if _non_empty_sequence(values) and (len(values) != 3 or not all(_finite_non_negative(v) for v in values)):
            errors.append(f"runtime_admission.context.{field} must contain three finite non-negative load averages")


def _validate_stats(payload: dict[str, Any], errors: list[str]) -> int | None:
    """Check declared samples, finite latencies, percentile ordering and bounded mean."""
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        errors.append("runtime_admission.stats must be an object")
        return None
    samples = stats.get("samples")
    if not _positive_int(samples):
        errors.append("runtime_admission.stats.samples must be a positive integer")
        return None
    for field in ("min_us", "median_us", "mean_us", "p95_us", "p99_us", "max_us"):
        if not _finite_non_negative(stats.get(field)):
            errors.append(f"runtime_admission.stats.{field} must be finite and non-negative")
    ordered = (
        stats.get("min_us"),
        stats.get("median_us"),
        stats.get("p95_us"),
        stats.get("p99_us"),
        stats.get("max_us"),
    )
    if all(_finite_non_negative(value) for value in ordered):
        min_us, median_us, p95_us, p99_us, max_us = (float(value) for value in ordered if _finite_non_negative(value))
        if min_us > median_us or median_us > p95_us or p95_us > p99_us or p99_us > max_us:
            errors.append("runtime_admission.stats percentiles must be monotonic")
        mean_us = stats.get("mean_us")
        if _finite_non_negative(mean_us) and not min_us <= mean_us <= max_us:
            errors.append("runtime_admission.stats.mean_us must be between min_us and max_us")
    return int(samples)


def validate_runtime_admission_evidence(report: str | Path = DEFAULT_REPORT) -> RuntimeAdmissionEvidenceAdmission:
    """Inspect benchmark declarations and canonical payload self-consistency.

    Parameters
    ----------
    report : str or pathlib.Path
        Caller-relative UTF-8 JSON-object path; defaults to the historical
        repository local regression report. Symlinks are followed.

    Returns
    -------
    RuntimeAdmissionEvidenceAdmission
        Structured reader findings, exact byte digest and typed declarations.
        Empty objects fail required fields. Context affinity is a nonempty
        integer CPU-ID list, load arrays each contain three nonnegative finite
        numbers, and recorded context strings are nonempty. Latencies are
        finite/nonnegative with monotonic percentiles and mean inside min/max.
        Duplicate keys and nonfinite floating JSON fail at every depth.
        Production-class declarations need a true claim flag, probe PASS and
        empty errors; failed local regression needs explanatory string errors
        and a false production flag. Probe PASS requires empty errors in either
        class. These checks do not inspect live hardware,
        verify provenance, enforce latency limits or recompute timing samples.

    Examples
    --------
    A decoded empty object retains its byte digest and fails schema checks.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = Path(directory) / "empty.json"
    ...     _ = path.write_text("{}", encoding="utf-8")
    ...     result = validate_runtime_admission_evidence(path)
    >>> result.status, result.samples, result.report_sha256 is not None
    ('fail', None, True)
    """
    errors: list[str] = []
    payload: dict[str, Any] = {}
    report_sha256: str | None = None
    try:
        payload, report_sha256 = _load_json(Path(report))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        errors.append(f"runtime_admission.report: {exc}")

    samples: int | None = None
    if report_sha256 is not None:
        if payload.get("schema_version") != RUNTIME_ADMISSION_BENCHMARK_SCHEMA_VERSION:
            errors.append(f"runtime_admission.schema_version must be {RUNTIME_ADMISSION_BENCHMARK_SCHEMA_VERSION!r}")
        evidence_class = payload.get("evidence_class")
        if not isinstance(evidence_class, str) or evidence_class not in BENCHMARK_EVIDENCE_CLASSES:
            errors.append("runtime_admission.evidence_class must be recognised")
        production_claim_allowed = payload.get("production_claim_allowed")
        if not isinstance(production_claim_allowed, bool):
            errors.append("runtime_admission.production_claim_allowed must be a boolean")
        elif evidence_class == "local_regression" and production_claim_allowed:
            errors.append("local runtime admission evidence must not allow production benchmark claims")
        if evidence_class == "production_benchmark" and production_claim_allowed is not True:
            errors.append("production runtime admission evidence must allow production benchmark claims")
        if not isinstance(payload.get("command"), str) or "bench_runtime_admission.py" not in str(
            payload.get("command")
        ):
            errors.append("runtime_admission.command must identify the runtime-admission benchmark")
        admission_status = payload.get("last_admission_status")
        if not isinstance(admission_status, str) or admission_status not in {"pass", "fail"}:
            errors.append("runtime_admission.last_admission_status must be 'pass' or 'fail'")
        admission_errors = payload.get("last_admission_errors")
        if not isinstance(admission_errors, list):
            errors.append("runtime_admission.last_admission_errors must be a list")
        elif not all(isinstance(error, str) and bool(error) for error in admission_errors):
            errors.append("runtime_admission.last_admission_errors must contain non-empty strings")
        elif evidence_class == "local_regression" and admission_status == "fail" and not admission_errors:
            errors.append("failed local runtime admission evidence must include fail-closed errors")
        if evidence_class == "production_benchmark" and admission_status != "pass":
            errors.append("production runtime admission evidence must pass strict runtime admission")
        if evidence_class == "production_benchmark" and admission_errors:
            errors.append("production runtime admission evidence must not carry admission errors")
        if admission_status == "pass" and admission_errors:
            errors.append("passed runtime admission evidence must not carry admission errors")
        warnings = payload.get("last_admission_warnings")
        if not isinstance(warnings, list):
            errors.append("runtime_admission.last_admission_warnings must be a list")
        elif not all(isinstance(warning, str) and bool(warning) for warning in warnings):
            errors.append("runtime_admission.last_admission_warnings must contain non-empty strings")
        _validate_context(payload, errors)
        samples = _validate_stats(payload, errors)
        _validate_payload_hash(payload, errors)

    admission_errors = payload.get("last_admission_errors") if payload else None
    admission_error_count = len(admission_errors) if isinstance(admission_errors, list) else None
    return RuntimeAdmissionEvidenceAdmission(
        status="pass" if not errors else "fail",
        errors=tuple(errors),
        report_sha256=report_sha256,
        payload_sha256=payload.get("payload_sha256") if isinstance(payload.get("payload_sha256"), str) else None,
        benchmark_evidence_class=payload.get("evidence_class")
        if isinstance(payload.get("evidence_class"), str)
        else None,
        production_claim_allowed=(
            payload.get("production_claim_allowed")
            if isinstance(payload.get("production_claim_allowed"), bool)
            else None
        ),
        admission_status=payload.get("last_admission_status")
        if isinstance(payload.get("last_admission_status"), str)
        else None,
        admission_error_count=admission_error_count,
        samples=samples,
    )


def main(argv: list[str] | None = None) -> int:
    """Print report admission JSON/text to stdout and return zero for PASS or one for findings.

    Parameters
    ----------
    argv : list[str] or None
        Optional ``--report`` path and ``--json-out`` arguments. ``None`` uses
        process arguments; argparse usage errors retain SystemExit(2).

    Returns
    -------
    int
        Reader admission exit code. No benchmark or host probe is run and no
        report file is created; the default path reads historical local data.
    """
    parser = argparse.ArgumentParser(description="Validate runtime-admission benchmark evidence")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Runtime-admission benchmark JSON report")
    parser.add_argument("--json-out", action="store_true")
    args = parser.parse_args(argv)

    result = validate_runtime_admission_evidence(args.report)
    payload = result.as_dict()
    if args.json_out:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Runtime admission evidence: {result.status}")
        for error in result.errors:
            print(f"ERROR {error}")
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
