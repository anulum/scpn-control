# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime admission evidence validation.
"""Admit PREEMPT_RT runtime-admission evidence for release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import TypeIs


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ADMISSION_EVIDENCE_SCHEMA_VERSION = "scpn-control.runtime-admission-evidence-admission.v1"
RUNTIME_ADMISSION_BENCHMARK_SCHEMA_VERSION = "scpn-control.runtime-admission-benchmark.v1"
DEFAULT_REPORT = ROOT / "validation" / "reports" / "runtime_admission_release_20260605T000000Z.json"
BENCHMARK_EVIDENCE_CLASSES = frozenset({"local_regression", "production_benchmark"})


@dataclass(frozen=True)
class RuntimeAdmissionEvidenceAdmission:
    """Admission result for runtime-admission release evidence."""

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
        """Return a stable JSON representation for top-level release evidence."""
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
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    blob = path.read_bytes()
    payload = json.loads(blob.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return payload, hashlib.sha256(blob).hexdigest()


def _sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _positive_int(value: object) -> TypeIs[int]:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _finite_non_negative(value: object) -> TypeIs[float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0.0


def _non_empty_sequence(value: object) -> bool:
    return isinstance(value, list | tuple) and bool(value)


def _validate_payload_hash(payload: dict[str, Any], errors: list[str]) -> None:
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
    context = payload.get("context")
    if not isinstance(context, dict):
        errors.append("runtime_admission.context must be an object")
        return
    if not _non_empty_sequence(context.get("cpu_affinity")):
        errors.append("runtime_admission.context.cpu_affinity must be a non-empty sequence")
    if not isinstance(context.get("platform"), str) or not context.get("platform"):
        errors.append("runtime_admission.context.platform must be recorded")
    if not isinstance(context.get("python"), str) or not context.get("python"):
        errors.append("runtime_admission.context.python must be recorded")
    if not isinstance(context.get("isolation_method"), str) or not context.get("isolation_method"):
        errors.append("runtime_admission.context.isolation_method must be recorded")
    if not _non_empty_sequence(context.get("loadavg_start")) or not _non_empty_sequence(context.get("loadavg_end")):
        errors.append("runtime_admission.context must record loadavg_start and loadavg_end")


def _validate_stats(payload: dict[str, Any], errors: list[str]) -> int | None:
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
    return int(samples)


def validate_runtime_admission_evidence(report: str | Path = DEFAULT_REPORT) -> RuntimeAdmissionEvidenceAdmission:
    """Validate a runtime-admission benchmark report before release admission."""
    errors: list[str] = []
    payload: dict[str, Any] = {}
    report_sha256: str | None = None
    try:
        payload, report_sha256 = _load_json(Path(report))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"runtime_admission.report: {exc}")

    samples: int | None = None
    if payload:
        if payload.get("schema_version") != RUNTIME_ADMISSION_BENCHMARK_SCHEMA_VERSION:
            errors.append(f"runtime_admission.schema_version must be {RUNTIME_ADMISSION_BENCHMARK_SCHEMA_VERSION!r}")
        evidence_class = payload.get("evidence_class")
        if evidence_class not in BENCHMARK_EVIDENCE_CLASSES:
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
        if admission_status not in {"pass", "fail"}:
            errors.append("runtime_admission.last_admission_status must be 'pass' or 'fail'")
        admission_errors = payload.get("last_admission_errors")
        if not isinstance(admission_errors, list):
            errors.append("runtime_admission.last_admission_errors must be a list")
        elif evidence_class == "local_regression" and admission_status == "fail" and not admission_errors:
            errors.append("failed local runtime admission evidence must include fail-closed errors")
        if evidence_class == "production_benchmark" and admission_status != "pass":
            errors.append("production runtime admission evidence must pass strict runtime admission")
        if evidence_class == "production_benchmark" and admission_errors:
            errors.append("production runtime admission evidence must not carry admission errors")
        if not isinstance(payload.get("last_admission_warnings"), list):
            errors.append("runtime_admission.last_admission_warnings must be a list")
        _validate_context(payload, errors)
        samples = _validate_stats(payload, errors)
        _validate_payload_hash(payload, errors)

    admission_errors = payload.get("last_admission_errors") if payload else None
    admission_error_count = len(admission_errors) if isinstance(admission_errors, list) else None
    return RuntimeAdmissionEvidenceAdmission(
        status="pass" if not errors else "fail",
        errors=tuple(errors),
        report_sha256=report_sha256,
        payload_sha256=payload.get("payload_sha256") if payload else None,
        benchmark_evidence_class=payload.get("evidence_class") if payload else None,
        production_claim_allowed=payload.get("production_claim_allowed") if payload else None,
        admission_status=payload.get("last_admission_status") if payload else None,
        admission_error_count=admission_error_count,
        samples=samples,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for runtime-admission evidence admission."""
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
