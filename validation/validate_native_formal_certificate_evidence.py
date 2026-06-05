#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native Formal Certificate Evidence Validation

"""Validate native AOT certificate benchmark evidence before admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "validation" / "reports" / "native_formal_aot_certificate_admission_20260604T103219Z.json"
RESULT_SCHEMA_VERSION = "scpn-control.native-formal-certificate-evidence.v1"
BENCHMARK_SCHEMA_VERSION = "scpn-control.native_formal_modes.v1"
BENCHMARK_CONTEXT_SCHEMA_VERSION = "scpn-control.benchmark-context.v1"
CERTIFICATE_SCHEMA_VERSION = "scpn-control.native-formal.aot-certificate.v1"
CERTIFICATE_ID = "bounded-petri-marking-sufficient-invariant"
DEFAULT_MAX_AOT_P99_CYCLE_US = 10.0
LOCAL_REGRESSION_EVIDENCE = "local_regression"
PRODUCTION_BENCHMARK_EVIDENCE = "production_benchmark"
ALLOWED_EVIDENCE_CLASSES = frozenset({LOCAL_REGRESSION_EVIDENCE, PRODUCTION_BENCHMARK_EVIDENCE})
UNISOLATED_METHODS = frozenset({"", "none", "unknown", "unspecified"})

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
JSONMapping: TypeAlias = dict[str, JSONValue]


@dataclass(frozen=True)
class NativeFormalCertificateEvidenceResult:
    """Strict admission result for native formal AOT certificate evidence."""

    status: str
    admitted_cases: tuple[str, ...]
    certificate_assumption_sha256: str | None
    benchmark_evidence_class: str | None
    production_claim_allowed: bool
    errors: tuple[str, ...]
    report_sha256: str | None

    def as_dict(self) -> JSONMapping:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": self.status,
            "admitted_cases": list(self.admitted_cases),
            "certificate_assumption_sha256": self.certificate_assumption_sha256,
            "benchmark_evidence_class": self.benchmark_evidence_class,
            "production_claim_allowed": self.production_claim_allowed,
            "errors": list(self.errors),
            "report_sha256": self.report_sha256,
        }


def _reject_duplicate_keys(pairs: list[tuple[str, JSONValue]]) -> JSONMapping:
    out: JSONMapping = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _load_json(path: Path) -> JSONMapping:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    return cast(JSONMapping, payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_finite_number(value: JSONValue) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _is_non_negative_int(value: JSONValue) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _mapping(value: JSONValue) -> JSONMapping | None:
    if isinstance(value, dict):
        return cast(JSONMapping, value)
    return None


def _string_list(value: JSONValue) -> list[str] | None:
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return cast(list[str], value)
    return None


def _int_list(value: JSONValue) -> list[int] | None:
    if (
        isinstance(value, list)
        and value
        and all(not isinstance(item, bool) and isinstance(item, int) and item >= 0 for item in value)
    ):
        return cast(list[int], value)
    return None


def _non_empty_string(value: JSONValue) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_benchmark_context(payload: JSONMapping, errors: list[str]) -> tuple[str | None, bool]:
    context = _mapping(payload.get("benchmark_context"))
    if context is None:
        errors.append("benchmark_context must be an object")
        return None, False

    if context.get("schema_version") != BENCHMARK_CONTEXT_SCHEMA_VERSION:
        errors.append(f"benchmark_context.schema_version must be {BENCHMARK_CONTEXT_SCHEMA_VERSION!r}")

    evidence_class_raw = context.get("evidence_class")
    evidence_class = evidence_class_raw if isinstance(evidence_class_raw, str) else None
    if evidence_class not in ALLOWED_EVIDENCE_CLASSES:
        errors.append("benchmark_context.evidence_class must be local_regression or production_benchmark")

    production_claim_allowed = context.get("production_claim_allowed")
    if not isinstance(production_claim_allowed, bool):
        errors.append("benchmark_context.production_claim_allowed must be a boolean")
        production_claim_allowed_bool = False
    else:
        production_claim_allowed_bool = production_claim_allowed

    command = _string_list(context.get("command"))
    if command is None:
        errors.append("benchmark_context.command must be a non-empty list of command arguments")
    elif not any("benchmark_native_formal_modes.py" in item for item in command):
        errors.append("benchmark_context.command must identify the native formal benchmark")
    if _int_list(context.get("affinity_cpus")) is None:
        errors.append("benchmark_context.affinity_cpus must be a non-empty integer list")
    if _int_list(context.get("reserved_core_set")) is None:
        errors.append("benchmark_context.reserved_core_set must be a non-empty integer list")

    for field in (
        "isolation_method",
        "host_load_before",
        "host_load_after",
        "cpu_governor",
        "cpu_frequency_context",
        "hardware_model",
        "os",
        "python",
        "claim_boundary",
    ):
        if not _non_empty_string(context.get(field)):
            errors.append(f"benchmark_context.{field} must be a non-empty string")

    runtime_versions = _mapping(context.get("runtime_versions"))
    if runtime_versions is None or not runtime_versions:
        errors.append("benchmark_context.runtime_versions must be a non-empty object")

    heavy_jobs = context.get("other_heavy_jobs_running")
    if not (isinstance(heavy_jobs, bool) or heavy_jobs == "unknown"):
        errors.append("benchmark_context.other_heavy_jobs_running must be a boolean or 'unknown'")

    if evidence_class == LOCAL_REGRESSION_EVIDENCE and production_claim_allowed_bool:
        errors.append("local regression evidence must not allow production benchmark claims")
    if evidence_class == PRODUCTION_BENCHMARK_EVIDENCE:
        isolation_method = str(context.get("isolation_method", "")).strip().lower()
        if isolation_method in UNISOLATED_METHODS:
            errors.append("production benchmark evidence requires an explicit CPU/core isolation method")
        if heavy_jobs == "unknown":
            errors.append("production benchmark evidence must declare whether other heavy jobs were running")
        if payload.get("workspace_dirty") is True:
            errors.append("production benchmark evidence must not come from a dirty workspace")
        if not production_claim_allowed_bool:
            errors.append("production benchmark evidence must set production_claim_allowed=true")

    return evidence_class, production_claim_allowed_bool


def _validate_summary_case(
    name: str,
    summary: JSONMapping,
    *,
    max_aot_p99_cycle_us: float,
) -> tuple[str | None, str | None, list[str]]:
    errors: list[str] = []
    if ":aot_certificate:" not in name:
        return None, None, errors

    certificate_count = summary.get("certificate_admitted_total")
    runs = summary.get("runs")
    if not _is_non_negative_int(certificate_count):
        errors.append(f"{name}: certificate_admitted_total must be a non-negative integer")
    if not _is_non_negative_int(runs) or int(cast(int, runs)) <= 0:
        errors.append(f"{name}: runs must be positive")
    elif certificate_count != runs:
        errors.append(f"{name}: every AOT run must admit a certificate")

    for field in (
        "formal_generated_total",
        "formal_submitted_total",
        "formal_checked_total",
        "formal_dropped_total",
        "formal_failures_total",
    ):
        if not _is_non_negative_int(summary.get(field)):
            errors.append(f"{name}: {field} must be a non-negative integer")

    generated = summary.get("formal_generated_total")
    submitted = summary.get("formal_submitted_total")
    checked = summary.get("formal_checked_total")
    dropped = summary.get("formal_dropped_total")
    failures = summary.get("formal_failures_total")
    if _is_non_negative_int(generated) and int(cast(int, generated)) <= 0:
        errors.append(f"{name}: AOT evidence must generate certificate checks")
    if _is_non_negative_int(generated) and _is_non_negative_int(submitted) and submitted != generated:
        errors.append(f"{name}: submitted checks must equal generated checks")
    if _is_non_negative_int(generated) and _is_non_negative_int(checked) and checked != generated:
        errors.append(f"{name}: checked proofs must equal generated checks")
    if dropped != 0:
        errors.append(f"{name}: dropped checks must be zero")
    if failures != 0:
        errors.append(f"{name}: formal failures must be zero")

    versions = summary.get("certificate_schema_versions")
    ids = summary.get("certificate_ids")
    digests = summary.get("certificate_assumption_sha256_values")
    if versions != [CERTIFICATE_SCHEMA_VERSION]:
        errors.append(f"{name}: certificate schema version mismatch")
    if ids != [CERTIFICATE_ID]:
        errors.append(f"{name}: certificate id mismatch")
    if not isinstance(digests, list) or len(digests) != 1 or not _is_sha256(digests[0]):
        errors.append(f"{name}: exactly one SHA-256 certificate digest is required")
        digest: str | None = None
    else:
        digest = cast(str, digests[0])

    avg_cycle = _mapping(summary.get("avg_cycle_us"))
    if avg_cycle is None:
        errors.append(f"{name}: avg_cycle_us must be an object")
    else:
        p99 = avg_cycle.get("p99")
        if not _is_finite_number(p99) or float(cast(float, p99)) <= 0.0:
            errors.append(f"{name}: avg_cycle_us.p99 must be positive and finite")
        elif float(cast(float, p99)) > max_aot_p99_cycle_us:
            errors.append(
                f"{name}: avg_cycle_us.p99 {float(cast(float, p99)):.6f} us exceeds {max_aot_p99_cycle_us:.6f} us"
            )

    return name, digest, errors


def validate_native_formal_certificate_evidence(
    report_path: str | Path = DEFAULT_REPORT,
    *,
    max_aot_p99_cycle_us: float = DEFAULT_MAX_AOT_P99_CYCLE_US,
) -> NativeFormalCertificateEvidenceResult:
    """Validate a native formal benchmark report for AOT certificate admission."""

    path = Path(report_path)
    errors: list[str] = []
    try:
        report_sha256 = _sha256_file(path)
        payload = _load_json(path)
    except (OSError, ValueError) as exc:
        return NativeFormalCertificateEvidenceResult("fail", (), None, None, False, (str(exc),), None)

    if payload.get("schema") != BENCHMARK_SCHEMA_VERSION:
        errors.append(f"schema must be {BENCHMARK_SCHEMA_VERSION!r}")

    benchmark_evidence_class, production_claim_allowed = _validate_benchmark_context(payload, errors)

    if not _is_finite_number(cast(JSONValue, max_aot_p99_cycle_us)) or max_aot_p99_cycle_us <= 0.0:
        errors.append("max_aot_p99_cycle_us must be positive and finite")

    summaries = _mapping(payload.get("summaries"))
    if summaries is None:
        errors.append("summaries must be an object")
        summaries = {}

    admitted_cases: list[str] = []
    observed_digests: set[str] = set()
    for case_name, raw_summary in summaries.items():
        summary = _mapping(raw_summary)
        if summary is None:
            errors.append(f"{case_name}: summary must be an object")
            continue
        admitted_case, digest, case_errors = _validate_summary_case(
            case_name,
            summary,
            max_aot_p99_cycle_us=max_aot_p99_cycle_us,
        )
        errors.extend(case_errors)
        if admitted_case is not None and not case_errors:
            admitted_cases.append(admitted_case)
        if digest is not None:
            observed_digests.add(digest)

    if not admitted_cases:
        errors.append("at least one AOT certificate case must be admitted")
    if len(observed_digests) > 1:
        errors.append("AOT certificate digest must be stable across admitted cases")

    status = "pass" if not errors else "fail"
    certificate_digest = next(iter(observed_digests)) if len(observed_digests) == 1 else None
    return NativeFormalCertificateEvidenceResult(
        status,
        tuple(sorted(admitted_cases)),
        certificate_digest,
        benchmark_evidence_class,
        production_claim_allowed,
        tuple(errors),
        report_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate native AOT formal-certificate benchmark evidence.")
    parser.add_argument(
        "report",
        nargs="?",
        default=str(DEFAULT_REPORT),
        help="native formal benchmark JSON report",
    )
    parser.add_argument(
        "--max-aot-p99-cycle-us",
        type=float,
        default=DEFAULT_MAX_AOT_P99_CYCLE_US,
        help="maximum admitted AOT avg_cycle_us.p99 threshold",
    )
    args = parser.parse_args(argv)

    result = validate_native_formal_certificate_evidence(
        args.report,
        max_aot_p99_cycle_us=args.max_aot_p99_cycle_us,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
