#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native Formal Certificate Evidence Validation

"""Inspect native AOT benchmark declarations without running a proof engine.

Read one UTF-8 object and hash those exact bytes. The reader checks benchmark
context, AOT case counts, certificate identifier/schema/digest spelling and a
positive finite p99 threshold. It does not reopen certificates, authenticate a
producer, rerun SMT, inspect the host or grant physical/control qualification.
Recorded p99 is an across-run summary, not a per-tick timing guarantee.

Caller-relative paths and symlinks are followed. Extra JSON metadata is ignored
except duplicate/nonfinite floating-token refusal; there is no containment or
input size/depth budget. The standalone standard-library CLI emits the result
as JSON; root validate and tracker53 consume the same public reader.
"""

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
    """Frozen findings and declared AOT benchmark metadata.

    Attributes
    ----------
    status : str
        pass when all reader checks succeed, otherwise fail.
    admitted_cases : tuple[str, ...]
        Sorted case labels whose AOT summary checks pass, even when global
        context/schema or digest instability makes overall status fail.
    certificate_assumption_sha256 : str or None
        One distinct syntactically valid digest observed across AOT summaries,
        including rejected summaries. None when zero or multiple are observed.
    benchmark_evidence_class : str or None
        Declared class string, including unknown values on FAIL.
    production_claim_allowed : bool
        Declared literal boolean, preserved on FAIL; other types become false.
        This is not host or controller qualification.
    errors : tuple[str, ...]
        Ordered read/context/argument/case/stability findings.
    report_sha256 : str or None
        Digest of exact decoded JSON-object bytes; read/decode/root-type error
        yields None. Hashes establish byte identity, not producer authenticity.
    """

    status: str
    admitted_cases: tuple[str, ...]
    certificate_assumption_sha256: str | None
    benchmark_evidence_class: str | None
    production_claim_allowed: bool
    errors: tuple[str, ...]
    report_sha256: str | None

    def as_dict(self) -> JSONMapping:
        """Return all fields plus result schema and fresh mutable finding/case lists."""
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
    """Refuse key shadowing at every JSON depth, including extra metadata."""
    out: JSONMapping = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _load_json(path: Path) -> tuple[JSONMapping, str]:
    """Read once, decode a UTF-8 object and hash the same exact bytes."""
    try:
        blob = path.read_bytes()
        payload = json.loads(
            blob.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_parse_json_float,
            parse_constant=_parse_json_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    return cast(JSONMapping, payload), hashlib.sha256(blob).hexdigest()


def _parse_json_float(token: str) -> float:
    """Refuse nonfinite JSON constants/exponents, including unrelated metadata."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite JSON number: {token}")
    return value


def _is_finite_number(value: JSONValue) -> bool:
    """Recognise finite integer/float declarations without bool coercion or overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_non_negative_int(value: JSONValue) -> bool:
    """Recognise nonnegative integer declarations, excluding booleans."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_sha256(value: object) -> bool:
    """Check lowercase SHA-256 spelling without authenticating a certificate."""
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _mapping(value: JSONValue) -> JSONMapping | None:
    """Select a JSON object without converting other JSON shapes."""
    if isinstance(value, dict):
        return value
    return None


def _string_list(value: JSONValue) -> list[str] | None:
    """Select a nonempty list of nonempty strings; whitespace strings are retained."""
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return cast(list[str], value)
    return None


def _int_list(value: JSONValue) -> list[int] | None:
    """Select a nonempty list of nonnegative integers with booleans refused."""
    if (
        isinstance(value, list)
        and value
        and all(not isinstance(item, bool) and isinstance(item, int) and item >= 0 for item in value)
    ):
        return cast(list[int], value)
    return None


def _non_empty_string(value: JSONValue) -> bool:
    """Recognise text containing at least one non-whitespace character."""
    return isinstance(value, str) and bool(value.strip())


def _validate_benchmark_context(payload: JSONMapping, errors: list[str]) -> tuple[str | None, bool]:
    """Check required context declarations and production-class metadata boundaries.

    Integer CPU lists need not be unique, online or mutually compatible. String
    loads/governors/isolation are not independently verified. runtime_versions
    needs a nonempty object but its contents remain unchecked. Production
    requires explicit isolation text, boolean heavy-job declaration and literal
    workspace_dirty=false; true heavy-job declarations are not rejected here.
    The returned production boolean is declared metadata, even on FAIL.
    """
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
        elif payload.get("workspace_dirty") is not False:
            errors.append("production benchmark evidence must declare workspace_dirty=false")
        if not production_claim_allowed_bool:
            errors.append("production benchmark evidence must set production_claim_allowed=true")

    return evidence_class, production_claim_allowed_bool


def _validate_summary_case(
    name: str,
    summary: JSONMapping,
    *,
    max_aot_p99_cycle_us: float,
) -> tuple[str | None, str | None, list[str]]:
    """Inspect counts, certificate spellings and declared p99 for one AOT label.

    Non-AOT labels return no admission or findings after the caller checks object
    shape. AOT generated/submitted/checked counts must agree and be positive;
    drops/failures must be zero and certificate count must equal positive runs.
    Return any valid digest even with other case findings. Other latency fields,
    execution rows and certificate bytes are not checked or recomputed.
    """
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
    """Inspect one persisted benchmark report through the public metadata API.

    Parameters
    ----------
    report_path : str or Path
        Caller-relative path or historical repository default. Read-only; no
        report, benchmark, certificate or proof artifact is generated.
    max_aot_p99_cycle_us : float
        Positive finite non-boolean numeric bound for each AOT avg_cycle_us.p99.
        Ill-typed/nonfinite/overflowing values return structured FAIL before
        evaluating case limits, with empty cases and no observed digest.

    Returns
    -------
    NativeFormalCertificateEvidenceResult
        Supported read, UTF-8, JSON, duplicate/nonfinite, recursion, schema,
        context, count and threshold refusals become findings. Case-level
        admitted labels and a single observed digest may remain on global FAIL.
        PASS admits declarations, not authenticity or certified control.

    Examples
    --------
    Empty metadata cannot establish an AOT certificate:

    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     path = Path(directory) / 'empty.json'
    ...     _ = path.write_text('{}', encoding='utf-8')
    ...     result = validate_native_formal_certificate_evidence(path)
    ...     (result.status, result.admitted_cases)
    ('fail', ())
    """
    path = Path(report_path)
    errors: list[str] = []
    try:
        payload, report_sha256 = _load_json(path)
    except (OSError, ValueError, RecursionError) as exc:
        return NativeFormalCertificateEvidenceResult("fail", (), None, None, False, (str(exc),), None)

    if payload.get("schema") != BENCHMARK_SCHEMA_VERSION:
        errors.append(f"schema must be {BENCHMARK_SCHEMA_VERSION!r}")

    benchmark_evidence_class, production_claim_allowed = _validate_benchmark_context(payload, errors)

    if not _is_finite_number(cast(JSONValue, max_aot_p99_cycle_us)) or max_aot_p99_cycle_us <= 0.0:
        errors.append("max_aot_p99_cycle_us must be positive and finite")
        return NativeFormalCertificateEvidenceResult(
            "fail", (), None, benchmark_evidence_class, production_claim_allowed, tuple(errors), report_sha256
        )

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
    """Run the read-only standard-library CLI and emit complete JSON findings.

    argv excludes the executable; None reads process arguments. The positional
    report defaults to the absolute historical repository path. Return zero on
    reader PASS, one on refusal. Argparse errors exit two. No report is written
    and no proof, benchmark or native loop is launched.
    """
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
