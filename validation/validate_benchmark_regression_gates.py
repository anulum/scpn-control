#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Validate persisted benchmark regression admission gates.

"""Check persisted local timing metadata and byte custody without running benchmarks.

Require nonempty versioned entries, exact report SHA-256, matching numeric
metrics/sample counters, inclusive upper thresholds, minimum samples, bounded
claim text and declared machine/platform context. Reports use us/ms/s as
declared; no unit conversion, fresh timing, isolation check, hardware origin or
physical/real-time admission is inferred. generated_utc is checked for presence,
not date format or freshness. Extra finite JSON fields remain uninterpreted.

Each file inspection captures one byte buffer for both SHA-256 and decoded
fields. Repeated report references reread the file per entry; separate reads
have no transaction or coherent snapshot.
The standard-library CLI reads local files and prints a versioned JSON result;
it never writes artifacts, executes a benchmark or contacts a remote service.
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
DEFAULT_MANIFEST = ROOT / "validation" / "reports" / "benchmark_regression_gates.json"
SCHEMA_VERSION = "scpn-control.benchmark-regression-gates.v1"
ALLOWED_UNITS = frozenset({"us", "ms", "s"})
SELF_DIGEST_FIELDS = frozenset({"payload_sha256", "report_payload_sha256"})

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
JSONMapping: TypeAlias = dict[str, JSONValue]


@dataclass(frozen=True)
class BenchmarkRegressionGateResult:
    """Immutable metadata-validation result, not proof of current measured performance.

    Validator status is pass/fail; admitted_gates retains manifest entry order
    only when every check passes, otherwise it is empty. errors contains ordered
    authored diagnostics. manifest_sha256 binds inspected raw bytes, including
    malformed input, or is empty when those bytes could not be read. Direct
    construction performs no validation; callers should use the public validator.
    """

    status: str
    admitted_gates: tuple[str, ...]
    errors: tuple[str, ...]
    manifest_sha256: str

    def as_dict(self) -> JSONMapping:
        """Return a new JSON carrier with schema version and list copies of tuple fields."""
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "admitted_gates": list(self.admitted_gates),
            "errors": list(self.errors),
            "manifest_sha256": self.manifest_sha256,
        }


class _JSONInspectionError(ValueError):
    """Carry an authored JSON refusal without exposing decoder/interpreter exception text."""


def _reject_duplicate_keys(pairs: list[tuple[str, JSONValue]]) -> JSONMapping:
    """Refuse duplicate keys at every object depth before field selection or hashing semantics."""
    out: JSONMapping = {}
    for key, value in pairs:
        if key in out:
            raise _JSONInspectionError("duplicate JSON key")
        out[key] = value
    return out


def _reject_nonfinite(token: str) -> None:
    """Refuse nonstandard JSON NaN/Infinity tokens, including uninterpreted metadata."""
    raise _JSONInspectionError("nonfinite JSON token")


def _finite_number(value: int | float) -> float:
    """Convert numeric inputs under one finite policy shared by decoding and metric checks.

    Callers exclude boolean fields. Refuse integer conversion overflow and
    nonfinite floating values with an authored error; no numeric clipping occurs.
    """
    try:
        out = float(value)
    except OverflowError as exc:
        raise _JSONInspectionError("number must be finite") from exc
    if not math.isfinite(out):
        raise _JSONInspectionError("number must be finite")
    return out


def _finite_json_float(token: str) -> float:
    """Decode a floating JSON token through the same finite conversion used for selected metrics."""
    return _finite_number(float(token))


def _decode_json(data: bytes) -> JSONMapping:
    """Decode the already captured UTF-8 bytes as an unambiguous finite JSON object.

    Refuse malformed/undecodable/non-object content with authored ValueError
    messages. Parsing performs no further file read, keeping byte SHA and fields
    coupled. Integer decoding limits and other decoder refusals map to a fixed
    message; floating overflow and duplicate/nonfinite hooks retain their reason.
    """
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_finite_json_float,
        )
    except _JSONInspectionError:
        raise
    except UnicodeError as exc:
        raise _JSONInspectionError("manifest/report must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise _JSONInspectionError("malformed JSON") from exc
    except ValueError as exc:
        raise _JSONInspectionError("JSON decoding refused") from exc
    if not isinstance(payload, dict):
        raise _JSONInspectionError("root must be a JSON object")
    return cast(JSONMapping, payload)


def _sha256_json(value: JSONValue) -> str:
    """Return the canonical SHA-256 digest for a JSON value."""
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256_hex(value: str) -> bool:
    """Return whether ``value`` is a SHA-256 hex digest."""
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _validate_self_digests(value: JSONValue, path: str, errors: list[str]) -> None:
    """Recursively validate embedded JSON self-digest fields.

    Only canonical self-digests are checked here. Artifact, dataset, model, and
    source file digests are intentionally left to their domain validators.
    """
    if isinstance(value, dict):
        for field in SELF_DIGEST_FIELDS:
            if field not in value:
                continue
            declared = value[field]
            field_path = f"{path}.{field}" if path else field
            if not isinstance(declared, str) or not _is_sha256_hex(declared):
                errors.append(f"{field_path} must be a SHA-256 hex digest")
                continue
            digest_payload = {key: item for key, item in value.items() if key != field}
            if _sha256_json(digest_payload) != declared.lower():
                errors.append(f"{field_path} self-digest mismatch")
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else key
            _validate_self_digests(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_self_digests(item, f"{path}[{index}]", errors)


def _as_text(value: JSONValue, field: str, errors: list[str]) -> str:
    """Require a nonblank string, preserving its original whitespace and case or recording refusal."""
    if isinstance(value, str) and value.strip():
        return value
    errors.append(f"{field} must be a non-empty string")
    return ""


def _as_number(value: JSONValue, field: str, errors: list[str]) -> float:
    """Convert a nonboolean numeric field, recording nonfinite/overflow refusal with a NaN sentinel."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        errors.append(f"{field} must be a finite number")
        return math.nan
    try:
        return _finite_number(value)
    except _JSONInspectionError:
        errors.append(f"{field} must be finite")
        return math.nan


def _as_positive_int(value: JSONValue, field: str, errors: list[str]) -> int:
    """Require a strictly positive integer sample counter; bools are rejected, failures return -1."""
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{field} must be a positive integer")
        return -1
    if value <= 0:
        errors.append(f"{field} must be positive")
        return -1
    return value


def _safe_repo_relative_uri_parts(uri: str) -> tuple[str, ...] | None:
    """Accept literal relative slash-separated components without normalization or URL/parent escapes."""
    if not uri or "\\" in uri or "://" in uri or uri.startswith(("/", "~", "file:")):
        return None
    parts = uri.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return tuple(str(part) for part in parts)


def _is_safe_repo_relative_uri(uri: str) -> bool:
    """Report lexical URI safety; filesystem/symlink containment is checked separately."""
    return _safe_repo_relative_uri_parts(uri) is not None


def _report_path(uri: str, report_root: Path) -> Path:
    """Resolve a lexically safe URI beneath an absolute report root and refuse symlink escape.

    ValueError signals unsafe components or containment failure. Resolution
    I/O/symlink-loop errors propagate for the owning validator to report safely.
    """
    parts = _safe_repo_relative_uri_parts(uri)
    if parts is None:
        raise ValueError(f"unsafe report path escapes report root: {uri}")
    path = (report_root / Path(*parts)).resolve()
    if not path.is_relative_to(report_root):
        raise ValueError(f"unsafe report path escapes report root: {uri}")
    return path


def _value_at_path(payload: JSONMapping, dotted_path: str) -> JSONValue:
    """Walk exact dot-separated object keys; empty/missing segments or nonobject parents raise KeyError.

    No list indexing or escaped dots are supported; the input is not mutated.
    """
    current: JSONValue = payload
    for segment in dotted_path.split("."):
        if not segment:
            raise KeyError(dotted_path)
        if not isinstance(current, dict) or segment not in current:
            raise KeyError(dotted_path)
        current = current[segment]
    return current


def _validate_hardware_context(
    report: JSONMapping,
    path: str,
    benchmark_id: str,
    errors: list[str],
) -> None:
    """Require declared machine and platform-or-kernel strings at a literal report object path.

    Append metadata refusals in place; strings are not authenticated hardware or
    isolation evidence, and a kernel string such as unknown is not RT approval.
    """
    try:
        context = _value_at_path(report, path)
    except KeyError:
        errors.append(f"{benchmark_id}: hardware context path missing: {path}")
        return
    if not isinstance(context, dict):
        errors.append(f"{benchmark_id}: hardware context must be an object")
        return
    machine = context.get("machine")
    platform = context.get("platform")
    rt_kernel = context.get("rt_kernel")
    if not isinstance(machine, str) or not machine.strip():
        errors.append(f"{benchmark_id}: hardware context must include machine")
    if not (isinstance(platform, str) and platform.strip() or isinstance(rt_kernel, str) and rt_kernel.strip()):
        errors.append(f"{benchmark_id}: hardware context must include platform or rt_kernel")


def _validate_entry(
    entry: JSONMapping,
    seen_ids: set[str],
    root_errors: list[str],
    report_root: Path,
) -> str | None:
    """Validate one entry against a captured report and append ordered errors without artifact writes.

    Entry-local shape errors stop its report read. Otherwise check containment,
    exact bytes/digest, optional declared self-digests, metric/counter equality,
    inclusive upper/minimum bounds and literal claim/context metadata. Us/ms/s
    labels are not converted or dimension-checked. Seen ids mutate to detect
    duplicates; final admission is decided by the parent across all entries.
    """
    local_errors: list[str] = []
    benchmark_id = _as_text(entry.get("benchmark_id"), "benchmark_id", local_errors)
    if benchmark_id in seen_ids:
        local_errors.append(f"{benchmark_id}: duplicate benchmark_id")
    seen_ids.add(benchmark_id)

    status = _as_text(entry.get("status"), f"{benchmark_id}.status", local_errors)
    if status != "pass":
        local_errors.append(f"{benchmark_id}: gate status must be pass")

    report_uri = _as_text(entry.get("report_uri"), f"{benchmark_id}.report_uri", local_errors)
    if not _is_safe_repo_relative_uri(report_uri):
        local_errors.append(f"{benchmark_id}: report_uri is not repository-relative")

    report_sha256 = _as_text(entry.get("report_sha256"), f"{benchmark_id}.report_sha256", local_errors)
    metric_path = _as_text(entry.get("metric_path"), f"{benchmark_id}.metric_path", local_errors)
    observed = _as_number(entry.get("observed"), f"{benchmark_id}.observed", local_errors)
    unit = _as_text(entry.get("unit"), f"{benchmark_id}.unit", local_errors)
    if unit and unit not in ALLOWED_UNITS:
        local_errors.append(f"{benchmark_id}: unsupported unit: {unit}")
    max_threshold = _as_number(entry.get("max_threshold"), f"{benchmark_id}.max_threshold", local_errors)
    if math.isfinite(max_threshold) and max_threshold <= 0.0:
        local_errors.append(f"{benchmark_id}: max_threshold must be positive")
    sample_count_path = _as_text(
        entry.get("sample_count_path"),
        f"{benchmark_id}.sample_count_path",
        local_errors,
    )
    sample_count = _as_positive_int(entry.get("sample_count"), f"{benchmark_id}.sample_count", local_errors)
    min_sample_count = _as_positive_int(
        entry.get("min_sample_count"),
        f"{benchmark_id}.min_sample_count",
        local_errors,
    )
    claim_status_path = _as_text(
        entry.get("claim_status_path"),
        f"{benchmark_id}.claim_status_path",
        local_errors,
    )
    required_claim_substring = _as_text(
        entry.get("required_claim_substring"),
        f"{benchmark_id}.required_claim_substring",
        local_errors,
    )
    hardware_context_path = _as_text(
        entry.get("hardware_context_path"),
        f"{benchmark_id}.hardware_context_path",
        local_errors,
    )

    if local_errors:
        root_errors.extend(local_errors)
        return None

    try:
        report_path = _report_path(report_uri, report_root)
    except (OSError, RuntimeError, ValueError):
        root_errors.append(f"{benchmark_id}: report_uri could not be resolved within report root")
        return None
    try:
        report_bytes = report_path.read_bytes()
    except FileNotFoundError:
        root_errors.append(f"{benchmark_id}: report does not exist: {report_uri}")
        return None
    except OSError:
        root_errors.append(f"{benchmark_id}: report could not be read")
        return None
    actual_report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    if actual_report_sha256 != report_sha256:
        root_errors.append(f"{benchmark_id}: report_sha256 mismatch")

    try:
        report = _decode_json(report_bytes)
    except _JSONInspectionError as exc:
        root_errors.append(f"{benchmark_id}: {exc}")
        return None
    _validate_self_digests(report, benchmark_id, root_errors)

    try:
        report_metric = _as_number(
            _value_at_path(report, metric_path),
            f"{benchmark_id}.report_metric",
            root_errors,
        )
    except KeyError:
        root_errors.append(f"{benchmark_id}: metric_path missing: {metric_path}")
        report_metric = math.nan
    if math.isfinite(observed) and math.isfinite(report_metric):
        if not math.isclose(observed, report_metric, rel_tol=1e-12, abs_tol=1e-9):
            root_errors.append(f"{benchmark_id}: observed metric does not match report")
    # Both fields are finite: entry-local numeric failures returned above.
    if observed > max_threshold:
        root_errors.append(f"{benchmark_id}: observed exceeds max_threshold")

    try:
        report_sample_count = _as_positive_int(
            _value_at_path(report, sample_count_path),
            f"{benchmark_id}.report_sample_count",
            root_errors,
        )
    except KeyError:
        root_errors.append(f"{benchmark_id}: sample_count_path missing: {sample_count_path}")
        report_sample_count = -1
    if report_sample_count != sample_count:
        root_errors.append(f"{benchmark_id}: sample_count does not match report")
    if sample_count < min_sample_count:
        root_errors.append(f"{benchmark_id}: sample_count below min_sample_count")

    try:
        claim_status = _value_at_path(report, claim_status_path)
    except KeyError:
        root_errors.append(f"{benchmark_id}: claim_status_path missing")
        claim_status = None
    if not isinstance(claim_status, str):
        root_errors.append(f"{benchmark_id}: claim_status must be a string")
    elif required_claim_substring not in claim_status:
        root_errors.append(f"{benchmark_id}: required claim boundary substring missing")

    _validate_hardware_context(report, hardware_context_path, benchmark_id, root_errors)
    return benchmark_id


def validate_benchmark_regression_gates(
    manifest_path: Path = DEFAULT_MANIFEST,
) -> BenchmarkRegressionGateResult:
    """Validate a local persisted manifest and return ordered all-or-none gate metadata admission.

    Parameters
    ----------
    manifest_path
        Local JSON file, relative to cwd. Reports resolve from ROOT when the
        resolved manifest belongs to this script repository; otherwise from
        its resolved parent. URIs must have literal relative components and
        remain within that root after symlink resolution.

    Returns
    -------
    BenchmarkRegressionGateResult
        Pass only with nonempty unique entries, matching report byte SHA-256,
        all selected finite metrics and positive sample counts, and qualifying
        local metadata. Metric equality uses rel_tol=1e-12 / abs_tol=1e-9;
        observed <= max_threshold and count >= minimum are inclusive. Thresholds
        must be positive, but metric labels are not converted or dimensionally
        checked. Optional self-digests exclude only their own field and hash
        recursively with sorted compact ASCII JSON; declared null is invalid.
        Report SHA strings must match raw lower-case hashes. Parsing/refusal
        and filesystem failures return fail, with no admitted ids or traceback.
        No artifacts, argv, benchmark state or report payloads are modified.
        Separate reads have no transactional snapshot or freshness guarantee.

    Examples
    --------
    Inspect the actual persisted repository evidence without timing any kernel:

    >>> result = validate_benchmark_regression_gates()
    >>> result.status, len(result.admitted_gates), result.errors
    ('pass', 4, ())
    >>> result.as_dict()['schema_version'] == SCHEMA_VERSION
    True
    """
    errors: list[str] = []
    admitted: list[str] = []
    manifest_sha256 = ""
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError:
        return BenchmarkRegressionGateResult("fail", (), ("manifest could not be read",), "")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest = _decode_json(manifest_bytes)
    except _JSONInspectionError as exc:
        return BenchmarkRegressionGateResult("fail", (), (str(exc),), manifest_sha256)
    resolved_manifest = manifest_path.resolve()
    report_root = ROOT if resolved_manifest.is_relative_to(ROOT) else resolved_manifest.parent

    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    gate_set_id = _as_text(manifest.get("gate_set_id"), "gate_set_id", errors)
    claim_boundary = _as_text(manifest.get("claim_boundary"), "claim_boundary", errors)
    if claim_boundary and ("bounded" not in claim_boundary.lower() or "unbounded" in claim_boundary.lower()):
        errors.append("claim_boundary must be bounded and must not be unbounded")
    _as_text(manifest.get("generated_utc"), "generated_utc", errors)

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("entries must be a non-empty list")
    else:
        seen_ids: set[str] = set()
        for index, raw_entry in enumerate(entries):
            if not isinstance(raw_entry, dict):
                errors.append(f"entries[{index}] must be an object")
                continue
            gate_id = _validate_entry(
                raw_entry,
                seen_ids,
                errors,
                report_root,
            )
            if gate_id is not None:
                admitted.append(gate_id)

    if gate_set_id and not admitted:
        errors.append(f"{gate_set_id}: no benchmark gates admitted")
    status = "pass" if not errors else "fail"
    return BenchmarkRegressionGateResult(
        status=status,
        admitted_gates=tuple(admitted) if status == "pass" else (),
        errors=tuple(errors),
        manifest_sha256=manifest_sha256,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse an optional local manifest path, using script-root defaults without mutating argv."""
    parser = argparse.ArgumentParser(description="Validate persisted benchmark regression gates.")
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="benchmark regression gate manifest",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the versioned result and return zero/pass or one/fail for local metadata inspection.

    Explicit argv is caller-owned; None reads process arguments without changing
    them. Relative paths belong to cwd; the omitted manifest is script-rooted.
    Argparse help/errors exit zero/two. Expected inspection failures produce
    authored JSON diagnostics on stdout, without decoder/OS exception text or
    traceback. This stdlib CLI never runs a benchmark or publishes a result.
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    result = validate_benchmark_regression_gates(args.manifest)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
