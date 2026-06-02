#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Validate persisted benchmark regression admission gates.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeAlias, cast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "validation" / "reports" / "benchmark_regression_gates.json"
SCHEMA_VERSION = "scpn-control.benchmark-regression-gates.v1"
ALLOWED_UNITS = frozenset({"us", "ms", "s"})

JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
JSONMapping: TypeAlias = dict[str, JSONValue]


@dataclass(frozen=True)
class BenchmarkRegressionGateResult:
    status: str
    admitted_gates: tuple[str, ...]
    errors: tuple[str, ...]
    manifest_sha256: str

    def as_dict(self) -> JSONMapping:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "admitted_gates": list(self.admitted_gates),
            "errors": list(self.errors),
            "manifest_sha256": self.manifest_sha256,
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


def _as_text(value: JSONValue, field: str, errors: list[str]) -> str:
    if isinstance(value, str) and value.strip():
        return value
    errors.append(f"{field} must be a non-empty string")
    return ""


def _as_number(value: JSONValue, field: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        errors.append(f"{field} must be a finite number")
        return math.nan
    out = float(value)
    if not math.isfinite(out):
        errors.append(f"{field} must be finite")
    return out


def _as_positive_int(value: JSONValue, field: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{field} must be a positive integer")
        return -1
    if value <= 0:
        errors.append(f"{field} must be positive")
        return -1
    return value


def _safe_repo_relative_uri_parts(uri: str) -> tuple[str, ...] | None:
    if not uri or "\\" in uri or "://" in uri or uri.startswith(("/", "~", "file:")):
        return None
    posix_path = PurePosixPath(uri)
    parts = posix_path.parts
    if not parts or posix_path.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return tuple(str(part) for part in parts)


def _is_safe_repo_relative_uri(uri: str) -> bool:
    return _safe_repo_relative_uri_parts(uri) is not None


def _report_path(uri: str, report_root: Path) -> Path:
    parts = _safe_repo_relative_uri_parts(uri)
    if parts is None:
        raise ValueError(f"unsafe report path escapes report root: {uri}")
    path = (report_root / Path(*parts)).resolve()
    if not path.is_relative_to(report_root):
        raise ValueError(f"unsafe report path escapes report root: {uri}")
    return path


def _value_at_path(payload: JSONMapping, dotted_path: str) -> JSONValue:
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
    if not (
        isinstance(platform, str)
        and platform.strip()
        or isinstance(rt_kernel, str)
        and rt_kernel.strip()
    ):
        errors.append(
            f"{benchmark_id}: hardware context must include platform or rt_kernel"
        )


def _validate_entry(
    entry: JSONMapping,
    seen_ids: set[str],
    root_errors: list[str],
    report_root: Path,
) -> str | None:
    local_errors: list[str] = []
    benchmark_id = _as_text(entry.get("benchmark_id"), "benchmark_id", local_errors)
    if benchmark_id in seen_ids:
        local_errors.append(f"{benchmark_id}: duplicate benchmark_id")
    seen_ids.add(benchmark_id)

    status = _as_text(entry.get("status"), f"{benchmark_id}.status", local_errors)
    if status != "pass":
        local_errors.append(f"{benchmark_id}: gate status must be pass")

    report_uri = _as_text(
        entry.get("report_uri"), f"{benchmark_id}.report_uri", local_errors
    )
    if not _is_safe_repo_relative_uri(report_uri):
        local_errors.append(f"{benchmark_id}: report_uri is not repository-relative")

    report_sha256 = _as_text(
        entry.get("report_sha256"), f"{benchmark_id}.report_sha256", local_errors
    )
    metric_path = _as_text(
        entry.get("metric_path"), f"{benchmark_id}.metric_path", local_errors
    )
    observed = _as_number(entry.get("observed"), f"{benchmark_id}.observed", local_errors)
    unit = _as_text(entry.get("unit"), f"{benchmark_id}.unit", local_errors)
    if unit and unit not in ALLOWED_UNITS:
        local_errors.append(f"{benchmark_id}: unsupported unit: {unit}")
    max_threshold = _as_number(
        entry.get("max_threshold"), f"{benchmark_id}.max_threshold", local_errors
    )
    if math.isfinite(max_threshold) and max_threshold <= 0.0:
        local_errors.append(f"{benchmark_id}: max_threshold must be positive")
    sample_count_path = _as_text(
        entry.get("sample_count_path"),
        f"{benchmark_id}.sample_count_path",
        local_errors,
    )
    sample_count = _as_positive_int(
        entry.get("sample_count"), f"{benchmark_id}.sample_count", local_errors
    )
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

    report_path = _report_path(report_uri, report_root)
    if not report_path.exists():
        root_errors.append(f"{benchmark_id}: report does not exist: {report_uri}")
        return None
    actual_report_sha256 = _sha256_file(report_path)
    if actual_report_sha256 != report_sha256:
        root_errors.append(f"{benchmark_id}: report_sha256 mismatch")

    try:
        report = _load_json(report_path)
    except ValueError as exc:
        root_errors.append(f"{benchmark_id}: {exc}")
        return None

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
    if math.isfinite(observed) and math.isfinite(max_threshold):
        if observed > max_threshold:
            root_errors.append(f"{benchmark_id}: observed exceeds max_threshold")

    try:
        report_sample_count = _as_positive_int(
            _value_at_path(report, sample_count_path),
            f"{benchmark_id}.report_sample_count",
            root_errors,
        )
    except KeyError:
        root_errors.append(
            f"{benchmark_id}: sample_count_path missing: {sample_count_path}"
        )
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
    errors: list[str] = []
    admitted: list[str] = []
    manifest_sha256 = _sha256_file(manifest_path) if manifest_path.exists() else ""
    try:
        manifest = _load_json(manifest_path)
    except ValueError as exc:
        return BenchmarkRegressionGateResult("fail", (), (str(exc),), manifest_sha256)
    resolved_manifest = manifest_path.resolve()
    report_root = ROOT if resolved_manifest.is_relative_to(ROOT) else manifest_path.parent

    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    gate_set_id = _as_text(manifest.get("gate_set_id"), "gate_set_id", errors)
    claim_boundary = _as_text(manifest.get("claim_boundary"), "claim_boundary", errors)
    if claim_boundary and (
        "bounded" not in claim_boundary.lower()
        or "unbounded" in claim_boundary.lower()
    ):
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
                cast(JSONMapping, raw_entry),
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
    parser = argparse.ArgumentParser(
        description="Validate persisted benchmark regression gates."
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="benchmark regression gate manifest",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    result = validate_benchmark_regression_gates(args.manifest)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
