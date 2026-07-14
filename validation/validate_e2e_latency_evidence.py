# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — E2E Latency Evidence Validation
"""Validate end-to-end control latency reports before real-time claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

E2E_LATENCY_SCHEMA_VERSION = "scpn-control.e2e-latency.v1"
E2E_LATENCY_CLAIM_BOUNDARY = (
    "local latency evidence only; not a hardware-in-the-loop real-time guarantee "
    "unless target_hardware.id, class, and rt_kernel are operator-qualified"
)
_UNQUALIFIED_VALUES = {"", "unknown", "unspecified", "unspecified-local", "local-host-unqualified"}


@dataclass(frozen=True)
class LatencyEvidenceReport:
    """Strict validation result for a latency evidence artifact."""

    status: str
    errors: tuple[str, ...]
    p95_us: float | None
    target_hardware_id: str | None
    target_hardware_class: str | None
    rt_kernel: str | None


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("latency evidence root must be a JSON object")
    return payload


def _qualified_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped.lower() in _UNQUALIFIED_VALUES:
        return None
    return stripped


def _finite_positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _payload_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("payload_sha256", None)
    blob = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_e2e_latency_evidence_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical schema-versioned latency evidence payload."""
    canonical = dict(payload)
    canonical["schema_version"] = E2E_LATENCY_SCHEMA_VERSION
    canonical["claim_status"] = E2E_LATENCY_CLAIM_BOUNDARY
    canonical["claim_boundary"] = E2E_LATENCY_CLAIM_BOUNDARY
    canonical.setdefault("evidence_class", "local_regression")
    canonical.setdefault("production_claim_allowed", False)
    canonical["payload_sha256"] = _payload_digest(canonical)
    return canonical


def _validate_percentiles(
    payload: dict[str, Any],
    section_name: str,
    errors: list[str],
) -> dict[str, float | None]:
    section = payload.get(section_name)
    if not isinstance(section, dict):
        errors.append(f"{section_name} must be an object")
        section = {}
    values = {key: _finite_positive_number(section.get(key)) for key in ("p50", "p95", "p99")}
    for key, value in values.items():
        if value is None:
            errors.append(f"{section_name}.{key} must be a positive finite number")
    p50, p95, p99 = values["p50"], values["p95"], values["p99"]
    if p50 is not None and p95 is not None and p99 is not None and not (p50 <= p95 <= p99):
        errors.append(f"{section_name} percentiles must satisfy p50 <= p95 <= p99")
    return values


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _validate_loadavg(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != 3:
        errors.append(f"context.{key} must contain three finite load-average values")
        return
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool) or not math.isfinite(float(item)):
            errors.append(f"context.{key} must contain three finite load-average values")
            return


def _validate_benchmark_context(payload: dict[str, Any], errors: list[str]) -> None:
    command = payload.get("command")
    if not isinstance(command, str) or "benchmarks/e2e_control_latency.py" not in command:
        errors.append("command must record the E2E benchmark invocation")

    if not _valid_utc_timestamp(payload.get("generated_utc")):
        errors.append("generated_utc must record an ISO-8601 UTC timestamp")

    if payload.get("evidence_class") != "local_regression":
        errors.append("evidence_class must be local_regression")
    if payload.get("production_claim_allowed") is not False:
        errors.append("production_claim_allowed must be false for local E2E latency reports")
    if payload.get("claim_boundary") != E2E_LATENCY_CLAIM_BOUNDARY:
        errors.append("claim_boundary must preserve the canonical local-evidence boundary")

    context = payload.get("context")
    if not isinstance(context, dict):
        errors.append("context must record benchmark host-load and isolation metadata")
        context = {}
    affinity = context.get("cpu_affinity")
    if (
        not isinstance(affinity, list)
        or not affinity
        or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in affinity)
    ):
        errors.append("context.cpu_affinity must record at least one CPU")
    isolation_method = context.get("isolation_method")
    if not isinstance(isolation_method, str) or not isolation_method.strip():
        errors.append("context.isolation_method must record the benchmark isolation method")
    _validate_loadavg(context, "loadavg_start", errors)
    _validate_loadavg(context, "loadavg_end", errors)
    if "governor" not in context:
        errors.append("context.governor must record CPU governor state or null when unavailable")
    heavy_jobs = context.get("heavy_jobs_running")
    if not isinstance(heavy_jobs, str) or not heavy_jobs.strip():
        errors.append("context.heavy_jobs_running must record whether concurrent heavy jobs were observed")


def validate_e2e_latency_evidence(
    path: str | Path,
    *,
    require_target_hardware: bool = True,
    max_e2e_p95_us: float | None = None,
) -> LatencyEvidenceReport:
    """Validate a persisted E2E latency report for publication admission."""

    payload = _load_json(path)
    errors: list[str] = []
    if payload.get("schema_version") != E2E_LATENCY_SCHEMA_VERSION:
        errors.append(f"schema_version must be {E2E_LATENCY_SCHEMA_VERSION!r}")
    declared_digest = payload.get("payload_sha256")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        errors.append("payload_sha256 must be a SHA-256 hex digest")
    elif _payload_digest(payload) != declared_digest.lower():
        errors.append("payload_sha256 does not match latency evidence payload")

    iterations = payload.get("iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        errors.append("iterations must be a positive integer")
    warmup = payload.get("warmup")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        errors.append("warmup must be a non-negative integer")

    target = payload.get("target_hardware")
    if not isinstance(target, dict):
        target = {}
        errors.append("target_hardware must be an object")

    hardware_id = _qualified_string(target.get("id"))
    hardware_class = _qualified_string(target.get("class"))
    rt_kernel = _qualified_string(target.get("rt_kernel"))
    if require_target_hardware:
        if hardware_id is None:
            errors.append("target_hardware.id must identify the measured hardware")
        if hardware_class is None:
            errors.append("target_hardware.class must identify the hardware class")
        if rt_kernel is None:
            errors.append("target_hardware.rt_kernel must identify scheduler or RT-kernel evidence")

    kernel_values = _validate_percentiles(payload, "kernel_only_us", errors)
    e2e_values = _validate_percentiles(payload, "e2e_us", errors)
    p95_us = e2e_values["p95"]
    if p95_us is None:
        pass
    elif max_e2e_p95_us is not None and p95_us > max_e2e_p95_us:
        errors.append(f"e2e_us.p95 exceeds admission threshold {max_e2e_p95_us}")

    overhead = _finite_positive_number(payload.get("e2e_overhead_factor"))
    if overhead is None:
        errors.append("e2e_overhead_factor must be a positive finite number")
    elif kernel_values["p50"] is not None and e2e_values["p50"] is not None:
        expected = e2e_values["p50"] / max(kernel_values["p50"], 0.1)
        if not math.isclose(overhead, round(expected, 1), rel_tol=0.0, abs_tol=0.1):
            errors.append("e2e_overhead_factor must match p50 e2e/kernel ratio")

    claim_status = payload.get("claim_status")
    if claim_status != E2E_LATENCY_CLAIM_BOUNDARY:
        errors.append("claim_status must preserve the canonical local-evidence boundary")

    _validate_benchmark_context(payload, errors)

    return LatencyEvidenceReport(
        status="pass" if not errors else "fail",
        errors=tuple(errors),
        p95_us=p95_us,
        target_hardware_id=hardware_id,
        target_hardware_class=hardware_class,
        rt_kernel=rt_kernel,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate E2E control-latency evidence")
    parser.add_argument("report", type=Path)
    parser.add_argument("--allow-local-unqualified", action="store_true")
    parser.add_argument("--max-e2e-p95-us", type=float, default=None)
    parser.add_argument("--json-out", action="store_true")
    args = parser.parse_args()

    result = validate_e2e_latency_evidence(
        args.report,
        require_target_hardware=not args.allow_local_unqualified,
        max_e2e_p95_us=args.max_e2e_p95_us,
    )
    payload = {
        "status": result.status,
        "errors": list(result.errors),
        "p95_us": result.p95_us,
        "target_hardware_id": result.target_hardware_id,
        "target_hardware_class": result.target_hardware_class,
        "rt_kernel": result.rt_kernel,
    }
    if args.json_out:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"E2E latency evidence: {result.status}")
        for error in result.errors:
            print(f"ERROR {error}")
    raise SystemExit(0 if result.status == "pass" else 1)


if __name__ == "__main__":
    main()
