# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CODAC runtime evidence contracts.
"""Validate CODAC runtime-evidence payloads independently of the live adapter."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

CODAC_RUNTIME_EVIDENCE_SCHEMA_VERSION = "scpn-control.codac-runtime-evidence.v3"
CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY = "bounded_codac_runtime_evidence_only"
CODAC_RUNTIME_EVIDENCE_QUALIFIED = "qualified_codac_runtime_evidence"


@dataclass(frozen=True)
class CODACRuntimeEvidence:
    """Tamper-evident CODAC/EPICS runtime admission evidence."""

    schema_version: str
    generated_utc: str
    controller_id: str
    plant_system: str
    pv_prefix: str
    cycle_hz: float
    cycle_budget_us: float
    timeout_budget_us: float
    deadline_us: float
    observed_cycle_p50_us: float
    observed_cycle_p95_us: float
    observed_cycle_p99_us: float
    observed_cycle_max_us: float
    input_channel_count: int
    output_channel_count: int
    interlock_pv_count: int
    interlock_checks: int
    interlock_blocks: int
    backpressure_events: int
    output_limits_enforced: bool
    epics_drive_limits_exported: bool
    interlock_fail_closed: bool
    epics_db_sha256: str
    opcua_nodeset_sha256: str
    facility_claim_allowed: bool
    claim_status: str
    payload_sha256: str


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False)
    except ValueError as exc:
        raise ValueError("CODAC evidence contains a non-finite JSON number") from exc


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key in CODAC runtime evidence: {key}")
        seen.add(key)
        out[key] = value
    return out


def _require_finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _validate_evidence_payload(
    payload: Mapping[str, Any],
    *,
    require_facility_claim: bool,
    expected_input_count: int,
    expected_output_count: int,
) -> CODACRuntimeEvidence:
    if payload.get("schema_version") != CODAC_RUNTIME_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("CODAC runtime evidence schema_version is unsupported")
    declared_digest = payload.get("payload_sha256")
    if not _is_sha256(declared_digest):
        raise ValueError("CODAC runtime evidence payload_sha256 must be a SHA-256 hex digest")
    if declared_digest != _payload_sha256(payload):
        raise ValueError("CODAC runtime evidence payload_sha256 does not match payload")

    generated_utc = payload.get("generated_utc")
    if not isinstance(generated_utc, str) or not generated_utc.endswith("Z"):
        raise ValueError("CODAC runtime evidence generated_utc must be a UTC timestamp ending in Z")
    controller_id = payload.get("controller_id")
    plant_system = payload.get("plant_system")
    pv_prefix = payload.get("pv_prefix")
    if not isinstance(controller_id, str) or not controller_id.strip():
        raise ValueError("CODAC runtime evidence controller_id must be non-empty")
    if not isinstance(plant_system, str) or not plant_system.strip():
        raise ValueError("CODAC runtime evidence plant_system must be non-empty")
    if not isinstance(pv_prefix, str) or not pv_prefix.strip():
        raise ValueError("CODAC runtime evidence pv_prefix must be non-empty")

    cycle_hz = _require_finite_nonnegative("cycle_hz", payload.get("cycle_hz"))
    if cycle_hz <= 0.0:
        raise ValueError("cycle_hz must be positive")
    cycle_budget_us = _require_finite_nonnegative("cycle_budget_us", payload.get("cycle_budget_us"))
    timeout_budget_us = _require_finite_nonnegative("timeout_budget_us", payload.get("timeout_budget_us"))
    deadline_us = _require_finite_nonnegative("deadline_us", payload.get("deadline_us"))
    if not math.isclose(deadline_us, min(cycle_budget_us, timeout_budget_us), rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("deadline_us must equal min(cycle_budget_us, timeout_budget_us)")

    observed_p50 = _require_finite_nonnegative("observed_cycle_p50_us", payload.get("observed_cycle_p50_us"))
    observed_p95 = _require_finite_nonnegative("observed_cycle_p95_us", payload.get("observed_cycle_p95_us"))
    observed_p99 = _require_finite_nonnegative("observed_cycle_p99_us", payload.get("observed_cycle_p99_us"))
    observed_max = _require_finite_nonnegative("observed_cycle_max_us", payload.get("observed_cycle_max_us"))
    if not (observed_p50 <= observed_p95 <= observed_p99 <= observed_max):
        raise ValueError("CODAC runtime cycle percentiles must be ordered")

    input_count = _require_nonnegative_int("input_channel_count", payload.get("input_channel_count"))
    output_count = _require_nonnegative_int("output_channel_count", payload.get("output_channel_count"))
    interlock_pv_count = _require_nonnegative_int("interlock_pv_count", payload.get("interlock_pv_count"))
    if input_count != expected_input_count or output_count != expected_output_count:
        raise ValueError("CODAC runtime evidence channel counts do not match current interface tables")
    if interlock_pv_count <= 0:
        raise ValueError("CODAC runtime evidence must bind at least one interlock PV")

    interlock_checks = _require_nonnegative_int("interlock_checks", payload.get("interlock_checks"))
    interlock_blocks = _require_nonnegative_int("interlock_blocks", payload.get("interlock_blocks"))
    backpressure_events = _require_nonnegative_int("backpressure_events", payload.get("backpressure_events"))
    if interlock_blocks > interlock_checks:
        raise ValueError("interlock_blocks cannot exceed interlock_checks")

    boundary_guards: dict[str, bool] = {}
    for guard_name in ("output_limits_enforced", "epics_drive_limits_exported", "interlock_fail_closed"):
        guard_value = payload.get(guard_name)
        if not isinstance(guard_value, bool):
            raise ValueError(f"CODAC runtime evidence {guard_name} must be boolean")
        boundary_guards[guard_name] = guard_value

    if not _is_sha256(payload.get("epics_db_sha256")):
        raise ValueError("epics_db_sha256 must be a SHA-256 hex digest")
    if not _is_sha256(payload.get("opcua_nodeset_sha256")):
        raise ValueError("opcua_nodeset_sha256 must be a SHA-256 hex digest")

    facility_claim_allowed = payload.get("facility_claim_allowed")
    claim_status = payload.get("claim_status")
    if not isinstance(facility_claim_allowed, bool):
        raise ValueError("facility_claim_allowed must be boolean")
    expected_status = CODAC_RUNTIME_EVIDENCE_QUALIFIED if facility_claim_allowed else CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY
    if claim_status != expected_status:
        raise ValueError("CODAC runtime evidence claim_status does not match facility_claim_allowed")

    if require_facility_claim:
        if not facility_claim_allowed:
            raise ValueError("CODAC runtime evidence is local-only and cannot support a facility claim")
        if observed_p99 > deadline_us or observed_max > deadline_us:
            raise ValueError("CODAC runtime evidence exceeds the configured cycle deadline")
        if interlock_checks <= 0 or interlock_blocks <= 0:
            raise ValueError("CODAC runtime evidence must exercise and block at least one interlock path")
        if backpressure_events != 0:
            raise ValueError("CODAC runtime evidence with backpressure events cannot support a facility claim")
        if not all(boundary_guards.values()):
            raise ValueError("CODAC runtime evidence requires all fail-closed boundary guards for a facility claim")

    if facility_claim_allowed:
        raise ValueError("CODAC facility claim requires independent runtime and hardware evidence")

    return CODACRuntimeEvidence(
        schema_version=str(payload["schema_version"]),
        generated_utc=generated_utc,
        controller_id=controller_id,
        plant_system=plant_system,
        pv_prefix=pv_prefix,
        cycle_hz=cycle_hz,
        cycle_budget_us=cycle_budget_us,
        timeout_budget_us=timeout_budget_us,
        deadline_us=deadline_us,
        observed_cycle_p50_us=observed_p50,
        observed_cycle_p95_us=observed_p95,
        observed_cycle_p99_us=observed_p99,
        observed_cycle_max_us=observed_max,
        input_channel_count=input_count,
        output_channel_count=output_count,
        interlock_pv_count=interlock_pv_count,
        interlock_checks=interlock_checks,
        interlock_blocks=interlock_blocks,
        backpressure_events=backpressure_events,
        output_limits_enforced=boundary_guards["output_limits_enforced"],
        epics_drive_limits_exported=boundary_guards["epics_drive_limits_exported"],
        interlock_fail_closed=boundary_guards["interlock_fail_closed"],
        epics_db_sha256=str(payload["epics_db_sha256"]),
        opcua_nodeset_sha256=str(payload["opcua_nodeset_sha256"]),
        facility_claim_allowed=facility_claim_allowed,
        claim_status=str(claim_status),
        payload_sha256=str(declared_digest),
    )
