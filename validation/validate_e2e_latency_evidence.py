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
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def validate_e2e_latency_evidence(
    path: str | Path,
    *,
    require_target_hardware: bool = True,
    max_e2e_p95_us: float | None = None,
) -> LatencyEvidenceReport:
    """Validate a persisted E2E latency report for publication admission."""

    payload = _load_json(path)
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")

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

    e2e = payload.get("e2e_us")
    if not isinstance(e2e, dict):
        e2e = {}
        errors.append("e2e_us must be an object")
    p95_us = _finite_positive_number(e2e.get("p95"))
    if p95_us is None:
        errors.append("e2e_us.p95 must be a positive finite number")
    elif max_e2e_p95_us is not None and p95_us > max_e2e_p95_us:
        errors.append(f"e2e_us.p95 exceeds admission threshold {max_e2e_p95_us}")

    claim_status = payload.get("claim_status")
    if not isinstance(claim_status, str) or "local latency evidence only" not in claim_status:
        errors.append("claim_status must preserve the local-evidence boundary")

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
