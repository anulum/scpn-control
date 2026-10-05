# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable Transport Latency Evidence Validation
"""Declared local differentiable latency reports contracts; no audit replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from validation.differentiable_latency_audit import _validate_audit
from validation.differentiable_latency_context import _validate_latency_order, _validate_runtime_metadata
from validation.differentiable_latency_fields import (
    BLOCKED_CLAIM_STATUSES,
    _load_json,
    _non_negative_int,
    _positive_int,
    _require_value,
)


def _validate_report(
    path: Path,
    *,
    kind: str,
    claim_status: str,
    require_admitted: bool,
    errors: list[dict[str, object]],
) -> dict[str, object]:
    """Validate a one-step/rollout declaration and derive its status from its errors.

    Parameters
    ----------
    path : pathlib.Path
        Caller-relative report path, read once through the JSON loader.
    kind : str
        one_step or rollout; controls index width and required n_steps.
    claim_status : str
        Fixed local audited-latency boundary for this kind.
    require_admitted : bool
        Retained helper-call compatibility argument. Aggregate policy refuses
        valid blocked reports; it does not grant full-fidelity readiness.
    errors : list[dict[str, object]]
        Shared ordered diagnostics appended in place.

    Returns
    -------
    dict[str, object]
        path/kind/status and declared backend/dtype/p95/audit flag. Status is
        fail if this report adds errors, pass for valid local fields, or blocked
        for a valid authored backend-unavailable declaration. Invalid reports
        cannot contribute to admitted_reports.

    Notes
    -----
    I/O/JSON/ValueError becomes an authored json diagnostic. No JAX execution,
    audit replay, timing measurement, source signature or hardware admission.
    """
    errors_before = len(errors)
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append({"path": str(path), "field": "json", "error": str(exc)})
        return {"path": str(path), "kind": kind, "status": "fail"}

    if payload.get("status") == "blocked":
        return _validate_blocked_report(path, payload, kind, errors)

    _require_value(path, payload, "schema_version", 1, errors)
    _require_value(path, payload, "backend", "jax", errors)
    _require_value(path, payload, "dtype", "float64", errors)
    _require_value(path, payload, "channel_count", 4, errors)
    _require_value(path, payload, "claim_status", claim_status, errors)
    if not _positive_int(payload.get("n_rho")) or int(payload.get("n_rho", 0)) < 3:
        errors.append({"path": str(path), "field": "n_rho", "error": "n_rho must be an integer >= 3"})
    if kind == "rollout" and not _positive_int(payload.get("n_steps")):
        errors.append({"path": str(path), "field": "n_steps", "error": "n_steps must be a positive integer"})
    if not _non_negative_int(payload.get("warmup_runs")):
        errors.append(
            {"path": str(path), "field": "warmup_runs", "error": "warmup_runs must be a non-negative integer"}
        )
    if not _positive_int(payload.get("timed_runs")):
        errors.append({"path": str(path), "field": "timed_runs", "error": "timed_runs must be a positive integer"})
    _validate_latency_order(path, payload, errors)
    _validate_runtime_metadata(path, payload.get("runtime_metadata"), errors)
    _validate_audit(path, payload, kind, errors)

    status = "fail" if len(errors) > errors_before else "pass"
    return {
        "path": str(path),
        "kind": kind,
        "status": status,
        "backend": payload.get("backend"),
        "dtype": payload.get("dtype"),
        "p95_ms": payload.get("p95_ms"),
        "audit_passed": payload.get("audit", {}).get("passed") if isinstance(payload.get("audit"), dict) else None,
    }


def _validate_blocked_report(
    path: Path, payload: dict[str, Any], kind: str, errors: list[dict[str, object]]
) -> dict[str, object]:
    """Validate an explicit no-latency-claim declaration without simulating absence.

    Parameters
    ----------
    path : pathlib.Path
        Path used in diagnostics.
    payload : dict[str, Any]
        Authored blocked declaration.
    kind : str
        Aggregate entry kind.
    errors : list[dict[str, object]]
        Ordered diagnostics appended in place.

    Returns
    -------
    dict[str, object]
        blocked only when schema, missing-JAX reason and no-claim boundary pass;
        otherwise fail. Backend/dtype/p95/audit fields are None.

    Notes
    -----
    Reading this declaration does not prove installed JAX is unavailable.
    """
    errors_before = len(errors)
    _require_value(path, payload, "schema_version", 1, errors)
    reason = payload.get("reason")
    if not isinstance(reason, str) or "JAX is required" not in reason:
        errors.append(
            {"path": str(path), "field": "reason", "error": "blocked report must explain missing JAX gradient backend"}
        )
    if payload.get("claim_status") not in BLOCKED_CLAIM_STATUSES:
        errors.append(
            {"path": str(path), "field": "claim_status", "error": "blocked report must make no latency claim"}
        )
    return {
        "path": str(path),
        "kind": kind,
        "status": "fail" if len(errors) > errors_before else "blocked",
        "backend": None,
        "dtype": None,
        "p95_ms": None,
        "audit_passed": None,
    }
