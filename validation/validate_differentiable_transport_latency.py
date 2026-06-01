#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable Transport Latency Evidence Validation
"""Validate persisted differentiable transport gradient-latency reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

ONE_STEP_CLAIM_STATUS = "local audited gradient-admission latency only; not a real-time control-loop guarantee"
ROLLOUT_CLAIM_STATUS = "local audited rollout source-gradient latency only; not a real-time control-loop guarantee"
READINESS_BLOCKED_CLAIM_STATUS = "bounded differentiable transport readiness only; full-fidelity claim remains blocked"
READINESS_ADMITTED_CLAIM_STATUS = "full-fidelity differentiable transport claim admitted"
CHANNEL_ORDER = ["electron_temperature", "ion_temperature", "electron_density", "impurity_density"]
BLOCKED_CLAIM_STATUSES = {
    "no latency claim; JAX gradient backend unavailable in this environment",
}


def validate_differentiable_transport_latency(
    one_step_report: str | Path,
    rollout_report: str | Path | None = None,
    *,
    readiness_report: str | Path | None = None,
    require_admitted: bool = False,
) -> dict[str, Any]:
    """Validate one-step and optional rollout differentiable transport reports."""

    one_step_path = Path(one_step_report)
    rollout_path = Path(rollout_report) if rollout_report is not None else None
    readiness_path = Path(readiness_report) if readiness_report is not None else None
    errors: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []

    entries.append(
        _validate_report(
            one_step_path,
            kind="one_step",
            claim_status=ONE_STEP_CLAIM_STATUS,
            require_admitted=require_admitted,
            errors=errors,
        )
    )
    if rollout_path is not None:
        entries.append(
            _validate_report(
                rollout_path,
                kind="rollout",
                claim_status=ROLLOUT_CLAIM_STATUS,
                require_admitted=require_admitted,
                errors=errors,
            )
        )

    admitted = [entry for entry in entries if entry.get("status") == "pass"]
    blocked = [entry for entry in entries if entry.get("status") == "blocked"]
    readiness_entry: dict[str, object] | None = None
    if readiness_path is not None:
        readiness_entry = _validate_readiness_report(readiness_path, errors)
    if require_admitted and blocked:
        for entry in blocked:
            errors.append(
                {
                    "path": entry["path"],
                    "field": "status",
                    "error": "admitted differentiable transport latency evidence is required",
                }
            )

    return {
        "status": "pass" if not errors else "fail",
        "one_step_report": str(one_step_path),
        "rollout_report": str(rollout_path) if rollout_path is not None else None,
        "readiness_report": str(readiness_path) if readiness_path is not None else None,
        "require_admitted": require_admitted,
        "admitted_reports": len(admitted),
        "blocked_reports": len(blocked),
        "readiness_entry": readiness_entry,
        "full_fidelity_ready": bool(readiness_entry and readiness_entry.get("full_fidelity_ready") is True),
        "entries": entries,
        "errors": errors,
    }


def _validate_readiness_report(path: Path, errors: list[dict[str, object]]) -> dict[str, object]:
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append({"path": str(path), "field": "json", "error": str(exc)})
        return {"path": str(path), "kind": "full_fidelity_readiness", "status": "fail"}

    if payload.get("status") == "blocked":
        entry = _validate_blocked_report(path, payload, "full_fidelity_readiness", errors)
        entry["full_fidelity_ready"] = False
        entry["blocked_reasons"] = ["jax_backend"]
        return entry

    _require_value(path, payload, "schema_version", 1, errors)
    _require_value(path, payload, "backend", "jax", errors)
    if not _positive_int(payload.get("n_rho")) or int(payload.get("n_rho", 0)) < 3:
        errors.append({"path": str(path), "field": "readiness.n_rho", "error": "n_rho must be an integer >= 3"})
    if not _positive_int(payload.get("rollout_steps")):
        errors.append(
            {"path": str(path), "field": "readiness.rollout_steps", "error": "rollout_steps must be positive"}
        )
    for field in (
        "campaign_sha256",
        "gradient_latency_report_sha256",
        "gradient_audit_sha256",
        "rollout_latency_report_sha256",
        "rollout_audit_sha256",
    ):
        if not _is_sha256_hex(payload.get(field)):
            errors.append({"path": str(path), "field": field, "error": "field must be a SHA-256 hex digest"})
    for field in ("external_reference_artifact_sha256", "controller_formal_artifact_sha256"):
        value = payload.get(field)
        if value is not None and not _is_sha256_hex(value):
            errors.append({"path": str(path), "field": field, "error": "field must be null or a SHA-256 hex digest"})

    channel_order = payload.get("channel_order")
    if channel_order != CHANNEL_ORDER:
        errors.append({"path": str(path), "field": "readiness.channel_order", "error": "unexpected channel order"})
    if payload.get("equilibrium_coupled") is not True:
        errors.append(
            {"path": str(path), "field": "readiness.equilibrium_coupled", "error": "equilibrium coupling required"}
        )
    if not isinstance(payload.get("external_reference_admitted"), bool):
        errors.append(
            {"path": str(path), "field": "readiness.external_reference_admitted", "error": "field must be boolean"}
        )
    full_fidelity_ready = payload.get("full_fidelity_claim_admissible")
    blocked_reasons = payload.get("blocked_reasons")
    if not isinstance(full_fidelity_ready, bool):
        errors.append(
            {"path": str(path), "field": "readiness.full_fidelity_claim_admissible", "error": "field must be boolean"}
        )
        full_fidelity_ready = False
    if not isinstance(blocked_reasons, list) or not all(isinstance(reason, str) for reason in blocked_reasons):
        errors.append(
            {"path": str(path), "field": "readiness.blocked_reasons", "error": "blocked_reasons must be string list"}
        )
        blocked_reasons = []
    if full_fidelity_ready and blocked_reasons:
        errors.append(
            {
                "path": str(path),
                "field": "readiness.blocked_reasons",
                "error": "full-fidelity readiness cannot have blocked reasons",
            }
        )
    if not full_fidelity_ready and not blocked_reasons:
        errors.append(
            {
                "path": str(path),
                "field": "readiness.blocked_reasons",
                "error": "blocked readiness must explain blocked reasons",
            }
        )
    expected_claim_status = READINESS_ADMITTED_CLAIM_STATUS if full_fidelity_ready else READINESS_BLOCKED_CLAIM_STATUS
    _require_value(path, payload, "claim_status", expected_claim_status, errors)
    return {
        "path": str(path),
        "kind": "full_fidelity_readiness",
        "status": "pass" if full_fidelity_ready else "blocked",
        "full_fidelity_ready": bool(full_fidelity_ready),
        "blocked_reasons": blocked_reasons,
    }


def _validate_report(
    path: Path,
    *,
    kind: str,
    claim_status: str,
    require_admitted: bool,
    errors: list[dict[str, object]],
) -> dict[str, object]:
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

    status = "pass"
    if require_admitted and payload.get("backend") != "jax":
        status = "fail"
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
        "status": "blocked",
        "backend": None,
        "dtype": None,
        "p95_ms": None,
        "audit_passed": None,
    }


def _validate_latency_order(path: Path, payload: dict[str, Any], errors: list[dict[str, object]]) -> None:
    p50 = _finite_non_negative(payload.get("p50_ms"))
    p95 = _finite_non_negative(payload.get("p95_ms"))
    max_ms = _finite_non_negative(payload.get("max_ms"))
    if p50 is None:
        errors.append({"path": str(path), "field": "p50_ms", "error": "p50_ms must be finite and non-negative"})
    if p95 is None:
        errors.append({"path": str(path), "field": "p95_ms", "error": "p95_ms must be finite and non-negative"})
    if max_ms is None:
        errors.append({"path": str(path), "field": "max_ms", "error": "max_ms must be finite and non-negative"})
    if p50 is not None and p95 is not None and max_ms is not None and not (p50 <= p95 <= max_ms):
        errors.append(
            {
                "path": str(path),
                "field": "latency",
                "error": "latency percentiles must satisfy p50_ms <= p95_ms <= max_ms",
            }
        )


def _validate_runtime_metadata(path: Path, metadata: object, errors: list[dict[str, object]]) -> None:
    if not isinstance(metadata, dict):
        errors.append({"path": str(path), "field": "runtime_metadata", "error": "runtime metadata must be an object"})
        return
    _require_value(path, metadata, "schema_version", 1, errors)
    measured = _finite_positive(metadata.get("measured_at_unix_s"))
    if measured is None:
        errors.append(
            {
                "path": str(path),
                "field": "runtime_metadata.measured_at_unix_s",
                "error": "measurement timestamp must be positive and finite",
            }
        )
    for field in (
        "python_version",
        "platform",
        "machine",
        "jax_version",
        "jaxlib_version",
        "jax_default_backend",
    ):
        value = metadata.get(field)
        if not isinstance(value, str) or not value:
            errors.append(
                {
                    "path": str(path),
                    "field": f"runtime_metadata.{field}",
                    "error": "field must be a non-empty string",
                }
            )
    if not isinstance(metadata.get("processor"), str):
        errors.append(
            {
                "path": str(path),
                "field": "runtime_metadata.processor",
                "error": "field must be a string",
            }
        )
    devices = metadata.get("jax_devices")
    if (
        not isinstance(devices, list)
        or not devices
        or not all(isinstance(device, str) and device for device in devices)
    ):
        errors.append(
            {
                "path": str(path),
                "field": "runtime_metadata.jax_devices",
                "error": "field must be a non-empty string list",
            }
        )
    if not isinstance(metadata.get("jax_enable_x64"), bool):
        errors.append(
            {
                "path": str(path),
                "field": "runtime_metadata.jax_enable_x64",
                "error": "field must be boolean",
            }
        )


def _validate_audit(path: Path, payload: dict[str, Any], kind: str, errors: list[dict[str, object]]) -> None:
    audit = payload.get("audit")
    if not isinstance(audit, dict):
        errors.append({"path": str(path), "field": "audit", "error": "audit must be an object"})
        return
    if audit.get("passed") is not True:
        errors.append(
            {"path": str(path), "field": "audit.passed", "error": "audit must pass for admitted latency evidence"}
        )
    tolerance = _finite_positive(audit.get("tolerance"))
    epsilon = _finite_positive(audit.get("epsilon"))
    loss = _finite_non_negative(audit.get("loss"))
    if tolerance is None:
        errors.append(
            {"path": str(path), "field": "audit.tolerance", "error": "audit tolerance must be positive and finite"}
        )
    if epsilon is None:
        errors.append(
            {"path": str(path), "field": "audit.epsilon", "error": "audit epsilon must be positive and finite"}
        )
    if loss is None:
        errors.append({"path": str(path), "field": "audit.loss", "error": "audit loss must be finite and non-negative"})
    source_error = _finite_non_negative(audit.get("source_max_abs_error"))
    if source_error is None:
        errors.append(
            {
                "path": str(path),
                "field": "audit.source_max_abs_error",
                "error": "source audit error must be finite and non-negative",
            }
        )
    elif tolerance is not None and source_error > tolerance:
        errors.append(
            {"path": str(path), "field": "audit.source_max_abs_error", "error": "source audit error exceeds tolerance"}
        )
    if kind == "one_step":
        chi_error = _finite_non_negative(audit.get("chi_max_abs_error"))
        if chi_error is None:
            errors.append(
                {
                    "path": str(path),
                    "field": "audit.chi_max_abs_error",
                    "error": "chi audit error must be finite and non-negative",
                }
            )
        elif tolerance is not None and chi_error > tolerance:
            errors.append(
                {"path": str(path), "field": "audit.chi_max_abs_error", "error": "chi audit error exceeds tolerance"}
            )
    _validate_indices(path, audit.get("checked_indices"), kind, payload, errors)


def _validate_indices(
    path: Path,
    indices: object,
    kind: str,
    payload: dict[str, Any],
    errors: list[dict[str, object]],
) -> None:
    if not isinstance(indices, list) or not indices:
        errors.append(
            {"path": str(path), "field": "audit.checked_indices", "error": "checked_indices must be a non-empty list"}
        )
        return
    width = 3 if kind == "rollout" else 2
    n_rho = int(payload.get("n_rho", 0)) if _positive_int(payload.get("n_rho")) else 0
    n_steps = int(payload.get("n_steps", 0)) if _positive_int(payload.get("n_steps")) else 0
    seen: set[tuple[int, ...]] = set()
    for raw in indices:
        if not isinstance(raw, list) or len(raw) != width or not all(_non_negative_int(value) for value in raw):
            errors.append(
                {
                    "path": str(path),
                    "field": "audit.checked_indices",
                    "error": f"indices must be {width}-element non-negative integer lists",
                }
            )
            return
        item = tuple(int(value) for value in raw)
        if item in seen:
            errors.append(
                {"path": str(path), "field": "audit.checked_indices", "error": "checked_indices must be unique"}
            )
            return
        seen.add(item)
        if kind == "one_step":
            channel, rho_idx = item
            if channel >= 4 or rho_idx >= n_rho:
                errors.append(
                    {"path": str(path), "field": "audit.checked_indices", "error": "one-step audit index out of domain"}
                )
                return
        else:
            step, channel, rho_idx = item
            if step >= n_steps or channel >= 4 or rho_idx >= n_rho:
                errors.append(
                    {"path": str(path), "field": "audit.checked_indices", "error": "rollout audit index out of domain"}
                )
                return


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(payload, dict):
        raise ValueError("report root must be a JSON object")
    return payload


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _require_value(
    path: Path, payload: dict[str, Any], field: str, expected: object, errors: list[dict[str, object]]
) -> None:
    if payload.get(field) != expected:
        errors.append({"path": str(path), "field": field, "error": f"field must be {expected!r}"})


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_positive(value: object) -> float | None:
    numeric = _finite_number(value)
    if numeric is None or numeric <= 0.0:
        return None
    return numeric


def _finite_non_negative(value: object) -> float | None:
    numeric = _finite_number(value)
    if numeric is None or numeric < 0.0:
        return None
    return numeric


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate differentiable transport gradient-latency evidence")
    parser.add_argument(
        "--one-step-report",
        type=Path,
        default=ROOT / "validation" / "reports" / "differentiable_transport_latency.json",
    )
    parser.add_argument(
        "--rollout-report",
        type=Path,
        default=ROOT / "validation" / "reports" / "differentiable_transport_rollout_latency.json",
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        default=ROOT / "validation" / "reports" / "differentiable_transport_full_fidelity_readiness.json",
    )
    parser.add_argument("--require-admitted", action="store_true")
    parser.add_argument("--json-out", action="store_true")
    args = parser.parse_args()

    report = validate_differentiable_transport_latency(
        args.one_step_report,
        args.rollout_report,
        readiness_report=args.readiness_report,
        require_admitted=args.require_admitted,
    )
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "Differentiable transport latency evidence: "
            f"{report['status']} admitted={report['admitted_reports']} blocked={report['blocked_reports']}"
        )
        for error in report["errors"]:
            print(f"ERROR {error['path']}:{error['field']}: {error['error']}")
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
