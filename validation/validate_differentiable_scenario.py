#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable scenario evidence validation
"""Validate persisted coupled differentiable scenario readiness evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "validation" / "reports" / "differentiable_scenario_readiness.json"
SCHEMA_VERSION = "scpn-control.differentiable-scenario-readiness.v1"


def validate_differentiable_scenario_report(path: str | Path = DEFAULT_REPORT) -> dict[str, Any]:
    """Validate a persisted differentiable-scenario readiness report."""

    report_path = Path(path)
    errors: list[dict[str, object]] = []
    try:
        payload = _load_json(report_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "fail", "report": str(report_path), "errors": [{"field": "json", "error": str(exc)}]}

    _require_value(payload, "schema_version", SCHEMA_VERSION, errors)
    if payload.get("status") == "blocked":
        _validate_blocked(payload, errors)
    else:
        _require_value(payload, "status", "pass", errors)
        _validate_campaign(payload.get("campaign_metadata"), errors)
        _validate_audit(payload.get("gradient_audit"), errors)
        _validate_readiness(payload.get("readiness"), errors)
        _validate_benchmark_context(payload.get("benchmark_context"), errors)
        _require_value(
            payload,
            "claim_status",
            "bounded coupled differentiable scenario evidence only; full-fidelity claim remains blocked",
            errors,
        )
    return {
        "status": "pass" if not errors else "fail",
        "report": str(report_path),
        "claim_admissible": _nested_bool(payload, "readiness", "claim_admissible"),
        "blocked_reasons": _nested_list(payload, "readiness", "blocked_reasons"),
        "errors": errors,
    }


def _validate_blocked(payload: dict[str, Any], errors: list[dict[str, object]]) -> None:
    reason = payload.get("reason")
    if not isinstance(reason, str) or "JAX" not in reason:
        errors.append({"field": "reason", "error": "blocked report must explain missing JAX backend"})
    _require_value(payload, "claim_status", "no coupled-scenario gradient claim; JAX backend unavailable", errors)


def _validate_campaign(value: Any, errors: list[dict[str, object]]) -> None:
    if not isinstance(value, dict):
        errors.append({"field": "campaign_metadata", "error": "campaign_metadata must be an object"})
        return
    _require_value(value, "schema_version", 1, errors, prefix="campaign_metadata.")
    _require_value(value, "backend", "jax", errors, prefix="campaign_metadata.")
    _require_value(value, "dtype", "float64", errors, prefix="campaign_metadata.")
    if not _positive_int(value.get("n_rho")) or int(value.get("n_rho", 0)) < 3:
        errors.append({"field": "campaign_metadata.n_rho", "error": "n_rho must be an integer >= 3"})
    if not _positive_int(value.get("n_steps")):
        errors.append({"field": "campaign_metadata.n_steps", "error": "n_steps must be a positive integer"})
    _require_value(value, "equilibrium_param_count", 2, errors, prefix="campaign_metadata.")
    _validate_pair(value.get("flux_grid_shape"), "campaign_metadata.flux_grid_shape", errors)
    if not _finite_positive(value.get("dt")):
        errors.append({"field": "campaign_metadata.dt", "error": "dt must be finite and positive"})
    if not _finite_positive(value.get("gradient_tolerance")):
        errors.append(
            {"field": "campaign_metadata.gradient_tolerance", "error": "gradient_tolerance must be finite and positive"}
        )
    if not isinstance(value.get("jax_enable_x64"), bool):
        errors.append({"field": "campaign_metadata.jax_enable_x64", "error": "field must be boolean"})
    params = value.get("equilibrium_params")
    if not isinstance(params, list) or len(params) != 2 or not all(_finite_number(param) for param in params):
        errors.append({"field": "campaign_metadata.equilibrium_params", "error": "expected two finite parameters"})
    if not _is_sha256_hex(value.get("inputs_sha256")):
        errors.append({"field": "campaign_metadata.inputs_sha256", "error": "field must be a SHA-256 hex digest"})


def _validate_audit(value: Any, errors: list[dict[str, object]]) -> None:
    if not isinstance(value, dict):
        errors.append({"field": "gradient_audit", "error": "gradient_audit must be an object"})
        return
    if not _finite_non_negative(value.get("loss")):
        errors.append({"field": "gradient_audit.loss", "error": "loss must be finite and non-negative"})
    if not _finite_positive(value.get("epsilon")):
        errors.append({"field": "gradient_audit.epsilon", "error": "epsilon must be finite and positive"})
    tolerance = value.get("tolerance")
    if not _finite_positive(tolerance):
        errors.append({"field": "gradient_audit.tolerance", "error": "tolerance must be finite and positive"})
    if value.get("checked_param_indices") != [0, 1]:
        errors.append({"field": "gradient_audit.checked_param_indices", "error": "expected both parameters checked"})
    source_indices = value.get("checked_source_indices")
    if not isinstance(source_indices, list) or not source_indices:
        errors.append({"field": "gradient_audit.checked_source_indices", "error": "expected non-empty source checks"})
    elif len({tuple(index) for index in source_indices if isinstance(index, list)}) != len(source_indices):
        errors.append({"field": "gradient_audit.checked_source_indices", "error": "source checks must be unique"})
    for field in ("param_max_abs_error", "source_max_abs_error"):
        if not _finite_non_negative(value.get(field)):
            errors.append({"field": f"gradient_audit.{field}", "error": "field must be finite and non-negative"})
    passed = value.get("passed")
    if not isinstance(passed, bool):
        errors.append({"field": "gradient_audit.passed", "error": "field must be boolean"})
    elif _finite_number(tolerance):
        max_error = max(
            float(value.get("param_max_abs_error", math.inf)), float(value.get("source_max_abs_error", math.inf))
        )
        if passed != bool(max_error <= float(tolerance)):
            errors.append({"field": "gradient_audit.passed", "error": "passed flag is inconsistent with tolerance"})


def _validate_readiness(value: Any, errors: list[dict[str, object]]) -> None:
    if not isinstance(value, dict):
        errors.append({"field": "readiness", "error": "readiness must be an object"})
        return
    _require_value(value, "schema_version", 1, errors, prefix="readiness.")
    _require_value(value, "backend", "jax", errors, prefix="readiness.")
    for field in ("campaign_sha256", "gradient_audit_sha256"):
        if not _is_sha256_hex(value.get(field)):
            errors.append({"field": f"readiness.{field}", "error": "field must be a SHA-256 hex digest"})
    if not _finite_positive(value.get("gradient_tolerance")):
        errors.append({"field": "readiness.gradient_tolerance", "error": "field must be finite and positive"})
    if value.get("audit_passed") is not True:
        errors.append({"field": "readiness.audit_passed", "error": "bounded evidence requires a passed audit"})
    if not _finite_non_negative(value.get("latency_p95_ms")):
        errors.append({"field": "readiness.latency_p95_ms", "error": "field must be finite and non-negative"})
    if not isinstance(value.get("traceability_passed"), bool):
        errors.append({"field": "readiness.traceability_passed", "error": "field must be boolean"})
    if value.get("claim_admissible") is not False:
        errors.append({"field": "readiness.claim_admissible", "error": "repository evidence must remain blocked"})
    reasons = value.get("blocked_reasons")
    if reasons != ["physics_traceability"]:
        errors.append({"field": "readiness.blocked_reasons", "error": "expected physics_traceability blocker"})
    _require_value(
        value,
        "claim_status",
        "bounded coupled differentiable scenario gradient evidence only",
        errors,
        prefix="readiness.",
    )


def _validate_benchmark_context(value: Any, errors: list[dict[str, object]]) -> None:
    if not isinstance(value, dict):
        errors.append({"field": "benchmark_context", "error": "benchmark_context must be an object"})
        return
    _require_value(
        value,
        "command",
        "python validation/benchmark_differentiable_scenario.py",
        errors,
        prefix="benchmark_context.",
    )
    _require_value(
        value,
        "isolation",
        "local_non_isolated_admission_smoke",
        errors,
        prefix="benchmark_context.",
    )
    if value.get("warmup_runs") != 1:
        errors.append({"field": "benchmark_context.warmup_runs", "error": "expected one warmup run"})
    if not _positive_int(value.get("timed_runs")):
        errors.append({"field": "benchmark_context.timed_runs", "error": "timed_runs must be positive"})
    durations = value.get("durations_ms")
    if not isinstance(durations, list) or not durations or not all(_finite_positive(item) for item in durations):
        errors.append({"field": "benchmark_context.durations_ms", "error": "durations_ms must be positive finite list"})
    if (
        _positive_int(value.get("timed_runs"))
        and isinstance(durations, list)
        and len(durations) != int(value["timed_runs"])
    ):
        errors.append({"field": "benchmark_context.durations_ms", "error": "duration count must match timed_runs"})


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                raise ValueError(f"duplicate JSON key: {key}")
            seen.add(key)
            result[key] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON payload must be an object")
    return payload


def _require_value(
    payload: dict[str, Any],
    field: str,
    expected: object,
    errors: list[dict[str, object]],
    *,
    prefix: str = "",
) -> None:
    if payload.get(field) != expected:
        errors.append({"field": f"{prefix}{field}", "error": f"expected {expected!r}"})


def _is_sha256_hex(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _finite_positive(value: object) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _finite_non_negative(value: object) -> bool:
    return _finite_number(value) and float(value) >= 0.0


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and value > 0


def _validate_pair(value: object, field: str, errors: list[dict[str, object]]) -> None:
    if not isinstance(value, list) or len(value) != 2 or not all(_positive_int(item) for item in value):
        errors.append({"field": field, "error": "field must be a two-integer shape"})


def _nested_bool(payload: dict[str, Any], parent: str, field: str) -> bool | None:
    value = payload.get(parent)
    if isinstance(value, dict) and isinstance(value.get(field), bool):
        return value[field]
    return None


def _nested_list(payload: dict[str, Any], parent: str, field: str) -> list[object] | None:
    value = payload.get(parent)
    if isinstance(value, dict) and isinstance(value.get(field), list):
        return value[field]
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for differentiable-scenario readiness validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", default=DEFAULT_REPORT)
    parser.add_argument("--json-out", action="store_true", help="print the validation result as JSON")
    args = parser.parse_args(argv)
    result = validate_differentiable_scenario_report(args.report)
    if args.json_out:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"differentiable scenario validation: {result['status']}")
        if result["errors"]:
            for error in result["errors"]:
                print(f"- {error['field']}: {error['error']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
