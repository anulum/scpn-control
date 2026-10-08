# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second report structure and consistency
"""Check declared analytic results and their content hash, without authenticating a producer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields
from datetime import UTC, datetime, timedelta

from validation.volt_second_models import (
    VoltSecondConfig,
    VoltSecondValidationResult,
    _finite_float,
)
from validation.volt_second_models import (
    VoltSecondEvidence as VoltSecondEvidence,
)

SCHEMA_VERSION = "scpn-control.volt-second-validation.v1"


_FIELDS = set(VoltSecondEvidence.__annotations__)
_CONFIG = {field.name for field in fields(VoltSecondConfig)}
_FLUX_ERRORS = ("inductive_rel_error", "ejima_rel_error", "resistive_ramp_rel_error", "flat_top_closure_rel_error")
_DECOMPOSITION = (
    "ramp_rel_error",
    "flat_top_rel_error",
    "ramp_down_rel_error",
    "sum_rel_error",
    "margin_abs_error",
    "max_rel_error",
)
_MONITOR = ("consumed_rel_error", "remaining_rel_error", "fraction_rel_error", "max_rel_error")
_RAMP_ERRORS = ("start_abs_error", "end_rel_error", "spacing_max_rel_error")
_SCALING_NAMES = {
    "inductive_current_linear",
    "inductive_inductance_linear",
    "ejima_major_radius_linear",
    "ejima_current_linear",
}


def evidence_digest(payload: Mapping[str, object]) -> str:
    """Hash the historical v1 JSON representation with its digest field empty.

    Parameters
    ----------
    payload : mapping of str to object
        Report fields to serialise. This hash does not authenticate a producer.

    Returns
    -------
    str
        SHA-256 hex digest with the existing sorted, ASCII JSON encoding.

    Raises
    ------
    ValueError
        Values cannot be represented as JSON.
    """
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    try:
        encoded = json.dumps(unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        raise ValueError("Volt-second evidence must contain JSON-serialisable values") from None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping(value: object, name: str, expected: set[str]) -> dict[str, object]:
    """Require exactly the declared string-keyed fields."""
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly the declared fields")
    return {name: value[name] for name in expected}


def _number(value: object, name: str, *, positive: bool = False) -> float:
    """Require a finite nonnegative number, or a strictly positive threshold."""
    number = _finite_float(name, value)
    if number < 0.0 or (positive and number == 0.0):
        raise ValueError(f"{name} has an invalid numerical domain")
    return number


def _boolean(value: object, name: str) -> bool:
    """Require a literal boolean without truthiness conversion."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _errors(value: object, name: str, names: tuple[str, ...]) -> dict[str, float]:
    """Read a complete finite nonnegative metric group."""
    group = _mapping(value, name, set(names))
    return {field: _number(group[field], name + "." + field) for field in names}


def _checked_header(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate schema, content digest, identity and UTC declaration."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported volt-second evidence schema_version")
    declared = payload.get("payload_sha256")
    if not isinstance(declared, str) or len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if declared != evidence_digest(payload):
        raise ValueError("payload_sha256 does not match payload")
    data = _mapping(payload, "volt-second evidence", _FIELDS)
    target = data["target_id"]
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target_id must be non-empty")
    stamp = data["generated_utc"]
    if not isinstance(stamp, str):
        raise ValueError("generated_utc must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("generated_utc must be a UTC timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("generated_utc must be a UTC timestamp")
    return data


def _scaling_errors(value: object) -> list[float]:
    """Check complete known linear laws and their declared ratio errors."""
    scaling = value
    if not isinstance(scaling, list) or len(scaling) != len(_SCALING_NAMES):
        raise ValueError("scaling must contain the four declared laws")
    seen: set[str] = set()
    scaling_errors: list[float] = []
    for item in scaling:
        check = _mapping(item, "scaling check", {"name", "measured_ratio", "expected_ratio", "rel_error"})
        name = check["name"]
        if not isinstance(name, str) or name not in _SCALING_NAMES or name in seen:
            raise ValueError("scaling laws must be known and unique")
        seen.add(name)
        measured = _number(check["measured_ratio"], "measured_ratio", positive=True)
        expected = _number(check["expected_ratio"], "expected_ratio", positive=True)
        error = _number(check["rel_error"], "rel_error")
        if expected != 2.0 or error != abs(measured - expected) / expected:
            raise ValueError("scaling values do not match the declared law")
        scaling_errors.append(error)
    return scaling_errors


def validate_evidence_payload(payload: Mapping[str, object]) -> bool:
    """Check report structure, finite domains, scaling and verdict consistency.

    Parameters
    ----------
    payload : mapping of str to object
        Complete v1 report. Errors must be finite and nonnegative, tolerances
        finite and positive, and every literal verdict must match its metrics.

    Returns
    -------
    bool
        Literal aggregate result; a consistent failed report returns False.
        Content integrity and self-consistency do not establish producer,
        source, freshness, experimental provenance or facility/control admission.

    Raises
    ------
    ValueError
        Schema, digest, fields, domains or declared verdicts are invalid.
    """
    data = _checked_header(payload)
    config = _mapping(data["config"], "config", _CONFIG)
    VoltSecondConfig(**{field: _finite_float("config." + field, value) for field, value in config.items()})
    exact = _number(data["exact_tol"], "exact_tol", positive=True)
    margin = _number(data["margin_abs_tol"], "margin_abs_tol", positive=True)
    flux = [_number(data[field], field) for field in _FLUX_ERRORS]
    decomposition = _errors(data["decomposition"], "decomposition", _DECOMPOSITION)
    monitor = _errors(data["monitor"], "monitor", _MONITOR)
    ramp = _mapping(data["ramp_optimizer"], "ramp_optimizer", {*_RAMP_ERRORS, "is_linear"})
    ramp_errors = {field: _number(ramp[field], "ramp_optimizer." + field) for field in _RAMP_ERRORS}
    linear = _boolean(ramp["is_linear"], "is_linear")
    if linear != (ramp_errors["spacing_max_rel_error"] < 1e-12):
        raise ValueError("ramp linearity does not match spacing error")
    if decomposition["max_rel_error"] != max(decomposition[field] for field in _DECOMPOSITION[:4]):
        raise ValueError("decomposition maximum does not match its phase errors")
    if monitor["max_rel_error"] != max(monitor[field] for field in _MONITOR[:3]):
        raise ValueError("monitor maximum does not match its errors")
    scaling_errors = _scaling_errors(data["scaling"])
    max_scaling = _number(data["max_scaling_rel_error"], "max_scaling_rel_error")
    if max_scaling != max(scaling_errors):
        raise ValueError("scaling maximum does not match its errors")
    checks = {
        "fluxes_passed": all(error < exact for error in flux),
        "decomposition_passed": decomposition["max_rel_error"] < exact and decomposition["margin_abs_error"] < margin,
        "monitor_passed": monitor["max_rel_error"] < exact,
        "optimizer_passed": linear and ramp_errors["start_abs_error"] < exact and ramp_errors["end_rel_error"] < exact,
        "scaling_passed": max_scaling < exact,
    }
    for field, expected_flag in checks.items():
        if _boolean(data[field], field) != expected_flag:
            raise ValueError("stage verdict does not match its metrics")
    aggregate = _boolean(data["passed"], "passed")
    if aggregate != all(checks.values()):
        raise ValueError("aggregate verdict does not match its stages")
    return aggregate


def build_evidence(result: VoltSecondValidationResult, *, target_id: str) -> VoltSecondEvidence:
    """Build a checked, content-hashed analytic v1 report."""
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: VoltSecondEvidence = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "target_id": target_id,
        "config": {
            "flux_budget_vs": result.config.flux_budget_vs,
            "plasma_inductance_uh": result.config.plasma_inductance_uh,
            "plasma_resistance_uohm": result.config.plasma_resistance_uohm,
            "major_radius_m": result.config.major_radius_m,
            "plasma_current_ma": result.config.plasma_current_ma,
            "bootstrap_current_ma": result.config.bootstrap_current_ma,
            "ramp_duration_s": result.config.ramp_duration_s,
            "flat_duration_s": result.config.flat_duration_s,
            "ramp_down_duration_s": result.config.ramp_down_duration_s,
            "standalone_ramp_flux_vs": result.config.standalone_ramp_flux_vs,
        },
        "exact_tol": result.exact_tol,
        "margin_abs_tol": result.margin_abs_tol,
        "inductive_rel_error": result.inductive_rel_error,
        "ejima_rel_error": result.ejima_rel_error,
        "resistive_ramp_rel_error": result.resistive_ramp_rel_error,
        "flat_top_closure_rel_error": result.flat_top_closure_rel_error,
        "decomposition": {
            "ramp_rel_error": result.decomposition.ramp_rel_error,
            "flat_top_rel_error": result.decomposition.flat_top_rel_error,
            "ramp_down_rel_error": result.decomposition.ramp_down_rel_error,
            "sum_rel_error": result.decomposition.sum_rel_error,
            "margin_abs_error": result.decomposition.margin_abs_error,
            "max_rel_error": result.decomposition.max_rel_error,
        },
        "monitor": {
            "consumed_rel_error": result.monitor.consumed_rel_error,
            "remaining_rel_error": result.monitor.remaining_rel_error,
            "fraction_rel_error": result.monitor.fraction_rel_error,
            "max_rel_error": result.monitor.max_rel_error,
        },
        "ramp_optimizer": {
            "start_abs_error": result.ramp_optimizer.start_abs_error,
            "end_rel_error": result.ramp_optimizer.end_rel_error,
            "spacing_max_rel_error": result.ramp_optimizer.spacing_max_rel_error,
            "is_linear": result.ramp_optimizer.is_linear,
        },
        "scaling": [
            {
                "name": check.name,
                "measured_ratio": check.measured_ratio,
                "expected_ratio": check.expected_ratio,
                "rel_error": check.rel_error,
            }
            for check in result.scaling
        ],
        "max_scaling_rel_error": result.max_scaling_rel_error,
        "fluxes_passed": result.fluxes_passed,
        "decomposition_passed": result.decomposition_passed,
        "monitor_passed": result.monitor_passed,
        "optimizer_passed": result.optimizer_passed,
        "scaling_passed": result.scaling_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = evidence_digest(payload)
    validate_evidence_payload(payload)
    return payload
