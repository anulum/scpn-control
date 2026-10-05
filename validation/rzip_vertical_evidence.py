# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Bounded RZIP vertical validation evidence contracts.

"""Serialize and check self-declared RZIP validation reports.

A seal detects changed bytes; it does not authenticate a producer. These
contracts check finite domains, fixed report structure and agreement between
reported metrics and verdicts. They do not establish facility calibration,
freshness, source identity or an independent physical validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from validation.rzip_vertical_models import RzipValidationResult

SCHEMA_VERSION = "scpn-control.rzip-vertical-stability-validation.v1"
_CONFIG_FIELDS = {"r0", "a", "kappa", "ip_ma", "b0", "m_eff_kg"}
_FIELDS = {
    "schema_version",
    "generated_utc",
    "target_id",
    "config",
    "unstable_indices",
    "stable_indices",
    "exact_tol",
    "marginal_tol",
    "max_growth_rel_error",
    "max_frequency_rel_error",
    "max_growth_time_rel_error",
    "marginal_growth_rate",
    "scaling",
    "max_scaling_rel_error",
    "wall",
    "growth_passed",
    "frequency_passed",
    "growth_time_passed",
    "marginal_passed",
    "scaling_passed",
    "wall_passed",
    "passed",
    "payload_sha256",
}
_SCALING_RATIOS = {"current_linear": 2.0, "index_sqrt": 2.0, "inertia_inverse_sqrt": 0.5}


def _digest(payload: Mapping[str, Any]) -> str:
    """Hash the historical JSON representation; field validators enforce finiteness."""
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    try:
        encoded = json.dumps(unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("RZIP evidence must contain JSON-serializable values") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping(value: object, name: str, fields: set[str]) -> Mapping[str, Any]:
    """Require an object with exactly the named schema fields."""
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly the declared fields")
    return value


def _number(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    """Decode a finite JSON number without accepting booleans or numeric text."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _boolean(value: object, name: str) -> bool:
    """Require an actual boolean verdict, without truthiness conversion."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    """Require a nonempty report array without admitting text or empty evidence."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty array")
    return value


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Check a sealed RZIP report's structure, domains and verdict consistency.

    Parameters
    ----------
    payload
        Complete v1 report from ``build_evidence``. Tolerances must be finite
        and positive; relative errors must be finite and non-negative. Growth
        rates use s^-1 and geometry/inertia retain metres, MA, tesla and kg.

    Returns
    -------
    bool
        The literal aggregate verdict after every reported check agrees with
        its metrics. A well-formed failing report returns ``False``. This is
        local report consistency, not producer authentication or admission to
        facility control.

    Raises
    ------
    ValueError
        For an unsupported schema, invalid seal/structure/domain or a verdict
        inconsistent with the declared metrics and thresholds.
    """
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported rzip vertical stability evidence schema_version")
    declared = payload.get("payload_sha256")
    if not isinstance(declared, str) or len(declared) != 64 or any(c not in "0123456789abcdef" for c in declared):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if declared != _digest(payload):
        raise ValueError("payload_sha256 does not match payload")
    _mapping(payload, "RZIP evidence", _FIELDS)
    target_id = payload["target_id"]
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be non-empty")
    stamp = payload["generated_utc"]
    if not isinstance(stamp, str):
        raise ValueError("generated_utc must be a UTC timestamp")
    try:
        generated = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generated_utc must be a UTC timestamp") from exc
    if generated.tzinfo is None or generated.utcoffset() != timedelta(0):
        raise ValueError("generated_utc must be a UTC timestamp")
    config = _mapping(payload["config"], "config", _CONFIG_FIELDS)
    numbers = {name: _number(config[name], f"config.{name}", positive=True) for name in _CONFIG_FIELDS}
    if numbers["a"] >= numbers["r0"]:
        raise ValueError("a must be smaller than r0 for tokamak ordering")
    for name, positive in (("unstable_indices", False), ("stable_indices", True)):
        for raw in _sequence(payload[name], name):
            index = _number(raw, name)
            if (positive and index <= 0.0) or (not positive and index >= 0.0):
                raise ValueError(f"{name} must have the declared sign")
    exact_tol = _number(payload["exact_tol"], "exact_tol", positive=True)
    marginal_tol = _number(payload["marginal_tol"], "marginal_tol", positive=True)
    errors = {
        name: _number(payload[name], name, nonnegative=True)
        for name in (
            "max_growth_rel_error",
            "max_frequency_rel_error",
            "max_growth_time_rel_error",
            "max_scaling_rel_error",
        )
    }
    marginal = _number(payload["marginal_growth_rate"], "marginal_growth_rate")
    scaling = _sequence(payload["scaling"], "scaling")
    if len(scaling) != len(_SCALING_RATIOS):
        raise ValueError("scaling must contain the three declared laws")
    scaling_errors = []
    seen: set[str] = set()
    for raw in scaling:
        check = _mapping(raw, "scaling check", {"name", "measured_ratio", "expected_ratio", "rel_error"})
        name = check["name"]
        if not isinstance(name, str) or name not in _SCALING_RATIOS or name in seen:
            raise ValueError("scaling laws must be known and unique")
        seen.add(name)
        measured = _number(check["measured_ratio"], "measured_ratio", positive=True)
        expected = _number(check["expected_ratio"], "expected_ratio", positive=True)
        error = _number(check["rel_error"], "rel_error", nonnegative=True)
        if expected != _SCALING_RATIOS[name] or error != abs(measured - expected) / expected:
            raise ValueError("scaling values do not agree with the declared law")
        scaling_errors.append(error)
    if errors["max_scaling_rel_error"] != max(scaling_errors):
        raise ValueError("max_scaling_rel_error does not match the scaling checks")
    wall = _mapping(
        payload["wall"],
        "wall",
        {"no_wall_growth_rate", "with_wall_growth_rate", "wall_slows_growth", "with_wall_finite"},
    )
    no_wall = _number(wall["no_wall_growth_rate"], "no_wall_growth_rate", positive=True)
    with_wall = _number(wall["with_wall_growth_rate"], "with_wall_growth_rate")
    wall_slows = _boolean(wall["wall_slows_growth"], "wall_slows_growth")
    wall_finite = _boolean(wall["with_wall_finite"], "with_wall_finite")
    if wall_slows != (with_wall < no_wall) or not wall_finite:
        raise ValueError("wall verdicts do not agree with finite growth rates")
    checks = {
        "growth_passed": errors["max_growth_rel_error"] < exact_tol,
        "frequency_passed": errors["max_frequency_rel_error"] < exact_tol,
        "growth_time_passed": errors["max_growth_time_rel_error"] < exact_tol,
        "marginal_passed": abs(marginal) < marginal_tol,
        "scaling_passed": errors["max_scaling_rel_error"] < exact_tol,
        "wall_passed": wall_slows and wall_finite,
    }
    for name, expected_verdict in checks.items():
        if _boolean(payload[name], name) != expected_verdict:
            raise ValueError(f"{name} does not agree with the reported metrics")
    passed = _boolean(payload["passed"], "passed")
    if passed != all(checks.values()):
        raise ValueError("passed does not agree with the individual checks")
    return passed


def build_evidence(result: RzipValidationResult, *, target_id: str) -> dict[str, Any]:
    """Serialize a bounded numerical result with its UTC receipt and JSON seal.

    Parameters
    ----------
    result
        Rigid-model result with finite domains and mutually consistent checks.
    target_id
        Nonblank label for the caller's local case, not a trusted producer or
        facility identity.

    Returns
    -------
    dict[str, Any]
        Detached v1 report with a UTC receipt and self-consistency seal. Both
        coherent passing and failing results are supported. No files are written.

    Raises
    ------
    ValueError
        The label, result domains or reported verdicts are invalid.

    Examples
    --------
    >>> from validation.validate_rzip_vertical_stability import validate_rzip_vertical_stability
    >>> report = build_evidence(validate_rzip_vertical_stability(), target_id="rigid-example")
    >>> validate_evidence_payload(report)
    True
    """
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "target_id": target_id,
        "config": {name: getattr(result.config, name) for name in sorted(_CONFIG_FIELDS)},
        "unstable_indices": list(result.unstable_indices),
        "stable_indices": list(result.stable_indices),
        "exact_tol": result.exact_tol,
        "marginal_tol": result.marginal_tol,
        "max_growth_rel_error": result.max_growth_rel_error,
        "max_frequency_rel_error": result.max_frequency_rel_error,
        "max_growth_time_rel_error": result.max_growth_time_rel_error,
        "marginal_growth_rate": result.marginal_growth_rate,
        "scaling": [
            {
                "name": c.name,
                "measured_ratio": c.measured_ratio,
                "expected_ratio": c.expected_ratio,
                "rel_error": c.rel_error,
            }
            for c in result.scaling
        ],
        "max_scaling_rel_error": result.max_scaling_rel_error,
        "wall": {
            "no_wall_growth_rate": result.wall.no_wall_growth_rate,
            "with_wall_growth_rate": result.wall.with_wall_growth_rate,
            "wall_slows_growth": result.wall.wall_slows_growth,
            "with_wall_finite": result.wall.with_wall_finite,
        },
        "growth_passed": result.growth_passed,
        "frequency_passed": result.frequency_passed,
        "growth_time_passed": result.growth_time_passed,
        "marginal_passed": result.marginal_passed,
        "scaling_passed": result.scaling_passed,
        "wall_passed": result.wall_passed,
        "passed": result.passed,
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _digest(payload)
    validate_evidence_payload(payload)
    return payload


def write_report(evidence: Mapping[str, Any], json_path: Path) -> None:
    """Write checked JSON and its same-stem Markdown report by direct replacement.

    Parameters
    ----------
    evidence
        Complete sealed v1 report; a coherent failing verdict is also writable.
    json_path
        Destination relative to the caller's cwd or absolute. The parent must
        exist. Existing JSON and same-stem Markdown files are overwritten.

    Returns
    -------
    None
        The JSON file is written first, followed by its Markdown sibling.

    Raises
    ------
    ValueError
        Report validation fails before any write.
    OSError
        A destination cannot be written. Failure after the JSON write can leave
        a partial pair; no rollback or exclusive publication is provided.

    Notes
    -----
    Writing a report does not authenticate its source or admit facility control.
    """
    validate_evidence_payload(evidence)
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = evidence["config"]
    wall = evidence["wall"]
    lines = [
        "",
        "# RZIP Rigid Vertical Stability Validation",
        "",
        f"- Schema: `{evidence['schema_version']}`",
        f"- Generated (UTC): {evidence['generated_utc']}",
        f"- Target: `{evidence['target_id']}`",
        f"- Geometry: R0={config['r0']} m, a={config['a']} m, kappa={config['kappa']}, Ip={config['ip_ma']} MA, M_eff={config['m_eff_kg']} kg",
        f"- Status: **{'pass' if evidence['passed'] else 'fail'}**",
        "",
        "## Exact no-wall references (largest eigenvalue of the 2x2 rigid block)",
        "",
        f"- Max unstable growth-rate rel error (n<0): {evidence['max_growth_rel_error']:.3e}",
        f"- Max stable oscillation-frequency rel error (n>0): {evidence['max_frequency_rel_error']:.3e}",
        f"- Max growth-time identity rel error: {evidence['max_growth_time_rel_error']:.3e}",
        f"- Marginal growth rate at n=0: {evidence['marginal_growth_rate']:.3e} (gate < {evidence['marginal_tol']:.1e})",
        f"- Exact-reference tolerance: {evidence['exact_tol']:.1e}",
        "",
        "## Exact scaling laws",
        "",
        "| law | measured ratio | expected | rel error |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {c['name']} | {c['measured_ratio']:.6f} | {c['expected_ratio']} | {c['rel_error']:.3e} |"
        for c in evidence["scaling"]
    ]
    lines += [
        "",
        "## Resistive-wall stabilisation",
        "",
        f"- No-wall growth rate: {wall['no_wall_growth_rate']:.4e} s^-1",
        f"- With-wall growth rate: {wall['with_wall_growth_rate']:.4e} s^-1",
        f"- Wall slows growth: {wall['wall_slows_growth']}; finite: {wall['with_wall_finite']}",
    ]
    json_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
