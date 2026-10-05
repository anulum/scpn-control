# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOL diagnostic evidence structure and self-digest.
"""Seal and validate complete v1 local algebraic observations without admission.

A self-digest detects byte-content changes against its declared digest. It
does not authenticate a producer, physical input, model implementation or source
provenance. A sender can reseal any internally consistent content.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from validation.sol_two_point_contracts.models import SOLConfig, SOLValidationResult, _positive_float

SOL_TWO_POINT_SCHEMA_VERSION = "scpn-control.sol-two-point-validation.v1"
_ERROR_FLAGS = {
    "connection_length_rel_error": "connection_passed",
    "max_flux_mapping_rel_error": "flux_mapping_passed",
    "max_conduction_rel_error": "conduction_passed",
    "max_pressure_balance_rel_error": "pressure_passed",
    "max_scaling_rel_error": "scaling_passed",
    "peak_heat_flux_rel_error": "peak_flux_passed",
}
_REQUIRED = {
    "schema_version",
    "generated_utc",
    "target_id",
    "config",
    "operating_points",
    "exact_tol",
    "scaling",
    "detachment",
    "detachment_passed",
    "passed",
    "payload_sha256",
    *_ERROR_FLAGS,
    *_ERROR_FLAGS.values(),
}
_SCALING_NAMES = ("power_-0.02", "major_radius_0.04", "b_pol_-0.92", "epsilon_0.42")


def _object(value: object, name: str, keys: set[str]) -> Mapping[str, Any]:
    """Require a mapping with exactly the specified string keys."""
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly its declared fields")
    return value


def _number(value: object, name: str) -> float:
    """Require a nonnegative finite Python number, excluding booleans/strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _flag(value: object, name: str) -> bool:
    """Require a literal bool; refuse truthy strings and integers."""
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _content(payload: Mapping[str, Any]) -> bool:
    """Validate v1 shapes, units domains and internally consistent flags/errors."""
    _object(payload, "payload", _REQUIRED)
    target = payload["target_id"]
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target_id must be a nonempty string")
    stamp = payload["generated_utc"]
    if not isinstance(stamp, str):
        raise ValueError("generated_utc must be a UTC second timestamp")
    try:
        parsed = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("generated_utc must be a UTC second timestamp") from exc
    if parsed.isoformat(timespec="seconds") + "Z" != stamp:
        raise ValueError("generated_utc must be a UTC second timestamp")
    config = _object(payload["config"], "config", {"r0", "a", "q95", "b_pol"})
    geometry = SOLConfig(**{key: _positive_float(key, config[key]) for key in config})
    points = payload["operating_points"]
    if not isinstance(points, list) or not points:
        raise ValueError("operating_points must be a nonempty array of pairs")
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("operating_points must contain length-two arrays")
        _positive_float("P_SOL_MW", point[0])
        _positive_float("n_u_19", point[1])
    tolerance = _positive_float("exact_tol", payload["exact_tol"])
    errors = {key: _number(payload[key], key) for key in _ERROR_FLAGS}
    scaling = payload["scaling"]
    if not isinstance(scaling, list) or len(scaling) != 4:
        raise ValueError("scaling must contain four ordered observations")
    epsilon_factor = 2.0 if geometry.epsilon < 0.5 else 0.5
    expected = (2.0**-0.02, 2.0**0.04, 2.0**-0.92, epsilon_factor**0.42)
    scaling_errors = []
    for value, name, expected_ratio in zip(scaling, _SCALING_NAMES, expected, strict=True):
        check = _object(value, "scaling observation", {"name", "measured_ratio", "expected_ratio", "rel_error"})
        measured = _positive_float("measured_ratio", check["measured_ratio"])
        declared_expected = _positive_float("expected_ratio", check["expected_ratio"])
        error = _number(check["rel_error"], "rel_error")
        if check["name"] != name or declared_expected != expected_ratio:
            raise ValueError("scaling name or expected ratio contradicts the declared probe")
        if error != abs(measured - declared_expected) / declared_expected:
            raise ValueError("scaling rel_error contradicts its ratios")
        scaling_errors.append(error)
    if errors["max_scaling_rel_error"] != max(scaling_errors):
        raise ValueError("max_scaling_rel_error contradicts the observations")
    flags = []
    for key, name in _ERROR_FLAGS.items():
        flag = _flag(payload[name], name)
        if flag != (errors[key] < tolerance):
            raise ValueError(f"{name} contradicts its strict error gate")
        flags.append(flag)
    detached = _object(
        payload["detachment"],
        "detachment",
        {
            "critical_density_19",
            "detached_below_critical",
            "detached_above_critical",
        },
    )
    _positive_float("critical_density_19", detached["critical_density_19"])
    below = _flag(detached["detached_below_critical"], "detached_below_critical")
    above = _flag(detached["detached_above_critical"], "detached_above_critical")
    detachment_passed = _flag(payload["detachment_passed"], "detachment_passed")
    if detachment_passed != all((not below, above)):
        raise ValueError("detachment_passed contradicts its boundary probes")
    flags.append(detachment_passed)
    passed = _flag(payload["passed"], "passed")
    if passed != all(flags):
        raise ValueError("passed contradicts the seven check flags")
    return passed


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash sorted ASCII finite JSON with the payload digest replaced by empty text."""
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    try:
        data = json.dumps(unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("payload must be serializable finite JSON") from exc
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build_evidence(result: SOLValidationResult, *, target_id: str) -> dict[str, Any]:
    """Seal a complete internally consistent local SOL result as schema v1 JSON.

    Parameters
    ----------
    result
        SOLValidationResult from the public diagnostic, or a manually constructed
        result with the same checked structure/flags. Config uses metres/tesla,
        operating points use MW and 1e19 m^-3, errors/ratios are dimensionless.
    target_id
        Nonempty descriptive string, preserved without trimming or authentication.

    Returns
    -------
    dict of str to Any
        Fresh JSON-compatible mapping, UTC wall-clock timestamp at second
        resolution and lowercase self-SHA256. A failing consistent result is
        sealed normally. No filesystem/state mutation or provenance is supplied.

    Raises
    ------
    ValueError
        Nonfinite/nonnumeric values, malformed structure or inconsistent gates.

    Notes
    -----
    This is an integrity self-digest, not a signature. No experimental, source,
    facility, safety, independent-reference or training admission is established.
    """
    payload = asdict(result)
    payload.update(
        schema_version=SOL_TWO_POINT_SCHEMA_VERSION,
        generated_utc=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        target_id=target_id,
        operating_points=[list(point) for point in result.operating_points],
        scaling=[asdict(check) for check in result.scaling],
        payload_sha256="",
    )
    _content(payload)
    payload["payload_sha256"] = _payload_sha256(payload)
    return payload


def validate_evidence_payload(payload: Mapping[str, Any]) -> bool:
    """Return the checked boolean outcome of a complete self-sealed v1 report.

    Parameters
    ----------
    payload
        Decoded JSON mapping with exactly the v1 fields. Arrays have explicit
        power/density pairs and four ordered scaling observations; all numbers
        are finite with declared positive/nonnegative domains, excluding bools.

    Returns
    -------
    bool
        True for internally consistent passing content, False for internally
        consistent failing content. This outcome is not scientific admission.

    Raises
    ------
    ValueError
        Unsupported schema, malformed/mismatched digest, invalid shapes, values,
        UTC-second timestamp, scaling arithmetic or contradictory check flags.

    Notes
    -----
    The self-digest uses sorted compact ASCII JSON with the seal field blank.
    Relative-error arithmetic/maxima and strict error gates are replayed from
    the declared observations. Physical formulas, source provenance, clock
    freshness, detachment input parameters and producer identity are not replayed
    or authenticated. V1 does not record all these inputs. Nested duplicate JSON
    keys must be rejected by a caller before decoding loses them.
    """
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SOL_TWO_POINT_SCHEMA_VERSION:
        raise ValueError("unsupported sol two-point evidence schema_version")
    declared = payload.get("payload_sha256")
    if not isinstance(declared, str) or len(declared) != 64 or any(ch not in "0123456789abcdef" for ch in declared):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if declared != _payload_sha256(payload):
        raise ValueError("payload_sha256 does not match payload")
    return _content(payload)
