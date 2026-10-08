# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density report structure and consistency
"""Check density report declarations without admitting physical or producer authenticity."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Callable

SCHEMA_VERSION = "scpn-control.density-control-validation.v3"
SOURCE_PATHS = (
    "src/scpn_control/control/density_controller.py",
    "validation/validate_density_control.py",
    "validation/density_control_evidence.py",
    "tools/inventory_file_output.py",
)
_CONFIG = {
    "n_rho",
    "n_chords",
    "major_radius_m",
    "minor_radius_m",
    "plasma_current_ma",
    "gas_puff_rate_per_s",
    "nbi_energy_kev",
    "nbi_power_mw",
    "recycling_outflux_per_s",
    "recycling_coeff",
    "pump_speed_m3_s",
    "edge_density_m3",
    "uniform_density_m3",
}
_GROUPS = {
    "greenwald_passed": ("greenwald_limit_rel_error", "greenwald_fraction_rel_error", "volume_element_rel_error"),
    "sources_passed": (
        "gas_puff_conservation_rel_error",
        "nbi_conservation_rel_error",
        "recycling_conservation_rel_error",
        "cryopump_sink_rel_error",
    ),
    "diffusion_passed": ("diffusion_uniform_invariance_abs_error",),
    "interferometry_passed": (
        "interferometer_uniform_projection_rel_error",
        "interferometer_signed_symmetry_rel_error",
    ),
}
_METRICS = {field for names in _GROUPS.values() for field in names}
_FIELDS = {
    "schema_version",
    "generated_utc",
    "target_id",
    "config",
    "exact_tol",
    "invariance_tol",
    "scaling",
    "max_scaling_rel_error",
    "scaling_passed",
    "passed",
    "runtime_source_sha256",
    "payload_sha256",
    *_GROUPS,
    *_METRICS,
}
_SCALING = {"current_linear": 2.0, "minor_radius_inverse_square": 0.25}


def _digest(payload: Mapping[str, object]) -> str:
    """Use the retained sorted ASCII JSON content-hash representation."""
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    try:
        encoded = json.dumps(unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        raise ValueError("Density evidence must contain JSON-serialisable values") from None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping(value: object, name: str, fields: set[str]) -> dict[str, object]:
    """Require exactly the declared object fields."""
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} must contain exactly the declared fields")
    return {field: value[field] for field in fields}


def _number(value: object, name: str, *, positive: bool = False) -> float:
    """Require a finite nonnegative JSON number without boolean coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        raise ValueError(f"{name} has an invalid numerical domain")
    return result


def _boolean(value: object, name: str) -> bool:
    """Require a literal boolean declaration."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _sha256(value: object) -> bool:
    """Recognise the declared lower-case SHA-256 digest syntax."""
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _header(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate schema, complete fields, content hash and declared UTC identity."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported density control evidence schema_version")
    if not _sha256(payload.get("payload_sha256")):
        raise ValueError("payload_sha256 must be a SHA-256 hex digest")
    if payload["payload_sha256"] != _digest(payload):
        raise ValueError("payload_sha256 does not match payload")
    data = _mapping(payload, "density evidence", _FIELDS)
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
    sources = _mapping(data["runtime_source_sha256"], "runtime source digests", set(SOURCE_PATHS))
    if any(not _sha256(digest) for digest in sources.values()):
        raise ValueError("Runtime source digests must be SHA-256 hex declarations")
    return data


def _scaling_errors(value: object) -> list[float]:
    """Check both declared Greenwald scaling laws and derived errors."""
    if not isinstance(value, list) or len(value) != len(_SCALING):
        raise ValueError("scaling must contain both declared laws")
    seen: set[str] = set()
    errors = []
    for item in value:
        check = _mapping(item, "scaling check", {"name", "measured_ratio", "expected_ratio", "rel_error"})
        name = check["name"]
        if not isinstance(name, str) or name not in _SCALING or name in seen:
            raise ValueError("scaling laws must be known and unique")
        seen.add(name)
        measured = _number(check["measured_ratio"], "measured_ratio", positive=True)
        expected = _number(check["expected_ratio"], "expected_ratio", positive=True)
        error = _number(check["rel_error"], "rel_error")
        if expected != _SCALING[name] or error != abs(measured - expected) / expected:
            raise ValueError("scaling values do not match the declared law")
        errors.append(error)
    return errors


def _validate_payload(payload: Mapping[str, object]) -> bool:
    """Check the complete v3 declaration through the owning facade's config."""
    from validation.validate_density_control import DensityConfig

    data = _header(payload)
    config = _mapping(data["config"], "density config", _CONFIG)
    constructor: Callable[..., DensityConfig] = DensityConfig
    constructor(**config)
    exact = _number(data["exact_tol"], "exact_tol", positive=True)
    invariance = _number(data["invariance_tol"], "invariance_tol", positive=True)
    metrics = {field: _number(data[field], field) for field in _METRICS}
    scaling_errors = _scaling_errors(data["scaling"])
    maximum = _number(data["max_scaling_rel_error"], "max_scaling_rel_error")
    if maximum != max(scaling_errors):
        raise ValueError("scaling maximum does not match its errors")
    checks = {
        flag: all(metrics[field] < (invariance if flag == "diffusion_passed" else exact) for field in names)
        for flag, names in _GROUPS.items()
    }
    checks["scaling_passed"] = maximum < exact
    for name, expected in checks.items():
        if _boolean(data[name], name) != expected:
            raise ValueError("stage verdict does not match its metrics")
    aggregate = _boolean(data["passed"], "passed")
    if aggregate != all(checks.values()):
        raise ValueError("aggregate verdict does not match its stages")
    return aggregate
