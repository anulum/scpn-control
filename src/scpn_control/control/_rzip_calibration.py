# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Self-consistency checks for bounded RZIP calibration.
"""Check declared calibration metrics without authenticating a reference source."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

_SCHEMA_VERSION = 1
_FACILITY_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "external_code_benchmark", "measured_discharge"}
)
_BOUNDED_REFERENCE_SOURCES = frozenset({"local_regression_reference", *_FACILITY_REFERENCE_SOURCES})
_FIELDS = frozenset(
    {
        "schema_version",
        "source",
        "source_id",
        "model_id",
        "vertical_inertia_kg",
        "wall_time_constant_s",
        "growth_rate_s_inv",
        "growth_time_ms",
        "reference_growth_rate_s_inv",
        "growth_rate_relative_error",
        "growth_rate_relative_tolerance",
        "facility_claim_allowed",
        "claim_status",
        "evidence_payload_sha256",
    }
)


def _payload_digest(payload: Mapping[str, object]) -> str:
    """Hash detached unsigned fields using the historical ASCII JSON representation."""
    unsigned = dict(payload)
    unsigned.pop("evidence_payload_sha256", None)
    blob = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _number(name: str, value: object, *, finite: bool = True) -> float:
    """Read a genuine JSON number, refusing booleans, coercions and overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside the working range") from exc
    if math.isnan(result) or (finite and not math.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(name: str, value: object) -> float:
    """Require a finite positive physical scale or comparison tolerance."""
    result = _number(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _claim_status(source: str, error: float | None, allowed: bool, *, finite_growth: bool) -> str:
    """Keep historical growing-source statuses and identify unsupported stable admission."""
    if source == "local_regression_reference":
        return "bounded local RZIP regression evidence only; external reference required for facility claims"
    if error is None:
        return "external RZIP reference source declared but quantitative growth-rate comparison is missing"
    if not finite_growth:
        return "external RZIP reference admission requires finite positive model growth"
    if allowed:
        return "external RZIP reference admission passed for declared tolerance"
    return "external RZIP reference admission failed declared growth-rate tolerance"


def _inspect_calibration_payload(payload: Mapping[str, object]) -> bool:
    """Return the consistent declared facility flag or raise for a malformed observation.

    Rates are in s^-1, growth time in ms, wall time in s and inertia in kg.
    Error/tolerance are dimensionless. Zero model growth requires positive
    infinite growth time, retained as historical extended-real JSON; that
    stable observation never admits a facility flag. Positive-growth time must
    agree with 1000/gamma. The historical comparison denominator floor is 1e-30.

    A self-seal and caller-declared source label provide content consistency,
    not an authenticated external result, calibration, freshness or deployment
    approval. No file is read or written and no model/reference is recomputed.
    """
    if set(payload) != _FIELDS:
        raise ValueError("RZIP calibration evidence fields do not match schema_version 1")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("RZIP calibration evidence schema_version is unsupported")
    source = payload["source"]
    if not isinstance(source, str) or source not in _BOUNDED_REFERENCE_SOURCES:
        raise ValueError("source must be a declared RZIP reference source")
    for name in ("source_id", "model_id"):
        value = payload[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"RZIP calibration evidence requires a non-empty {name}")
    _positive("vertical_inertia_kg", payload["vertical_inertia_kg"])
    _positive("wall_time_constant_s", payload["wall_time_constant_s"])
    tolerance = _positive("growth_rate_relative_tolerance", payload["growth_rate_relative_tolerance"])
    try:
        gamma = _number("growth_rate_s_inv", payload["growth_rate_s_inv"])
    except ValueError as exc:
        raise ValueError("RZIP facility claim requires a finite model growth rate") from exc
    if gamma < 0:
        raise ValueError("RZIP model growth rate must be non-negative")
    tau = _number("growth_time_ms", payload["growth_time_ms"], finite=False)
    expected_tau = 1000 / gamma if gamma > 0 else math.inf
    if tau <= 0 or not math.isclose(tau, expected_tau, rel_tol=1e-12, abs_tol=0):
        raise ValueError("growth_time_ms must agree with the model growth rate")
    reference_raw = payload["reference_growth_rate_s_inv"]
    error_raw = payload["growth_rate_relative_error"]
    error: float | None = None
    if reference_raw is None:
        if error_raw is not None:
            raise ValueError("RZIP facility claim requires a reference growth rate when an error is declared")
    else:
        reference = _positive("reference_growth_rate_s_inv", reference_raw)
        if error_raw is None:
            raise ValueError("RZIP facility claim requires finite growth-rate comparison error for a reference")
        error = _number("growth_rate_relative_error", error_raw)
        expected_error = abs(gamma - reference) / max(abs(reference), 1e-30)
        if error < 0 or not math.isclose(error, expected_error, rel_tol=1e-12, abs_tol=0):
            raise ValueError("RZIP growth-rate tolerance comparison error is inconsistent")
    flag = payload["facility_claim_allowed"]
    if type(flag) is not bool:
        raise ValueError("facility_claim_allowed must be a literal boolean")
    allowed = bool(
        source in _FACILITY_REFERENCE_SOURCES
        and gamma > 0
        and math.isfinite(tau)
        and error is not None
        and error <= tolerance
    )
    if flag != allowed:
        raise ValueError("RZIP facility claim is not admissible: facility_claim_allowed is inconsistent")
    if payload["claim_status"] != _claim_status(source, error, allowed, finite_growth=gamma > 0 and math.isfinite(tau)):
        raise ValueError("RZIP calibration evidence claim_status is inconsistent")
    if payload["evidence_payload_sha256"] != _payload_digest(payload):
        raise ValueError("RZIP calibration evidence payload digest mismatch")
    return allowed
