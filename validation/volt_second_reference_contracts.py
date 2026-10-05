#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Volt-second declaration contracts


"""Check volt-second declaration identity, provenance presence and original numeric domains."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_ALLOWED_SOURCES = {"documented_public_reference", "measured_loop_voltage_replay", "external_scenario_benchmark"}
_ALLOWED_EXTERNAL_CODES = {"TRANSP", "TSC", "ASTRA", "JINTRAC", "PROCESS"}
_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)
_REQUIRED_UNITS = {
    "flux": "V s",
    "voltage": "V",
    "current": "A",
    "current_MA": "MA",
    "time": "s",
    "resistance": "ohm",
    "inductance": "H",
    "radius": "m",
    "dimensionless": "1",
}
_REQUIRED_MACHINE_FIELDS = ("Phi_CS_Vs", "L_plasma_H", "R_plasma_Ohm", "Ip_MA", "R0_m")
_MAXIMUM_ERROR_METRICS = (
    "total_flux_relative_error",
    "flat_top_duration_relative_error",
    "ejima_flux_relative_error",
    "bootstrap_current_abs_error_MA",
    "margin_abs_error_Vs",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Check original schema1.0, provenance presence, format-only SHA, units, machine/count and declared bounds; return identity metadata without referenced-byte or physics authentication."""
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "artifact root must be an object"})
        return None
    if payload.get("schema_version") != "1.0":
        errors.append({"path": str(path), "field": "schema_version", "error": "schema_version must be '1.0'"})
    for field in _REQUIRED_STR_FIELDS:
        if not _has_nonempty_str(payload, field):
            errors.append({"path": str(path), "field": field, "error": "field must be a non-empty string"})
    digest = payload.get("reference_artifact_sha256")
    if isinstance(digest, str) and not _SHA256_RE.fullmatch(digest):
        errors.append(
            {"path": str(path), "field": "reference_artifact_sha256", "error": "field must be a SHA-256 hex digest"}
        )
    source = payload.get("source")
    if not isinstance(source, str) or source not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": str(path),
                "field": "source",
                "error": "source must be documented_public_reference, measured_loop_voltage_replay, or external_scenario_benchmark",
            }
        )
    _validate_source_provenance(path, payload, errors)
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare volt-second contracts"})
    if not _valid_machine_metadata(payload.get("machine_metadata")):
        errors.append(
            {
                "path": str(path),
                "field": "machine_metadata",
                "error": "machine_metadata must declare finite positive scenario parameters",
            }
        )
    count = payload.get("reference_case_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        errors.append({"path": str(path), "field": "reference_case_count", "error": "field must be a positive integer"})
    _validate_metric_block(path, payload.get("metrics"), payload.get("tolerances"), errors)
    if any(error["path"] == str(path) for error in errors):
        return None
    return {
        "path": str(path),
        "source": str(payload["source"]),
        "model_id": str(payload["model_id"]),
        "model_version": str(payload["model_version"]),
        "reference_dataset_id": str(payload["reference_dataset_id"]),
        "reference_case_count": int(payload["reference_case_count"]),
    }


def _validate_source_provenance(path: Path, payload: dict[str, object], errors: list[dict[str, object]]) -> None:
    """Require public URL/DOI, replay shot/diagnostic or named external-code/URI presence; do not parse, resolve or authenticate references."""
    source = payload.get("source")
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "reference",
                "error": "documented public references require reference_url or reference_doi",
            }
        )
    if source == "measured_loop_voltage_replay":
        if not _has_nonempty_str(payload, "shot_id"):
            errors.append(
                {"path": str(path), "field": "shot_id", "error": "measured loop-voltage replays require shot_id"}
            )
        if not _has_nonempty_str(payload, "diagnostic_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "diagnostic_uri",
                    "error": "measured loop-voltage replays require diagnostic_uri",
                }
            )
    if source == "external_scenario_benchmark":
        external_code = payload.get("external_code")
        if not isinstance(external_code, str) or external_code not in _ALLOWED_EXTERNAL_CODES:
            errors.append(
                {
                    "path": str(path),
                    "field": "external_code",
                    "error": "external_code must be TRANSP, TSC, ASTRA, JINTRAC, or PROCESS",
                }
            )
        if not _has_nonempty_str(payload, "reference_artifact_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "reference_artifact_uri",
                    "error": "external scenario benchmarks require reference_artifact_uri",
                }
            )


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare five nonnegative finite errors to positive finite declared bounds, admitting equality; compute no flux or scenario physics."""
    if not isinstance(metrics, dict):
        errors.append({"path": str(path), "field": "metrics", "error": "metrics must be an object"})
        return
    if not isinstance(tolerances, dict):
        errors.append({"path": str(path), "field": "tolerances", "error": "tolerances must be an object"})
        return
    for field in _MAXIMUM_ERROR_METRICS:
        metric = metrics.get(field)
        tolerance = tolerances.get(field)
        if not _is_nonnegative_finite(metric):
            errors.append({"path": str(path), "field": field, "error": "metric must be finite and non-negative"})
            continue
        if not _is_positive_finite(tolerance):
            errors.append({"path": str(path), "field": field, "error": "tolerance must be finite and positive"})
            continue
        if float(metric) > float(tolerance):
            errors.append({"path": str(path), "field": field, "error": "metric exceeds declared tolerance"})


def _valid_machine_metadata(value: object) -> bool:
    """Require five positive finite scenario numbers in declared units, without deriving conversions or physical consistency."""
    return isinstance(value, dict) and all(_is_positive_finite(value.get(field)) for field in _REQUIRED_MACHINE_FIELDS)


def _valid_units(value: object) -> bool:
    """Require original flux V s, voltage V, current A/MA, s, ohm, H, m and dimensionless1 labels; extras are allowed."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require nonblank URL or DOI strings, without URI policy or citation verification."""
    return any(_has_nonempty_str(payload, field) for field in ("reference_url", "reference_doi"))


def _has_nonempty_str(payload: dict[str, object], field: str) -> bool:
    """Accept strings with nonblank stripped content while preserving their original values."""
    value = payload.get(field)
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Admit nonboolean int/float only when binary64 conversion is finite, refusing overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Require an admitted finite number greater than or equal to zero."""
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Require an admitted finite number strictly greater than zero."""
    return _is_finite_number(value) and float(value) > 0.0
