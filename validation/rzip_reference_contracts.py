#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP declaration contracts


"""Check original RZIP declaration identity, artifact URI policy and numeric domains."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

from validation.reference_uri import reference_artifact_uri_error

_ALLOWED_SOURCES = {"documented_public_reference", "external_code_benchmark", "measured_discharge"}
_ALLOWED_EXTERNAL_CODES = {"CREATE-L", "CREATE-NL", "TSC", "reference_rzip"}
_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)
_REQUIRED_POSITIVE_PARAMETERS = (
    "major_radius_m",
    "minor_radius_m",
    "elongation",
    "plasma_current_A",
    "toroidal_field_T",
    "wall_time_constant_s",
)
_REQUIRED_UNITS = {
    "vertical_displacement": "m",
    "growth_rate": "s^-1",
    "growth_time": "ms",
    "coil_current": "A",
    "time": "s",
}
_MAXIMUM_ERROR_METRICS = (
    "growth_rate_relative_error",
    "vertical_displacement_rmse_m",
    "closed_loop_pole_real_abs_error",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Check original schema1.0, identity/source presence, format-only digest, declared RZIP inputs, units, count and error bounds; return metadata without physics or referenced-byte authentication."""
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "artifact root must be an object"})
        return None
    if payload.get("schema_version") != "1.0":
        errors.append({"path": str(path), "field": "schema_version", "error": "schema_version must be '1.0'"})
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
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
                "error": "source must be documented_public_reference, external_code_benchmark, or measured_discharge",
            }
        )
    _validate_source_provenance(path, payload, errors)
    if not _valid_physical_parameters(payload.get("physical_parameters")):
        errors.append(
            {
                "path": str(path),
                "field": "physical_parameters",
                "error": "physical_parameters must declare finite RZIP model inputs",
            }
        )
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare RZIP reference units"})
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
    """Require public citation presence, external named code and original shared artifact-URI policy, or measured shot/diagnostic presence; no URI retrieval or producer authentication."""
    source = payload.get("source")
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "reference",
                "error": "documented public reference artifacts require reference_url or reference_doi",
            }
        )
    if source == "external_code_benchmark":
        external_code = payload.get("external_code")
        if not isinstance(external_code, str) or external_code not in _ALLOWED_EXTERNAL_CODES:
            errors.append(
                {
                    "path": str(path),
                    "field": "external_code",
                    "error": "external_code must be CREATE-L, CREATE-NL, TSC, or reference_rzip",
                }
            )
        uri_error = reference_artifact_uri_error(payload.get("reference_artifact_uri"), "reference_artifact_uri")
        if uri_error is not None:
            errors.append({"path": str(path), "field": "reference_artifact_uri", "error": uri_error})
    if source == "measured_discharge":
        if not _has_nonempty_str(payload, "shot_id"):
            errors.append(
                {"path": str(path), "field": "shot_id", "error": "measured discharge references require shot_id"}
            )
        if not _has_nonempty_str(payload, "diagnostic_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "diagnostic_uri",
                    "error": "measured discharge references require diagnostic_uri",
                }
            )


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare three nonnegative finite growth/displacement/pole errors to positive finite bounds, admitting equality without RZIP dynamics recomputation."""
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


def _valid_physical_parameters(value: object) -> bool:
    """Require original six positive finite inputs and signed finite vertical field index; do not impose geometric ordering or negative-only index."""
    if not isinstance(value, dict):
        return False
    if not all(_is_positive_finite(value.get(field)) for field in _REQUIRED_POSITIVE_PARAMETERS):
        return False
    return _is_finite_number(value.get("vertical_field_index"))


def _valid_units(value: object) -> bool:
    """Require original vertical m, growth s^-1, growth time ms, coil A and time s labels; admit extras without conversion."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require nonblank URL or DOI strings without citation verification or external artifact-URI policy."""
    return any(_has_nonempty_str(payload, field) for field in ("reference_url", "reference_doi"))


def _has_nonempty_str(payload: dict[str, object], field: str) -> bool:
    """Accept strings with nonblank stripped content, retaining original values."""
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
