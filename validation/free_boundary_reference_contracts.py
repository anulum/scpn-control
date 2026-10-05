#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary declaration contracts

"""Check original free-boundary identity, equilibrium counts/timing and declared tracking bounds."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_ALLOWED_SOURCES = {"documented_public_reference", "measured_free_boundary_replay", "external_equilibrium_benchmark"}
_ALLOWED_EXTERNAL_CODES = {"EFIT", "P-EFIT", "CREATE-NL", "TSC"}
_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)
_REQUIRED_UNITS = {
    "position": "m",
    "flux": "Wb/rad",
    "current": "MA",
    "time": "s",
    "tracking_error": "1",
}
_MAXIMUM_ERROR_METRICS = (
    "shape_rms_abs_error",
    "x_point_position_abs_error_m",
    "x_point_flux_abs_error",
    "divertor_rms_abs_error",
    "coil_current_relative_error",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Check original schema1.0 identity, format-only SHA, source presence, equilibrium metadata, units and declared tracking bounds; authenticate no reference bytes or physical replay."""
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
                "error": "source must be documented_public_reference, measured_free_boundary_replay, or external_equilibrium_benchmark",
            }
        )
    _validate_source_provenance(path, payload, errors)
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare free-boundary SI contracts"})
    if not _valid_equilibrium_metadata(payload.get("equilibrium_metadata")):
        errors.append(
            {
                "path": str(path),
                "field": "equilibrium_metadata",
                "error": "equilibrium_metadata must declare finite coil, objective, and timing counts",
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
    """Require original URL/DOI or measured shot/diagnostic or named equilibrium code/URI presence without parsing or fetching."""
    source = payload.get("source")
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "reference",
                "error": "documented public references require reference_url or reference_doi",
            }
        )
    if source == "measured_free_boundary_replay":
        if not _has_nonempty_str(payload, "shot_id"):
            errors.append(
                {"path": str(path), "field": "shot_id", "error": "measured free-boundary replays require shot_id"}
            )
        if not _has_nonempty_str(payload, "diagnostic_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "diagnostic_uri",
                    "error": "measured free-boundary replays require diagnostic_uri",
                }
            )
    if source == "external_equilibrium_benchmark":
        external_code = payload.get("external_code")
        if not isinstance(external_code, str) or external_code not in _ALLOWED_EXTERNAL_CODES:
            errors.append(
                {
                    "path": str(path),
                    "field": "external_code",
                    "error": "external_code must be EFIT, P-EFIT, CREATE-NL, or TSC",
                }
            )
        if not _has_nonempty_str(payload, "reference_artifact_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "reference_artifact_uri",
                    "error": "external benchmarks require reference_artifact_uri",
                }
            )


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare five finite nonnegative declared tracking errors with positive finite bounds; equality passes without recomputing shape, flux, divertor or coil tracking."""
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


def _valid_equilibrium_metadata(value: object) -> bool:
    """Require three uncapped positive nonboolean coil/boundary/divertor integer counts and positive finite control interval/slew declarations without cross-count relations."""
    if not isinstance(value, dict):
        return False
    integer_fields = ("coil_count", "boundary_point_count", "divertor_point_count")
    if any(isinstance(v := value.get(field), bool) or not isinstance(v, int) or v <= 0 for field in integer_fields):
        return False
    return all(_is_positive_finite(value.get(field)) for field in ("control_dt_s", "coil_slew_limit_MA_s"))


def _valid_units(value: object) -> bool:
    """Require original m, Wb/rad, MA, s and dimensionless tracking labels; extras pass without conversion."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require original nonblank URL or DOI presence without reference verification."""
    return any(_has_nonempty_str(payload, field) for field in ("reference_url", "reference_doi"))


def _has_nonempty_str(payload: dict[str, object], field: str) -> bool:
    """Check nonblank original strings without parsing references."""
    value = payload.get(field)
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Admit nonboolean int/float with finite binary64 conversion, refusing overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Admit finite numbers at least zero."""
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Admit finite numbers strictly above zero."""
    return _is_finite_number(value) and float(value) > 0.0
