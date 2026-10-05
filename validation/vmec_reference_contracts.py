#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — VMEC reference declaration contracts

"""Check VMEC reference declarations without running or authenticating a stellarator equilibrium."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

from validation.reference_uri import reference_artifact_uri_error

_ALLOWED_SOURCES = {"documented_public_reference", "real_vmec_run"}
_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)
_REQUIRED_UNITS = {
    "R_mn": "m",
    "Z_mn": "m",
    "B_mn": "T",
    "pressure": "Pa",
    "iota": "1",
}
_MAXIMUM_ERROR_METRICS = (
    "surface_R_rmse_m",
    "surface_Z_rmse_m",
    "iota_rmse",
    "force_residual_relative",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Check schema1.0 source/identity/SHA/count/units/Fourier and declared errors; return accepted identity metadata only. Referenced bytes, execution, identity and equilibrium computations are unauthenticated."""
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
                "error": "source must be documented_public_reference or real_vmec_run",
            }
        )
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "reference",
                "error": "documented public reference artifacts require reference_url or reference_doi",
            }
        )
    if source == "real_vmec_run":
        uri_error = reference_artifact_uri_error(payload.get("vmec_artifact_uri"), "vmec_artifact_uri")
        if uri_error is not None:
            errors.append({"path": str(path), "field": "vmec_artifact_uri", "error": uri_error})
    if not _valid_fourier_truncation(payload.get("fourier_truncation")):
        errors.append(
            {
                "path": str(path),
                "field": "fourier_truncation",
                "error": "fourier_truncation must declare positive m_pol and n_fp plus non-negative n_tor",
            }
        )
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare VMEC-lite reference units"})
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


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare four declared nonnegative errors with strictly positive finite bounds; equality passes, no equilibrium or force residual is computed."""
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


def _valid_fourier_truncation(value: object) -> bool:
    """Require positive integer m_pol/n_fp and nonnegative integer n_tor, excluding booleans; no coefficient arrays are inspected."""
    if not isinstance(value, dict):
        return False
    return (
        _is_positive_int(value.get("m_pol"))
        and _is_nonnegative_int(value.get("n_tor"))
        and _is_positive_int(value.get("n_fp"))
    )


def _valid_units(value: object) -> bool:
    """Require exact m, T, Pa and dimensionless labels for declared Fourier/pressure/iota outputs; allow extra fields."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require nonblank URL or DOI presence without resolution, referenced-byte hashing or citation authentication."""
    return any(
        isinstance(payload.get(field), str) and str(payload[field]).strip()
        for field in ("reference_url", "reference_doi")
    )


def _is_positive_int(value: object) -> bool:
    """Recognize positive JSON integers, excluding booleans."""
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _is_nonnegative_int(value: object) -> bool:
    """Recognize nonnegative JSON integers, excluding booleans."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_number(value: object) -> TypeGuard[int | float]:
    """Recognize finite binary64-convertible JSON numbers without raising on huge integers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Recognize representable finite nonnegative errors, excluding booleans."""
    return _finite_number(value) and value >= 0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Recognize representable strictly positive finite error bounds, excluding booleans."""
    return _finite_number(value) and value > 0
