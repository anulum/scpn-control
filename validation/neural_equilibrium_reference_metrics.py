#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural equilibrium reference artifact validator

"""Declared neural-reference metric, shape, unit and URI preconditions; no array or model execution."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_METRICS = (
    "psi_rmse_Wb",
    "pressure_rmse_Pa",
    "q_profile_rmse",
    "boundary_rmse_m",
    "axis_position_error_m",
)
_REQUIRED_UNITS = {
    "psi": "Wb/rad",
    "pressure": "Pa",
    "q_profile": "1",
    "boundary": "m",
}


def _validate_metric_block(
    path: str,
    metrics: object,
    tolerances: object,
    errors: list[dict[str, object]],
) -> None:
    """Append shape/type/range findings for five declared scalar errors and positive tolerances.

    Values are compared in their declared units: flux Wb/rad despite the legacy
    Wb field name, pressure Pa, dimensionless q, and boundary/axis metres. No
    actual error computation or interpretation of metric algorithms is implied.
    """
    if not isinstance(metrics, dict):
        errors.append({"path": path, "field": "metrics", "error": "metrics must be an object"})
        return
    if not isinstance(tolerances, dict):
        errors.append({"path": path, "field": "tolerances", "error": "tolerances must be an object"})
        return
    for field in _REQUIRED_METRICS:
        metric = metrics.get(field)
        tolerance = tolerances.get(field)
        if not _is_nonnegative_finite(metric):
            errors.append({"path": path, "field": field, "error": "metric must be finite and non-negative"})
            continue
        if not _is_positive_finite(tolerance):
            errors.append({"path": path, "field": field, "error": "tolerance must be finite and positive"})
            continue
        if float(metric) > float(tolerance):
            errors.append({"path": path, "field": field, "error": "metric exceeds declared tolerance"})


def _valid_grid_shape(value: object) -> bool:
    """Require exactly two positive nonboolean integer grid counts; no array shape is read."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    return all(not isinstance(item, bool) and isinstance(item, int) and item > 0 for item in value)


def _valid_units(value: object) -> bool:
    """Require the four exact unit labels while permitting unconsumed dictionary entries."""
    if not isinstance(value, dict):
        return False
    return all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require a nonblank string URL or DOI declaration without parsing or fetching it."""
    for field in ("reference_url", "reference_doi"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _artifact_uri_error(value: object) -> str | None:
    """Check historical POSIX relative or admitted-prefix URI spelling without resolving data.

    Blank/nonstring/NUL, absolute POSIX paths and parent components are refused.
    Literal http/https/doi/s3/gs prefixes pass without URL parsing or downloads.
    This uses host Path semantics, not Windows-drive validation on POSIX.
    """
    if not isinstance(value, str) or not value.strip():
        return "artefact URI must be a non-empty string"
    ref = value.strip()
    if "\x00" in ref:
        return "artefact URI must not contain NUL bytes"
    if ref.startswith(("http://", "https://", "doi:", "s3://", "gs://")):
        return None
    # The declaration is a document, not a path on this machine: POSIX rules on every platform.
    path = PurePosixPath(ref)
    if path.is_absolute():
        return "artefact URI must be relative or an admitted external reference URI"
    if any(part == ".." for part in path.parts):
        return "artefact URI must not contain traversal"
    return None


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Require a nonboolean finite representable number at least zero.

    Unrepresentable arbitrary-size integer conversion is a refused value, not an
    overflow escaping the public declaration validator. No unit conversion occurs.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value)) and value >= 0.0
    except OverflowError:
        return False


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Require a nonboolean finite representable number greater than zero.

    Unrepresentable arbitrary-size integer conversion is a refused value, not an
    overflow escaping the public declaration validator. No unit conversion occurs.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value)) and value > 0.0
    except OverflowError:
        return False
