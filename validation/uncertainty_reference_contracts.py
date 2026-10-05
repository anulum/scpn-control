#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Uncertainty reference declaration contracts

"""Validate uncertainty-reference declarations without running or authenticating a referenced campaign."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_ALLOWED_SOURCES = {"real_uq_campaign", "documented_public_reference"}
_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)
_REQUIRED_UNITS = {"tau_E": "s", "P_fusion": "MW", "Q": "1", "sigma": "same_as_quantity"}
_MAXIMUM_ERROR_METRICS = ("tau_E_relative_error", "P_fusion_relative_error", "Q_relative_error")
_MINIMUM_SCORE_METRICS = ("percentile_monotonicity_fraction",)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Check one schema1.0 declaration, append field findings and return accepted identity metadata.

    Source labels, nonempty identity strings, exact SHA256 spelling, nonblank
    campaign URI/public citation presence, units/count and declared numerical
    tolerances are validated. Referenced bytes, identity/time authenticity and
    metric execution are outside this declaration-only contract.
    """
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
                "error": "source must be real_uq_campaign or documented_public_reference",
            }
        )
    if source == "real_uq_campaign" and not _has_campaign_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "campaign_artifact_uri",
                "error": "real UQ campaigns require campaign_artifact_uri",
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
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare uncertainty output units"})
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
    """Compare three declared nonnegative errors and one monotonicity fraction with their declared bounds.

    Integers/floats must be finite and binary64-convertible, excluding booleans.
    Error limits are strictly positive; scores/minima are inclusive [0,1].
    Equality passes, extra fields are ignored, and no uncertainty is propagated.
    """
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
    for field in _MINIMUM_SCORE_METRICS:
        metric = metrics.get(field)
        tolerance = tolerances.get(f"{field}_min")
        if not _is_unit_interval(metric):
            errors.append({"path": str(path), "field": field, "error": "score must be finite in [0, 1]"})
            continue
        if not _is_unit_interval(tolerance):
            errors.append(
                {"path": str(path), "field": field, "error": "minimum score tolerance must be finite in [0, 1]"}
            )
            continue
        if float(metric) < float(tolerance):
            errors.append({"path": str(path), "field": field, "error": "score is below declared minimum"})


def _valid_units(value: object) -> bool:
    """Require exact s, MW, dimensionless and same_as_quantity unit labels; permit extra declared units."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require one nonblank public URL/DOI declaration without resolving or authenticating it."""
    return any(
        isinstance(payload.get(field), str) and str(payload[field]).strip()
        for field in ("reference_url", "reference_doi")
    )


def _has_campaign_reference(payload: dict[str, object]) -> bool:
    """Retain campaign URI nonblank presence only, without lexical parsing, resolution or authentication."""
    value = payload.get("campaign_artifact_uri")
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: object) -> TypeGuard[int | float]:
    """Recognize finite binary64-convertible JSON numbers without raising on very large integers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Recognize representable finite errors at or above zero, excluding booleans."""
    return _finite_number(value) and value >= 0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Recognize representable finite error bounds strictly above zero, excluding booleans."""
    return _finite_number(value) and value > 0


def _is_unit_interval(value: object) -> TypeGuard[int | float]:
    """Recognize representable finite monotonicity fractions/minima in inclusive [0,1]."""
    return _finite_number(value) and 0 <= value <= 1
