# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural turbulence reference artifact validator

"""Validate original turbulence identities, feature order, units and declared error/score bounds."""

from __future__ import annotations

import re
from pathlib import Path

from validation.neural_turbulence_reference_domains import (
    _has_gk_campaign_reference,
    _has_public_reference,
    _is_nonnegative_finite,
    _is_positive_finite,
    _is_unit_interval,
    _valid_units,
)

_ALLOWED_SOURCES = {"real_gk_campaign", "documented_public_reference"}

_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "trained_weights_sha256",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)

_REQUIRED_FEATURE_SCHEMA = (
    "R_LTi",
    "R_LTe",
    "R_Ln",
    "q",
    "s_hat",
    "alpha_MHD",
    "Ti_Te",
    "nu_star",
    "Z_eff",
    "epsilon",
)


_MAXIMUM_ERROR_METRICS = (
    "Q_i_rmse_gB",
    "Q_e_rmse_gB",
    "Gamma_e_rmse_gB",
    "flux_relative_mae",
)

_MINIMUM_SCORE_METRICS = ("critical_gradient_accuracy",)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Inspect schema1.0, seven nonblank identities, two exact hex shapes, feature/unit/count declarations and inclusive bounds; no referenced bytes or metrics are authenticated."""
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "artifact root must be an object"})
        return None
    if payload.get("schema_version") != "1.0":
        errors.append({"path": str(path), "field": "schema_version", "error": "schema_version must be '1.0'"})
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": str(path), "field": field, "error": "field must be a non-empty string"})
    for field in ("trained_weights_sha256", "reference_artifact_sha256"):
        value = payload.get(field)
        if isinstance(value, str) and not _SHA256_RE.fullmatch(value):
            errors.append({"path": str(path), "field": field, "error": "field must be a SHA-256 hex digest"})
    source = payload.get("source")
    if not isinstance(source, str) or source not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": str(path),
                "field": "source",
                "error": "source must be real_gk_campaign or documented_public_reference",
            }
        )
    if source == "real_gk_campaign" and not _has_gk_campaign_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "campaign_artifact_uri",
                "error": "real GK campaign artifacts require campaign_artifact_uri",
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
    if payload.get("feature_schema") != list(_REQUIRED_FEATURE_SCHEMA):
        errors.append(
            {"path": str(path), "field": "feature_schema", "error": "feature_schema must match neural turbulence order"}
        )
    if not _valid_units(payload.get("units")):
        errors.append(
            {"path": str(path), "field": "units", "error": "units must declare neural turbulence target units"}
        )
    count = payload.get("reference_sample_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        errors.append(
            {"path": str(path), "field": "reference_sample_count", "error": "field must be a positive integer"}
        )
    _validate_metric_block(path, payload.get("metrics"), payload.get("tolerances"), errors)
    if any(error["path"] == str(path) for error in errors):
        return None
    return {
        "path": str(path),
        "source": str(payload["source"]),
        "model_id": str(payload["model_id"]),
        "model_version": str(payload["model_version"]),
        "reference_dataset_id": str(payload["reference_dataset_id"]),
        "reference_sample_count": int(payload["reference_sample_count"]),
    }


def _validate_metric_block(
    path: Path,
    metrics: object,
    tolerances: object,
    errors: list[dict[str, object]],
) -> None:
    """Require four nonnegative errors within positive inclusive bounds and a zero-to-one score at least its declared minimum."""
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
