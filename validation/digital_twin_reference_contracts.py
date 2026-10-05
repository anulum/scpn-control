#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin declaration contracts

"""Check original digital twin identity, provenance and declared error bounds."""

from __future__ import annotations

import re
from pathlib import Path

from validation.digital_twin_reference_domains import (
    _has_nonempty_str,
    _has_public_reference,
    _is_nonnegative_finite,
    _is_positive_finite,
    _valid_actuator_metadata,
    _valid_grid_metadata,
    _valid_units,
)

_ALLOWED_SOURCES = {
    "documented_public_reference",
    "measured_discharge_replay",
    "external_integrated_modelling",
}
_ALLOWED_EXTERNAL_CODES = {"ASTRA", "IMAS", "JINTRAC", "TRANSP", "TSC"}
_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)
_MAXIMUM_ERROR_METRICS = (
    "final_avg_temp_relative_error",
    "q_profile_rmse",
    "actuator_lag_abs_error",
    "ids_roundtrip_abs_error",
    "island_mask_f1_error",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Check original schema1.0 identity, format-only SHA, grid/actuator declarations, units and error bounds; return identity without simulating a twin or authenticating references."""
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
                "error": "source must be documented_public_reference, measured_discharge_replay, or external_integrated_modelling",
            }
        )
    _validate_source_provenance(path, payload, errors)
    if not _valid_grid_metadata(payload.get("grid_metadata")):
        errors.append(
            {
                "path": str(path),
                "field": "grid_metadata",
                "error": "grid_metadata must declare finite topology and IDS export metadata",
            }
        )
    if not _valid_actuator_metadata(payload.get("actuator_metadata")):
        errors.append(
            {
                "path": str(path),
                "field": "actuator_metadata",
                "error": "actuator_metadata must declare finite actuator and sensor metadata",
            }
        )
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare digital twin reference units"})
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
    """Require public URL/DOI presence, replay shot/diagnostic presence or original named external integrated model and URI presence, without fetching references."""
    source = payload.get("source")
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "reference",
                "error": "documented public reference artifacts require reference_url or reference_doi",
            }
        )
    if source == "measured_discharge_replay":
        if not _has_nonempty_str(payload, "shot_id"):
            errors.append(
                {"path": str(path), "field": "shot_id", "error": "measured discharge replays require shot_id"}
            )
        if not _has_nonempty_str(payload, "diagnostic_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "diagnostic_uri",
                    "error": "measured discharge replays require diagnostic_uri",
                }
            )
    if source == "external_integrated_modelling":
        external_code = payload.get("external_code")
        if not isinstance(external_code, str) or external_code not in _ALLOWED_EXTERNAL_CODES:
            errors.append(
                {
                    "path": str(path),
                    "field": "external_code",
                    "error": "external_code must be ASTRA, IMAS, JINTRAC, TRANSP, or TSC",
                }
            )
        if not _has_nonempty_str(payload, "reference_artifact_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "reference_artifact_uri",
                    "error": "external integrated modelling artifacts require reference_artifact_uri",
                }
            )


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare five nonnegative finite declared twin errors with positive finite bounds; equality passes without recomputing profiles, lag, IDS or islands."""
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
