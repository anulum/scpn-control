# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOC reference artifact validator

"""Inspect original SOC identities, declared provenance and bounded error metadata."""

from __future__ import annotations

import re
from pathlib import Path

from validation.soc_reference_domains import (
    _has_nonempty_str,
    _has_public_reference,
    _is_nonnegative_finite,
    _is_positive_finite,
    _valid_lattice_metadata,
    _valid_learning_metadata,
    _valid_units,
)

_ALLOWED_SOURCES = {
    "documented_public_reference",
    "measured_turbulence_replay",
    "external_gyrokinetic_reference",
}

_ALLOWED_EXTERNAL_CODES = {"CGYRO", "GENE", "GS2", "TGLF", "QuaLiKiz"}

_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)

_MAXIMUM_ERROR_METRICS = (
    "mean_turbulence_relative_error",
    "flow_mean_abs_error",
    "policy_action_accuracy_error",
    "reward_relative_error",
    "core_temperature_relative_error",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Validate schema1.0, six identities, exact hex shape, source, lattice/learning declarations, units, counts and errors without loading referenced data."""
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
                "error": "source must be documented_public_reference, measured_turbulence_replay, or external_gyrokinetic_reference",
            }
        )
    _validate_source_provenance(path, payload, errors)
    if not _valid_lattice_metadata(payload.get("lattice_metadata")):
        errors.append(
            {
                "path": str(path),
                "field": "lattice_metadata",
                "error": "lattice_metadata must declare finite SOC runtime metadata",
            }
        )
    if not _valid_learning_metadata(payload.get("learning_metadata")):
        errors.append(
            {
                "path": str(path),
                "field": "learning_metadata",
                "error": "learning_metadata must declare finite Q-learning metadata",
            }
        )
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare SOC reference units"})
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
    """Require original public citation, measured shot/diagnostic text or admitted external code/artifact text without fetching or URI parsing."""
    source = payload.get("source")
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "reference",
                "error": "documented public reference artifacts require reference_url or reference_doi",
            }
        )
    if source == "measured_turbulence_replay":
        if not _has_nonempty_str(payload, "shot_id"):
            errors.append(
                {"path": str(path), "field": "shot_id", "error": "measured turbulence replays require shot_id"}
            )
        if not _has_nonempty_str(payload, "diagnostic_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "diagnostic_uri",
                    "error": "measured turbulence replays require diagnostic_uri",
                }
            )
    if source == "external_gyrokinetic_reference":
        external_code = payload.get("external_code")
        if not isinstance(external_code, str) or external_code not in _ALLOWED_EXTERNAL_CODES:
            errors.append(
                {
                    "path": str(path),
                    "field": "external_code",
                    "error": "external_code must be CGYRO, GENE, GS2, TGLF, or QuaLiKiz",
                }
            )
        if not _has_nonempty_str(payload, "reference_artifact_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "reference_artifact_uri",
                    "error": "external gyrokinetic references require reference_artifact_uri",
                }
            )


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare five finite nonnegative declared errors to positive inclusive declared bounds without recomputing physical metrics."""
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
