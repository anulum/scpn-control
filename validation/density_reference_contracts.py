# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density reference artifact validator

"""Inspect density declaration identities and source-specific metadata with unchanged scientific acceptance rules.

Referenced digests and source labels remain declarations, not authenticated evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

from validation.density_reference_domains import (
    _has_nonempty_str,
    _has_public_reference,
    _valid_actuator_metadata,
    _valid_radial_grid,
    _valid_units,
    _validate_metric_block,
)
from validation.reference_uri import reference_artifact_uri_error

_ALLOWED_SOURCES = {"documented_public_reference", "measured_fuelling_campaign", "external_integrated_modelling"}


_ALLOWED_EXTERNAL_CODES = {"ASTRA", "TRANSP", "JINTRAC", "TGLF"}


_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Check one decoded declaration and append findings; return its declared identity only when valid."""
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
                "error": "source must be documented_public_reference, measured_fuelling_campaign, or external_integrated_modelling",
            }
        )
    _validate_source_provenance(path, payload, errors)
    if not _valid_radial_grid(payload.get("radial_grid")):
        errors.append(
            {
                "path": str(path),
                "field": "radial_grid",
                "error": "radial_grid must declare positive density-model geometry",
            }
        )
    if not _valid_actuator_metadata(payload.get("actuator_metadata")):
        errors.append(
            {
                "path": str(path),
                "field": "actuator_metadata",
                "error": "actuator_metadata must declare finite fuelling and exhaust settings",
            }
        )
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare density reference units"})
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
    """Check source-specific strings and URI syntax without fetching or authenticating references."""
    source = payload.get("source")
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "reference",
                "error": "documented public reference artifacts require reference_url or reference_doi",
            }
        )
    if source == "measured_fuelling_campaign":
        if not _has_nonempty_str(payload, "shot_id"):
            errors.append(
                {"path": str(path), "field": "shot_id", "error": "measured fuelling campaigns require shot_id"}
            )
        if not _has_nonempty_str(payload, "diagnostic_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "diagnostic_uri",
                    "error": "measured fuelling campaigns require diagnostic_uri",
                }
            )
    if source == "external_integrated_modelling":
        external_code = payload.get("external_code")
        if not isinstance(external_code, str) or external_code not in _ALLOWED_EXTERNAL_CODES:
            errors.append(
                {
                    "path": str(path),
                    "field": "external_code",
                    "error": "external_code must be ASTRA, TRANSP, JINTRAC, or TGLF",
                }
            )
        uri_error = reference_artifact_uri_error(payload.get("reference_artifact_uri"), "reference_artifact_uri")
        if uri_error is not None:
            errors.append({"path": str(path), "field": "reference_artifact_uri", "error": uri_error})
