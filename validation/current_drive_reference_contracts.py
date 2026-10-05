# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Current-drive reference artifact validator

"""Inspect original current-drive reference identity and provenance declarations without source authentication.

A passing declaration authenticates neither reference bytes nor external/facility evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

from validation.current_drive_reference_domains import (
    _has_nonempty_str,
    _has_public_reference,
    _valid_source_metadata,
    _valid_units,
    _validate_metric_block,
)

_ALLOWED_SOURCES = {
    "documented_public_reference",
    "ray_tracing_benchmark",
    "fokker_planck_benchmark",
    "measured_deposition_replay",
}


_ALLOWED_EXTERNAL_CODES = {"TORBEAM", "GENRAY", "CQL3D", "NUBEAM", "TRANSP", "LUKE"}


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
    """Check one declaration and append field errors; return identity only when its path has no errors."""
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
                "error": "source must be documented_public_reference, ray_tracing_benchmark, fokker_planck_benchmark, or measured_deposition_replay",
            }
        )
    _validate_source_provenance(path, payload, errors)
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare current-drive contracts"})
    if not _valid_source_metadata(payload.get("source_metadata")):
        errors.append(
            {
                "path": str(path),
                "field": "source_metadata",
                "error": "source_metadata must declare finite positive source and grid parameters",
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
    """Check source-specific nonblank provenance fields without fetching their targets."""
    source = payload.get("source")
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "reference",
                "error": "documented public references require reference_url or reference_doi",
            }
        )
    if source == "measured_deposition_replay":
        if not _has_nonempty_str(payload, "shot_id"):
            errors.append(
                {"path": str(path), "field": "shot_id", "error": "measured deposition replays require shot_id"}
            )
        if not _has_nonempty_str(payload, "diagnostic_uri"):
            errors.append(
                {
                    "path": str(path),
                    "field": "diagnostic_uri",
                    "error": "measured deposition replays require diagnostic_uri",
                }
            )
    if isinstance(source, str) and source in {"ray_tracing_benchmark", "fokker_planck_benchmark"}:
        external_code = payload.get("external_code")
        if not isinstance(external_code, str) or external_code not in _ALLOWED_EXTERNAL_CODES:
            errors.append(
                {
                    "path": str(path),
                    "field": "external_code",
                    "error": "external_code must be TORBEAM, GENRAY, CQL3D, NUBEAM, TRANSP, or LUKE",
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
