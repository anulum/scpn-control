#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Blob transport declaration contracts

"""Check original SOL blob identity, body consistency, geometry and declared bounds."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path

from validation.blob_transport_reference_domains import (
    _artifact_uri_error,
    _has_measured_campaign,
    _has_public_reference,
    _is_nonnegative_finite,
    _is_positive_finite,
    _positive_ordered_pair,
    _strictly_increasing_nonnegative,
    _valid_units,
)

_ALLOWED_SOURCES = {"measured_probe_campaign", "documented_public_reference"}
_SCHEMA_VERSION = "scpn-control.blob-transport-reference.v1"
_REQUIRED_STR_FIELDS = (
    "source",
    "reference_dataset_id",
    "executed_at",
    "reference_artifact_uri",
    "profile_artifact_uri",
    "detector_artifact_uri",
    "reference_artifact_sha256",
    "profile_artifact_sha256",
    "detector_artifact_sha256",
    "payload_sha256",
)
_REQUIRED_METRICS = (
    "radial_velocity_rmse_m_s",
    "density_profile_relative_l2",
    "wall_flux_relative_error",
    "event_duration_relative_error",
    "event_size_relative_error",
)
_ARTIFACT_URI_FIELDS = ("reference_artifact_uri", "profile_artifact_uri", "detector_artifact_uri")
_SHA256_FIELDS = (
    "reference_artifact_sha256",
    "profile_artifact_sha256",
    "detector_artifact_sha256",
    "payload_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Check original named schema, lexical references, format SHAs, canonical body consistency and declared SOL domains without authenticating producers or computing transport."""
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "artifact root must be an object"})
        return None
    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append(
            {"path": str(path), "field": "schema_version", "error": f"schema_version must be '{_SCHEMA_VERSION}'"}
        )
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": str(path), "field": field, "error": "field must be a non-empty string"})
    for field in _SHA256_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and not _SHA256_RE.fullmatch(value):
            errors.append({"path": str(path), "field": field, "error": "field must be a SHA-256 hex digest"})
    for field in _ARTIFACT_URI_FIELDS:
        error = _artifact_uri_error(payload.get(field))
        if error is not None:
            errors.append({"path": str(path), "field": field, "error": error})
    source = payload.get("source")
    if not isinstance(source, str) or source not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": str(path),
                "field": "source",
                "error": "source must be measured_probe_campaign or documented_public_reference",
            }
        )
    if payload.get("source") == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": str(path),
                "field": "reference",
                "error": "documented public reference artifacts require reference_url or reference_doi",
            }
        )
    if payload.get("source") == "measured_probe_campaign" and not _has_measured_campaign(payload):
        errors.append(
            {
                "path": str(path),
                "field": "campaign",
                "error": "measured campaigns require machine and shot_id or campaign_id",
            }
        )
    digest = payload.get("payload_sha256")
    if isinstance(digest, str) and _SHA256_RE.fullmatch(digest):
        expected = canonical_artifact_sha256(payload)
        if not hmac.compare_digest(digest.lower(), expected):
            errors.append({"path": str(path), "field": "payload_sha256", "error": "canonical payload digest mismatch"})
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare SOL blob reference units"})
    if not _strictly_increasing_nonnegative(payload.get("separatrix_to_wall_coordinates_m")):
        errors.append(
            {
                "path": str(path),
                "field": "separatrix_to_wall_coordinates_m",
                "error": "coordinates must be finite non-negative and strictly increasing",
            }
        )
    if not _positive_ordered_pair(payload.get("detector_time_domain_s")):
        errors.append(
            {
                "path": str(path),
                "field": "detector_time_domain_s",
                "error": "detector time domain must be two finite non-negative increasing times",
            }
        )
    if not _positive_ordered_pair(payload.get("blob_size_range_m")):
        errors.append(
            {
                "path": str(path),
                "field": "blob_size_range_m",
                "error": "blob size range must be two positive increasing sizes",
            }
        )
    _validate_geometry(path, payload.get("magnetic_geometry"), errors)
    _validate_metric_block(path, payload.get("metrics"), payload.get("tolerances"), errors)
    if any(error["path"] == str(path) for error in errors):
        return None
    return {
        "path": str(path),
        "source": str(payload["source"]),
        "reference_dataset_id": str(payload["reference_dataset_id"]),
        "payload_sha256": str(payload["payload_sha256"]).lower(),
    }


def _validate_geometry(path: Path, geometry: object, errors: list[dict[str, object]]) -> None:
    """Require five original positive finite R0/B0/parallel-length/Te/density declarations without conversion or magnetic reconstruction."""
    if not isinstance(geometry, dict):
        errors.append({"path": str(path), "field": "magnetic_geometry", "error": "magnetic_geometry must be an object"})
        return
    for field in ("R0_m", "B0_T", "L_parallel_m", "Te_eV", "density_m3"):
        value = geometry.get(field)
        if not _is_positive_finite(value):
            errors.append(
                {"path": str(path), "field": f"magnetic_geometry.{field}", "error": "field must be finite and positive"}
            )


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare five finite nonnegative declared velocity/profile/wall/duration/size errors with positive finite inclusive bounds."""
    if not isinstance(metrics, dict):
        errors.append({"path": str(path), "field": "metrics", "error": "metrics must be an object"})
        return
    if not isinstance(tolerances, dict):
        errors.append({"path": str(path), "field": "tolerances", "error": "tolerances must be an object"})
        return
    for field in _REQUIRED_METRICS:
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


def canonical_artifact_sha256(payload: dict[str, object]) -> str:
    """Return SHA256 of original compact sorted ASCII-escaped JSON excluding payload_sha256.

    This is caller-recomputable body consistency, without producer or reference-byte
    authentication. Original JSON serialization and errors remain unchanged.

    Examples
    --------
    >>> canonical_artifact_sha256({"value": 1}) == canonical_artifact_sha256({"value": 1, "payload_sha256": "ignored"})
    True
    """
    canonical_payload = dict(payload)
    canonical_payload.pop("payload_sha256", None)
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
