# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ELM reference artifact validator

"""Inspect ELM identities and declared errors while retaining compact ASCII body hashing."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path

from validation.elm_reference_domains import (
    _artifact_uri_error,
    _has_measured_campaign,
    _has_public_reference,
    _is_nonnegative_finite,
    _is_positive_finite,
    _positive_ordered_pair,
    _strictly_increasing_unit_grid,
    _valid_elm_fraction_range,
    _valid_units,
)

_ALLOWED_SOURCES = {"measured_hmode_campaign", "documented_public_reference"}

_SCHEMA_VERSION = "scpn-control.elm-reference.v1"

_REQUIRED_STR_FIELDS = (
    "source",
    "reference_dataset_id",
    "executed_at",
    "pre_crash_profile_uri",
    "post_crash_profile_uri",
    "event_catalog_uri",
    "rmp_artifact_uri",
    "pre_crash_profile_sha256",
    "post_crash_profile_sha256",
    "event_catalog_sha256",
    "rmp_artifact_sha256",
    "payload_sha256",
)

_REQUIRED_METRICS = (
    "elm_frequency_relative_error",
    "crash_energy_fraction_error",
    "pedestal_temperature_drop_relative_error",
    "pedestal_density_drop_relative_error",
    "rmp_suppression_window_error_s",
    "peak_heat_flux_relative_error",
)

_ARTIFACT_URI_FIELDS = ("pre_crash_profile_uri", "post_crash_profile_uri", "event_catalog_uri", "rmp_artifact_uri")

_SHA256_FIELDS = (
    "pre_crash_profile_sha256",
    "post_crash_profile_sha256",
    "event_catalog_sha256",
    "rmp_artifact_sha256",
    "payload_sha256",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Inspect original twelve text identities, five digest shapes, four lexical URIs, measured/public provenance, grids, independent time windows, energy fractions and six declared error bounds without fetching bytes."""
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
                "error": "source must be measured_hmode_campaign or documented_public_reference",
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
    if payload.get("source") == "measured_hmode_campaign" and not _has_measured_campaign(payload):
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
        observed = str(payload["payload_sha256"])
        if not hmac.compare_digest(observed.lower(), expected):
            errors.append({"path": str(path), "field": "payload_sha256", "error": "canonical payload digest mismatch"})
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare ELM/RMP reference units"})
    if not _strictly_increasing_unit_grid(payload.get("pedestal_rho_grid")):
        errors.append(
            {
                "path": str(path),
                "field": "pedestal_rho_grid",
                "error": "pedestal rho grid must be finite and strictly increasing on [0, 1]",
            }
        )
    if not _positive_ordered_pair(payload.get("event_time_window_s")):
        errors.append(
            {
                "path": str(path),
                "field": "event_time_window_s",
                "error": "event time window must be two finite non-negative increasing times",
            }
        )
    if not _valid_elm_fraction_range(payload.get("elm_energy_fraction_range")):
        errors.append(
            {
                "path": str(path),
                "field": "elm_energy_fraction_range",
                "error": "ELM energy fraction range must lie within Type-I bounds [0.04, 0.15]",
            }
        )
    if not _positive_ordered_pair(payload.get("rmp_suppression_window_s")):
        errors.append(
            {
                "path": str(path),
                "field": "rmp_suppression_window_s",
                "error": "RMP suppression window must be two finite non-negative increasing times",
            }
        )
    _validate_metric_block(path, payload.get("metrics"), payload.get("tolerances"), errors)
    if any(error["path"] == str(path) for error in errors):
        return None
    return {
        "path": str(path),
        "source": str(payload["source"]),
        "reference_dataset_id": str(payload["reference_dataset_id"]),
        "payload_sha256": str(payload["payload_sha256"]).lower(),
    }


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare six nonnegative finite declared errors to positive finite inclusive tolerances without calculating physical metrics."""
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
    """Return original compact sorted ASCII SHA-256 excluding payload_sha256.

    This is a consistency digest supplied by the artifact author, not reference
    authentication. Other fields and Unicode escaping remain in the digest.

    Parameters
    ----------
    payload : dict[str, object]
        JSON-serializable declared artifact body.

    Returns
    -------
    str
        Lowercase hexadecimal digest of the original canonical serialization.
    """
    canonical_payload = dict(payload)
    canonical_payload.pop("payload_sha256", None)
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
