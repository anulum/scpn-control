#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NTM reference artifact validator

"""Check NTM source, identity, units, lexical artifact URIs and original canonical body-hash declarations."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path

from validation.ntm_reference_domains import _validate_domains

_ALLOWED_SOURCES = {"measured_ntm_campaign", "documented_public_reference"}

_SCHEMA_VERSION = "scpn-control.ntm-reference.v1"

_REQUIRED_STR_FIELDS = (
    "source",
    "reference_dataset_id",
    "executed_at",
    "q_profile_uri",
    "rational_surface_uri",
    "island_width_trace_uri",
    "eccd_alignment_uri",
    "q_profile_sha256",
    "rational_surface_sha256",
    "island_width_trace_sha256",
    "eccd_alignment_sha256",
    "payload_sha256",
)

_REQUIRED_UNITS = {
    "island_width": "m",
    "time": "s",
    "current": "A",
    "q": "1",
    "rho": "1",
    "power": "W",
    "deposition_width": "m",
    "island_growth_rate": "m/s",
}

_ARTIFACT_URI_FIELDS = ("q_profile_uri", "rational_surface_uri", "island_width_trace_uri", "eccd_alignment_uri")

_SHA256_FIELDS = (
    "q_profile_sha256",
    "rational_surface_sha256",
    "island_width_trace_sha256",
    "eccd_alignment_sha256",
    "payload_sha256",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Check schema, source, identity, URI, units, digests and delegated declared domains; return accepted identity metadata only, without experiment or referenced-byte authentication."""
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
                "error": "source must be measured_ntm_campaign or documented_public_reference",
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
    if payload.get("source") == "measured_ntm_campaign" and not _has_measured_campaign(payload):
        errors.append(
            {
                "path": str(path),
                "field": "campaign",
                "error": "measured NTM campaigns require machine and shot_id or campaign_id",
            }
        )
    digest = payload.get("payload_sha256")
    if isinstance(digest, str) and _SHA256_RE.fullmatch(digest):
        expected = canonical_artifact_sha256(payload)
        observed = str(payload["payload_sha256"])
        if not hmac.compare_digest(observed.lower(), expected):
            errors.append({"path": str(path), "field": "payload_sha256", "error": "canonical payload digest mismatch"})
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare NTM reference units"})
    _validate_domains(path, payload, errors)
    if any(error["path"] == str(path) for error in errors):
        return None
    return {
        "path": str(path),
        "source": str(payload["source"]),
        "reference_dataset_id": str(payload["reference_dataset_id"]),
        "payload_sha256": str(payload["payload_sha256"]).lower(),
    }


def canonical_artifact_sha256(payload: dict[str, object]) -> str:
    """Hash a copy excluding payload_sha256 with original sorted ASCII compact JSON. Body consistency is not authenticity; unsupported public inputs propagate serialization failures."""
    canonical_payload = dict(payload)
    canonical_payload.pop("payload_sha256", None)
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_uri_error(value: object) -> str | None:
    """Retain lexical nonblank/no-NUL URI rule: admitted http/https/doi/s3/gs prefixes or relative paths without parent segments. No URL parsing, resolution or retrieval."""
    if not isinstance(value, str) or not value.strip():
        return "artifact URI must be a non-empty string"
    ref = value.strip()
    if "\x00" in ref:
        return "artifact URI must not contain NUL bytes"
    if ref.startswith(("http://", "https://", "doi:", "s3://", "gs://")):
        return None
    artifact_path = Path(ref)
    if artifact_path.is_absolute():
        return "artifact URI must be relative or an admitted external reference URI"
    if any(part == ".." for part in artifact_path.parts):
        return "artifact URI must not contain traversal"
    return None


def _valid_units(value: object) -> bool:
    """Require original island/deposition m, time s, current A, q/rho dimensionless, power W and growth m/s labels; allow extras."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require nonblank URL or DOI presence without resolution or producer authentication."""
    for field in ("reference_url", "reference_doi"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _has_measured_campaign(payload: dict[str, object]) -> bool:
    """Require nonblank machine and shot_id or campaign_id strings without facility authentication."""
    machine = payload.get("machine")
    shot_id = payload.get("shot_id")
    campaign_id = payload.get("campaign_id")
    return (
        isinstance(machine, str)
        and bool(machine.strip())
        and any(isinstance(value, str) and bool(value.strip()) for value in (shot_id, campaign_id))
    )
