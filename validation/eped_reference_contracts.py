#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EPED reference artifact validator

"""Check EPED identity, provenance, URI, units and canonical payload declarations."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path, PurePosixPath

from validation.eped_reference_geometry import _validate_geometry

_ALLOWED_SOURCES = {"measured_pedestal_database", "documented_public_reference"}

_SCHEMA_VERSION = "scpn-control.eped-reference.v1"

_REQUIRED_STR_FIELDS = (
    "source",
    "reference_dataset_id",
    "executed_at",
    "pedestal_profile_uri",
    "eped_prediction_uri",
    "bootstrap_current_uri",
    "peeling_ballooning_uri",
    "pedestal_profile_sha256",
    "eped_prediction_sha256",
    "bootstrap_current_sha256",
    "peeling_ballooning_sha256",
    "payload_sha256",
)

_REQUIRED_UNITS = {
    "width": "psi_N",
    "pressure": "Pa",
    "temperature": "eV",
    "density": "m^-3",
    "current": "A",
    "beta": "1",
    "shape": "1",
}

_ARTIFACT_URI_FIELDS = (
    "pedestal_profile_uri",
    "eped_prediction_uri",
    "bootstrap_current_uri",
    "peeling_ballooning_uri",
)

_SHA256_FIELDS = (
    "pedestal_profile_sha256",
    "eped_prediction_sha256",
    "bootstrap_current_sha256",
    "peeling_ballooning_sha256",
    "payload_sha256",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Validate schema, source, digests, URI, units and delegated declared numbers; return identity metadata only. No referenced bytes or science are authenticated."""
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
                "error": "source must be measured_pedestal_database or documented_public_reference",
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
    if payload.get("source") == "measured_pedestal_database" and not _has_measured_campaign(payload):
        errors.append(
            {
                "path": str(path),
                "field": "campaign",
                "error": "measured pedestal databases require machine and shot_id or campaign_id",
            }
        )
    digest = payload.get("payload_sha256")
    if isinstance(digest, str) and _SHA256_RE.fullmatch(digest):
        expected = canonical_artifact_sha256(payload)
        observed = str(payload["payload_sha256"])
        if not hmac.compare_digest(observed.lower(), expected):
            errors.append({"path": str(path), "field": "payload_sha256", "error": "canonical payload digest mismatch"})
    if not _valid_units(payload.get("units")):
        errors.append(
            {"path": str(path), "field": "units", "error": "units must declare EPED pedestal reference units"}
        )
    _validate_geometry(path, payload, errors)
    if any(error["path"] == str(path) for error in errors):
        return None
    return {
        "path": str(path),
        "source": str(payload["source"]),
        "reference_dataset_id": str(payload["reference_dataset_id"]),
        "payload_sha256": str(payload["payload_sha256"]).lower(),
    }


def canonical_artifact_sha256(payload: dict[str, object]) -> str:
    """Hash a copy excluding payload_sha256 using sorted ASCII-escaped compact JSON; preserve original serialization. No source authenticity or referenced-byte checks. Unsupported public inputs propagate JSON errors."""
    canonical_payload = dict(payload)
    canonical_payload.pop("payload_sha256", None)
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_uri_error(value: object) -> str | None:
    """Retain lexical URI rules: nonblank, no NUL; http/https/doi/s3/gs prefixes or relative paths without literal parent segments. No URL parsing, resolution or download."""
    if not isinstance(value, str) or not value.strip():
        return "artifact URI must be a non-empty string"
    ref = value.strip()
    if "\x00" in ref:
        return "artifact URI must not contain NUL bytes"
    if ref.startswith(("http://", "https://", "doi:", "s3://", "gs://")):
        return None
    # The declaration is a document, not a path on this machine: POSIX rules on every platform.
    artifact_path = PurePosixPath(ref)
    if artifact_path.is_absolute():
        return "artifact URI must be relative or an admitted external reference URI"
    if any(part == ".." for part in artifact_path.parts):
        return "artifact URI must not contain traversal"
    return None


def _valid_units(value: object) -> bool:
    """Require original psi_N, Pa, eV, m^-3, A and dimensionless unit labels; allow extra fields."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require a nonblank reference URL or DOI without resolution or authentication."""
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
