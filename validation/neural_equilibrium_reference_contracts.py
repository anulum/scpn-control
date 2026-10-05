#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural equilibrium reference artifact validator

"""Validate one neural-reference declaration and canonical checksum without model/provenance authentication."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path

from validation.neural_equilibrium_reference_metrics import (
    _artifact_uri_error,
    _has_public_reference,
    _valid_grid_shape,
    _valid_units,
    _validate_metric_block,
)
from validation.reference_uri import external_executable_path_error

ROOT = Path(__file__).resolve().parents[1]

_ALLOWED_SOURCES = {"real_pefit", "documented_public_reference"}
_SCHEMA_VERSION = "scpn-control.neural-equilibrium-reference.v1"
_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "trained_weights_sha256",
    "reference_dataset_id",
    "reference_artifact_uri",
    "prediction_artifact_uri",
    "reference_artifact_sha256",
    "prediction_artifact_sha256",
    "payload_sha256",
    "executed_at",
)
_REQUIRED_TARGET_SCHEMA = ("psi", "pressure", "q_profile", "lcfs_boundary", "magnetic_axis")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ARTIFACT_URI_FIELDS = ("reference_artifact_uri", "prediction_artifact_uri")
_SHA256_FIELDS = (
    "trained_weights_sha256",
    "reference_artifact_sha256",
    "prediction_artifact_sha256",
    "payload_sha256",
)


def _validate_artifact(
    path: Path,
    raw_payload: bytes,
    payload: object,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    """Validate one decoded declaration, append field findings and return accepted identity metadata.

    Required strings, hex checksums, exact targets/units, grid/count, allowed
    source labels, lexical URI/binary paths and declared error tolerances are
    checked. Canonical payload hashing excludes its own checksum; digest format
    is checked before constant-time comparison. Original captured JSON bytes
    determine the file digest. No referenced bytes, executable, DOI, timestamp,
    model version or numerical metric computation is verified here.
    """
    if not isinstance(payload, dict):
        errors.append({"path": _portable_path(path), "field": "root", "error": "artefact root must be an object"})
        return None
    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "schema_version",
                "error": f"schema_version must be '{_SCHEMA_VERSION}'",
            }
        )
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": _portable_path(path), "field": field, "error": "field must be a non-empty string"})
    for field in _SHA256_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and not _SHA256_RE.fullmatch(value):
            errors.append({"path": _portable_path(path), "field": field, "error": "field must be a SHA-256 hex digest"})
    for field in _ARTIFACT_URI_FIELDS:
        error = _artifact_uri_error(payload.get(field))
        if error is not None:
            errors.append({"path": _portable_path(path), "field": field, "error": error})
    if payload.get("target_schema") != list(_REQUIRED_TARGET_SCHEMA):
        errors.append(
            {
                "path": _portable_path(path),
                "field": "target_schema",
                "error": "target_schema must match equilibrium outputs",
            }
        )
    if isinstance(payload.get("payload_sha256"), str) and _SHA256_RE.fullmatch(payload["payload_sha256"]):
        expected = canonical_artifact_sha256(payload)
        observed = str(payload["payload_sha256"])
        if not hmac.compare_digest(observed.lower(), expected):
            errors.append(
                {"path": _portable_path(path), "field": "payload_sha256", "error": "canonical payload digest mismatch"}
            )
    if not isinstance(payload.get("source"), str) or payload["source"] not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "source",
                "error": "source must be real_pefit or documented_public_reference",
            }
        )
    if payload.get("source") == "real_pefit":
        binary_path_error = external_executable_path_error(payload.get("binary_path"))
        if binary_path_error is not None:
            errors.append({"path": _portable_path(path), "field": "binary_path", "error": binary_path_error})
    if payload.get("source") == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": _portable_path(path),
                "field": "reference",
                "error": "documented public reference artefacts require reference_url or reference_doi",
            }
        )
    if not _valid_grid_shape(payload.get("grid_shape")):
        errors.append(
            {"path": _portable_path(path), "field": "grid_shape", "error": "grid_shape must be two positive integers"}
        )
    if not _valid_units(payload.get("units")):
        errors.append(
            {
                "path": _portable_path(path),
                "field": "units",
                "error": "units must declare psi, pressure, q_profile, and boundary",
            }
        )
    count = payload.get("reference_equilibria_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "reference_equilibria_count",
                "error": "field must be a positive integer",
            }
        )
    _validate_metric_block(_portable_path(path), payload.get("metrics"), payload.get("tolerances"), errors)
    if any(error["path"] == _portable_path(path) for error in errors):
        return None
    return {
        "path": _portable_path(path),
        "source": str(payload["source"]),
        "model_id": str(payload["model_id"]),
        "model_version": str(payload["model_version"]),
        "trained_weights_sha256": str(payload["trained_weights_sha256"]).lower(),
        "reference_dataset_id": str(payload["reference_dataset_id"]),
        "reference_equilibria_count": int(payload["reference_equilibria_count"]),
        "artifact_file_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "payload_sha256": str(payload["payload_sha256"]).lower(),
    }


def canonical_artifact_sha256(payload: dict[str, object]) -> str:
    """Hash compact sorted ASCII JSON declaration content excluding payload_sha256 without mutation.

    Nonfinite/nonserializable values raise ValueError/TypeError. Key order and
    source whitespace do not affect this checksum; spelling of retained values
    does. It is an unkeyed consistency checksum, not source authentication.

    Examples
    --------
    >>> canonical_artifact_sha256({"model_id": "declaration", "payload_sha256": "ignored"}) == canonical_artifact_sha256({"model_id": "declaration"})
    True
    """
    canonical_payload = dict(payload)
    canonical_payload.pop("payload_sha256", None)
    encoded = json.dumps(
        canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _portable_path(path: Path) -> str:
    """Return resolved canonical-relative spelling when contained, otherwise preserve supplied path spelling."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.as_posix()
