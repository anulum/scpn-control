#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural transport reference identity, schemas and declared bounds


"""Check original neural transport identities, schemas, lexical provenance and body consistency."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path

from validation.neural_transport_reference_domains import (
    _artifact_uri_error,
    _has_public_reference,
    _is_nonnegative_finite,
    _is_positive_finite,
    _is_unit_interval,
    _valid_units,
)
from validation.reference_uri import external_executable_path_error

_ALLOWED_SOURCES = {"real_qualikiz", "documented_public_reference"}
_SCHEMA_VERSION = "scpn-control.neural-transport-reference.v1"
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
_REQUIRED_FEATURE_SCHEMA = (
    "R_LTi",
    "R_LTe",
    "R_Ln",
    "q",
    "s_hat",
    "alpha",
    "Ti_Te",
    "Zeff",
    "collisionality",
    "beta_e",
)
_MAXIMUM_ERROR_METRICS = (
    "chi_i_rmse_m2_s",
    "chi_e_rmse_m2_s",
    "D_e_rmse_m2_s",
    "chi_i_relative_mae",
)
_MINIMUM_SCORE_METRICS = ("unstable_branch_accuracy",)
_REQUIRED_TARGET_SCHEMA = ("chi_i", "chi_e", "D_e", "unstable_branch")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ARTIFACT_URI_FIELDS = ("reference_artifact_uri", "prediction_artifact_uri")
_SHA256_FIELDS = (
    "trained_weights_sha256",
    "reference_artifact_sha256",
    "prediction_artifact_sha256",
    "payload_sha256",
)


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    """Validate original identities and exact hex shapes, schema orders, units, counts and declared bounds; body SHA consistency authenticates no producer or referenced bytes."""
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "artifact root must be an object"})
        return None
    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append(
            {
                "path": str(path),
                "field": "schema_version",
                "error": f"schema_version must be '{_SCHEMA_VERSION}'",
            }
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
    if payload.get("target_schema") != list(_REQUIRED_TARGET_SCHEMA):
        errors.append(
            {"path": str(path), "field": "target_schema", "error": "target_schema must match transport outputs"}
        )
    if isinstance(payload.get("payload_sha256"), str) and _SHA256_RE.fullmatch(payload["payload_sha256"]):
        expected = canonical_artifact_sha256(payload)
        observed = str(payload["payload_sha256"])
        if not hmac.compare_digest(observed.lower(), expected):
            errors.append({"path": str(path), "field": "payload_sha256", "error": "canonical payload digest mismatch"})
    source = payload.get("source")
    if not isinstance(source, str) or source not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": str(path),
                "field": "source",
                "error": "source must be real_qualikiz or documented_public_reference",
            }
        )
    if source == "real_qualikiz":
        binary_path_error = external_executable_path_error(payload.get("binary_path"))
        if binary_path_error is not None:
            errors.append({"path": str(path), "field": "binary_path", "error": binary_path_error})
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
            {"path": str(path), "field": "feature_schema", "error": "feature_schema must match QLKNN-10D order"}
        )
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare transport target units"})
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
        "payload_sha256": str(payload["payload_sha256"]).lower(),
    }


def _validate_metric_block(
    path: Path,
    metrics: object,
    tolerances: object,
    errors: list[dict[str, object]],
) -> None:
    """Compare four finite nonnegative errors to positive inclusive bounds and branch accuracy to an inclusive zero-to-one minimum score."""
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


def canonical_artifact_sha256(payload: dict[str, object]) -> str:
    """Return SHA256 of original compact sorted ASCII-escaped JSON excluding payload_sha256.

    This is caller-recomputable consistency without weight, producer or artifact
    authentication; original serialization domain and errors remain unchanged.

    Examples
    --------
    >>> canonical_artifact_sha256({"value": 1}) == canonical_artifact_sha256({"value": 1, "payload_sha256": "ignored"})
    True
    """
    canonical_payload = dict(payload)
    canonical_payload.pop("payload_sha256", None)
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
