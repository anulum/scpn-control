# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK OOD author declaration and report contracts

"""Preserve v2 OOD author/report identities and canonical hashes; accepted metadata is not installed deployment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from validation.gk_ood_reference_domains import (
    _ACCEPTANCE_FIELDS,
    _EXPECTED_FEATURE_SCHEMA,
    _THRESHOLD_FIELDS,
    _portable_path,
    _validate_mahalanobis_metric,
    _validate_numeric_object,
    _validate_training_distribution,
)

_ALLOWED_SOURCES = {"published_gk_campaign", "real_external_gk_campaign", "facility_gk_campaign"}
_REQUIRED_STR_FIELDS = ("campaign_id", "source", "evaluated_at")
_REPORT_SCHEMA = "scpn-control.gk-ood-calibration-report.v2"
_ARTIFACT_SCHEMA = "scpn-control.gk-ood-calibration-artifact.v2"
_BLOCKED_REASON = "Requires persisted published, external-code, or facility GK OOD calibration artifacts."


def _validate_artifact(
    path: Path,
    raw_payload: bytes,
    payload: object,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    """Inspect declared campaign identities, numeric domains, covariance labels and inclusive rate comparisons.

    Raw captured bytes bind artifact_sha256; canonical JSON binds decoded author
    metadata. Original false-positive/negative <= maximum and recall >= minimum
    comparisons remain. No actual dataset, covariance, published run, scientific
    source, held-out metrics or installed/current detector state is authenticated.
    """
    if not isinstance(payload, dict):
        errors.append({"path": _portable_path(path), "field": "root", "error": "artifact root must be an object"})
        return None
    if payload.get("schema_version") != _ARTIFACT_SCHEMA:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "schema_version",
                "error": f"schema_version must be '{_ARTIFACT_SCHEMA}'",
            }
        )
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": _portable_path(path), "field": field, "error": "field must be a non-empty string"})
    if not isinstance(payload.get("source"), str) or payload["source"] not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "source",
                "error": "source must identify published, external GK, or facility campaign evidence",
            }
        )
    if payload.get("feature_schema") != _EXPECTED_FEATURE_SCHEMA:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "feature_schema",
                "error": "feature_schema must match the declared 10D GK OOD vector",
            }
        )
    _validate_training_distribution(path, payload.get("training_distribution"), errors)
    _validate_numeric_object(path, payload.get("thresholds"), _THRESHOLD_FIELDS, "thresholds", errors)
    _validate_mahalanobis_metric(path, payload.get("mahalanobis_metric"), errors)
    _validate_numeric_object(path, payload.get("acceptance"), _ACCEPTANCE_FIELDS, "acceptance", errors)
    if any(error["path"] == _portable_path(path) for error in errors):
        return None

    acceptance = payload["acceptance"]
    false_positive_rate = float(acceptance["false_positive_rate"])
    false_negative_rate = float(acceptance["false_negative_rate"])
    max_false_positive_rate = float(acceptance["max_false_positive_rate"])
    max_false_negative_rate = float(acceptance["max_false_negative_rate"])
    ood_recall = float(acceptance["ood_recall"])
    min_ood_recall = float(acceptance["min_ood_recall"])

    if false_positive_rate > max_false_positive_rate:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "false_positive_rate",
                "error": "false positive rate exceeds acceptance bound",
            }
        )
    if false_negative_rate > max_false_negative_rate:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "false_negative_rate",
                "error": "false negative rate exceeds acceptance bound",
            }
        )
    if ood_recall < min_ood_recall:
        errors.append(
            {"path": _portable_path(path), "field": "ood_recall", "error": "OOD recall below acceptance bound"}
        )
    if any(error["path"] == _portable_path(path) for error in errors):
        return None

    return {
        "path": _portable_path(path),
        "schema_version": str(payload["schema_version"]),
        "campaign_id": str(payload["campaign_id"]),
        "source": str(payload["source"]),
        "artifact_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "canonical_payload_sha256": _json_sha256(payload),
        "mahalanobis_metric": {
            "calibration_method": str(payload["mahalanobis_metric"]["calibration_method"]),
            "covariance_inverse_sha256": str(payload["mahalanobis_metric"]["covariance_inverse_sha256"]),
            "positive_definite": bool(payload["mahalanobis_metric"]["positive_definite"]),
        },
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "ood_recall": ood_recall,
    }


def _new_report(root: Path, *, require_campaign_artifacts: bool) -> dict[str, Any]:
    """Construct the original v2 report and false claim defaults; root path formatting remains portable."""
    return {
        "schema_version": _REPORT_SCHEMA,
        "status": "pass",
        "root": _portable_path(root),
        "payload_sha256": None,
        "campaign_artifacts": 0,
        "require_campaign_artifacts": bool(require_campaign_artifacts),
        "public_claims": {
            "deployment_calibration_admitted": False,
            "full_gk_operating_envelope_admitted": False,
            "blocked_reason": _BLOCKED_REASON,
        },
        "feature_schema": list(_EXPECTED_FEATURE_SCHEMA),
        "entries": [],
        "errors": [],
    }


def _json_sha256(payload: object) -> str:
    """Hash original compact sorted ASCII JSON without mutation or source authentication.

    The historical direct Python nonfinite JSON extension is unchanged; public
    stored declarations refuse nonfinite numbers before invoking this helper.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finalise_report(report: dict[str, Any]) -> dict[str, Any]:
    """Preserve original accepted-metadata flag and canonical report hash with payload_sha256=None.

    deployment_calibration_admitted means only passing declared campaign metadata
    with at least one accepted artifact. It installs no calibration and grants no
    measured/held-out/physical/control/operator acceptance; full-envelope stays false.
    """
    admitted = report["status"] == "pass" and report["campaign_artifacts"] > 0
    report["public_claims"]["deployment_calibration_admitted"] = admitted
    payload = dict(report)
    payload["payload_sha256"] = None
    report["payload_sha256"] = _json_sha256(payload)
    return report
