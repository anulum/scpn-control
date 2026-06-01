#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural transport reference artifact validator

"""Validate persisted neural-transport reference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
_REQUIRED_UNITS = {
    "chi_i": "m^2/s",
    "chi_e": "m^2/s",
    "D_e": "m^2/s",
    "input_gradients": "dimensionless",
}
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


def validate_neural_transport_reference(
    artifact_root: str | Path,
    *,
    require_reference_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate neural transport surrogate evidence against persisted references."""
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report: dict[str, Any] = {
        "status": "pass",
        "root": str(root),
        "reference_artifacts": 0,
        "require_reference_artifacts": bool(require_reference_artifacts),
        "entries": [],
        "errors": [],
    }
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]

    if require_reference_artifacts and not paths:
        errors.append(
            {"path": str(root), "field": "artifact_root", "error": "no neural transport reference artifacts found"}
        )

    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
            entry = _validate_artifact(path, payload, errors)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "field": "json", "error": str(exc)})
            continue
        if entry is not None:
            entries.append(entry)
            report["reference_artifacts"] += 1

    if require_reference_artifacts and report["reference_artifacts"] == 0 and not errors:
        errors.append(
            {"path": str(root), "field": "artifact_root", "error": "no neural transport reference artifacts found"}
        )
    if errors:
        report["status"] = "fail"
    return report


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
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
        if isinstance(value, str) and not _SHA256_RE.match(value):
            errors.append({"path": str(path), "field": field, "error": "field must be a SHA-256 hex digest"})
    for field in _ARTIFACT_URI_FIELDS:
        error = _artifact_uri_error(payload.get(field))
        if error is not None:
            errors.append({"path": str(path), "field": field, "error": error})
    if payload.get("target_schema") != list(_REQUIRED_TARGET_SCHEMA):
        errors.append(
            {"path": str(path), "field": "target_schema", "error": "target_schema must match transport outputs"}
        )
    if isinstance(payload.get("payload_sha256"), str):
        expected = canonical_artifact_sha256(payload)
        observed = str(payload["payload_sha256"])
        if not hmac.compare_digest(observed.lower(), expected):
            errors.append({"path": str(path), "field": "payload_sha256", "error": "canonical payload digest mismatch"})
    source = payload.get("source")
    if source not in _ALLOWED_SOURCES:
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


def _valid_units(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    for field in ("reference_url", "reference_doi"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def canonical_artifact_sha256(payload: dict[str, object]) -> str:
    """Return the tamper-evident digest for a neural transport reference artifact."""
    canonical_payload = dict(payload)
    canonical_payload.pop("payload_sha256", None)
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_uri_error(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "artifact URI must be a non-empty string"
    ref = value.strip()
    if "\x00" in ref:
        return "artifact URI must not contain NUL bytes"
    if ref.startswith(("http://", "https://", "doi:", "s3://", "gs://")):
        return None
    path = Path(ref)
    if path.is_absolute():
        return "artifact URI must be relative or an admitted external reference URI"
    if any(part == ".." for part in path.parts):
        return "artifact URI must not contain traversal"
    return None


def _is_nonnegative_finite(value: object) -> bool:
    return (
        not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value)) and value >= 0.0
    )


def _is_positive_finite(value: object) -> bool:
    return (
        not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value)) and value > 0.0
    )


def _is_unit_interval(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and 0.0 <= value <= 1.0
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "validation" / "reports" / "neural_transport_reference"),
        help="Directory or JSON artifact containing persisted neural transport reference evidence",
    )
    parser.add_argument(
        "--require-reference-artifacts",
        action="store_true",
        help="Fail if no neural transport reference artifacts are present",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_neural_transport_reference(
        args.artifact_root,
        require_reference_artifacts=args.require_reference_artifacts,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Neural transport reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
