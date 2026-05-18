#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK OOD calibration artifact validator

"""Validate persisted gyrokinetic OOD calibration campaign artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_FEATURE_SCHEMA = [
    "R_L_Ti",
    "R_L_Te",
    "R_L_ne",
    "q",
    "s_hat",
    "alpha_MHD",
    "Te_Ti",
    "Z_eff",
    "nu_star",
    "beta_e",
]
_ALLOWED_SOURCES = {"published_gk_campaign", "real_external_gk_campaign", "facility_gk_campaign"}
_REQUIRED_STR_FIELDS = ("campaign_id", "source", "evaluated_at")
_THRESHOLD_FIELDS = ("mahalanobis", "soft_sigma", "ensemble_disagreement")
_ACCEPTANCE_FIELDS = (
    "false_positive_rate",
    "false_negative_rate",
    "max_false_positive_rate",
    "max_false_negative_rate",
    "ood_recall",
    "min_ood_recall",
)


def validate_gk_ood_calibration(
    artifact_root: str | Path,
    *,
    require_campaign_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate OOD calibration artifacts and deployment acceptance metrics."""
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report: dict[str, Any] = {
        "status": "pass",
        "root": str(root),
        "campaign_artifacts": 0,
        "require_campaign_artifacts": bool(require_campaign_artifacts),
        "entries": [],
        "errors": [],
    }
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]

    if require_campaign_artifacts and not paths:
        errors.append({"path": str(root), "field": "artifact_root", "error": "no GK OOD calibration artifacts found"})

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
            report["campaign_artifacts"] += 1

    if require_campaign_artifacts and report["campaign_artifacts"] == 0 and not errors:
        errors.append({"path": str(root), "field": "artifact_root", "error": "no GK OOD calibration artifacts found"})
    if errors:
        report["status"] = "fail"
    return report


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "artifact root must be an object"})
        return None
    if payload.get("schema_version") != "1.0":
        errors.append({"path": str(path), "field": "schema_version", "error": "schema_version must be '1.0'"})
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": str(path), "field": field, "error": "field must be a non-empty string"})
    if payload.get("source") not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": str(path),
                "field": "source",
                "error": "source must identify published, external GK, or facility campaign evidence",
            }
        )
    if payload.get("feature_schema") != _EXPECTED_FEATURE_SCHEMA:
        errors.append(
            {
                "path": str(path),
                "field": "feature_schema",
                "error": "feature_schema must match the declared 10D GK OOD vector",
            }
        )
    _validate_training_distribution(path, payload.get("training_distribution"), errors)
    _validate_numeric_object(path, payload.get("thresholds"), _THRESHOLD_FIELDS, "thresholds", errors)
    _validate_numeric_object(path, payload.get("acceptance"), _ACCEPTANCE_FIELDS, "acceptance", errors)
    if any(error["path"] == str(path) for error in errors):
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
            {"path": str(path), "field": "false_positive_rate", "error": "false positive rate exceeds acceptance bound"}
        )
    if false_negative_rate > max_false_negative_rate:
        errors.append(
            {"path": str(path), "field": "false_negative_rate", "error": "false negative rate exceeds acceptance bound"}
        )
    if ood_recall < min_ood_recall:
        errors.append({"path": str(path), "field": "ood_recall", "error": "OOD recall below acceptance bound"})
    if any(error["path"] == str(path) for error in errors):
        return None

    return {
        "path": str(path),
        "campaign_id": str(payload["campaign_id"]),
        "source": str(payload["source"]),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "ood_recall": ood_recall,
    }


def _validate_training_distribution(path: Path, payload: object, errors: list[dict[str, object]]) -> None:
    if not isinstance(payload, dict):
        errors.append(
            {"path": str(path), "field": "training_distribution", "error": "training_distribution must be an object"}
        )
        return
    if not isinstance(payload.get("dataset_id"), str) or not str(payload.get("dataset_id")).strip():
        errors.append({"path": str(path), "field": "dataset_id", "error": "dataset_id must be a non-empty string"})
    sample_count = payload.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        errors.append({"path": str(path), "field": "sample_count", "error": "sample_count must be a positive integer"})
    for field in ("mean", "std"):
        values = payload.get(field)
        if (
            not isinstance(values, list)
            or len(values) != len(_EXPECTED_FEATURE_SCHEMA)
            or not all(_is_number(value) for value in values)
        ):
            errors.append({"path": str(path), "field": field, "error": "field must be a numeric 10-element array"})


def _validate_numeric_object(
    path: Path,
    payload: object,
    fields: tuple[str, ...],
    parent: str,
    errors: list[dict[str, object]],
) -> None:
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": parent, "error": f"{parent} must be an object"})
        return
    for field in fields:
        value = payload.get(field)
        if not _is_number(value):
            errors.append({"path": str(path), "field": field, "error": "field must be numeric"})


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)


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
        default=str(ROOT / "validation" / "reports" / "gk_ood_calibration"),
        help="Directory or JSON artifact containing persisted GK OOD calibration evidence",
    )
    parser.add_argument(
        "--require-campaign-artifacts", action="store_true", help="Fail if no calibration artifacts are present"
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_gk_ood_calibration(args.artifact_root, require_campaign_artifacts=args.require_campaign_artifacts)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK OOD calibration: {report['status']} campaign_artifacts={report['campaign_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
