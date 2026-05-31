#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin reference artifact validator

"""Validate persisted tokamak digital-twin reference artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_ALLOWED_SOURCES = {
    "documented_public_reference",
    "measured_discharge_replay",
    "external_integrated_modelling",
}
_ALLOWED_EXTERNAL_CODES = {"ASTRA", "IMAS", "JINTRAC", "TRANSP", "TSC"}
_REQUIRED_STR_FIELDS = (
    "source",
    "model_id",
    "model_version",
    "reference_dataset_id",
    "reference_artifact_sha256",
    "executed_at",
)
_REQUIRED_GRID_FIELDS = ("grid_size", "time_steps", "seed", "state_variables", "has_ids_export")
_REQUIRED_ACTUATOR_FIELDS = (
    "actuator_tau_steps",
    "actuator_rate_limit",
    "actuator_bias",
    "sensor_dropout_prob",
    "sensor_noise_std",
)
_REQUIRED_UNITS = {
    "temperature": "keV",
    "density": "m^-3",
    "q_profile": "1",
    "actuator_action": "1",
    "time": "step",
    "ids_pulse": "1",
}
_MAXIMUM_ERROR_METRICS = (
    "final_avg_temp_relative_error",
    "q_profile_rmse",
    "actuator_lag_abs_error",
    "ids_roundtrip_abs_error",
    "island_mask_f1_error",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_digital_twin_reference(
    artifact_root: str | Path,
    *,
    require_reference_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate digital-twin evidence against persisted references."""
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
        errors.append({"path": str(root), "field": "artifact_root", "error": "no digital twin reference artifacts found"})

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
        errors.append({"path": str(root), "field": "artifact_root", "error": "no digital twin reference artifacts found"})
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
    digest = payload.get("reference_artifact_sha256")
    if isinstance(digest, str) and not _SHA256_RE.match(digest):
        errors.append({"path": str(path), "field": "reference_artifact_sha256", "error": "field must be a SHA-256 hex digest"})
    source = payload.get("source")
    if source not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": str(path),
                "field": "source",
                "error": "source must be documented_public_reference, measured_discharge_replay, or external_integrated_modelling",
            }
        )
    _validate_source_provenance(path, payload, errors)
    if not _valid_grid_metadata(payload.get("grid_metadata")):
        errors.append({"path": str(path), "field": "grid_metadata", "error": "grid_metadata must declare finite topology and IDS export metadata"})
    if not _valid_actuator_metadata(payload.get("actuator_metadata")):
        errors.append({"path": str(path), "field": "actuator_metadata", "error": "actuator_metadata must declare finite actuator and sensor metadata"})
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare digital twin reference units"})
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
    source = payload.get("source")
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append({"path": str(path), "field": "reference", "error": "documented public reference artifacts require reference_url or reference_doi"})
    if source == "measured_discharge_replay":
        if not _has_nonempty_str(payload, "shot_id"):
            errors.append({"path": str(path), "field": "shot_id", "error": "measured discharge replays require shot_id"})
        if not _has_nonempty_str(payload, "diagnostic_uri"):
            errors.append({"path": str(path), "field": "diagnostic_uri", "error": "measured discharge replays require diagnostic_uri"})
    if source == "external_integrated_modelling":
        external_code = payload.get("external_code")
        if external_code not in _ALLOWED_EXTERNAL_CODES:
            errors.append({"path": str(path), "field": "external_code", "error": "external_code must be ASTRA, IMAS, JINTRAC, TRANSP, or TSC"})
        if not _has_nonempty_str(payload, "reference_artifact_uri"):
            errors.append({"path": str(path), "field": "reference_artifact_uri", "error": "external integrated modelling artifacts require reference_artifact_uri"})


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
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


def _valid_grid_metadata(value: object) -> bool:
    if not isinstance(value, dict) or not all(field in value for field in _REQUIRED_GRID_FIELDS):
        return False
    grid_size = value.get("grid_size")
    time_steps = value.get("time_steps")
    seed = value.get("seed")
    state_variables = value.get("state_variables")
    has_ids_export = value.get("has_ids_export")
    return (
        not isinstance(grid_size, bool)
        and isinstance(grid_size, int)
        and grid_size >= 4
        and not isinstance(time_steps, bool)
        and isinstance(time_steps, int)
        and time_steps > 0
        and not isinstance(seed, bool)
        and isinstance(seed, int)
        and seed >= 0
        and isinstance(state_variables, list)
        and bool(state_variables)
        and all(isinstance(item, str) and item.strip() for item in state_variables)
        and isinstance(has_ids_export, bool)
    )


def _valid_actuator_metadata(value: object) -> bool:
    if not isinstance(value, dict) or not all(field in value for field in _REQUIRED_ACTUATOR_FIELDS):
        return False
    tau_steps = value.get("actuator_tau_steps")
    if isinstance(tau_steps, bool) or not isinstance(tau_steps, int) or tau_steps < 0:
        return False
    if not all(_is_finite_number(value.get(field)) for field in _REQUIRED_ACTUATOR_FIELDS if field != "actuator_tau_steps"):
        return False
    rate_limit = value.get("actuator_rate_limit")
    dropout = value.get("sensor_dropout_prob")
    noise = value.get("sensor_noise_std")
    return float(rate_limit) >= 0.0 and 0.0 <= float(dropout) <= 1.0 and float(noise) >= 0.0


def _valid_units(value: object) -> bool:
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    return any(_has_nonempty_str(payload, field) for field in ("reference_url", "reference_doi"))


def _has_nonempty_str(payload: dict[str, object], field: str) -> bool:
    value = payload.get(field)
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _is_nonnegative_finite(value: object) -> bool:
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> bool:
    return _is_finite_number(value) and float(value) > 0.0


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
        default=str(ROOT / "validation" / "reports" / "digital_twin_reference"),
        help="Directory or JSON artifact containing persisted digital twin reference evidence",
    )
    parser.add_argument("--require-reference-artifacts", action="store_true", help="Fail if no digital twin reference artifacts are present")
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_digital_twin_reference(args.artifact_root, require_reference_artifacts=args.require_reference_artifacts)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Digital twin reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
