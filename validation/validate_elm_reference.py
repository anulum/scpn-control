#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ELM reference artifact validator

"""Validate persisted ELM crash and RMP suppression reference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import TypeIs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
_REQUIRED_UNITS = {
    "time": "s",
    "energy": "J",
    "density": "m^-3",
    "temperature": "eV",
    "pressure": "Pa",
    "rmp_perturbation": "dimensionless",
    "heat_flux": "MW/m^2",
}
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


def validate_elm_reference(
    artifact_root: str | Path,
    *,
    require_reference_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate ELM/RMP evidence against persisted measured or public reference artifacts."""
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
        errors.append({"path": str(root), "field": "artifact_root", "error": "no ELM reference artifacts found"})

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
        errors.append({"path": str(root), "field": "artifact_root", "error": "no ELM reference artifacts found"})
    if errors:
        report["status"] = "fail"
    return report


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
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
        if isinstance(value, str) and not _SHA256_RE.match(value):
            errors.append({"path": str(path), "field": field, "error": "field must be a SHA-256 hex digest"})
    for field in _ARTIFACT_URI_FIELDS:
        error = _artifact_uri_error(payload.get(field))
        if error is not None:
            errors.append({"path": str(path), "field": field, "error": error})
    if payload.get("source") not in _ALLOWED_SOURCES:
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
    if isinstance(payload.get("payload_sha256"), str):
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
    """Return the tamper-evident digest for an ELM reference artifact."""
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
    artifact_path = Path(ref)
    if artifact_path.is_absolute():
        return "artifact URI must be relative or an admitted external reference URI"
    if any(part == ".." for part in artifact_path.parts):
        return "artifact URI must not contain traversal"
    return None


def _valid_units(value: object) -> bool:
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    for field in ("reference_url", "reference_doi"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _has_measured_campaign(payload: dict[str, object]) -> bool:
    machine = payload.get("machine")
    shot_id = payload.get("shot_id")
    campaign_id = payload.get("campaign_id")
    return (
        isinstance(machine, str)
        and bool(machine.strip())
        and any(isinstance(value, str) and bool(value.strip()) for value in (shot_id, campaign_id))
    )


def _strictly_increasing_unit_grid(value: object) -> bool:
    if not isinstance(value, list | tuple) or len(value) < 2:
        return False
    last = -math.inf
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(float(item)):
            return False
        current = float(item)
        if current < 0.0 or current > 1.0 or current <= last:
            return False
        last = current
    return True


def _valid_elm_fraction_range(value: object) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    lo, hi = value
    if not _is_positive_finite(lo) or not _is_positive_finite(hi):
        return False
    return 0.04 <= float(lo) <= float(hi) <= 0.15


def _positive_ordered_pair(value: object) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    lo, hi = value
    if not _is_nonnegative_finite(lo) or not _is_positive_finite(hi):
        return False
    return float(hi) > float(lo)


def _is_nonnegative_finite(value: object) -> TypeIs[float]:
    return (
        not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value)) and value >= 0.0
    )


def _is_positive_finite(value: object) -> TypeIs[float]:
    return (
        not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value)) and value > 0.0
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
        default=str(ROOT / "validation" / "reports" / "elm_reference"),
        help="Directory or JSON artifact containing persisted ELM reference evidence",
    )
    parser.add_argument(
        "--require-reference-artifacts",
        action="store_true",
        help="Fail if no ELM reference artifacts are present",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_elm_reference(args.artifact_root, require_reference_artifacts=args.require_reference_artifacts)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"ELM reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
