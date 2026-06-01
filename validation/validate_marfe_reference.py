#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MARFE reference artifact validator

"""Validate persisted MARFE radiation-condensation reference artifacts."""

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

_ALLOWED_SOURCES = {"measured_marfe_campaign", "documented_public_reference"}
_SCHEMA_VERSION = "scpn-control.marfe-reference.v1"
_REQUIRED_STR_FIELDS = (
    "source",
    "reference_dataset_id",
    "executed_at",
    "temperature_profile_uri",
    "density_limit_uri",
    "radiation_curve_uri",
    "power_balance_uri",
    "temperature_profile_sha256",
    "density_limit_sha256",
    "radiation_curve_sha256",
    "power_balance_sha256",
    "payload_sha256",
)
_REQUIRED_UNITS = {
    "temperature": "eV",
    "density": "m^-3",
    "power": "W",
    "current": "A",
    "impurity_fraction": "1",
    "length": "m",
    "growth_rate": "s^-1",
}
_REQUIRED_METRICS = (
    "onset_temperature_relative_error",
    "density_limit_relative_error",
    "greenwald_fraction_error",
    "front_temperature_min_relative_error",
    "radiation_growth_rate_relative_error",
)
_ARTIFACT_URI_FIELDS = ("temperature_profile_uri", "density_limit_uri", "radiation_curve_uri", "power_balance_uri")
_SHA256_FIELDS = (
    "temperature_profile_sha256",
    "density_limit_sha256",
    "radiation_curve_sha256",
    "power_balance_sha256",
    "payload_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_marfe_reference(
    artifact_root: str | Path,
    *,
    require_reference_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate MARFE evidence against persisted measured or public reference artifacts."""
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
        errors.append({"path": str(root), "field": "artifact_root", "error": "no MARFE reference artifacts found"})

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
        errors.append({"path": str(root), "field": "artifact_root", "error": "no MARFE reference artifacts found"})
    if errors:
        report["status"] = "fail"
    return report


def _validate_artifact(path: Path, payload: object, errors: list[dict[str, object]]) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "artifact root must be an object"})
        return None
    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append({"path": str(path), "field": "schema_version", "error": f"schema_version must be '{_SCHEMA_VERSION}'"})
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
                "error": "source must be measured_marfe_campaign or documented_public_reference",
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
    if payload.get("source") == "measured_marfe_campaign" and not _has_measured_campaign(payload):
        errors.append(
            {"path": str(path), "field": "campaign", "error": "measured MARFE campaigns require machine and shot_id or campaign_id"}
        )
    if isinstance(payload.get("payload_sha256"), str):
        expected = canonical_artifact_sha256(payload)
        observed = str(payload["payload_sha256"])
        if not hmac.compare_digest(observed.lower(), expected):
            errors.append({"path": str(path), "field": "payload_sha256", "error": "canonical payload digest mismatch"})
    if not _valid_units(payload.get("units")):
        errors.append({"path": str(path), "field": "units", "error": "units must declare MARFE reference units"})
    if not _ordered_positive_grid(payload.get("temperature_scan_eV")):
        errors.append(
            {"path": str(path), "field": "temperature_scan_eV", "error": "temperature scan must be finite, positive, and strictly increasing"}
        )
    if not _ordered_positive_grid(payload.get("density_scan_m3")):
        errors.append(
            {"path": str(path), "field": "density_scan_m3", "error": "density scan must be finite, positive, and strictly increasing"}
        )
    if not _valid_impurity_fraction_range(payload.get("impurity_fraction_range")):
        errors.append(
            {"path": str(path), "field": "impurity_fraction_range", "error": "impurity fraction range must lie inside (0, 1]"}
        )
    if not _valid_geometry(payload.get("geometry")):
        errors.append(
            {
                "path": str(path),
                "field": "geometry",
                "error": "geometry must declare finite R0_m, a_m, q95, and connection_length_m with tokamak ordering",
            }
        )
    if not _valid_power_balance(payload.get("power_balance")):
        errors.append(
            {
                "path": str(path),
                "field": "power_balance",
                "error": "power_balance must declare positive P_SOL_W and non-negative q_perp_W_m2",
            }
        )
    impurity = payload.get("impurity")
    if not isinstance(impurity, str) or not impurity.strip():
        errors.append({"path": str(path), "field": "impurity", "error": "impurity must be a non-empty string"})
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
    """Return the tamper-evident digest for a MARFE reference artifact."""
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
    return isinstance(machine, str) and bool(machine.strip()) and any(
        isinstance(value, str) and bool(value.strip()) for value in (shot_id, campaign_id)
    )


def _ordered_positive_grid(value: object) -> bool:
    if not isinstance(value, list | tuple) or len(value) < 2:
        return False
    last = -math.inf
    for item in value:
        if not _is_positive_finite(item):
            return False
        current = float(item)
        if current <= last:
            return False
        last = current
    return True


def _valid_impurity_fraction_range(value: object) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    lo, hi = value
    if not _is_positive_finite(lo) or not _is_positive_finite(hi):
        return False
    return 0.0 < float(lo) <= float(hi) <= 1.0


def _valid_geometry(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    r0 = value.get("R0_m")
    minor = value.get("a_m")
    q95 = value.get("q95")
    connection = value.get("connection_length_m")
    if not all(_is_positive_finite(item) for item in (r0, minor, q95, connection)):
        return False
    return float(minor) < float(r0)


def _valid_power_balance(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    p_sol = value.get("P_SOL_W")
    q_perp = value.get("q_perp_W_m2")
    return _is_positive_finite(p_sol) and _is_nonnegative_finite(q_perp)


def _is_finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _is_nonnegative_finite(value: object) -> bool:
    return _is_finite(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> bool:
    return _is_finite(value) and float(value) > 0.0


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
        default=str(ROOT / "validation" / "reports" / "marfe_reference"),
        help="Directory or JSON artifact containing persisted MARFE reference evidence",
    )
    parser.add_argument(
        "--require-reference-artifacts",
        action="store_true",
        help="Fail if no MARFE reference artifacts are present",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_marfe_reference(args.artifact_root, require_reference_artifacts=args.require_reference_artifacts)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"MARFE reference: {report['status']} reference_artifacts={report['reference_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
