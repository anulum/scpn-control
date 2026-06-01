#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK geometry reference validator

"""Validate Miller geometry output against immutable reference cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from scpn_control.core.gk_geometry import miller_geometry

_REQUIRED_CASES = {"circular_cyclone_limit", "shaped_positive_triangularity", "high_shear_local_equilibrium"}
_REQUIRED_HEADER_FIELDS = (
    "spdx_license_id",
    "commercial_license",
    "concepts_copyright",
    "code_copyright",
    "orcid",
    "contact",
    "file",
)
_REQUIRED_SAMPLE_FIELDS = (
    "theta",
    "R",
    "Z",
    "jacobian",
    "g_rr",
    "g_rt",
    "g_tt",
    "B_toroidal",
    "b_dot_grad_theta",
)
_GEOMETRY_PARAMETERS = {
    "R0",
    "a",
    "rho",
    "kappa",
    "delta",
    "s_kappa",
    "s_delta",
    "q",
    "s_hat",
    "alpha_MHD",
    "dR_dr",
    "B0",
}
_ABS_TOLERANCE = 1.0e-11
_REL_TOLERANCE = 1.0e-10
_REPORT_SCHEMA = "scpn-control.gk-geometry-reference.v2"
_UNITS = {
    "theta": "rad",
    "R": "m",
    "Z": "m",
    "jacobian": "m2",
    "g_rr": "dimensionless",
    "g_rt": "m",
    "g_tt": "m2",
    "B_toroidal": "T",
    "b_dot_grad_theta": "m-1",
}
_FULL_EQUILIBRIUM_BLOCKED_REASON = (
    "Requires independent Miller-geometry implementation or external equilibrium-code evidence."
)


def validate_gk_geometry_reference(reference_path: str | Path) -> dict[str, Any]:
    """Validate repository Miller geometry against stored reference cases."""
    path = Path(reference_path)
    report = _new_report(path)
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]
    try:
        raw_payload = path.read_text(encoding="utf-8")
        report["reference_file_sha256"] = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        payload = json.loads(raw_payload, object_pairs_hook=_reject_duplicate_json_keys)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report["status"] = "fail"
        errors.append({"path": str(path), "field": "json", "error": str(exc)})
        return _finalise_report(report)

    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "reference root must be an object"})
        report["status"] = "fail"
        return _finalise_report(report)
    _validate_header(path, payload, errors)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append({"path": str(path), "field": "cases", "error": "cases must be a non-empty array"})
        report["status"] = "fail"
        return _finalise_report(report)

    seen_cases: set[str] = set()
    for index, case_payload in enumerate(cases):
        entry = _validate_case(path, index, case_payload, errors)
        if entry is not None:
            case_name = str(entry["case"])
            if case_name in seen_cases:
                errors.append({"path": str(path), "index": index, "field": "case", "error": f"duplicate case: {case_name}"})
                continue
            seen_cases.add(case_name)
            entries.append(entry)
    missing_cases = sorted(_REQUIRED_CASES - seen_cases)
    for case_name in missing_cases:
        errors.append({"path": str(path), "field": "case", "error": f"missing required reference case: {case_name}"})
    report["cases"] = len(entries)
    if errors:
        report["status"] = "fail"
    return _finalise_report(report)


def _validate_header(path: Path, payload: dict[str, Any], errors: list[dict[str, object]]) -> None:
    for field in _REQUIRED_HEADER_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                {"path": str(path), "field": field, "error": "reference file requires canonical header metadata"}
            )
    if payload.get("schema_version") != "1.0":
        errors.append({"path": str(path), "field": "schema_version", "error": "schema_version must be '1.0'"})


def _validate_case(
    path: Path,
    index: int,
    case_payload: object,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    if not isinstance(case_payload, dict):
        errors.append({"path": str(path), "index": index, "field": "case", "error": "case must be an object"})
        return None
    case_name = case_payload.get("case")
    if not isinstance(case_name, str) or not case_name.strip():
        errors.append({"path": str(path), "index": index, "field": "case", "error": "case must be a non-empty string"})
        return None
    parameters = case_payload.get("parameters")
    sample_points = case_payload.get("sample_points")
    if not isinstance(parameters, dict):
        errors.append(
            {"path": str(path), "index": index, "field": "parameters", "error": "parameters must be an object"}
        )
        return None
    if not isinstance(sample_points, list) or not sample_points:
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "sample_points",
                "error": "sample_points must be a non-empty array",
            }
        )
        return None
    if not _parameters_are_numeric(path, index, parameters, errors):
        return None

    geometry_args = {key: float(value) for key, value in parameters.items() if key in _GEOMETRY_PARAMETERS}
    geometry = miller_geometry(**geometry_args, n_theta=4, n_period=1)
    max_abs_error = 0.0
    for sample_index, sample in enumerate(sample_points):
        if not isinstance(sample, dict):
            errors.append(
                {"path": str(path), "index": index, "field": "sample_points", "error": "sample point must be an object"}
            )
            continue
        actual = _actual_values_at_theta(parameters, geometry, float(sample.get("theta", np.nan)))
        for field in _REQUIRED_SAMPLE_FIELDS:
            expected_value = sample.get(field)
            if isinstance(expected_value, bool) or not isinstance(expected_value, int | float):
                errors.append(
                    {"path": str(path), "index": index, "field": field, "error": "sample field must be numeric"}
                )
                continue
            abs_error = abs(actual[field] - float(expected_value))
            max_abs_error = max(max_abs_error, abs_error)
            if not np.isclose(actual[field], float(expected_value), rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE):
                errors.append(
                    {
                        "path": str(path),
                        "index": index,
                        "sample_index": sample_index,
                        "field": field,
                        "error": f"geometry value drifted: expected {expected_value}, got {actual[field]}",
                    }
                )
    if any(error.get("index") == index for error in errors):
        return None
    return {
        "case": case_name,
        "samples": len(sample_points),
        "max_abs_error": max_abs_error,
        "case_sha256": _json_sha256(case_payload),
    }


def _parameters_are_numeric(
    path: Path,
    index: int,
    parameters: dict[object, object],
    errors: list[dict[str, object]],
) -> bool:
    required_parameters = {"R0", "a", "rho", "kappa", "delta", "dR_dr", "q", "B0"}
    ok = True
    for field in sorted(required_parameters):
        value = parameters.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append({"path": str(path), "index": index, "field": field, "error": "parameter must be numeric"})
            ok = False
    return ok


def _actual_values_at_theta(parameters: dict[object, object], geometry: Any, theta: float) -> dict[str, float]:
    index = int(np.argmin(np.abs(geometry.theta - theta)))
    R = float(geometry.R[index])
    return {
        "theta": float(geometry.theta[index]),
        "R": R,
        "Z": float(geometry.Z[index]),
        "jacobian": float(geometry.jacobian[index]),
        "g_rr": float(geometry.g_rr[index]),
        "g_rt": float(geometry.g_rt[index]),
        "g_tt": float(geometry.g_tt[index]),
        "B_toroidal": float(parameters["B0"]) * float(parameters["R0"]) / R,
        "b_dot_grad_theta": float(geometry.b_dot_grad_theta[index]),
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _new_report(path: Path) -> dict[str, Any]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "status": "pass",
        "reference_path": _portable_path(path),
        "reference_file_sha256": None,
        "payload_sha256": None,
        "tolerances": {"absolute": _ABS_TOLERANCE, "relative": _REL_TOLERANCE},
        "units": dict(_UNITS),
        "public_claims": {
            "bounded_local_miller_geometry_reference": False,
            "full_equilibrium_reconstruction": False,
            "full_equilibrium_blocked_reason": _FULL_EQUILIBRIUM_BLOCKED_REASON,
        },
        "cases": 0,
        "entries": [],
        "errors": [],
    }


def _json_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _finalise_report(report: dict[str, Any]) -> dict[str, Any]:
    report["public_claims"]["bounded_local_miller_geometry_reference"] = report["status"] == "pass"
    payload = dict(report)
    payload["payload_sha256"] = None
    report["payload_sha256"] = _json_sha256(payload)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-path",
        default=str(ROOT / "validation" / "reference_data" / "gk_geometry" / "miller_reference_cases.json"),
        help="Immutable Miller geometry reference case JSON",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_gk_geometry_reference(args.reference_path)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK geometry reference: {report['status']} cases={report['cases']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
