#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Linear GK cross-code evidence validator

"""Validate real external-code evidence for linear gyrokinetic agreement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from validation.reference_uri import external_executable_path_error

ROOT = Path(__file__).resolve().parents[1]

_ALLOWED_EXTERNAL_CODES = {"TGLF", "GENE", "GS2", "CGYRO", "GYRO", "QuaLiKiz"}
_REQUIRED_STR_FIELDS = (
    "case",
    "external_code",
    "source",
    "binary_path",
    "code_version",
    "run_id",
    "executed_at",
    "units",
)
_REQUIRED_FLOAT_FIELDS = (
    "gamma_max_cs_over_a",
    "omega_r_cs_over_a",
    "k_y_rho_s_at_max",
    "native_gamma_max_cs_over_a",
    "native_omega_r_cs_over_a",
    "native_k_y_rho_s_at_max",
)
_MAX_GAMMA_RELATIVE_ERROR = 0.20
_MAX_OMEGA_RELATIVE_ERROR = 0.30
_MAX_KY_ABSOLUTE_ERROR = 0.10


def validate_gk_crosscode_evidence(
    evidence_root: str | Path,
    *,
    require_external_runs: bool = False,
) -> dict[str, Any]:
    """Validate immutable real-binary external GK comparison evidence."""
    root = Path(evidence_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report: dict[str, Any] = {
        "status": "pass",
        "root": str(root),
        "external_runs": 0,
        "require_external_runs": bool(require_external_runs),
        "entries": [],
        "errors": [],
    }
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]

    if require_external_runs and not paths:
        errors.append(
            {"path": str(root), "field": "evidence_root", "error": "no real external GK evidence reports found"}
        )

    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
            entry = _validate_evidence_payload(path, payload, errors)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "field": "json", "error": str(exc)})
            continue
        if entry is not None:
            entries.append(entry)
            report["external_runs"] += 1

    if require_external_runs and report["external_runs"] == 0 and not errors:
        errors.append(
            {"path": str(root), "field": "evidence_root", "error": "no real external GK evidence reports found"}
        )
    if errors:
        report["status"] = "fail"
    return report


def _validate_evidence_payload(
    path: Path, payload: object, errors: list[dict[str, object]]
) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "evidence report root must be an object"})
        return None
    if payload.get("schema_version") != "1.0":
        errors.append({"path": str(path), "field": "schema_version", "error": "schema_version must be '1.0'"})
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": str(path), "field": field, "error": "field must be a non-empty string"})
    for field in _REQUIRED_FLOAT_FIELDS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append({"path": str(path), "field": field, "error": "field must be numeric"})
    if payload.get("source") != "real_binary":
        errors.append({"path": str(path), "field": "source", "error": "source must be real_binary"})
    if payload.get("external_code") not in _ALLOWED_EXTERNAL_CODES:
        errors.append({"path": str(path), "field": "external_code", "error": "unsupported external GK code"})
    binary_path_error = external_executable_path_error(payload.get("binary_path"))
    if binary_path_error is not None:
        errors.append({"path": str(path), "field": "binary_path", "error": binary_path_error})
    if any(error["path"] == str(path) for error in errors):
        return None

    gamma = float(payload["gamma_max_cs_over_a"])
    native_gamma = float(payload["native_gamma_max_cs_over_a"])
    omega = float(payload["omega_r_cs_over_a"])
    native_omega = float(payload["native_omega_r_cs_over_a"])
    ky = float(payload["k_y_rho_s_at_max"])
    native_ky = float(payload["native_k_y_rho_s_at_max"])
    gamma_error = abs(native_gamma - gamma) / max(abs(gamma), 1e-12)
    omega_error = abs(native_omega - omega) / max(abs(omega), 1e-12)
    ky_error = abs(native_ky - ky)

    if gamma_error > _MAX_GAMMA_RELATIVE_ERROR:
        errors.append({"path": str(path), "field": "gamma_max_cs_over_a", "error": "native/external gamma mismatch"})
    if omega_error > _MAX_OMEGA_RELATIVE_ERROR:
        errors.append({"path": str(path), "field": "omega_r_cs_over_a", "error": "native/external omega mismatch"})
    if ky_error > _MAX_KY_ABSOLUTE_ERROR:
        errors.append({"path": str(path), "field": "k_y_rho_s_at_max", "error": "native/external k_y mismatch"})
    if any(error["path"] == str(path) for error in errors):
        return None

    return {
        "path": str(path),
        "case": str(payload["case"]),
        "external_code": str(payload["external_code"]),
        "run_id": str(payload["run_id"]),
        "gamma_relative_error": gamma_error,
        "omega_relative_error": omega_error,
        "k_y_absolute_error": ky_error,
    }


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
        "--evidence-root",
        default=str(ROOT / "validation" / "reports" / "gk_crosscode"),
        help="Directory or JSON report containing real external-code comparison evidence",
    )
    parser.add_argument(
        "--require-external-runs",
        action="store_true",
        help="Fail if no real external-code evidence reports are present",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_gk_crosscode_evidence(args.evidence_root, require_external_runs=args.require_external_runs)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK cross-code evidence: {report['status']} external_runs={report['external_runs']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
