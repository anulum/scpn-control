#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — External GK interface artifact validator

"""Validate persisted external gyrokinetic interface parser artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from validation.reference_uri import external_executable_path_error

ROOT = Path(__file__).resolve().parents[1]

_ALLOWED_CODES = {"TGLF", "GENE", "GS2", "CGYRO", "QuaLiKiz"}
_ALLOWED_SOURCES = {"real_executable", "documented_public_reference"}
_REQUIRED_STR_FIELDS = (
    "interface_code",
    "source",
    "code_version",
    "run_id",
    "executed_at",
    "input_deck_sha256",
    "output_artifact_sha256",
    "parser_version",
    "units",
)
_REQUIRED_NUMERIC_FIELDS = (
    "chi_i_m2_s",
    "chi_e_m2_s",
    "D_e_m2_s",
    "gamma_max_cs_over_a",
    "omega_r_cs_over_a",
    "k_y_rho_s_at_max",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_gk_interface_artifacts(
    artifact_root: str | Path,
    *,
    require_interface_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate external GK parser artifacts and reject mock-only evidence."""
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report: dict[str, Any] = {
        "status": "pass",
        "root": str(root),
        "interface_artifacts": 0,
        "require_interface_artifacts": bool(require_interface_artifacts),
        "entries": [],
        "errors": [],
    }
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]

    if require_interface_artifacts and not paths:
        errors.append(
            {"path": str(root), "field": "artifact_root", "error": "no external GK interface artifacts found"}
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
            report["interface_artifacts"] += 1

    if require_interface_artifacts and report["interface_artifacts"] == 0 and not errors:
        errors.append(
            {"path": str(root), "field": "artifact_root", "error": "no external GK interface artifacts found"}
        )
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
    for field in ("input_deck_sha256", "output_artifact_sha256"):
        value = payload.get(field)
        if isinstance(value, str) and not _SHA256_RE.match(value):
            errors.append({"path": str(path), "field": field, "error": "field must be a SHA-256 hex digest"})
    for field in _REQUIRED_NUMERIC_FIELDS:
        if not _is_finite_number(payload.get(field)):
            errors.append({"path": str(path), "field": field, "error": "field must be finite numeric"})
    if payload.get("interface_code") not in _ALLOWED_CODES:
        errors.append({"path": str(path), "field": "interface_code", "error": "unsupported external GK interface code"})
    if payload.get("source") not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": str(path),
                "field": "source",
                "error": "source must be real_executable or documented_public_reference",
            }
        )

    source = payload.get("source")
    if source == "real_executable":
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
    if any(error["path"] == str(path) for error in errors):
        return None

    chi_i = float(payload["chi_i_m2_s"])
    chi_e = float(payload["chi_e_m2_s"])
    d_e = float(payload["D_e_m2_s"])
    gamma = float(payload["gamma_max_cs_over_a"])
    ky = float(payload["k_y_rho_s_at_max"])
    if chi_i < 0.0:
        errors.append({"path": str(path), "field": "chi_i_m2_s", "error": "transport coefficient must be non-negative"})
    if chi_e < 0.0:
        errors.append({"path": str(path), "field": "chi_e_m2_s", "error": "transport coefficient must be non-negative"})
    if d_e < 0.0:
        errors.append({"path": str(path), "field": "D_e_m2_s", "error": "transport coefficient must be non-negative"})
    if gamma < 0.0:
        errors.append(
            {"path": str(path), "field": "gamma_max_cs_over_a", "error": "dominant growth rate must be non-negative"}
        )
    if ky <= 0.0:
        errors.append({"path": str(path), "field": "k_y_rho_s_at_max", "error": "dominant wavenumber must be positive"})
    if any(error["path"] == str(path) for error in errors):
        return None

    return {
        "path": str(path),
        "interface_code": str(payload["interface_code"]),
        "source": str(payload["source"]),
        "run_id": str(payload["run_id"]),
        "code_version": str(payload["code_version"]),
    }


def _has_public_reference(payload: dict[str, object]) -> bool:
    for field in ("reference_url", "reference_doi"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


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
        default=str(ROOT / "validation" / "reports" / "gk_interfaces"),
        help="Directory or JSON artifact containing persisted external GK interface evidence",
    )
    parser.add_argument(
        "--require-interface-artifacts", action="store_true", help="Fail if no external interface artifacts are present"
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_gk_interface_artifacts(
        args.artifact_root, require_interface_artifacts=args.require_interface_artifacts
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK interface artifacts: {report['status']} interface_artifacts={report['interface_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
