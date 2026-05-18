#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX GK parity artifact validator

"""Validate persisted JAX gyrokinetic backend parity artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_ALLOWED_CASES = {"cyclone_base_case", "tem_kinetic_electron", "electromagnetic_kbm", "stable_mode"}
_ALLOWED_BACKENDS = {"cpu", "gpu", "tpu"}
_REQUIRED_STR_FIELDS = (
    "case",
    "backend",
    "jax_version",
    "jaxlib_version",
    "platform",
    "device_kind",
    "dtype",
    "executed_at",
)
_REQUIRED_FLOAT_FIELDS = (
    "native_gamma_max_cs_over_a",
    "jax_gamma_max_cs_over_a",
    "native_omega_r_cs_over_a",
    "jax_omega_r_cs_over_a",
    "gamma_relative_tolerance",
    "omega_absolute_tolerance",
)


def validate_jax_gk_parity(
    artifact_root: str | Path,
    *,
    require_parity_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate persisted JAX/native GK parity artifacts and backend metadata."""
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report: dict[str, Any] = {
        "status": "pass",
        "root": str(root),
        "parity_artifacts": 0,
        "require_parity_artifacts": bool(require_parity_artifacts),
        "entries": [],
        "errors": [],
    }
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]

    if require_parity_artifacts and not paths:
        errors.append({"path": str(root), "field": "artifact_root", "error": "no JAX GK parity artifacts found"})

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
            report["parity_artifacts"] += 1

    if require_parity_artifacts and report["parity_artifacts"] == 0 and not errors:
        errors.append({"path": str(root), "field": "artifact_root", "error": "no JAX GK parity artifacts found"})
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
    if not isinstance(payload.get("x64_enabled"), bool):
        errors.append({"path": str(path), "field": "x64_enabled", "error": "field must be boolean"})
    for field in _REQUIRED_FLOAT_FIELDS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append({"path": str(path), "field": field, "error": "field must be numeric"})
    if payload.get("case") not in _ALLOWED_CASES:
        errors.append({"path": str(path), "field": "case", "error": "unsupported JAX GK parity case"})
    if payload.get("backend") not in _ALLOWED_BACKENDS:
        errors.append({"path": str(path), "field": "backend", "error": "backend must be cpu, gpu, or tpu"})
    if any(error["path"] == str(path) for error in errors):
        return None

    native_gamma = float(payload["native_gamma_max_cs_over_a"])
    jax_gamma = float(payload["jax_gamma_max_cs_over_a"])
    native_omega = float(payload["native_omega_r_cs_over_a"])
    jax_omega = float(payload["jax_omega_r_cs_over_a"])
    gamma_tolerance = float(payload["gamma_relative_tolerance"])
    omega_tolerance = float(payload["omega_absolute_tolerance"])
    if gamma_tolerance <= 0.0:
        errors.append({"path": str(path), "field": "gamma_relative_tolerance", "error": "tolerance must be positive"})
    if omega_tolerance <= 0.0:
        errors.append({"path": str(path), "field": "omega_absolute_tolerance", "error": "tolerance must be positive"})

    gamma_error = abs(jax_gamma - native_gamma) / max(abs(native_gamma), 1e-12)
    omega_error = abs(jax_omega - native_omega)
    if gamma_error > gamma_tolerance:
        errors.append({"path": str(path), "field": "gamma_max_cs_over_a", "error": "JAX/native gamma mismatch"})
    if omega_error > omega_tolerance:
        errors.append({"path": str(path), "field": "omega_r_cs_over_a", "error": "JAX/native omega mismatch"})
    if any(error["path"] == str(path) for error in errors):
        return None

    return {
        "path": str(path),
        "case": str(payload["case"]),
        "backend": str(payload["backend"]),
        "dtype": str(payload["dtype"]),
        "x64_enabled": bool(payload["x64_enabled"]),
        "gamma_relative_error": gamma_error,
        "omega_absolute_error": omega_error,
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
        "--artifact-root",
        default=str(ROOT / "validation" / "reports" / "jax_gk_parity"),
        help="Directory or JSON artifact containing persisted JAX/native GK parity evidence",
    )
    parser.add_argument(
        "--require-parity-artifacts", action="store_true", help="Fail if no persisted parity artifacts are present"
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_jax_gk_parity(args.artifact_root, require_parity_artifacts=args.require_parity_artifacts)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"JAX GK parity: {report['status']} parity_artifacts={report['parity_artifacts']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
