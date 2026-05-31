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
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_SCHEMA_VERSION = "scpn-control.jax-gk-parity.v1"
_ALLOWED_CASES = {"cyclone_base_case", "tem_kinetic_electron", "electromagnetic_kbm", "stable_mode"}
_ALLOWED_BACKENDS = {"cpu", "gpu", "tpu"}
_REQUIRED_STR_FIELDS = (
    "schema_version",
    "case",
    "backend",
    "jax_version",
    "jaxlib_version",
    "platform",
    "device_kind",
    "dtype",
    "executed_at",
    "solver_contract",
    "normalisation",
    "evidence_boundary",
    "solver_kwargs_sha256",
    "payload_sha256",
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
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": str(path), "field": field, "error": "field must be a non-empty string"})
    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append(
            {"path": str(path), "field": "schema_version", "error": f"schema_version must be {_SCHEMA_VERSION!r}"}
        )
    if not isinstance(payload.get("x64_enabled"), bool):
        errors.append({"path": str(path), "field": "x64_enabled", "error": "field must be boolean"})
    if not isinstance(payload.get("external_validation_required"), bool) or not payload.get("external_validation_required"):
        errors.append(
            {
                "path": str(path),
                "field": "external_validation_required",
                "error": "JAX GK parity artifacts must keep external validation required",
            }
        )
    if not isinstance(payload.get("admitted_for_control"), bool) or payload.get("admitted_for_control"):
        errors.append(
            {
                "path": str(path),
                "field": "admitted_for_control",
                "error": "JAX GK parity artifacts are not control-admission evidence",
            }
        )
    for field in _REQUIRED_FLOAT_FIELDS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            errors.append({"path": str(path), "field": field, "error": "field must be finite numeric"})
    if payload.get("case") not in _ALLOWED_CASES:
        errors.append({"path": str(path), "field": "case", "error": "unsupported JAX GK parity case"})
    if payload.get("backend") not in _ALLOWED_BACKENDS:
        errors.append({"path": str(path), "field": "backend", "error": "backend must be cpu, gpu, or tpu"})
    if payload.get("solver_contract") != "native_linear_gk_local_dispersion":
        errors.append({"path": str(path), "field": "solver_contract", "error": "unsupported solver contract"})
    if payload.get("normalisation") != "c_s_over_a":
        errors.append({"path": str(path), "field": "normalisation", "error": "normalisation must be c_s_over_a"})
    if payload.get("evidence_boundary") != "backend_parity_only":
        errors.append({"path": str(path), "field": "evidence_boundary", "error": "unsupported evidence boundary"})
    if not _is_sha256_hex(payload.get("solver_kwargs_sha256")):
        errors.append({"path": str(path), "field": "solver_kwargs_sha256", "error": "field must be SHA-256 hex"})
    if not _is_sha256_hex(payload.get("payload_sha256")):
        errors.append({"path": str(path), "field": "payload_sha256", "error": "field must be SHA-256 hex"})
    elif _sha256_json(payload) != payload.get("payload_sha256"):
        errors.append({"path": str(path), "field": "payload_sha256", "error": "payload digest mismatch"})
    if not isinstance(payload.get("solver_kwargs"), dict) or not payload["solver_kwargs"]:
        errors.append({"path": str(path), "field": "solver_kwargs", "error": "solver_kwargs must be a non-empty object"})
    elif _sha256_json(payload["solver_kwargs"], include_payload_field=True) != payload.get("solver_kwargs_sha256"):
        errors.append({"path": str(path), "field": "solver_kwargs_sha256", "error": "solver kwargs digest mismatch"})
    if any(error["path"] == str(path) for error in errors):
        return None

    native_gamma = float(payload["native_gamma_max_cs_over_a"])
    jax_gamma = float(payload["jax_gamma_max_cs_over_a"])
    native_omega = float(payload["native_omega_r_cs_over_a"])
    jax_omega = float(payload["jax_omega_r_cs_over_a"])
    gamma_tolerance = float(payload["gamma_relative_tolerance"])
    omega_tolerance = float(payload["omega_absolute_tolerance"])
    if native_gamma < 0.0 or jax_gamma < 0.0:
        errors.append({"path": str(path), "field": "gamma_max_cs_over_a", "error": "growth rates must be non-negative"})
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
        "payload_sha256": str(payload["payload_sha256"]),
        "evidence_boundary": str(payload["evidence_boundary"]),
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _sha256_json(payload: dict[str, Any], *, include_payload_field: bool = False) -> str:
    digest_payload = dict(payload) if include_payload_field else {k: v for k, v in payload.items() if k != "payload_sha256"}
    encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
