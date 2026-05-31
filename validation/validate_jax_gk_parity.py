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
    "case_parameters_sha256",
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
    require_cases: tuple[str, ...] | list[str] | set[str] | None = None,
    require_backends: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Validate persisted JAX/native GK parity artifacts and backend metadata."""
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    required_cases = _normalise_required_values(require_cases, _ALLOWED_CASES, "case")
    required_backends = _normalise_required_values(require_backends, _ALLOWED_BACKENDS, "backend")
    report: dict[str, Any] = {
        "status": "pass",
        "root": str(root),
        "parity_artifacts": 0,
        "require_parity_artifacts": bool(require_parity_artifacts),
        "required_cases": sorted(required_cases),
        "required_backends": sorted(required_backends),
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

    _validate_required_coverage(
        root,
        entries,
        errors,
        required_cases=required_cases,
        required_backends=required_backends,
    )
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
    if not _is_sha256_hex(payload.get("case_parameters_sha256")):
        errors.append({"path": str(path), "field": "case_parameters_sha256", "error": "field must be SHA-256 hex"})
    if not _is_sha256_hex(payload.get("payload_sha256")):
        errors.append({"path": str(path), "field": "payload_sha256", "error": "field must be SHA-256 hex"})
    elif _sha256_json(payload) != payload.get("payload_sha256"):
        errors.append({"path": str(path), "field": "payload_sha256", "error": "payload digest mismatch"})
    if not isinstance(payload.get("solver_kwargs"), dict) or not payload["solver_kwargs"]:
        errors.append({"path": str(path), "field": "solver_kwargs", "error": "solver_kwargs must be a non-empty object"})
    elif _sha256_json(payload["solver_kwargs"], include_payload_field=True) != payload.get("solver_kwargs_sha256"):
        errors.append({"path": str(path), "field": "solver_kwargs_sha256", "error": "solver kwargs digest mismatch"})
    if not isinstance(payload.get("case_parameters"), dict) or not payload["case_parameters"]:
        errors.append({"path": str(path), "field": "case_parameters", "error": "case_parameters must be a non-empty object"})
    elif _sha256_json(payload["case_parameters"], include_payload_field=True) != payload.get("case_parameters_sha256"):
        errors.append({"path": str(path), "field": "case_parameters_sha256", "error": "case parameters digest mismatch"})
    if not isinstance(payload.get("case_acceptance"), dict) or not payload["case_acceptance"]:
        errors.append({"path": str(path), "field": "case_acceptance", "error": "case_acceptance must be a non-empty object"})
    native_mode_types = _string_list(payload.get("native_mode_types"))
    jax_mode_types = _string_list(payload.get("jax_mode_types"))
    if not native_mode_types:
        errors.append({"path": str(path), "field": "native_mode_types", "error": "mode spectrum must be a non-empty string list"})
    if not jax_mode_types:
        errors.append({"path": str(path), "field": "jax_mode_types", "error": "mode spectrum must be a non-empty string list"})
    if not isinstance(payload.get("native_dominant_mode_type"), str) or not str(payload.get("native_dominant_mode_type")).strip():
        errors.append({"path": str(path), "field": "native_dominant_mode_type", "error": "field must be a non-empty string"})
    if not isinstance(payload.get("jax_dominant_mode_type"), str) or not str(payload.get("jax_dominant_mode_type")).strip():
        errors.append({"path": str(path), "field": "jax_dominant_mode_type", "error": "field must be a non-empty string"})
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
    native_dominant = str(payload["native_dominant_mode_type"]).strip()
    jax_dominant = str(payload["jax_dominant_mode_type"]).strip()
    if native_mode_types != jax_mode_types:
        errors.append({"path": str(path), "field": "mode_types", "error": "JAX/native mode spectrum mismatch"})
    if native_dominant != jax_dominant:
        errors.append({"path": str(path), "field": "dominant_mode_type", "error": "JAX/native dominant mode mismatch"})
    if native_dominant not in native_mode_types or jax_dominant not in jax_mode_types:
        errors.append({"path": str(path), "field": "dominant_mode_type", "error": "dominant mode missing from spectrum"})
    _validate_case_acceptance(
        path,
        payload["case_acceptance"],
        native_mode_types,
        jax_mode_types,
        max(native_gamma, jax_gamma),
        errors,
    )
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
        "native_dominant_mode_type": native_dominant,
        "jax_dominant_mode_type": jax_dominant,
    }


def _normalise_required_values(values: tuple[str, ...] | list[str] | set[str] | None, allowed: set[str], label: str) -> set[str]:
    if values is None:
        return set()
    out: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text not in allowed:
            raise ValueError(f"unsupported required {label}: {text}")
        out.add(text)
    return out


def _validate_required_coverage(
    root: Path,
    entries: list[dict[str, object]],
    errors: list[dict[str, object]],
    *,
    required_cases: set[str],
    required_backends: set[str],
) -> None:
    observed_cases = {str(entry["case"]) for entry in entries if "case" in entry}
    observed_backends = {str(entry["backend"]) for entry in entries if "backend" in entry}
    observed_pairs = {(str(entry["case"]), str(entry["backend"])) for entry in entries if "case" in entry and "backend" in entry}

    for case in sorted(required_cases - observed_cases):
        errors.append({"path": str(root), "field": "required_case", "error": f"missing required case: {case}"})
    for backend in sorted(required_backends - observed_backends):
        errors.append({"path": str(root), "field": "required_backend", "error": f"missing required backend: {backend}"})
    if required_cases and required_backends:
        for case in sorted(required_cases):
            for backend in sorted(required_backends):
                if (case, backend) not in observed_pairs:
                    errors.append(
                        {
                            "path": str(root),
                            "field": "required_case_backend",
                            "error": f"missing required case/backend evidence: {case}/{backend}",
                        }
                    )


def _validate_case_acceptance(
    path: Path,
    case_acceptance: object,
    native_mode_types: list[str],
    jax_mode_types: list[str],
    gamma_max: float,
    errors: list[dict[str, object]],
) -> None:
    if not isinstance(case_acceptance, dict):
        errors.append({"path": str(path), "field": "case_acceptance", "error": "case_acceptance must be an object"})
        return
    required_mode_types = _string_list(case_acceptance.get("required_mode_types"))
    if not required_mode_types:
        errors.append({"path": str(path), "field": "case_acceptance.required_mode_types", "error": "required mode types missing"})
    for mode_type in required_mode_types:
        if mode_type not in native_mode_types or mode_type not in jax_mode_types:
            errors.append(
                {
                    "path": str(path),
                    "field": "case_acceptance.required_mode_types",
                    "error": f"required mode absent from native/JAX spectra: {mode_type}",
                }
            )
    gamma_bound = case_acceptance.get("max_gamma_max_cs_over_a")
    if gamma_bound is not None:
        if isinstance(gamma_bound, bool) or not isinstance(gamma_bound, int | float) or not math.isfinite(float(gamma_bound)):
            errors.append({"path": str(path), "field": "case_acceptance.max_gamma_max_cs_over_a", "error": "gamma bound must be finite numeric or null"})
        elif gamma_max > float(gamma_bound):
            errors.append({"path": str(path), "field": "case_acceptance.max_gamma_max_cs_over_a", "error": "growth exceeds case bound"})


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return []
        out.append(item.strip())
    return out


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


def _split_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


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
    parser.add_argument("--require-cases", help="Comma-separated required parity cases")
    parser.add_argument("--require-backends", help="Comma-separated required JAX backends")
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_jax_gk_parity(
        args.artifact_root,
        require_parity_artifacts=args.require_parity_artifacts,
        require_cases=_split_csv(args.require_cases),
        require_backends=_split_csv(args.require_backends),
    )
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
