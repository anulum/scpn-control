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

from validation.reference_uri import external_executable_path_error

_ALLOWED_CODES = {"TGLF", "GENE", "GS2", "CGYRO", "QuaLiKiz"}
_ALLOWED_SOURCES = {"real_executable", "documented_public_reference"}
_SCHEMA_VERSION = "scpn-control.gk-interface-artifact.v1"
_REQUIRED_STR_FIELDS = (
    "interface_code",
    "source",
    "code_version",
    "run_id",
    "executed_at",
    "input_deck_uri",
    "output_artifact_uri",
    "parsed_output_uri",
    "input_deck_sha256",
    "output_artifact_sha256",
    "parsed_output_sha256",
    "payload_sha256",
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
_ARTIFACT_URI_FIELDS = ("input_deck_uri", "output_artifact_uri", "parsed_output_uri")
_SHA256_FIELDS = (
    "input_deck_sha256",
    "output_artifact_sha256",
    "parsed_output_sha256",
    "payload_sha256",
)
_REQUIRED_UNIT_TOKENS = ("m^2/s", "c_s/a", "k_y*rho_s")
_REPORT_SCHEMA = "scpn-control.gk-interface-artifact-report.v2"
_BLOCKED_REASON = "Requires persisted real-executable or documented public-reference GK interface artefacts."


def validate_gk_interface_artifacts(
    artifact_root: str | Path,
    *,
    require_interface_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate external GK parser artefacts and reject mock-only evidence."""
    root = Path(artifact_root)
    paths = sorted(root.glob("*.json")) if root.is_dir() else ([root] if root.is_file() else [])
    report = _new_report(root, require_interface_artifacts=require_interface_artifacts)
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]

    if require_interface_artifacts and not paths:
        errors.append(
            {"path": _portable_path(root), "field": "artifact_root", "error": "no external GK interface artefacts found"}
        )

    seen_code_runs: set[tuple[str, str]] = set()
    for path in paths:
        try:
            raw_payload = path.read_text(encoding="utf-8")
            payload = json.loads(raw_payload, object_pairs_hook=_reject_duplicate_json_keys)
            entry = _validate_artifact(path, raw_payload, payload, errors)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": _portable_path(path), "field": "json", "error": str(exc)})
            continue
        if entry is not None:
            code_run = (str(entry["interface_code"]), str(entry["run_id"]))
            if code_run in seen_code_runs:
                errors.append(
                    {
                        "path": _portable_path(path),
                        "field": "run_id",
                        "error": f"duplicate interface_code/run_id: {code_run[0]} {code_run[1]}",
                    }
                )
                continue
            seen_code_runs.add(code_run)
            entries.append(entry)
            report["interface_artifacts"] += 1

    if require_interface_artifacts and report["interface_artifacts"] == 0 and not errors:
        errors.append(
            {"path": _portable_path(root), "field": "artifact_root", "error": "no external GK interface artefacts found"}
        )
    if errors:
        report["status"] = "fail"
    return _finalise_report(report)


def _validate_artifact(
    path: Path,
    raw_payload: str,
    payload: object,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        errors.append({"path": _portable_path(path), "field": "root", "error": "artefact root must be an object"})
        return None
    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "schema_version",
                "error": f"schema_version must be '{_SCHEMA_VERSION}'",
            }
        )
    for field in _REQUIRED_STR_FIELDS:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append({"path": _portable_path(path), "field": field, "error": "field must be a non-empty string"})
    for field in _SHA256_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and not _SHA256_RE.match(value):
            errors.append({"path": _portable_path(path), "field": field, "error": "field must be a SHA-256 hex digest"})
    for field in _ARTIFACT_URI_FIELDS:
        error = _artifact_uri_error(payload.get(field))
        if error is not None:
            errors.append({"path": _portable_path(path), "field": field, "error": error})
    units = payload.get("units")
    if isinstance(units, str) and not all(token in units for token in _REQUIRED_UNIT_TOKENS):
        errors.append(
            {
                "path": _portable_path(path),
                "field": "units",
                "error": "units must declare m^2/s transport, c_s/a frequencies, and k_y*rho_s wavenumber",
            }
        )
    if isinstance(payload.get("payload_sha256"), str):
        expected = canonical_artifact_sha256(payload)
        observed = str(payload["payload_sha256"])
        if not hmac.compare_digest(observed.lower(), expected):
            errors.append(
                {"path": _portable_path(path), "field": "payload_sha256", "error": "canonical payload digest mismatch"}
            )
    for field in _REQUIRED_NUMERIC_FIELDS:
        if not _is_finite_number(payload.get(field)):
            errors.append({"path": _portable_path(path), "field": field, "error": "field must be finite numeric"})
    if payload.get("interface_code") not in _ALLOWED_CODES:
        errors.append(
            {"path": _portable_path(path), "field": "interface_code", "error": "unsupported external GK interface code"}
        )
    if payload.get("source") not in _ALLOWED_SOURCES:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "source",
                "error": "source must be real_executable or documented_public_reference",
            }
        )

    source = payload.get("source")
    if source == "real_executable":
        binary_path_error = external_executable_path_error(payload.get("binary_path"))
        if binary_path_error is not None:
            errors.append({"path": _portable_path(path), "field": "binary_path", "error": binary_path_error})
    if source == "documented_public_reference" and not _has_public_reference(payload):
        errors.append(
            {
                "path": _portable_path(path),
                "field": "reference",
                "error": "documented public reference artefacts require reference_url or reference_doi",
            }
        )
    if any(error["path"] == _portable_path(path) for error in errors):
        return None

    chi_i = float(payload["chi_i_m2_s"])
    chi_e = float(payload["chi_e_m2_s"])
    d_e = float(payload["D_e_m2_s"])
    gamma = float(payload["gamma_max_cs_over_a"])
    ky = float(payload["k_y_rho_s_at_max"])
    if chi_i < 0.0:
        errors.append(
            {"path": _portable_path(path), "field": "chi_i_m2_s", "error": "transport coefficient must be non-negative"}
        )
    if chi_e < 0.0:
        errors.append(
            {"path": _portable_path(path), "field": "chi_e_m2_s", "error": "transport coefficient must be non-negative"}
        )
    if d_e < 0.0:
        errors.append(
            {"path": _portable_path(path), "field": "D_e_m2_s", "error": "transport coefficient must be non-negative"}
        )
    if gamma < 0.0:
        errors.append(
            {
                "path": _portable_path(path),
                "field": "gamma_max_cs_over_a",
                "error": "dominant growth rate must be non-negative",
            }
        )
    if ky <= 0.0:
        errors.append(
            {"path": _portable_path(path), "field": "k_y_rho_s_at_max", "error": "dominant wavenumber must be positive"}
        )
    if any(error["path"] == _portable_path(path) for error in errors):
        return None

    return {
        "path": _portable_path(path),
        "interface_code": str(payload["interface_code"]),
        "source": str(payload["source"]),
        "run_id": str(payload["run_id"]),
        "code_version": str(payload["code_version"]),
        "artifact_file_sha256": hashlib.sha256(raw_payload.encode("utf-8")).hexdigest(),
        "payload_sha256": str(payload["payload_sha256"]).lower(),
    }


def _has_public_reference(payload: dict[str, object]) -> bool:
    for field in ("reference_url", "reference_doi"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def canonical_artifact_sha256(payload: dict[str, object]) -> str:
    """Return the tamper-evident digest for an external GK interface artifact."""
    canonical_payload = dict(payload)
    canonical_payload.pop("payload_sha256", None)
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_uri_error(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "artefact URI must be a non-empty string"
    ref = value.strip()
    if "\x00" in ref:
        return "artefact URI must not contain NUL bytes"
    if ref.startswith(("http://", "https://", "doi:", "s3://", "gs://")):
        return None
    path = Path(ref)
    if path.is_absolute():
        return "artefact URI must be relative or an admitted external reference URI"
    if any(part == ".." for part in path.parts):
        return "artefact URI must not contain traversal"
    return None


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _new_report(root: Path, *, require_interface_artifacts: bool) -> dict[str, Any]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "status": "pass",
        "root": _portable_path(root),
        "payload_sha256": None,
        "interface_artifacts": 0,
        "require_interface_artifacts": bool(require_interface_artifacts),
        "public_claims": {
            "external_interface_artifacts_admitted": False,
            "full_gk_cross_code_claim_admitted": False,
            "blocked_reason": _BLOCKED_REASON,
        },
        "entries": [],
        "errors": [],
    }


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.as_posix()


def _json_sha256(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _finalise_report(report: dict[str, Any]) -> dict[str, Any]:
    admitted = report["status"] == "pass" and report["interface_artifacts"] > 0
    report["public_claims"]["external_interface_artifacts_admitted"] = admitted
    payload = dict(report)
    payload["payload_sha256"] = None
    report["payload_sha256"] = _json_sha256(payload)
    return report


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
