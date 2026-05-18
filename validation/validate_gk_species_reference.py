#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species reference validator

"""Validate gyrokinetic species normalisation and collision reference cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.core.gk_species import GKSpecies, collision_frequencies

ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_CASES = {
    "deuterium_cbc_main_ion",
    "kinetic_electron_cbc",
    "carbon_impurity_edge",
    "hot_deuterium_extreme_temperature",
}
_REQUIRED_HEADER_FIELDS = (
    "spdx_license_id",
    "commercial_license",
    "concepts_copyright",
    "code_copyright",
    "orcid",
    "contact",
    "file",
)
_REQUIRED_SPECIES_FIELDS = ("mass_amu", "charge_e", "temperature_keV", "density_19", "R_L_T", "R_L_n")
_REQUIRED_COLLISION_FIELDS = ("n_e_19", "T_e_keV", "Z_eff", "ln_lambda")
_EXPECTED_FIELDS = ("mass_kg", "thermal_speed_m_per_s", "larmor_radius_per_tesla_m", "nu_D_s^-1", "nu_E_s^-1")
_ABS_TOLERANCE = 1.0e-12
_REL_TOLERANCE = 1.0e-10


def validate_gk_species_reference(reference_path: str | Path) -> dict[str, Any]:
    """Validate repository species and collision outputs against reference cases."""
    path = Path(reference_path)
    report: dict[str, Any] = {"status": "pass", "reference_path": str(path), "cases": 0, "entries": [], "errors": []}
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {**report, "status": "fail", "errors": [{"path": str(path), "field": "json", "error": str(exc)}]}

    if not isinstance(payload, dict):
        errors.append({"path": str(path), "field": "root", "error": "reference root must be an object"})
        report["status"] = "fail"
        return report
    _validate_header(path, payload, errors)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append({"path": str(path), "field": "cases", "error": "cases must be a non-empty array"})
        report["status"] = "fail"
        return report

    seen_cases: set[str] = set()
    for index, case_payload in enumerate(cases):
        entry = _validate_case(path, index, case_payload, errors)
        if entry is not None:
            seen_cases.add(str(entry["case"]))
            entries.append(entry)
    for case_name in sorted(_REQUIRED_CASES - seen_cases):
        errors.append({"path": str(path), "field": "case", "error": f"missing required reference case: {case_name}"})
    report["cases"] = len(entries)
    if errors:
        report["status"] = "fail"
    return report


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
    path: Path, index: int, case_payload: object, errors: list[dict[str, object]]
) -> dict[str, object] | None:
    if not isinstance(case_payload, dict):
        errors.append({"path": str(path), "index": index, "field": "case", "error": "case must be an object"})
        return None
    case_name = case_payload.get("case")
    if not isinstance(case_name, str) or not case_name.strip():
        errors.append({"path": str(path), "index": index, "field": "case", "error": "case must be a non-empty string"})
        return None
    species_payload = case_payload.get("species")
    collision_payload = case_payload.get("collision")
    expected = case_payload.get("expected")
    if not isinstance(species_payload, dict):
        errors.append({"path": str(path), "index": index, "field": "species", "error": "species must be an object"})
        return None
    if not isinstance(collision_payload, dict):
        errors.append({"path": str(path), "index": index, "field": "collision", "error": "collision must be an object"})
        return None
    if not isinstance(expected, dict):
        errors.append({"path": str(path), "index": index, "field": "expected", "error": "expected must be an object"})
        return None
    if not _object_fields_are_numeric(path, index, species_payload, _REQUIRED_SPECIES_FIELDS, errors):
        return None
    if not _object_fields_are_numeric(path, index, collision_payload, _REQUIRED_COLLISION_FIELDS, errors):
        return None
    if not _object_fields_are_numeric(path, index, expected, _EXPECTED_FIELDS, errors):
        return None

    try:
        species = GKSpecies(
            mass_amu=float(species_payload["mass_amu"]),
            charge_e=float(species_payload["charge_e"]),
            temperature_keV=float(species_payload["temperature_keV"]),
            density_19=float(species_payload["density_19"]),
            R_L_T=float(species_payload["R_L_T"]),
            R_L_n=float(species_payload["R_L_n"]),
            is_adiabatic=bool(species_payload.get("is_adiabatic", False)),
        )
        nu_d, nu_e = collision_frequencies(
            species,
            n_e_19=float(collision_payload["n_e_19"]),
            T_e_keV=float(collision_payload["T_e_keV"]),
            Z_eff=float(collision_payload["Z_eff"]),
            ln_lambda=float(collision_payload["ln_lambda"]),
        )
    except ValueError as exc:
        errors.append({"path": str(path), "index": index, "field": "case", "error": str(exc)})
        return None

    actual = {
        "mass_kg": species.mass_kg,
        "thermal_speed_m_per_s": species.thermal_speed,
        "larmor_radius_per_tesla_m": species.larmor_radius,
        "nu_D_s^-1": nu_d,
        "nu_E_s^-1": nu_e,
    }
    max_relative_error = 0.0
    for field in _EXPECTED_FIELDS:
        expected_value = float(expected[field])
        actual_value = actual[field]
        relative_error = abs(actual_value - expected_value) / max(abs(expected_value), _ABS_TOLERANCE)
        max_relative_error = max(max_relative_error, relative_error)
        if not np.isclose(actual_value, expected_value, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE):
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": field,
                    "error": f"species reference drifted: expected {expected_value}, got {actual_value}",
                }
            )
    if any(error.get("index") == index for error in errors):
        return None
    return {"case": case_name, "max_relative_error": max_relative_error}


def _object_fields_are_numeric(
    path: Path,
    index: int,
    payload: dict[object, object],
    fields: tuple[str, ...],
    errors: list[dict[str, object]],
) -> bool:
    ok = True
    for field in fields:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append({"path": str(path), "index": index, "field": field, "error": "field must be numeric"})
            ok = False
    return ok


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
        "--reference-path",
        default=str(ROOT / "validation" / "reference_data" / "gk_species" / "species_collision_reference_cases.json"),
        help="Immutable GK species and collision reference case JSON",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_gk_species_reference(args.reference_path)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK species reference: {report['status']} cases={report['cases']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
