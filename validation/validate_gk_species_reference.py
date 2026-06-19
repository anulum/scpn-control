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
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def ensure_repo_src_on_path() -> None:
    """Allow direct script execution from a source checkout without installation."""

    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)


ensure_repo_src_on_path()

from scpn_control.core.gk_species import (
    GKSpecies,
    VelocityGrid,
    bessel_j0,
    collision_frequencies,
    diamagnetic_frequencies,
    pitch_angle_operator,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = "scpn-control.gk-species-reference.v3"
EXPECTED_UNITS = {
    "mass_kg": "kg",
    "thermal_speed_m_per_s": "m/s",
    "larmor_radius_per_tesla_m": "m/T",
    "nu_D_s^-1": "s^-1",
    "nu_E_s^-1": "s^-1",
    "omega_star_density": "dimensionless",
    "omega_star_temperature": "dimensionless",
    "omega_star_pressure": "dimensionless",
}
FULL_FIDELITY_BLOCKERS = (
    "field_particle_momentum_conservation_evidence",
    "external_fokker_planck_reference",
)

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
_REQUIRED_SPECIES_FIELDS = (
    "mass_amu",
    "charge_e",
    "temperature_keV",
    "density_19",
    "R_L_T",
    "R_L_n",
)
_REQUIRED_COLLISION_FIELDS = ("n_e_19", "T_e_keV", "Z_eff", "ln_lambda")
_REQUIRED_DRIVE_FIELDS = ("k_y_rho_s",)
_EXPECTED_FIELDS = (
    "mass_kg",
    "thermal_speed_m_per_s",
    "larmor_radius_per_tesla_m",
    "nu_D_s^-1",
    "nu_E_s^-1",
    "omega_star_density",
    "omega_star_temperature",
    "omega_star_pressure",
)
_ABS_TOLERANCE = 1.0e-12
_REL_TOLERANCE = 1.0e-10


def _canonical_json(value: Any) -> str:
    """Serialise evidence deterministically for SHA-256 binding."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_payload(value: Any) -> str:
    """Return the canonical SHA-256 digest for a JSON-compatible payload."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _payload_without_digest(report: dict[str, Any]) -> dict[str, Any]:
    """Return a report copy with the self-referential digest field removed."""

    payload = dict(report)
    payload.pop("payload_sha256", None)
    return payload


def _portable_reference_path(path: Path) -> str:
    """Return a stable in-repository path for persisted reports."""

    try:
        return f"<repo-root>/{path.resolve().relative_to(ROOT).as_posix()}"
    except ValueError:
        return str(path)


def verify_payload_digest(report: dict[str, Any]) -> bool:
    """Return true when a persisted GK species report matches its digest."""

    digest = report.get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    return digest == _sha256_payload(_payload_without_digest(report))


def _finalise_report(report: dict[str, Any], reference_path: Path) -> dict[str, Any]:
    """Attach schema, claim boundary, and payload digest to a report."""

    try:
        reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    except OSError:
        reference_sha256 = None
    status = report.get("status")
    report.update(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "reference_sha256": reference_sha256,
            "bounded_operator_reference_admitted": status == "pass",
            "full_fidelity_claim_admitted": False,
            "blocked_reasons": list(FULL_FIDELITY_BLOCKERS),
            "claim_status": (
                "bounded GK species and test-particle collision reference admitted; "
                "full collision-operator claim remains blocked"
            ),
            "tolerances": {"absolute": _ABS_TOLERANCE, "relative": _REL_TOLERANCE},
        }
    )
    report["payload_sha256"] = _sha256_payload(report)
    return report


def validate_gk_species_reference(reference_path: str | Path) -> dict[str, Any]:
    """Validate repository species and collision outputs against reference cases."""
    path = Path(reference_path)
    report: dict[str, Any] = {
        "status": "pass",
        "reference_path": _portable_reference_path(path),
        "cases": 0,
        "entries": [],
        "operator_checks": {},
        "errors": [],
    }
    entries: list[dict[str, object]] = report["entries"]
    errors: list[dict[str, object]] = report["errors"]
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.update(
            {
                "status": "fail",
                "errors": [{"path": str(path), "field": "json", "error": str(exc)}],
            }
        )
        return _finalise_report(report, path)

    if not isinstance(payload, dict):
        errors.append(
            {
                "path": str(path),
                "field": "root",
                "error": "reference root must be an object",
            }
        )
        report["status"] = "fail"
        return _finalise_report(report, path)
    _validate_header(path, payload, errors)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append(
            {
                "path": str(path),
                "field": "cases",
                "error": "cases must be a non-empty array",
            }
        )
        report["status"] = "fail"
        return _finalise_report(report, path)

    seen_cases: set[str] = set()
    for index, case_payload in enumerate(cases):
        entry = _validate_case(path, index, case_payload, errors)
        if entry is not None:
            case_name = str(entry["case"])
            if case_name in seen_cases:
                errors.append(
                    {
                        "path": str(path),
                        "index": index,
                        "field": "case",
                        "error": f"duplicate case: {case_name}",
                    }
                )
            else:
                seen_cases.add(case_name)
                entries.append(entry)
    for case_name in sorted(_REQUIRED_CASES - seen_cases):
        errors.append(
            {
                "path": str(path),
                "field": "case",
                "error": f"missing required reference case: {case_name}",
            }
        )
    operator_checks = _validate_operator_checks(
        path, payload.get("operator_checks"), errors
    )
    if operator_checks is not None:
        report["operator_checks"] = operator_checks
    report["cases"] = len(entries)
    if errors:
        report["status"] = "fail"
    return _finalise_report(report, path)


def _validate_header(
    path: Path, payload: dict[str, Any], errors: list[dict[str, object]]
) -> None:
    for field in _REQUIRED_HEADER_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                {
                    "path": str(path),
                    "field": field,
                    "error": "reference file requires canonical header metadata",
                }
            )
    if payload.get("schema_version") != "1.0":
        errors.append(
            {
                "path": str(path),
                "field": "schema_version",
                "error": "schema_version must be '1.0'",
            }
        )


def _validate_case(
    path: Path, index: int, case_payload: object, errors: list[dict[str, object]]
) -> dict[str, object] | None:
    if not isinstance(case_payload, dict):
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "case",
                "error": "case must be an object",
            }
        )
        return None
    case_name = case_payload.get("case")
    if not isinstance(case_name, str) or not case_name.strip():
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "case",
                "error": "case must be a non-empty string",
            }
        )
        return None
    species_payload = case_payload.get("species")
    collision_payload = case_payload.get("collision")
    drive_payload = case_payload.get("drive")
    expected = case_payload.get("expected")
    if not isinstance(species_payload, dict):
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "species",
                "error": "species must be an object",
            }
        )
        return None
    if not isinstance(collision_payload, dict):
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "collision",
                "error": "collision must be an object",
            }
        )
        return None
    if not isinstance(drive_payload, dict):
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "drive",
                "error": "drive must be an object",
            }
        )
        return None
    if not isinstance(expected, dict):
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "expected",
                "error": "expected must be an object",
            }
        )
        return None
    if not _object_fields_are_numeric(
        path, index, species_payload, _REQUIRED_SPECIES_FIELDS, errors
    ):
        return None
    if not _object_fields_are_numeric(
        path, index, collision_payload, _REQUIRED_COLLISION_FIELDS, errors
    ):
        return None
    if not _object_fields_are_numeric(
        path, index, drive_payload, _REQUIRED_DRIVE_FIELDS, errors
    ):
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
        omega = diamagnetic_frequencies(
            species, k_y_rho_s=float(drive_payload["k_y_rho_s"])
        )
    except ValueError as exc:
        errors.append(
            {"path": str(path), "index": index, "field": "case", "error": str(exc)}
        )
        return None

    actual = {
        "mass_kg": species.mass_kg,
        "thermal_speed_m_per_s": species.thermal_speed,
        "larmor_radius_per_tesla_m": species.larmor_radius,
        "nu_D_s^-1": nu_d,
        "nu_E_s^-1": nu_e,
        "omega_star_density": omega.density,
        "omega_star_temperature": omega.temperature,
        "omega_star_pressure": omega.pressure,
    }
    max_relative_error = 0.0
    for field in _EXPECTED_FIELDS:
        expected_value = float(expected[field])
        actual_value = actual[field]
        relative_error = abs(actual_value - expected_value) / max(
            abs(expected_value), _ABS_TOLERANCE
        )
        max_relative_error = max(max_relative_error, relative_error)
        if not np.isclose(
            actual_value, expected_value, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE
        ):
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
    return {
        "case": case_name,
        "case_sha256": _sha256_payload(case_payload),
        "max_relative_error": max_relative_error,
        "units": EXPECTED_UNITS,
        "actual": actual,
    }


def _validate_operator_checks(
    path: Path,
    payload: object,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks",
                "error": "operator_checks must be an object",
            }
        )
        return None

    report: dict[str, object] = {}
    _validate_bessel_checks(path, payload.get("bessel_j0"), report, errors)
    _validate_velocity_grid_check(path, payload.get("velocity_grid"), report, errors)
    _validate_pitch_angle_check(
        path, payload.get("pitch_angle_operator"), report, errors
    )
    return report if report else None


def _validate_bessel_checks(
    path: Path,
    payload: object,
    report: dict[str, object],
    errors: list[dict[str, object]],
) -> None:
    if not isinstance(payload, list) or not payload:
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.bessel_j0",
                "error": "bessel_j0 must be a non-empty array",
            }
        )
        return
    entries: list[dict[str, float]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": "bessel_j0",
                    "error": "case must be an object",
                }
            )
            continue
        if not _object_fields_are_numeric(
            path, index, item, ("argument", "expected"), errors
        ):
            continue
        argument = float(item["argument"])
        expected = float(item["expected"])
        actual = float(bessel_j0(np.asarray([argument], dtype=np.float64))[0])
        rel_error = _relative_error(actual, expected)
        if not np.isclose(actual, expected, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE):
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": "operator_checks.bessel_j0",
                    "error": f"Bessel J0 drifted: expected {expected}, got {actual}",
                }
            )
        entries.append(
            {"argument": argument, "actual": actual, "relative_error": rel_error}
        )
    report["bessel_j0"] = entries


def _validate_velocity_grid_check(
    path: Path,
    payload: object,
    report: dict[str, object],
    errors: list[dict[str, object]],
) -> None:
    if not isinstance(payload, dict):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.velocity_grid",
                "error": "velocity_grid must be an object",
            }
        )
        return
    if not _object_fields_are_numeric(
        path,
        -1,
        payload,
        (
            "n_energy",
            "n_lambda",
            "energy_weight_sum",
            "lambda_weight_sum",
            "energy_min",
            "energy_max",
        ),
        errors,
    ):
        return
    try:
        grid = VelocityGrid(
            n_energy=int(payload["n_energy"]), n_lambda=int(payload["n_lambda"])
        )
    except ValueError as exc:
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.velocity_grid",
                "error": str(exc),
            }
        )
        return
    actual = {
        "energy_weight_sum": float(np.sum(grid.energy_weights)),
        "lambda_weight_sum": float(np.sum(grid.lambda_weights)),
        "energy_min": float(grid.energy[0]),
        "energy_max": float(grid.energy[-1]),
    }
    max_relative_error = 0.0
    for field, actual_value in actual.items():
        expected = float(payload[field])
        max_relative_error = max(
            max_relative_error, _relative_error(actual_value, expected)
        )
        if not np.isclose(
            actual_value, expected, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE
        ):
            errors.append(
                {
                    "path": str(path),
                    "field": f"operator_checks.velocity_grid.{field}",
                    "error": f"velocity-grid reference drifted: expected {expected}, got {actual_value}",
                }
            )
    report["velocity_grid"] = {
        "n_energy": grid.n_energy,
        "n_lambda": grid.n_lambda,
        "actual": actual,
        "max_relative_error": max_relative_error,
    }


def _validate_pitch_angle_check(
    path: Path,
    payload: object,
    report: dict[str, object],
    errors: list[dict[str, object]],
) -> None:
    if not isinstance(payload, dict):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator",
                "error": "pitch_angle_operator must be an object",
            }
        )
        return
    lam_payload = payload.get("lambda_grid")
    if not isinstance(lam_payload, list) or not lam_payload:
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator.lambda_grid",
                "error": "lambda_grid must be a non-empty array",
            }
        )
        return
    try:
        lam = np.asarray(
            [_numeric_scalar(value) for value in lam_payload], dtype=np.float64
        )
        b_ratio = _numeric_scalar(payload.get("B_ratio"))
        expected_constant = _numeric_scalar(payload.get("constant_nullspace_max_abs"))
        expected_nonzero = int(
            _numeric_scalar(payload.get("tridiagonal_nonzero_entries"))
        )
    except ValueError as exc:
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator",
                "error": str(exc),
            }
        )
        return
    try:
        matrix = pitch_angle_operator(len(lam), lam, B_ratio=b_ratio)
    except ValueError as exc:
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator",
                "error": str(exc),
            }
        )
        return
    constant_residual = float(np.max(np.abs(matrix @ np.ones_like(lam))))
    nonzero_entries = int(np.count_nonzero(np.abs(matrix) > 0.0))
    outside_band = [
        (int(i), int(j))
        for i, row in enumerate(matrix)
        for j, value in enumerate(row)
        if abs(i - j) > 1 and abs(value) > _ABS_TOLERANCE
    ]
    if not np.isclose(
        constant_residual, expected_constant, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE
    ):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator.constant_nullspace_max_abs",
                "error": f"pitch-angle nullspace drifted: expected {expected_constant}, got {constant_residual}",
            }
        )
    if nonzero_entries != expected_nonzero:
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator.tridiagonal_nonzero_entries",
                "error": f"pitch-angle sparsity drifted: expected {expected_nonzero}, got {nonzero_entries}",
            }
        )
    if outside_band:
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator.tridiagonal",
                "error": f"operator has off-tridiagonal entries: {outside_band}",
            }
        )
    report["pitch_angle_operator"] = {
        "n_lambda": len(lam),
        "B_ratio": b_ratio,
        "constant_nullspace_max_abs": constant_residual,
        "tridiagonal_nonzero_entries": nonzero_entries,
    }


def _numeric_scalar(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(float(value))
    ):
        raise ValueError("field must be a finite numeric scalar")
    return float(value)


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), _ABS_TOLERANCE)


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
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": field,
                    "error": "field must be numeric",
                }
            )
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
        default=str(
            ROOT
            / "validation"
            / "reference_data"
            / "gk_species"
            / "species_collision_reference_cases.json"
        ),
        help="Immutable GK species and collision reference case JSON",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument("--json-out", action="store_true", help="Emit JSON report")
    args = parser.parse_args(argv)

    report = validate_gk_species_reference(args.reference_path)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.json_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GK species reference: {report['status']} cases={report['cases']}")
        for error in report["errors"]:
            print(f"ERROR {error['path']}: {error['error']}", file=sys.stderr)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
