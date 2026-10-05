# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species bounded case comparison

"""GK species bounded case comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_control.core.gk_species import GKSpecies, collision_frequencies, diamagnetic_frequencies
from validation.gk_species_reference_contracts import (
    _ABS_TOLERANCE,
    _EXPECTED_FIELDS,
    _REL_TOLERANCE,
    EXPECTED_UNITS,
    _sha256_payload,
)
from validation.gk_species_reference_numeric import _numeric_scalar, _object_fields_are_numeric

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


def _validate_case(
    path: Path, index: int, case_payload: object, errors: list[dict[str, object]]
) -> dict[str, object] | None:
    """Compare original species, collision and drive fields with finite input and computed scalars."""
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
    if not _object_fields_are_numeric(path, index, species_payload, _REQUIRED_SPECIES_FIELDS, errors):
        return None
    if not _object_fields_are_numeric(path, index, collision_payload, _REQUIRED_COLLISION_FIELDS, errors):
        return None
    if not _object_fields_are_numeric(path, index, drive_payload, _REQUIRED_DRIVE_FIELDS, errors):
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
        omega = diamagnetic_frequencies(species, k_y_rho_s=float(drive_payload["k_y_rho_s"]))
    except (ValueError, ArithmeticError):
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "case",
                "error": "species or collision parameters are outside the computed finite domain",
            }
        )
        return None

    try:
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
        for value in actual.values():
            _numeric_scalar(value)
    except (ValueError, ArithmeticError):
        errors.append(
            {"path": str(path), "index": index, "field": "case", "error": "computed species fields must be finite"}
        )
        return None
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
                    "error": "species reference drifted beyond declared tolerance",
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
