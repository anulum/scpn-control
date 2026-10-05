# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species reference validation tests

"""Exercise original species scalar domains through persisted public comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from validation.validate_gk_species_reference import validate_gk_species_reference

REFERENCE = (
    Path(__file__).resolve().parents[1] / "validation/reference_data/gk_species/species_collision_reference_cases.json"
)
FIELDS = (
    [("species", field) for field in ("mass_amu", "charge_e", "temperature_keV", "density_19", "R_L_T", "R_L_n")]
    + [("collision", field) for field in ("n_e_19", "T_e_keV", "Z_eff", "ln_lambda")]
    + [("drive", "k_y_rho_s")]
    + [
        ("expected", field)
        for field in (
            "mass_kg",
            "thermal_speed_m_per_s",
            "larmor_radius_per_tesla_m",
            "nu_D_s^-1",
            "nu_E_s^-1",
            "omega_star_density",
            "omega_star_temperature",
            "omega_star_pressure",
        )
    ]
)


def declaration() -> dict[str, Any]:
    """Read the actual four-species canonical reference without rewriting it."""
    return cast(dict[str, Any], json.loads(REFERENCE.read_text()))


def inspect(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a canonical-reference mutation and execute the public species reader."""
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload))
    return validate_gk_species_reference(path)


@pytest.mark.parametrize(("block", "field"), FIELDS)
@pytest.mark.parametrize("value", [None, [], "1", True, float("nan"), float("inf"), 10**400])
def test_required_scalar_findings(tmp_path: Path, block: str, field: str, value: object) -> None:
    """Each physical/expected scalar refuses malformed and nonfinite values before coefficients or digest serialization."""
    payload = declaration()
    payload["cases"][0][block][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == field for error in report["errors"])
    assert report["full_fidelity_claim_admitted"] is False


@pytest.mark.parametrize(
    ("block", "field", "value"),
    [
        ("species", "mass_amu", 0),
        ("species", "charge_e", 0),
        ("species", "temperature_keV", 0),
        ("species", "density_19", 0),
        ("collision", "n_e_19", 0),
        ("collision", "T_e_keV", 0),
        ("collision", "Z_eff", 0),
        ("collision", "ln_lambda", 0),
        ("drive", "k_y_rho_s", -1),
    ],
)
def test_actual_physical_domain_refusal(tmp_path: Path, block: str, field: str, value: float) -> None:
    """Actual unchanged species/collision physics rejects nonphysical inputs with a fixed case finding."""
    payload = declaration()
    payload["cases"][0][block][field] = value
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and any(error["field"] == "case" for error in report["errors"])
    assert all("must be positive" not in error["error"] for error in report["errors"])


@pytest.mark.parametrize("value", [False, [], "false", None])
def test_original_adiabatic_coercion(tmp_path: Path, value: object) -> None:
    """The declared adiabatic flag retains original bool coercion without altering coefficient comparisons."""
    payload = declaration()
    payload["cases"][0]["species"]["is_adiabatic"] = value
    assert inspect(tmp_path, payload)["status"] == "pass"


def test_actual_tiny_mass_underflow(tmp_path: Path) -> None:
    """A positive representable mass whose SI product underflows cannot escape as a division error or be admitted."""
    payload = declaration()
    payload["cases"][0]["species"]["mass_amu"] = 5e-324
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail" and report["bounded_operator_reference_admitted"] is False


def test_computed_larmor_coefficient_underflow(tmp_path: Path) -> None:
    """A finite nonzero charge whose Coulomb product vanishes produces a fixed computed-property finding."""
    payload = declaration()
    payload["cases"][0]["species"]["charge_e"] = 5e-324
    report = inspect(tmp_path, payload)
    assert report["status"] == "fail"
    assert any(error["error"] == "computed species fields must be finite" for error in report["errors"])
