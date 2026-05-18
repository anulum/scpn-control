# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species reference validation tests

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_gk_species_reference import validate_gk_species_reference


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "gk_species" / "species_collision_reference_cases.json"


def test_repository_species_reference_cases_pass() -> None:
    report = validate_gk_species_reference(REFERENCE_CASES)

    assert report["status"] == "pass"
    assert report["cases"] == 4
    assert {entry["case"] for entry in report["entries"]} == {
        "deuterium_cbc_main_ion",
        "kinetic_electron_cbc",
        "carbon_impurity_edge",
        "hot_deuterium_extreme_temperature",
    }


def test_species_reference_gate_rejects_missing_required_case(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["cases"] = [payload["cases"][0]]
    path = tmp_path / "species_collision_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_species_reference(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "case" for error in report["errors"])


def test_species_reference_gate_rejects_collision_drift(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["cases"][0]["expected"]["nu_D_s^-1"] *= 2.0
    path = tmp_path / "species_collision_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_species_reference(path)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "nu_D_s^-1"
