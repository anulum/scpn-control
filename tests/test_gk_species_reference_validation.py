# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species reference validation tests

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from validation import validate_gk_species_reference as gk_species_ref
from validation.validate_gk_species_reference import validate_gk_species_reference

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASES = ROOT / "validation" / "reference_data" / "gk_species" / "species_collision_reference_cases.json"


def test_repo_src_bootstrap_supports_direct_script_execution(monkeypatch) -> None:
    repo_src = str(Path(gk_species_ref.__file__).resolve().parents[1] / "src")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != repo_src])

    gk_species_ref.ensure_repo_src_on_path()

    assert sys.path[0] == repo_src


def test_repository_species_reference_cases_pass() -> None:
    report = validate_gk_species_reference(REFERENCE_CASES)

    assert report["status"] == "pass"
    assert report["schema_version"] == gk_species_ref.REPORT_SCHEMA_VERSION
    assert (
        report["reference_path"]
        == "<repo-root>/validation/reference_data/gk_species/species_collision_reference_cases.json"
    )
    assert report["cases"] == 4
    assert report["bounded_operator_reference_admitted"] is True
    assert report["full_fidelity_claim_admitted"] is False
    assert report["blocked_reasons"] == [
        "field_particle_momentum_conservation_evidence",
        "external_fokker_planck_reference",
    ]
    assert gk_species_ref.verify_payload_digest(report) is True
    assert {entry["case"] for entry in report["entries"]} == {
        "deuterium_cbc_main_ion",
        "kinetic_electron_cbc",
        "carbon_impurity_edge",
        "hot_deuterium_extreme_temperature",
    }
    assert all(len(entry["case_sha256"]) == 64 for entry in report["entries"])
    assert all(set(entry["units"]) == set(gk_species_ref.EXPECTED_UNITS) for entry in report["entries"])
    assert report["operator_checks"]["bessel_j0"][1]["actual"] == pytest.approx(0.938469807240813)
    assert report["operator_checks"]["velocity_grid"]["actual"]["energy_weight_sum"] == pytest.approx(6.0)
    assert report["operator_checks"]["velocity_grid"]["actual"]["lambda_weight_sum"] == pytest.approx(1.0)
    assert report["operator_checks"]["pitch_angle_operator"]["constant_nullspace_max_abs"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert report["operator_checks"]["pitch_angle_operator"]["tridiagonal_nonzero_entries"] == 9


def test_species_reference_gate_rejects_missing_required_case(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["cases"] = [payload["cases"][0]]
    path = tmp_path / "species_collision_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_species_reference(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "case" for error in report["errors"])
    assert gk_species_ref.verify_payload_digest(report) is True


def test_species_reference_gate_rejects_duplicate_case_name(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["cases"][1] = payload["cases"][0]
    path = tmp_path / "species_collision_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_species_reference(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "case" and "duplicate" in error["error"] for error in report["errors"])


def test_species_reference_gate_rejects_collision_drift(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["cases"][0]["expected"]["nu_D_s^-1"] *= 2.0
    path = tmp_path / "species_collision_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_species_reference(path)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "nu_D_s^-1"


def test_species_reference_gate_rejects_diamagnetic_drive_drift(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["cases"][1]["expected"]["omega_star_pressure"] *= -1.0
    path = tmp_path / "species_collision_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_species_reference(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "omega_star_pressure" for error in report["errors"])


def test_species_reference_gate_rejects_bessel_drift(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["operator_checks"]["bessel_j0"][2]["expected"] = 0.5
    path = tmp_path / "species_collision_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_species_reference(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "operator_checks.bessel_j0" for error in report["errors"])


def test_species_reference_gate_rejects_velocity_grid_drift(tmp_path: Path) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["operator_checks"]["velocity_grid"]["lambda_weight_sum"] = 0.75
    path = tmp_path / "species_collision_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_species_reference(path)

    assert report["status"] == "fail"
    assert any(error["field"] == "operator_checks.velocity_grid.lambda_weight_sum" for error in report["errors"])


def test_species_reference_gate_rejects_pitch_angle_operator_drift(
    tmp_path: Path,
) -> None:
    payload = json.loads(REFERENCE_CASES.read_text(encoding="utf-8"))
    payload["operator_checks"]["pitch_angle_operator"]["tridiagonal_nonzero_entries"] = 7
    path = tmp_path / "species_collision_reference_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_species_reference(path)

    assert report["status"] == "fail"
    assert any(
        error["field"] == "operator_checks.pitch_angle_operator.tridiagonal_nonzero_entries"
        for error in report["errors"]
    )


def test_species_reference_report_digest_rejects_tampering() -> None:
    report = validate_gk_species_reference(REFERENCE_CASES)
    report["entries"][0]["max_relative_error"] = 1.0

    assert gk_species_ref.verify_payload_digest(report) is False
