# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_density_reference import validate_density_reference


def _valid_density_reference_artifact() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1088/0029-5515/39/12/301",
        "model_id": "density-control-particle-source",
        "model_version": "0.19.0",
        "reference_dataset_id": "density-fuelling-reference-2026-05-20",
        "reference_artifact_sha256": "5" * 64,
        "reference_case_count": 8,
        "executed_at": "2026-05-20T05:00:00Z",
        "radial_grid": {"n_rho": 64, "major_radius_m": 6.2, "minor_radius_m": 2.0},
        "actuator_metadata": {
            "gas_puff_rate_particles_s": 3.0e21,
            "pellet_radius_mm": 2.0,
            "pellet_speed_m_s": 500.0,
            "nbi_energy_keV": 80.0,
            "nbi_power_MW": 5.0,
            "cryopump_speed_m3_s": 8.0,
            "recycling_coefficient": 0.97,
        },
        "units": {
            "density": "m^-3",
            "particle_rate": "s^-1",
            "radius": "m",
            "diffusivity": "m^2/s",
            "pinch_velocity": "m/s",
            "time": "s",
            "greenwald_fraction": "1",
        },
        "metrics": {
            "pellet_deposition_rmse": 0.025,
            "recycling_source_relative_error": 0.04,
            "greenwald_fraction_abs_error": 0.015,
            "density_profile_relative_error": 0.05,
        },
        "tolerances": {
            "pellet_deposition_rmse": 0.05,
            "recycling_source_relative_error": 0.08,
            "greenwald_fraction_abs_error": 0.03,
            "density_profile_relative_error": 0.1,
        },
    }


def test_strict_density_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no density reference artifacts found"


def test_density_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "density_public_reference.json"
    artifact.write_text(json.dumps(_valid_density_reference_artifact()), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_case_count"] == 8


def test_density_gate_accepts_measured_fuelling_campaign(tmp_path: Path) -> None:
    payload = _valid_density_reference_artifact()
    payload["source"] = "measured_fuelling_campaign"
    payload.pop("reference_doi")
    payload["shot_id"] = "DIII-D:163303"
    payload["diagnostic_uri"] = "mdsplus://DIII-D/163303/electron_density"
    artifact = tmp_path / "measured_density_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "measured_fuelling_campaign"


def test_density_gate_accepts_external_integrated_modelling(tmp_path: Path) -> None:
    payload = _valid_density_reference_artifact()
    payload["source"] = "external_integrated_modelling"
    payload.pop("reference_doi")
    payload["external_code"] = "ASTRA"
    payload["reference_artifact_uri"] = "file:///validation/reports/density/astra_fuelling_profile.nc"
    artifact = tmp_path / "external_density_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "external_integrated_modelling"


def test_density_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_density_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_density_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_density_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_density_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["pellet_deposition_rmse"] = 0.2
    artifact = tmp_path / "bad_density_metric.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "pellet_deposition_rmse"


def test_density_gate_rejects_missing_actuator_metadata(tmp_path: Path) -> None:
    payload = _valid_density_reference_artifact()
    actuator_metadata = cast(dict[str, object], payload["actuator_metadata"])
    actuator_metadata.pop("pellet_speed_m_s")
    artifact = tmp_path / "bad_density_metadata.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_density_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "actuator_metadata"
