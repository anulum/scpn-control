# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_digital_twin_reference import validate_digital_twin_reference


def _valid_digital_twin_reference_artifact() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1016/j.fusengdes.2018.02.001",
        "model_id": "tokamak-digital-twin-topology-runtime",
        "model_version": "0.19.0",
        "reference_dataset_id": "digital-twin-reference-2026-05-20",
        "reference_artifact_sha256": "7" * 64,
        "reference_case_count": 5,
        "executed_at": "2026-05-20T07:00:00Z",
        "grid_metadata": {
            "grid_size": 32,
            "time_steps": 128,
            "seed": 42,
            "state_variables": ["temperature", "density", "q_profile", "island_mask"],
            "has_ids_export": True,
        },
        "actuator_metadata": {
            "actuator_tau_steps": 3,
            "actuator_rate_limit": 0.2,
            "actuator_bias": 0.01,
            "sensor_dropout_prob": 0.02,
            "sensor_noise_std": 0.005,
        },
        "units": {
            "temperature": "keV",
            "density": "m^-3",
            "q_profile": "1",
            "actuator_action": "1",
            "time": "step",
            "ids_pulse": "1",
        },
        "metrics": {
            "final_avg_temp_relative_error": 0.03,
            "q_profile_rmse": 0.04,
            "actuator_lag_abs_error": 0.02,
            "ids_roundtrip_abs_error": 0.0,
            "island_mask_f1_error": 0.06,
        },
        "tolerances": {
            "final_avg_temp_relative_error": 0.07,
            "q_profile_rmse": 0.08,
            "actuator_lag_abs_error": 0.05,
            "ids_roundtrip_abs_error": 0.01,
            "island_mask_f1_error": 0.12,
        },
    }


def test_strict_digital_twin_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_digital_twin_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no digital twin reference artifacts found"


def test_digital_twin_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "digital_twin_public_reference.json"
    artifact.write_text(json.dumps(_valid_digital_twin_reference_artifact()), encoding="utf-8")

    report = validate_digital_twin_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_case_count"] == 5


def test_digital_twin_gate_accepts_measured_discharge_replay(tmp_path: Path) -> None:
    payload = _valid_digital_twin_reference_artifact()
    payload["source"] = "measured_discharge_replay"
    payload.pop("reference_doi")
    payload["shot_id"] = "DIII-D:163303"
    payload["diagnostic_uri"] = "mdsplus://DIII-D/163303/digital_twin_replay"
    artifact = tmp_path / "measured_digital_twin_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_digital_twin_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "measured_discharge_replay"


def test_digital_twin_gate_accepts_external_integrated_modelling(tmp_path: Path) -> None:
    payload = _valid_digital_twin_reference_artifact()
    payload["source"] = "external_integrated_modelling"
    payload.pop("reference_doi")
    payload["external_code"] = "TRANSP"
    payload["reference_artifact_uri"] = "file:///validation/reports/digital_twin/transp_replay_cases.nc"
    artifact = tmp_path / "external_digital_twin_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_digital_twin_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "external_integrated_modelling"


def test_digital_twin_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_digital_twin_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_digital_twin_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_digital_twin_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_digital_twin_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_digital_twin_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["q_profile_rmse"] = 0.2
    artifact = tmp_path / "bad_digital_twin_metric.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_digital_twin_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "q_profile_rmse"


def test_digital_twin_gate_rejects_missing_actuator_metadata(tmp_path: Path) -> None:
    payload = _valid_digital_twin_reference_artifact()
    actuator_metadata = cast(dict[str, object], payload["actuator_metadata"])
    actuator_metadata.pop("actuator_tau_steps")
    artifact = tmp_path / "bad_digital_twin_metadata.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_digital_twin_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "actuator_metadata"
