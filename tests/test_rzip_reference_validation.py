# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RZIP reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_rzip_reference import validate_rzip_reference


def _valid_rzip_reference_artifact() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1109/27.55730",
        "model_id": "rzip-rigid-vertical-stability",
        "model_version": "0.19.0",
        "reference_dataset_id": "rzip-lazarus-reference-2026-05-20",
        "reference_artifact_sha256": "4" * 64,
        "reference_case_count": 5,
        "executed_at": "2026-05-20T04:00:00Z",
        "physical_parameters": {
            "major_radius_m": 2.0,
            "minor_radius_m": 0.5,
            "elongation": 1.7,
            "plasma_current_A": 1.0e6,
            "toroidal_field_T": 1.0,
            "vertical_field_index": -0.5,
            "wall_time_constant_s": 0.01,
        },
        "units": {
            "vertical_displacement": "m",
            "growth_rate": "s^-1",
            "growth_time": "ms",
            "coil_current": "A",
            "time": "s",
        },
        "metrics": {
            "growth_rate_relative_error": 0.035,
            "vertical_displacement_rmse_m": 0.0012,
            "closed_loop_pole_real_abs_error": 0.08,
        },
        "tolerances": {
            "growth_rate_relative_error": 0.05,
            "vertical_displacement_rmse_m": 0.003,
            "closed_loop_pole_real_abs_error": 0.12,
        },
    }


def test_strict_rzip_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_rzip_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no RZIP reference artifacts found"


def test_rzip_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "lazarus_rzip_reference.json"
    artifact.write_text(json.dumps(_valid_rzip_reference_artifact()), encoding="utf-8")

    report = validate_rzip_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_case_count"] == 5


def test_rzip_gate_accepts_external_code_benchmark(tmp_path: Path) -> None:
    payload = _valid_rzip_reference_artifact()
    payload["source"] = "external_code_benchmark"
    payload.pop("reference_doi")
    payload["external_code"] = "CREATE-L"
    payload["reference_artifact_uri"] = "file:///validation/reports/rzip/create_l_growth_cases.nc"
    artifact = tmp_path / "create_l_rzip_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_rzip_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "external_code_benchmark"


def test_rzip_gate_accepts_measured_discharge_reference(tmp_path: Path) -> None:
    payload = _valid_rzip_reference_artifact()
    payload["source"] = "measured_discharge"
    payload.pop("reference_doi")
    payload["shot_id"] = "DIII-D:163303"
    payload["diagnostic_uri"] = "mdsplus://DIII-D/163303/vertical_position"
    artifact = tmp_path / "measured_rzip_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_rzip_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "measured_discharge"


def test_rzip_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_rzip_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_rzip_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_rzip_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_rzip_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_rzip_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["growth_rate_relative_error"] = 0.2
    artifact = tmp_path / "bad_growth_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_rzip_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "growth_rate_relative_error"


def test_rzip_gate_rejects_missing_physical_parameter_metadata(tmp_path: Path) -> None:
    payload = _valid_rzip_reference_artifact()
    parameters = cast(dict[str, object], payload["physical_parameters"])
    parameters.pop("wall_time_constant_s")
    artifact = tmp_path / "bad_parameter_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_rzip_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "physical_parameters"
