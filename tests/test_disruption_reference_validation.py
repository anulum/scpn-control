# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_disruption_reference import validate_disruption_reference


def _valid_disruption_reference_artifact() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1088/0029-5515/57/1/016016",
        "model_id": "disruption-mitigation-contract-layer",
        "model_version": "0.19.0",
        "reference_dataset_id": "disruption-mitigation-reference-2026-05-20",
        "reference_artifact_sha256": "6" * 64,
        "reference_case_count": 7,
        "executed_at": "2026-05-20T06:00:00Z",
        "signal_window": {
            "sample_count": 512,
            "sample_period_s": 0.001,
            "pre_disruption_duration_s": 0.3,
            "current_quench_duration_ms": 18.0,
            "thermal_quench_duration_ms": 4.0,
        },
        "mitigation_metadata": {
            "neon_quantity_mol": 0.04,
            "argon_quantity_mol": 0.01,
            "xenon_quantity_mol": 0.002,
            "total_impurity_mol": 0.052,
            "mitigation_strength": 0.72,
            "tbr_reference": 1.12,
        },
        "units": {
            "time": "s",
            "quench_time": "ms",
            "current": "MA",
            "energy": "MJ",
            "impurity_inventory": "mol",
            "risk": "1",
            "tbr": "1",
        },
        "metrics": {
            "risk_after_abs_error": 0.025,
            "detection_lead_time_abs_error_ms": 4.0,
            "halo_current_relative_error": 0.08,
            "runaway_beam_relative_error": 0.09,
            "tbr_abs_error": 0.015,
        },
        "tolerances": {
            "risk_after_abs_error": 0.05,
            "detection_lead_time_abs_error_ms": 8.0,
            "halo_current_relative_error": 0.15,
            "runaway_beam_relative_error": 0.18,
            "tbr_abs_error": 0.03,
        },
    }


def test_strict_disruption_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_disruption_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no disruption reference artifacts found"


def test_disruption_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "disruption_public_reference.json"
    artifact.write_text(json.dumps(_valid_disruption_reference_artifact()), encoding="utf-8")

    report = validate_disruption_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_case_count"] == 7


def test_disruption_gate_accepts_measured_campaign(tmp_path: Path) -> None:
    payload = _valid_disruption_reference_artifact()
    payload["source"] = "measured_disruption_campaign"
    payload.pop("reference_doi")
    payload["shot_id"] = "DIII-D:163303"
    payload["diagnostic_uri"] = "mdsplus://DIII-D/163303/disruption_window"
    artifact = tmp_path / "measured_disruption_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_disruption_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "measured_disruption_campaign"


def test_disruption_gate_accepts_external_benchmark(tmp_path: Path) -> None:
    payload = _valid_disruption_reference_artifact()
    payload["source"] = "external_benchmark"
    payload.pop("reference_doi")
    payload["external_code"] = "JOREK"
    payload["reference_artifact_uri"] = "file:///validation/reports/disruption/jorek_mitigation_cases.nc"
    artifact = tmp_path / "external_disruption_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_disruption_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "external_benchmark"


def test_disruption_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_disruption_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_disruption_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_disruption_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_disruption_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_disruption_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["halo_current_relative_error"] = 0.4
    artifact = tmp_path / "bad_disruption_metric.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_disruption_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "halo_current_relative_error"


def test_disruption_gate_rejects_missing_mitigation_metadata(tmp_path: Path) -> None:
    payload = _valid_disruption_reference_artifact()
    mitigation_metadata = cast(dict[str, object], payload["mitigation_metadata"])
    mitigation_metadata.pop("mitigation_strength")
    artifact = tmp_path / "bad_disruption_metadata.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_disruption_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "mitigation_metadata"
