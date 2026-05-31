# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Orbit reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_orbit_reference import validate_orbit_reference


def _valid_orbit_reference_artifact() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1016/0021-9991(81)90038-4",
        "model_id": "guiding-centre-orbit-following",
        "model_version": "0.19.0",
        "reference_dataset_id": "goldston-first-orbit-reference-2026-05-20",
        "reference_artifact_sha256": "1" * 64,
        "reference_case_count": 12,
        "executed_at": "2026-05-20T02:00:00Z",
        "units": {
            "orbit_width": "m",
            "loss_fraction": "1",
            "energy": "keV",
            "magnetic_field": "T",
        },
        "metrics": {
            "banana_width_relative_error": 0.015,
            "first_orbit_loss_abs_error": 0.018,
            "passing_trapped_classification_accuracy": 0.96,
        },
        "tolerances": {
            "banana_width_relative_error": 0.03,
            "first_orbit_loss_abs_error": 0.03,
            "passing_trapped_classification_accuracy_min": 0.90,
        },
    }


def test_strict_orbit_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no orbit reference artifacts found"


def test_orbit_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "goldston_orbit_reference.json"
    artifact.write_text(json.dumps(_valid_orbit_reference_artifact()), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_case_count"] == 12


def test_orbit_gate_accepts_real_campaign_artifact(tmp_path: Path) -> None:
    payload = _valid_orbit_reference_artifact()
    payload["source"] = "real_orbit_campaign"
    payload.pop("reference_doi")
    payload["campaign_artifact_uri"] = "file:///validation/reports/orbit/alpha_orbits.h5"
    artifact = tmp_path / "real_orbit_campaign.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "real_orbit_campaign"


def test_orbit_gate_rejects_traversing_campaign_artifact_uri(tmp_path: Path) -> None:
    payload = _valid_orbit_reference_artifact()
    payload["source"] = "real_orbit_campaign"
    payload.pop("reference_doi")
    payload["campaign_artifact_uri"] = "https://example.invalid/validation/../alpha_orbits.h5"
    artifact = tmp_path / "bad_real_orbit_campaign.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "campaign_artifact_uri"
    assert "stable remote artifact path" in report["errors"][0]["error"]


def test_orbit_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_orbit_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_orbit_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_orbit_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_orbit_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["first_orbit_loss_abs_error"] = 0.08
    artifact = tmp_path / "bad_loss_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "first_orbit_loss_abs_error"


def test_orbit_gate_rejects_missing_unit_contract(tmp_path: Path) -> None:
    payload = _valid_orbit_reference_artifact()
    payload["units"] = {"orbit_width": "m"}
    artifact = tmp_path / "bad_units_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_orbit_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "units"
