# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ELM reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_elm_reference import canonical_artifact_sha256, validate_elm_reference


def _valid_elm_reference_artifact() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scpn-control.elm-reference.v1",
        "source": "measured_hmode_campaign",
        "machine": "DIII-D",
        "shot_id": "hmode-elm-170000",
        "reference_dataset_id": "diiid-hmode-elm-2026-05-18",
        "executed_at": "2026-05-18T16:00:00Z",
        "pre_crash_profile_uri": "elm/diiid/pre_crash_profiles.npz",
        "post_crash_profile_uri": "elm/diiid/post_crash_profiles.npz",
        "event_catalog_uri": "elm/diiid/event_catalog.json",
        "rmp_artifact_uri": "elm/diiid/rmp_suppression.json",
        "pre_crash_profile_sha256": "1" * 64,
        "post_crash_profile_sha256": "2" * 64,
        "event_catalog_sha256": "3" * 64,
        "rmp_artifact_sha256": "4" * 64,
        "units": {
            "time": "s",
            "energy": "J",
            "density": "m^-3",
            "temperature": "eV",
            "pressure": "Pa",
            "rmp_perturbation": "dimensionless",
            "heat_flux": "MW/m^2",
        },
        "pedestal_rho_grid": [0.86, 0.9, 0.94, 0.98],
        "event_time_window_s": [4.0, 4.4],
        "elm_energy_fraction_range": [0.05, 0.12],
        "rmp_suppression_window_s": [4.1, 4.3],
        "metrics": {
            "elm_frequency_relative_error": 0.08,
            "crash_energy_fraction_error": 0.015,
            "pedestal_temperature_drop_relative_error": 0.07,
            "pedestal_density_drop_relative_error": 0.06,
            "rmp_suppression_window_error_s": 0.015,
            "peak_heat_flux_relative_error": 0.10,
        },
        "tolerances": {
            "elm_frequency_relative_error": 0.12,
            "crash_energy_fraction_error": 0.025,
            "pedestal_temperature_drop_relative_error": 0.10,
            "pedestal_density_drop_relative_error": 0.10,
            "rmp_suppression_window_error_s": 0.03,
            "peak_heat_flux_relative_error": 0.15,
        },
    }
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def test_strict_elm_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_elm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no ELM reference artifacts found"


def test_elm_gate_accepts_measured_hmode_campaign(tmp_path: Path) -> None:
    artifact = tmp_path / "diiid_elm_reference.json"
    artifact.write_text(json.dumps(_valid_elm_reference_artifact()), encoding="utf-8")

    report = validate_elm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "measured_hmode_campaign"


def test_elm_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    payload = _valid_elm_reference_artifact()
    payload["source"] = "documented_public_reference"
    payload.pop("machine")
    payload.pop("shot_id")
    payload["reference_doi"] = "10.1088/0029-5515/54/9/093011"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "published_elm_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_elm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "documented_public_reference"


def test_elm_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_elm_reference_artifact()
    payload["source"] = "synthetic"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "synthetic_elm_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_elm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_elm_gate_rejects_missing_measured_campaign_identity(tmp_path: Path) -> None:
    payload = _valid_elm_reference_artifact()
    payload.pop("shot_id")
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "missing_campaign_identity.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_elm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "campaign"


def test_elm_gate_rejects_energy_fraction_outside_type_i_bounds(tmp_path: Path) -> None:
    payload = _valid_elm_reference_artifact()
    payload["elm_energy_fraction_range"] = [0.01, 0.20]
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_energy_fraction.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_elm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "elm_energy_fraction_range"


def test_elm_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_elm_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["elm_frequency_relative_error"] = 0.2
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_frequency_error.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_elm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "elm_frequency_relative_error"


def test_elm_gate_rejects_tampered_payload_digest(tmp_path: Path) -> None:
    payload = _valid_elm_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["peak_heat_flux_relative_error"] = 0.05
    artifact = tmp_path / "tampered_elm_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_elm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "payload_sha256"


def test_elm_gate_rejects_traversing_artifact_uri(tmp_path: Path) -> None:
    payload = _valid_elm_reference_artifact()
    payload["event_catalog_uri"] = "../event_catalog.json"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "traversing_elm_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_elm_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "event_catalog_uri"
