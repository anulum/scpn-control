# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Blob transport reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_blob_transport_reference import canonical_artifact_sha256, validate_blob_transport_reference


def _valid_blob_reference_artifact() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scpn-control.blob-transport-reference.v1",
        "source": "measured_probe_campaign",
        "machine": "DIII-D",
        "shot_id": "sol-probe-170000",
        "reference_dataset_id": "diiid-sol-blob-probe-2026-05-18",
        "executed_at": "2026-05-18T15:30:00Z",
        "reference_artifact_uri": "blob_transport/diiid/reference_events.npz",
        "profile_artifact_uri": "blob_transport/diiid/sol_profiles.npz",
        "detector_artifact_uri": "blob_transport/diiid/detector_events.json",
        "reference_artifact_sha256": "1" * 64,
        "profile_artifact_sha256": "2" * 64,
        "detector_artifact_sha256": "3" * 64,
        "units": {
            "radius": "m",
            "time": "s",
            "velocity": "m/s",
            "density": "m^-3",
            "temperature": "eV",
            "magnetic_field": "T",
            "wall_flux": "m^-2 s^-1",
        },
        "separatrix_to_wall_coordinates_m": [0.0, 0.01, 0.03, 0.06],
        "detector_time_domain_s": [0.0, 0.25],
        "blob_size_range_m": [0.002, 0.04],
        "magnetic_geometry": {
            "R0_m": 1.67,
            "B0_T": 2.1,
            "L_parallel_m": 18.0,
            "Te_eV": 35.0,
            "density_m3": 4.0e18,
        },
        "metrics": {
            "radial_velocity_rmse_m_s": 210.0,
            "density_profile_relative_l2": 0.08,
            "wall_flux_relative_error": 0.11,
            "event_duration_relative_error": 0.10,
            "event_size_relative_error": 0.09,
        },
        "tolerances": {
            "radial_velocity_rmse_m_s": 300.0,
            "density_profile_relative_l2": 0.12,
            "wall_flux_relative_error": 0.15,
            "event_duration_relative_error": 0.15,
            "event_size_relative_error": 0.15,
        },
    }
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def test_strict_blob_transport_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_blob_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no blob transport reference artifacts found"


def test_blob_transport_gate_accepts_measured_probe_campaign(tmp_path: Path) -> None:
    artifact = tmp_path / "diiid_blob_reference.json"
    artifact.write_text(json.dumps(_valid_blob_reference_artifact()), encoding="utf-8")

    report = validate_blob_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "measured_probe_campaign"


def test_blob_transport_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    payload = _valid_blob_reference_artifact()
    payload["source"] = "documented_public_reference"
    payload.pop("machine")
    payload.pop("shot_id")
    payload["reference_doi"] = "10.1063/1.3659319"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "published_blob_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_blob_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "documented_public_reference"


def test_blob_transport_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_blob_reference_artifact()
    payload["source"] = "synthetic"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "synthetic_blob_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_blob_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_blob_transport_gate_rejects_missing_measured_campaign_identity(tmp_path: Path) -> None:
    payload = _valid_blob_reference_artifact()
    payload.pop("shot_id")
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "missing_campaign_identity.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_blob_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "campaign"


def test_blob_transport_gate_rejects_nonmonotone_sol_profile_coordinates(tmp_path: Path) -> None:
    payload = _valid_blob_reference_artifact()
    payload["separatrix_to_wall_coordinates_m"] = [0.0, 0.02, 0.02, 0.05]
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_profile_coordinates.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_blob_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "separatrix_to_wall_coordinates_m"


def test_blob_transport_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_blob_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["wall_flux_relative_error"] = 0.25
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_wall_flux_error.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_blob_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "wall_flux_relative_error"


def test_blob_transport_gate_rejects_tampered_payload_digest(tmp_path: Path) -> None:
    payload = _valid_blob_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["density_profile_relative_l2"] = 0.05
    artifact = tmp_path / "tampered_blob_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_blob_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "payload_sha256"


def test_blob_transport_gate_rejects_traversing_artifact_uri(tmp_path: Path) -> None:
    payload = _valid_blob_reference_artifact()
    payload["profile_artifact_uri"] = "../sol_profiles.npz"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "traversing_blob_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_blob_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "profile_artifact_uri"
