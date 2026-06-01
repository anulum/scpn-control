# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MARFE reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_marfe_reference import canonical_artifact_sha256, validate_marfe_reference


def _valid_marfe_reference_artifact() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scpn-control.marfe-reference.v1",
        "source": "measured_marfe_campaign",
        "machine": "JET",
        "shot_id": "marfe-density-limit-90321",
        "reference_dataset_id": "jet-marfe-density-limit-2026-05-22",
        "executed_at": "2026-05-22T11:15:00Z",
        "temperature_profile_uri": "marfe/jet/edge_temperature_profiles.npz",
        "density_limit_uri": "marfe/jet/density_limit_trace.json",
        "radiation_curve_uri": "marfe/jet/radiation_curve_w.json",
        "power_balance_uri": "marfe/jet/power_balance.json",
        "temperature_profile_sha256": "1" * 64,
        "density_limit_sha256": "2" * 64,
        "radiation_curve_sha256": "3" * 64,
        "power_balance_sha256": "4" * 64,
        "units": {
            "temperature": "eV",
            "density": "m^-3",
            "power": "W",
            "current": "A",
            "impurity_fraction": "1",
            "length": "m",
            "growth_rate": "s^-1",
        },
        "impurity": "W",
        "temperature_scan_eV": [20.0, 50.0, 100.0, 500.0, 2000.0],
        "density_scan_m3": [1.0e19, 3.0e19, 6.0e19, 1.0e20],
        "impurity_fraction_range": [1.0e-5, 5.0e-4],
        "geometry": {"R0_m": 2.96, "a_m": 0.95, "q95": 3.4, "connection_length_m": 31.6},
        "power_balance": {"P_SOL_W": 6.0e6, "q_perp_W_m2": 1.5e5},
        "metrics": {
            "onset_temperature_relative_error": 0.08,
            "density_limit_relative_error": 0.09,
            "greenwald_fraction_error": 0.04,
            "front_temperature_min_relative_error": 0.10,
            "radiation_growth_rate_relative_error": 0.11,
        },
        "tolerances": {
            "onset_temperature_relative_error": 0.12,
            "density_limit_relative_error": 0.12,
            "greenwald_fraction_error": 0.08,
            "front_temperature_min_relative_error": 0.15,
            "radiation_growth_rate_relative_error": 0.15,
        },
    }
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def test_strict_marfe_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_marfe_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no MARFE reference artifacts found"


def test_marfe_gate_accepts_measured_campaign(tmp_path: Path) -> None:
    artifact = tmp_path / "jet_marfe_reference.json"
    artifact.write_text(json.dumps(_valid_marfe_reference_artifact()), encoding="utf-8")

    report = validate_marfe_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "measured_marfe_campaign"


def test_marfe_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    payload = _valid_marfe_reference_artifact()
    payload["source"] = "documented_public_reference"
    payload.pop("machine")
    payload.pop("shot_id")
    payload["reference_doi"] = "10.1016/0022-3115(87)90365-3"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "published_marfe_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_marfe_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "documented_public_reference"


def test_marfe_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_marfe_reference_artifact()
    payload["source"] = "synthetic"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "synthetic_marfe_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_marfe_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_marfe_gate_rejects_missing_measured_campaign_identity(tmp_path: Path) -> None:
    payload = _valid_marfe_reference_artifact()
    payload.pop("shot_id")
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "missing_campaign_identity.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_marfe_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "campaign"


def test_marfe_gate_rejects_nonmonotone_temperature_scan(tmp_path: Path) -> None:
    payload = _valid_marfe_reference_artifact()
    payload["temperature_scan_eV"] = [20.0, 100.0, 50.0]
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_temperature_scan.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_marfe_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "temperature_scan_eV"


def test_marfe_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_marfe_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["density_limit_relative_error"] = 0.3
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_density_limit_error.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_marfe_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "density_limit_relative_error"


def test_marfe_gate_rejects_tampered_payload_digest(tmp_path: Path) -> None:
    payload = _valid_marfe_reference_artifact()
    power_balance = cast(dict[str, object], payload["power_balance"])
    power_balance["q_perp_W_m2"] = 2.0e5
    artifact = tmp_path / "tampered_marfe_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_marfe_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "payload_sha256"


def test_marfe_gate_rejects_traversing_artifact_uri(tmp_path: Path) -> None:
    payload = _valid_marfe_reference_artifact()
    payload["radiation_curve_uri"] = "../radiation_curve_w.json"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "traversing_marfe_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_marfe_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "radiation_curve_uri"
