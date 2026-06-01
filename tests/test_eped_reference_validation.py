# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EPED reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_eped_reference import canonical_artifact_sha256, validate_eped_reference


def _valid_eped_reference_artifact() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scpn-control.eped-reference.v1",
        "source": "measured_pedestal_database",
        "machine": "DIII-D",
        "shot_id": "pedestal-174082",
        "reference_dataset_id": "diiid-eped-pedestal-2026-05-21",
        "executed_at": "2026-05-21T15:30:00Z",
        "pedestal_profile_uri": "eped/diiid/pedestal_profiles.npz",
        "eped_prediction_uri": "eped/diiid/eped_prediction.json",
        "bootstrap_current_uri": "eped/diiid/bootstrap_current.npz",
        "peeling_ballooning_uri": "eped/diiid/peeling_ballooning_boundary.json",
        "pedestal_profile_sha256": "1" * 64,
        "eped_prediction_sha256": "2" * 64,
        "bootstrap_current_sha256": "3" * 64,
        "peeling_ballooning_sha256": "4" * 64,
        "units": {
            "width": "psi_N",
            "pressure": "Pa",
            "temperature": "eV",
            "density": "m^-3",
            "current": "A",
            "beta": "1",
            "shape": "1",
        },
        "rho_grid": [0.82, 0.88, 0.94, 1.0],
        "pedestal_width_range_psi_n": [0.035, 0.07],
        "beta_limit_range": [0.018, 0.032],
        "shaping": {"kappa": 1.72, "delta": 0.36, "R0_m": 1.67, "a_m": 0.61},
        "metrics": {
            "pedestal_width_relative_error": 0.06,
            "pedestal_height_relative_error": 0.08,
            "pressure_limit_relative_error": 0.07,
            "bootstrap_current_relative_error": 0.09,
            "collisionality_width_order_error": 0.0,
        },
        "tolerances": {
            "pedestal_width_relative_error": 0.10,
            "pedestal_height_relative_error": 0.12,
            "pressure_limit_relative_error": 0.10,
            "bootstrap_current_relative_error": 0.12,
            "collisionality_width_order_error": 0.01,
        },
    }
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def test_strict_eped_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_eped_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no EPED reference artifacts found"


def test_eped_gate_accepts_measured_pedestal_database(tmp_path: Path) -> None:
    artifact = tmp_path / "diiid_eped_reference.json"
    artifact.write_text(json.dumps(_valid_eped_reference_artifact()), encoding="utf-8")

    report = validate_eped_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "measured_pedestal_database"


def test_eped_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    payload = _valid_eped_reference_artifact()
    payload["source"] = "documented_public_reference"
    payload.pop("machine")
    payload.pop("shot_id")
    payload["reference_doi"] = "10.1063/1.3122146"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "published_eped_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_eped_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "documented_public_reference"


def test_eped_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_eped_reference_artifact()
    payload["source"] = "synthetic"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "synthetic_eped_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_eped_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_eped_gate_rejects_missing_measured_campaign_identity(tmp_path: Path) -> None:
    payload = _valid_eped_reference_artifact()
    payload.pop("shot_id")
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "missing_campaign_identity.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_eped_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "campaign"


def test_eped_gate_rejects_nonmonotone_rho_grid(tmp_path: Path) -> None:
    payload = _valid_eped_reference_artifact()
    payload["rho_grid"] = [0.82, 0.94, 0.91, 1.0]
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_rho_grid.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_eped_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "rho_grid"


def test_eped_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_eped_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["pedestal_height_relative_error"] = 0.2
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_height_error.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_eped_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "pedestal_height_relative_error"


def test_eped_gate_rejects_tampered_payload_digest(tmp_path: Path) -> None:
    payload = _valid_eped_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["bootstrap_current_relative_error"] = 0.04
    artifact = tmp_path / "tampered_eped_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_eped_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "payload_sha256"


def test_eped_gate_rejects_traversing_artifact_uri(tmp_path: Path) -> None:
    payload = _valid_eped_reference_artifact()
    payload["bootstrap_current_uri"] = "../bootstrap_current.npz"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "traversing_eped_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_eped_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "bootstrap_current_uri"
