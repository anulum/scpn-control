# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural equilibrium reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_neural_equilibrium_reference import (
    canonical_artifact_sha256,
    validate_neural_equilibrium_reference,
)


def _valid_pefit_reference_artifact() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scpn-control.neural-equilibrium-reference.v1",
        "source": "real_pefit",
        "model_id": "neural-equilibrium-sparc-pca-mlp",
        "model_version": "0.19.0",
        "trained_weights_sha256": "a" * 64,
        "reference_dataset_id": "pefit-sparc-reference-2026-05-18",
        "binary_path": "/opt/pefit/bin/pefit",
        "reference_artifact_uri": "pefit/sparc/reference_equilibria.npz",
        "prediction_artifact_uri": "pefit/sparc/scpn_predictions.npz",
        "reference_artifact_sha256": "b" * 64,
        "prediction_artifact_sha256": "c" * 64,
        "reference_equilibria_count": 32,
        "executed_at": "2026-05-18T12:30:00Z",
        "grid_shape": [129, 129],
        "target_schema": ["psi", "pressure", "q_profile", "lcfs_boundary", "magnetic_axis"],
        "units": {
            "psi": "Wb/rad",
            "pressure": "Pa",
            "q_profile": "1",
            "boundary": "m",
        },
        "metrics": {
            "psi_rmse_Wb": 2.0e-4,
            "pressure_rmse_Pa": 250.0,
            "q_profile_rmse": 0.018,
            "boundary_rmse_m": 0.006,
            "axis_position_error_m": 0.004,
        },
        "tolerances": {
            "psi_rmse_Wb": 5.0e-4,
            "pressure_rmse_Pa": 500.0,
            "q_profile_rmse": 0.03,
            "boundary_rmse_m": 0.01,
            "axis_position_error_m": 0.01,
        },
    }
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def test_strict_neural_equilibrium_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["schema_version"] == "scpn-control.neural-equilibrium-reference-report.v2"
    assert len(report["payload_sha256"]) == 64
    assert report["public_claims"] == {
        "reference_artifacts_admitted": False,
        "predictive_equilibrium_claim_admitted": False,
        "blocked_reason": (
            "Requires persisted real P-EFIT or documented public-reference neural equilibrium artefacts."
        ),
    }
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no neural equilibrium reference artefacts found"


def test_neural_equilibrium_gate_accepts_real_pefit_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "sparc_pefit_reference.json"
    artifact.write_text(json.dumps(_valid_pefit_reference_artifact()), encoding="utf-8")

    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "real_pefit"
    assert report["entries"][0]["reference_equilibria_count"] == 32
    assert len(report["entries"][0]["artifact_file_sha256"]) == 64
    assert report["entries"][0]["path"] == artifact.as_posix()
    assert report["public_claims"]["reference_artifacts_admitted"] is True


def test_neural_equilibrium_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    payload = _valid_pefit_reference_artifact()
    payload["source"] = "documented_public_reference"
    payload.pop("binary_path")
    payload["reference_url"] = "https://example.invalid/equilibrium-reference"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "public_equilibrium_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "documented_public_reference"


def test_neural_equilibrium_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_pefit_reference_artifact()
    payload["source"] = "synthetic"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "synthetic_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_neural_equilibrium_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_pefit_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["q_profile_rmse"] = 0.05
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_q_profile_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "q_profile_rmse"


def test_neural_equilibrium_gate_rejects_missing_unit_contract(tmp_path: Path) -> None:
    payload = _valid_pefit_reference_artifact()
    payload["units"] = {"psi": "Wb/rad", "pressure": "Pa"}
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "missing_units_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "units"


def test_neural_equilibrium_gate_rejects_relative_pefit_binary(tmp_path: Path) -> None:
    payload = _valid_pefit_reference_artifact()
    payload["binary_path"] = "pefit"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "relative_binary_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "binary_path"


def test_neural_equilibrium_gate_rejects_tampered_payload_digest(tmp_path: Path) -> None:
    payload = _valid_pefit_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["psi_rmse_Wb"] = 3.0e-4
    artifact = tmp_path / "tampered_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "payload_sha256"


def test_neural_equilibrium_gate_rejects_missing_prediction_digest(tmp_path: Path) -> None:
    payload = _valid_pefit_reference_artifact()
    payload.pop("prediction_artifact_sha256")
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "missing_prediction_digest.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "prediction_artifact_sha256"


def test_neural_equilibrium_gate_rejects_traversing_artifact_uri(tmp_path: Path) -> None:
    payload = _valid_pefit_reference_artifact()
    payload["reference_artifact_uri"] = "../reference_equilibria.npz"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "traversing_reference_uri.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "reference_artifact_uri"


def test_neural_equilibrium_gate_rejects_duplicate_reference_set(tmp_path: Path) -> None:
    payload = _valid_pefit_reference_artifact()
    for index in range(2):
        artifact = tmp_path / f"duplicate_reference_{index}.json"
        artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_equilibrium_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert any(error["field"] == "reference_dataset_id" and "duplicate" in error["error"] for error in report["errors"])
