# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural transport reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_neural_transport_reference import (
    canonical_artifact_sha256,
    validate_neural_transport_reference,
)


def _valid_qualikiz_reference_artifact() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scpn-control.neural-transport-reference.v1",
        "source": "documented_public_reference",
        "reference_doi": "10.1016/j.cpc.2016.09.003",
        "model_id": "qlknn-10d-transport-surrogate",
        "model_version": "0.19.0",
        "trained_weights_sha256": "c" * 64,
        "reference_dataset_id": "qualikiz-transport-reference-2026-05-18",
        "reference_artifact_uri": "qualikiz/reference_targets.npz",
        "prediction_artifact_uri": "qualikiz/scpn_predictions.npz",
        "reference_artifact_sha256": "d" * 64,
        "prediction_artifact_sha256": "e" * 64,
        "reference_sample_count": 512,
        "executed_at": "2026-05-18T14:00:00Z",
        "feature_schema": [
            "R_LTi",
            "R_LTe",
            "R_Ln",
            "q",
            "s_hat",
            "alpha",
            "Ti_Te",
            "Zeff",
            "collisionality",
            "beta_e",
        ],
        "target_schema": ["chi_i", "chi_e", "D_e", "unstable_branch"],
        "units": {
            "chi_i": "m^2/s",
            "chi_e": "m^2/s",
            "D_e": "m^2/s",
            "input_gradients": "dimensionless",
        },
        "metrics": {
            "chi_i_rmse_m2_s": 0.18,
            "chi_e_rmse_m2_s": 0.16,
            "D_e_rmse_m2_s": 0.07,
            "chi_i_relative_mae": 0.08,
            "unstable_branch_accuracy": 0.94,
        },
        "tolerances": {
            "chi_i_rmse_m2_s": 0.25,
            "chi_e_rmse_m2_s": 0.25,
            "D_e_rmse_m2_s": 0.12,
            "chi_i_relative_mae": 0.12,
            "unstable_branch_accuracy_min": 0.90,
        },
    }
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    return payload


def test_strict_neural_transport_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no neural transport reference artifacts found"


def test_neural_transport_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "qualikiz_reference.json"
    artifact.write_text(json.dumps(_valid_qualikiz_reference_artifact()), encoding="utf-8")

    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_sample_count"] == 512


def test_neural_transport_gate_accepts_real_qualikiz_artifact(tmp_path: Path) -> None:
    payload = _valid_qualikiz_reference_artifact()
    payload["source"] = "real_qualikiz"
    payload.pop("reference_doi")
    payload["binary_path"] = "/opt/qualikiz/QuaLiKiz"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "real_qualikiz_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "real_qualikiz"


def test_neural_transport_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_qualikiz_reference_artifact()
    payload["source"] = "synthetic"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "synthetic_transport_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_neural_transport_gate_rejects_uri_binary_path(tmp_path: Path) -> None:
    payload = _valid_qualikiz_reference_artifact()
    payload["source"] = "real_qualikiz"
    payload.pop("reference_doi")
    payload["binary_path"] = "file:///opt/qualikiz/QuaLiKiz"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "uri_binary_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "binary_path"


def test_neural_transport_gate_rejects_traversing_binary_path(tmp_path: Path) -> None:
    payload = _valid_qualikiz_reference_artifact()
    payload["source"] = "real_qualikiz"
    payload.pop("reference_doi")
    payload["binary_path"] = "/opt/qualikiz/../QuaLiKiz"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "traversing_binary_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "binary_path"


def test_neural_transport_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_qualikiz_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["chi_i_relative_mae"] = 0.20
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_transport_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "chi_i_relative_mae"


def test_neural_transport_gate_rejects_missing_feature_schema(tmp_path: Path) -> None:
    payload = _valid_qualikiz_reference_artifact()
    payload["feature_schema"] = ["R_LTi", "R_LTe"]
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "bad_feature_schema_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "feature_schema"


def test_neural_transport_gate_rejects_tampered_payload_digest(tmp_path: Path) -> None:
    payload = _valid_qualikiz_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["chi_i_rmse_m2_s"] = 0.21
    artifact = tmp_path / "tampered_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "payload_sha256"


def test_neural_transport_gate_rejects_missing_prediction_digest(tmp_path: Path) -> None:
    payload = _valid_qualikiz_reference_artifact()
    payload.pop("prediction_artifact_sha256")
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "missing_prediction_digest.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "prediction_artifact_sha256"


def test_neural_transport_gate_rejects_traversing_artifact_uri(tmp_path: Path) -> None:
    payload = _valid_qualikiz_reference_artifact()
    payload["reference_artifact_uri"] = "../reference_targets.npz"
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    artifact = tmp_path / "traversing_reference_uri.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_transport_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "reference_artifact_uri"
