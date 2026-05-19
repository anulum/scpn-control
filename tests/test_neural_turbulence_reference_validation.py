# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural turbulence reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_neural_turbulence_reference import validate_neural_turbulence_reference


def _valid_turbulence_reference_artifact() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1063/5.0005374",
        "model_id": "qlknn-class-neural-turbulence-surrogate",
        "model_version": "0.19.0",
        "trained_weights_sha256": "e" * 64,
        "reference_dataset_id": "qlknn-turbulence-reference-2026-05-20",
        "reference_artifact_sha256": "f" * 64,
        "reference_sample_count": 384,
        "executed_at": "2026-05-20T01:10:00Z",
        "feature_schema": [
            "R_LTi",
            "R_LTe",
            "R_Ln",
            "q",
            "s_hat",
            "alpha_MHD",
            "Ti_Te",
            "nu_star",
            "Z_eff",
            "epsilon",
        ],
        "units": {
            "Q_i": "gyroBohm",
            "Q_e": "gyroBohm",
            "Gamma_e": "gyroBohm",
            "input_gradients": "dimensionless",
        },
        "metrics": {
            "Q_i_rmse_gB": 0.22,
            "Q_e_rmse_gB": 0.18,
            "Gamma_e_rmse_gB": 0.10,
            "flux_relative_mae": 0.09,
            "critical_gradient_accuracy": 0.93,
        },
        "tolerances": {
            "Q_i_rmse_gB": 0.30,
            "Q_e_rmse_gB": 0.25,
            "Gamma_e_rmse_gB": 0.15,
            "flux_relative_mae": 0.12,
            "critical_gradient_accuracy_min": 0.90,
        },
    }


def test_strict_neural_turbulence_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_neural_turbulence_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no neural turbulence reference artifacts found"


def test_neural_turbulence_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "qlknn_turbulence_reference.json"
    artifact.write_text(json.dumps(_valid_turbulence_reference_artifact()), encoding="utf-8")

    report = validate_neural_turbulence_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_sample_count"] == 384


def test_neural_turbulence_gate_accepts_real_gk_artifact(tmp_path: Path) -> None:
    payload = _valid_turbulence_reference_artifact()
    payload["source"] = "real_gk_campaign"
    payload.pop("reference_doi")
    payload["campaign_artifact_uri"] = "file:///validation/reports/gk_campaigns/cbc_fluxes.h5"
    artifact = tmp_path / "real_gk_campaign_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_turbulence_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "real_gk_campaign"


def test_neural_turbulence_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_turbulence_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_turbulence_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_turbulence_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_neural_turbulence_gate_rejects_score_below_minimum(tmp_path: Path) -> None:
    payload = _valid_turbulence_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["critical_gradient_accuracy"] = 0.81
    artifact = tmp_path / "bad_gradient_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_turbulence_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "critical_gradient_accuracy"


def test_neural_turbulence_gate_rejects_missing_flux_unit_contract(tmp_path: Path) -> None:
    payload = _valid_turbulence_reference_artifact()
    payload["units"] = {"Q_i": "gyroBohm", "Q_e": "gyroBohm"}
    artifact = tmp_path / "bad_units_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_neural_turbulence_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "units"
