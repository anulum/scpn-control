# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural equilibrium campaign tests

from __future__ import annotations

import json
from pathlib import Path

from validation.publish_mast_efm_neural_equilibrium_campaign import (
    CAMPAIGN_SCHEMA,
    CampaignInput,
    build_campaign_report,
    write_campaign_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_campaign_report_aggregates_metrics_and_keeps_claim_blocked(tmp_path: Path) -> None:
    sas_root = tmp_path / "sas"
    candidate = sas_root / "converted/neural_equilibrium_reference/candidate.json"
    reference_path = sas_root / "converted/neural_equilibrium_reference/mast_efm_shot_30419_reference.npz"
    prediction_path = (
        sas_root / "converted/neural_equilibrium_reference/evaluation_predictions/mast_efm_shot_30419_prediction.npz"
    )
    evaluation = (
        sas_root / "converted/neural_equilibrium_reference/evaluation_predictions/mast_efm_shot_30419_evaluation.json"
    )
    _write_json(
        candidate,
        {
            "payload_sha256": "a" * 64,
            "reference_dataset_id": "mast-efm-test",
            "reference_equilibria_count": 2,
            "shots": [{"output_path": str(reference_path), "sha256": "b" * 64}],
        },
    )
    _write_json(
        evaluation,
        {
            "reference_path": str(reference_path),
            "prediction_path": str(prediction_path),
            "reference_artifact_sha256": "b" * 64,
            "prediction_artifact_sha256": "c" * 64,
            "weights_sha256": "d" * 64,
            "reference_equilibria_count": 2,
            "grid_shape": [65, 129],
            "admission_ready": False,
            "strict_artifact_emitted": False,
            "feature_mapping_notes": {
                "Ip_MA": "fallback: unavailable in converted EFM bundle; synthetic-domain centre used",
                "R_axis_m": "source: magnetic_axis_r_m",
            },
            "metrics": {
                "psi_rmse_Wb_per_rad": 1.5,
                "magnetic_axis_rmse_m": 0.7,
                "boundary_mean_distance_m": 0.4,
                "boundary_p95_distance_m": 1.2,
                "pressure_rmse_Pa": None,
                "q_profile_rmse": None,
            },
        },
    )

    report = build_campaign_report(
        CampaignInput(candidate_report=candidate, evaluation_reports=(evaluation,), sas_root=sas_root)
    )

    assert report["schema_version"] == CAMPAIGN_SCHEMA
    assert report["status"] == "blocked"
    assert report["admission_ready"] is False
    assert report["evaluated_reference_equilibria_count"] == 2
    assert report["aggregate_metrics"]["psi_rmse_Wb_per_rad_mean"] == 1.5
    assert report["aggregate_metrics"]["pressure_rmse_Pa_mean"] is None
    assert (
        report["shots"][0]["reference_path"]
        == "converted/neural_equilibrium_reference/mast_efm_shot_30419_reference.npz"
    )
    assert report["fallback_features"] == ["Ip_MA"]
    assert len(report["payload_sha256"]) == 64


def test_write_campaign_report_emits_markdown_boundary(tmp_path: Path) -> None:
    report = {
        "schema_version": CAMPAIGN_SCHEMA,
        "status": "blocked",
        "reference_dataset_id": "mast-efm-test",
        "shot_ids": [30419],
        "evaluated_reference_equilibria_count": 2,
        "candidate_payload_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "aggregate_metrics": {
            "psi_rmse_Wb_per_rad_mean": 1.5,
            "psi_rmse_Wb_per_rad_max": 1.5,
            "magnetic_axis_rmse_m_mean": 0.7,
            "boundary_mean_distance_m_mean": 0.4,
            "boundary_p95_distance_m_mean": 1.2,
            "pressure_rmse_Pa_mean": None,
            "q_profile_rmse_mean": None,
        },
        "shots": [
            {
                "shot_id": 30419,
                "reference_equilibria_count": 2,
                "metrics": {
                    "psi_rmse_Wb_per_rad": 1.5,
                    "magnetic_axis_rmse_m": 0.7,
                    "boundary_mean_distance_m": 0.4,
                    "boundary_p95_distance_m": 1.2,
                },
            }
        ],
        "blocked_reason": "predictive claims remain blocked",
        "fallback_features": ["Ip_MA"],
        "next_processing_steps": ["train full-output model"],
    }

    write_campaign_report(report, tmp_path / "campaign.json", tmp_path / "campaign.md")

    markdown = (tmp_path / "campaign.md").read_text(encoding="utf-8")
    assert "MAST EFM Neural-Equilibrium Campaign Evidence" in markdown
    assert "predictive claims remain blocked" in markdown
    assert "`Ip_MA`" in markdown
