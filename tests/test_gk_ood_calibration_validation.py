# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK OOD calibration validation tests

from __future__ import annotations

import json
from pathlib import Path

from validation.validate_gk_ood_calibration import validate_gk_ood_calibration


def _valid_calibration_report() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "campaign_id": "qlknn-public-cbc-shift-2026-05-18",
        "source": "published_gk_campaign",
        "feature_schema": [
            "R_L_Ti",
            "R_L_Te",
            "R_L_ne",
            "q",
            "s_hat",
            "alpha_MHD",
            "Te_Ti",
            "Z_eff",
            "nu_star",
            "beta_e",
        ],
        "training_distribution": {
            "dataset_id": "qlknn-10d-public",
            "sample_count": 10000,
            "mean": [6.0, 6.0, 2.0, 2.0, 1.0, 0.3, 1.0, 1.5, 0.5, 0.02],
            "std": [5.0, 5.0, 3.0, 1.2, 1.0, 0.5, 0.5, 0.8, 1.5, 0.02],
        },
        "thresholds": {
            "mahalanobis": 4.0,
            "soft_sigma": 2.0,
            "ensemble_disagreement": 0.3,
        },
        "acceptance": {
            "false_positive_rate": 0.02,
            "false_negative_rate": 0.01,
            "max_false_positive_rate": 0.05,
            "max_false_negative_rate": 0.05,
            "ood_recall": 0.99,
            "min_ood_recall": 0.95,
        },
        "evaluated_at": "2026-05-18T06:45:00Z",
    }


def test_strict_ood_calibration_gate_requires_campaign_artifacts(tmp_path: Path) -> None:
    report = validate_gk_ood_calibration(tmp_path, require_campaign_artifacts=True)

    assert report["status"] == "fail"
    assert report["campaign_artifacts"] == 0
    assert report["errors"][0]["error"] == "no GK OOD calibration artifacts found"


def test_ood_calibration_gate_accepts_valid_campaign_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "qlknn_public_cbc_shift.json"
    artifact.write_text(json.dumps(_valid_calibration_report()), encoding="utf-8")

    report = validate_gk_ood_calibration(tmp_path, require_campaign_artifacts=True)

    assert report["status"] == "pass"
    assert report["campaign_artifacts"] == 1
    assert report["entries"][0]["campaign_id"] == "qlknn-public-cbc-shift-2026-05-18"


def test_ood_calibration_gate_rejects_missing_feature_schema(tmp_path: Path) -> None:
    payload = _valid_calibration_report()
    payload["feature_schema"] = ["R_L_Ti"]
    artifact = tmp_path / "bad_schema.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_ood_calibration(tmp_path, require_campaign_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "feature_schema"


def test_ood_calibration_gate_rejects_false_negative_regression(tmp_path: Path) -> None:
    payload = _valid_calibration_report()
    acceptance = payload["acceptance"]
    assert isinstance(acceptance, dict)
    acceptance["false_negative_rate"] = 0.2
    artifact = tmp_path / "bad_acceptance.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_gk_ood_calibration(tmp_path, require_campaign_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "false_negative_rate"
