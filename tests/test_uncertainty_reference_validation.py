# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Uncertainty reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_uncertainty_reference import validate_uncertainty_reference


def _valid_uncertainty_reference_artifact() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1088/0029-5515/39/12/302",
        "model_id": "ipb98y2-monte-carlo-uq",
        "model_version": "0.19.0",
        "reference_dataset_id": "ipb98y2-uq-reference-2026-05-20",
        "reference_artifact_sha256": "2" * 64,
        "reference_case_count": 8,
        "executed_at": "2026-05-20T02:15:00Z",
        "units": {
            "tau_E": "s",
            "P_fusion": "MW",
            "Q": "1",
            "sigma": "same_as_quantity",
        },
        "metrics": {
            "tau_E_relative_error": 0.025,
            "P_fusion_relative_error": 0.06,
            "Q_relative_error": 0.07,
            "percentile_monotonicity_fraction": 1.0,
        },
        "tolerances": {
            "tau_E_relative_error": 0.05,
            "P_fusion_relative_error": 0.10,
            "Q_relative_error": 0.10,
            "percentile_monotonicity_fraction_min": 1.0,
        },
    }


def test_strict_uncertainty_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_uncertainty_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no uncertainty reference artifacts found"


def test_uncertainty_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "ipb98y2_uncertainty_reference.json"
    artifact.write_text(json.dumps(_valid_uncertainty_reference_artifact()), encoding="utf-8")

    report = validate_uncertainty_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_case_count"] == 8


def test_uncertainty_gate_accepts_real_campaign_artifact(tmp_path: Path) -> None:
    payload = _valid_uncertainty_reference_artifact()
    payload["source"] = "real_uq_campaign"
    payload.pop("reference_doi")
    payload["campaign_artifact_uri"] = "file:///validation/reports/uq/ipb98y2_samples.parquet"
    artifact = tmp_path / "real_uq_campaign.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_uncertainty_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "real_uq_campaign"


def test_uncertainty_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_uncertainty_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_uq_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_uncertainty_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_uncertainty_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_uncertainty_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["Q_relative_error"] = 0.25
    artifact = tmp_path / "bad_q_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_uncertainty_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "Q_relative_error"


def test_uncertainty_gate_rejects_missing_unit_contract(tmp_path: Path) -> None:
    payload = _valid_uncertainty_reference_artifact()
    payload["units"] = {"tau_E": "s"}
    artifact = tmp_path / "bad_units_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_uncertainty_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "units"
