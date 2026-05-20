# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOC reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_soc_reference import validate_soc_reference


def _valid_soc_reference_artifact() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1063/1.871048",
        "model_id": "advanced-soc-turbulence-learning",
        "model_version": "0.19.0",
        "reference_dataset_id": "soc-reference-2026-05-20",
        "reference_artifact_sha256": "8" * 64,
        "reference_case_count": 6,
        "executed_at": "2026-05-20T08:00:00Z",
        "lattice_metadata": {
            "size": 60,
            "time_steps": 10000,
            "seed": 42,
            "z_crit_base": 6.0,
            "flow_generation": 0.2,
            "flow_damping": 0.05,
            "shear_efficiency": 3.0,
            "max_sub_steps": 50,
        },
        "learning_metadata": {
            "alpha": 0.1,
            "gamma": 0.95,
            "epsilon": 0.1,
            "n_states_turb": 5,
            "n_states_flow": 5,
            "n_actions": 3,
            "reward_definition": "temperature-growth-minus-turbulence-penalty",
        },
        "units": {
            "lattice_gradient": "1",
            "flow": "1",
            "shear": "1",
            "topple_count": "1",
            "q_value": "1",
            "reward": "1",
            "time": "step",
        },
        "metrics": {
            "mean_turbulence_relative_error": 0.04,
            "flow_mean_abs_error": 0.03,
            "policy_action_accuracy_error": 0.08,
            "reward_relative_error": 0.06,
            "core_temperature_relative_error": 0.05,
        },
        "tolerances": {
            "mean_turbulence_relative_error": 0.08,
            "flow_mean_abs_error": 0.05,
            "policy_action_accuracy_error": 0.12,
            "reward_relative_error": 0.1,
            "core_temperature_relative_error": 0.09,
        },
    }


def test_strict_soc_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_soc_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no SOC reference artifacts found"


def test_soc_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "soc_public_reference.json"
    artifact.write_text(json.dumps(_valid_soc_reference_artifact()), encoding="utf-8")

    report = validate_soc_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_case_count"] == 6


def test_soc_gate_accepts_measured_turbulence_replay(tmp_path: Path) -> None:
    payload = _valid_soc_reference_artifact()
    payload["source"] = "measured_turbulence_replay"
    payload.pop("reference_doi")
    payload["shot_id"] = "DIII-D:163303"
    payload["diagnostic_uri"] = "mdsplus://DIII-D/163303/turbulence_replay"
    artifact = tmp_path / "measured_soc_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_soc_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "measured_turbulence_replay"


def test_soc_gate_accepts_external_gyrokinetic_reference(tmp_path: Path) -> None:
    payload = _valid_soc_reference_artifact()
    payload["source"] = "external_gyrokinetic_reference"
    payload.pop("reference_doi")
    payload["external_code"] = "GENE"
    payload["reference_artifact_uri"] = "file:///validation/reports/soc/gene_turbulence_cases.nc"
    artifact = tmp_path / "external_soc_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_soc_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "external_gyrokinetic_reference"


def test_soc_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_soc_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_soc_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_soc_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_soc_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_soc_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["policy_action_accuracy_error"] = 0.3
    artifact = tmp_path / "bad_soc_metric.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_soc_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "policy_action_accuracy_error"


def test_soc_gate_rejects_missing_learning_metadata(tmp_path: Path) -> None:
    payload = _valid_soc_reference_artifact()
    learning_metadata = cast(dict[str, object], payload["learning_metadata"])
    learning_metadata.pop("n_actions")
    artifact = tmp_path / "bad_soc_learning_metadata.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_soc_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "learning_metadata"
