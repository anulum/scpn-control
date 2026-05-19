# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — VMEC reference validation tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from validation.validate_vmec_reference import validate_vmec_reference


def _valid_vmec_reference_artifact() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source": "documented_public_reference",
        "reference_doi": "10.1063/1.864116",
        "model_id": "vmec-lite-spectral-equilibrium",
        "model_version": "0.19.0",
        "reference_dataset_id": "vmec-lite-w7x-reference-2026-05-20",
        "reference_artifact_sha256": "3" * 64,
        "reference_case_count": 6,
        "executed_at": "2026-05-20T03:00:00Z",
        "fourier_truncation": {"m_pol": 3, "n_tor": 2, "n_fp": 5},
        "units": {
            "R_mn": "m",
            "Z_mn": "m",
            "B_mn": "T",
            "pressure": "Pa",
            "iota": "1",
        },
        "metrics": {
            "surface_R_rmse_m": 0.008,
            "surface_Z_rmse_m": 0.006,
            "iota_rmse": 0.012,
            "force_residual_relative": 0.04,
        },
        "tolerances": {
            "surface_R_rmse_m": 0.015,
            "surface_Z_rmse_m": 0.015,
            "iota_rmse": 0.02,
            "force_residual_relative": 0.08,
        },
    }


def test_strict_vmec_gate_requires_reference_artifacts(tmp_path: Path) -> None:
    report = validate_vmec_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["reference_artifacts"] == 0
    assert report["errors"][0]["error"] == "no VMEC reference artifacts found"


def test_vmec_gate_accepts_documented_public_reference(tmp_path: Path) -> None:
    artifact = tmp_path / "w7x_vmec_reference.json"
    artifact.write_text(json.dumps(_valid_vmec_reference_artifact()), encoding="utf-8")

    report = validate_vmec_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["reference_artifacts"] == 1
    assert report["entries"][0]["source"] == "documented_public_reference"
    assert report["entries"][0]["reference_case_count"] == 6


def test_vmec_gate_accepts_real_vmec_artifact(tmp_path: Path) -> None:
    payload = _valid_vmec_reference_artifact()
    payload["source"] = "real_vmec_run"
    payload.pop("reference_doi")
    payload["vmec_artifact_uri"] = "file:///validation/reports/vmec/wout_w7x.nc"
    artifact = tmp_path / "real_vmec_run.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_vmec_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "pass"
    assert report["entries"][0]["source"] == "real_vmec_run"


def test_vmec_gate_rejects_synthetic_source(tmp_path: Path) -> None:
    payload = _valid_vmec_reference_artifact()
    payload["source"] = "synthetic"
    artifact = tmp_path / "synthetic_vmec_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_vmec_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "source"


def test_vmec_gate_rejects_metric_outside_tolerance(tmp_path: Path) -> None:
    payload = _valid_vmec_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["surface_R_rmse_m"] = 0.05
    artifact = tmp_path / "bad_surface_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_vmec_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "surface_R_rmse_m"


def test_vmec_gate_rejects_missing_fourier_truncation(tmp_path: Path) -> None:
    payload = _valid_vmec_reference_artifact()
    payload["fourier_truncation"] = {"m_pol": 3, "n_tor": 2}
    artifact = tmp_path / "bad_truncation_reference.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_vmec_reference(tmp_path, require_reference_artifacts=True)

    assert report["status"] == "fail"
    assert report["errors"][0]["field"] == "fourier_truncation"
