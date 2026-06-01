# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium training tests

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validation.build_mast_efm_neural_equilibrium_dataset import DATASET_SCHEMA, FEATURE_NAMES
from validation.plan_neural_equilibrium_training_campaign import REPORT_SCHEMA as PLAN_SCHEMA
from validation.train_mast_efm_neural_equilibrium import (
    TRAINING_SCHEMA,
    TrainingInputs,
    build_training_report,
    write_report,
)


def _write_payloads(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dataset = tmp_path / "mast_efm_supervised_dataset.npz"
    n = 6
    features = np.column_stack([np.linspace(1.0 + col, 2.0 + col, n) for col in range(len(FEATURE_NAMES))])
    split = np.array(["train", "train", "train", "validation", "test", "test"])
    z, r = 3, 4
    base = np.arange(n * z * r, dtype=np.float64).reshape(n, z, r) / 10.0
    psirz_mask = np.ones_like(base, dtype=bool)
    base[0, 0, 0] = np.nan
    psirz_mask[0, 0, 0] = False
    lcfs_r = np.tile(np.linspace(0.4, 0.8, 5), (n, 1))
    lcfs_z = np.tile(np.linspace(-0.2, 0.2, 5), (n, 1))
    lcfs_mask = np.ones((n, 5), dtype=bool)
    lcfs_mask[0, 4] = False
    lcfs_r[0, 4] = np.nan
    lcfs_z[0, 4] = np.nan
    np.savez_compressed(
        dataset,
        features=features,
        feature_names=np.asarray(FEATURE_NAMES),
        split=split,
        shot_id=np.array([1, 1, 1, 2, 3, 3]),
        time_s=np.linspace(0.0, 0.5, n),
        r_grid_m=np.linspace(0.0, 1.0, r),
        z_grid_m=np.linspace(-1.0, 1.0, z),
        psirz_Wb_per_rad=base,
        psirz_valid_mask=psirz_mask,
        psi_axis_Wb_per_rad=np.linspace(0.0, 0.1, n),
        psi_boundary_Wb_per_rad=np.linspace(1.0, 1.1, n),
        pprime_Pa_per_Wb_rad=np.column_stack([np.linspace(1.0, 2.0, n), np.linspace(2.0, 3.0, n)]),
        pprime_valid_mask=np.ones((n, 2), dtype=bool),
        q_profile=np.column_stack([np.linspace(2.0, 3.0, n), np.linspace(3.0, 4.0, n)]),
        q_profile_valid_mask=np.ones((n, 2), dtype=bool),
        lcfs_r_m=lcfs_r,
        lcfs_z_m=lcfs_z,
        lcfs_valid_mask=lcfs_mask,
        lcfs_point_count=np.array([4, 5, 5, 5, 5, 5]),
        magnetic_axis_r_m=np.linspace(0.6, 0.7, n),
        magnetic_axis_z_m=np.linspace(-0.01, 0.01, n),
    )
    import hashlib

    sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    dataset_report = tmp_path / "dataset.json"
    dataset_report.write_text(
        json.dumps(
            {
                "schema_version": DATASET_SCHEMA,
                "status": "blocked",
                "dataset_sha256": sha,
                "split_counts": {"train": 3, "validation": 1, "test": 2},
                "fallback_features": [],
            }
        ),
        encoding="utf-8",
    )
    campaign_plan = tmp_path / "plan.json"
    campaign_plan.write_text(json.dumps({"schema_version": PLAN_SCHEMA, "status": "prepared"}), encoding="utf-8")
    weights = tmp_path / "weights.npz"
    return dataset, dataset_report, campaign_plan, weights


def test_training_report_default_is_dry_run_and_does_not_write_weights(tmp_path: Path) -> None:
    dataset, dataset_report, campaign_plan, weights = _write_payloads(tmp_path)

    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report, campaign_plan=campaign_plan, dataset_path=dataset, weights_out=weights
        )
    )

    assert report["schema_version"] == TRAINING_SCHEMA
    assert report["status"] == "prepared"
    assert report["execution_mode"] == "dry_run"
    assert report["dataset_exists_on_this_host"] is True
    assert report["dataset_metadata"]["split_counts"] == {"train": 3, "validation": 1, "test": 2}
    assert report["fallback_features"] == []
    assert all("fallback" not in item for item in report["blocked_before_admission"])
    assert report["holdout_metrics"] is None
    assert not weights.exists()


def test_training_report_execute_writes_weights_and_holdout_metrics(tmp_path: Path) -> None:
    dataset, dataset_report, campaign_plan, weights = _write_payloads(tmp_path)

    report = build_training_report(
        TrainingInputs(
            dataset_report=dataset_report,
            campaign_plan=campaign_plan,
            dataset_path=dataset,
            weights_out=weights,
            execute=True,
            max_flux_components=2,
        )
    )

    assert report["status"] == "executed"
    assert report["execution_mode"] == "execute"
    assert report["weights_sha256"]
    assert report["holdout_metrics"]["validation"]["psi_rmse_Wb_per_rad"] is not None
    assert report["holdout_metrics"]["test"]["magnetic_axis_rmse_m"] is not None
    with np.load(weights, allow_pickle=False) as payload:
        assert payload["flux_components"].shape == (2, 12)
        assert payload["axis_regression"].shape[1] == 2


def test_write_report_records_execute_command_and_admission_boundary(tmp_path: Path) -> None:
    report = {
        "schema_version": TRAINING_SCHEMA,
        "status": "prepared",
        "execution_mode": "dry_run",
        "dataset_path": "/sas/dataset.npz",
        "dataset_sha256": "a" * 64,
        "dataset_exists_on_this_host": False,
        "weights_path": "/sas/weights.npz",
        "claim_boundary": "not predictive EFIT/P-EFIT admission evidence",
        "run_command": "python validation/train_mast_efm_neural_equilibrium.py --execute",
        "required_targets": ["psirz_Wb_per_rad"],
        "blocked_before_admission": ["strict reference gate"],
        "holdout_metrics": None,
    }

    write_report(report, tmp_path / "report.json", tmp_path / "report.md")

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "MAST EFM Neural-Equilibrium Training Launch" in markdown
    assert "--execute" in markdown
    assert "not predictive EFIT/P-EFIT admission evidence" in markdown
