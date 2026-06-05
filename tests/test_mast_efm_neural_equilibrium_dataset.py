# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium dataset tests

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validation.build_mast_efm_neural_equilibrium_dataset import (
    DATASET_SCHEMA,
    DatasetInput,
    build_dataset,
    write_report,
)


def _write_reference(path: Path, shot_id: int, *, n: int = 2, lcfs_points: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    psirz = np.full((n, 3, 4), float(shot_id), dtype=np.float64)
    lcfs_r = np.linspace(0.5, 0.9, lcfs_points)
    lcfs_z = np.linspace(-0.2, 0.2, lcfs_points)
    np.savez_compressed(
        path,
        time_s=np.arange(n, dtype=np.float64),
        r_grid_m=np.linspace(0.4, 1.0, 4),
        z_grid_m=np.linspace(-0.2, 0.2, 3),
        psirz_Wb_per_rad=psirz,
        psirz_valid_mask=np.ones_like(psirz, dtype=bool),
        psi_axis_Wb_per_rad=np.full(n, 0.1),
        psi_boundary_Wb_per_rad=np.full(n, 1.1),
        Ip_MA=np.linspace(0.8, 0.9, n),
        Bt_T=np.linspace(0.5, 0.6, n),
        ffprime_rms_T_rad=np.linspace(2.0, 4.0, n),
        pprime_Pa_per_Wb_rad=np.ones((n, 5)),
        pprime_valid_mask=np.ones((n, 5), dtype=bool),
        q_profile=np.full((n, 5), 2.5),
        q_profile_valid_mask=np.ones((n, 5), dtype=bool),
        lcfs_r_m=np.tile(lcfs_r, (n, 1)),
        lcfs_z_m=np.tile(lcfs_z, (n, 1)),
        lcfs_valid_mask=np.ones((n, lcfs_points), dtype=bool),
        magnetic_axis_r_m=np.full(n, 0.7),
        magnetic_axis_z_m=np.zeros(n),
        shot_id=np.full(n, shot_id),
    )


def test_build_dataset_writes_supervised_npz_and_compact_report(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    ref_a = storage_root / "converted/neural_equilibrium_reference/mast_efm_shot_1_reference.npz"
    ref_b = storage_root / "converted/neural_equilibrium_reference/mast_efm_shot_2_reference.npz"
    ref_c = storage_root / "converted/neural_equilibrium_reference/mast_efm_shot_3_reference.npz"
    _write_reference(ref_a, 1, lcfs_points=3)
    _write_reference(ref_b, 2, lcfs_points=4)
    _write_reference(ref_c, 3, lcfs_points=5)
    candidate = storage_root / "converted/neural_equilibrium_reference/candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "payload_sha256": "a" * 64,
                "reference_dataset_id": "mast-efm-test",
                "shots": [
                    {"shot_id": 1, "output_path": str(ref_a)},
                    {"shot_id": 2, "output_path": str(ref_b)},
                    {"shot_id": 3, "output_path": str(ref_c)},
                ],
            }
        ),
        encoding="utf-8",
    )
    output_npz = storage_root / "processed/neural_equilibrium/mast_efm_supervised_dataset.npz"

    report = build_dataset(
        DatasetInput(
            candidate_report=candidate,
            storage_root=storage_root,
            output_npz=output_npz,
            train_shots=(1,),
            validation_shots=(2,),
            test_shots=(3,),
        )
    )

    assert report["schema_version"] == DATASET_SCHEMA
    assert report["status"] == "blocked"
    assert report["equilibria_count"] == 6
    assert report["split_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert report["fallback_features"] == []
    assert report["feature_source_policy"]["Ip_MA"]["source_key"] == "Ip_MA"
    assert report["feature_source_policy"]["Bt_T"]["source_key"] == "Bt_T"
    assert report["feature_source_policy"]["ffprime_scale"]["source_key"] == "ffprime_rms_T_rad"
    assert report["ragged_target_policy"]["max_lcfs_points"] == 5
    assert report["dataset_path"] == "processed/neural_equilibrium/mast_efm_supervised_dataset.npz"
    assert len(report["dataset_sha256"]) == 64
    assert output_npz.exists()
    with np.load(output_npz, allow_pickle=False) as payload:
        assert payload["features"].shape == (6, 12)
        assert np.allclose(payload["features"][:2, 0], [0.8, 0.9])
        assert np.allclose(payload["features"][:2, 1], [0.5, 0.6])
        assert np.all(payload["features"][:, 5] > 0.0)
        assert payload["psirz_Wb_per_rad"].shape == (6, 3, 4)
        assert payload["lcfs_r_m"].shape == (6, 5)
        assert payload["lcfs_point_count"].tolist() == [3, 3, 4, 4, 5, 5]
        assert np.isnan(payload["lcfs_r_m"][0, 3])
        assert not bool(payload["lcfs_valid_mask"][0, 3])
        assert payload["split"].tolist() == ["train", "train", "validation", "validation", "test", "test"]


def test_write_report_records_split_and_admission_boundary(tmp_path: Path) -> None:
    report = {
        "schema_version": DATASET_SCHEMA,
        "status": "blocked",
        "reference_dataset_id": "mast-efm-test",
        "dataset_path": "processed/dataset.npz",
        "dataset_sha256": "a" * 64,
        "equilibria_count": 6,
        "grid_shape": [3, 4],
        "ragged_target_policy": {
            "keys": ["lcfs_r_m", "lcfs_z_m", "lcfs_valid_mask"],
            "padding": "NaN for coordinates and False for validity mask",
            "point_count_key": "lcfs_point_count",
            "max_lcfs_points": 5,
        },
        "split_counts": {"train": 2, "validation": 2, "test": 2},
        "split_policy": {
            "train_shots": [1],
            "validation_shots": [2],
            "test_shots": [3],
            "policy": "shot-held-out deterministic split",
        },
        "target_keys": ["psirz_Wb_per_rad"],
        "blocked_reason": "predictive claims remain blocked",
        "fallback_features": [],
        "feature_source_policy": {
            "Ip_MA": {"source_key": "Ip_MA", "transform": "identity_MA"},
            "Bt_T": {"source_key": "Bt_T", "transform": "identity_T"},
            "ffprime_scale": {"source_key": "ffprime_rms_T_rad", "transform": "campaign_median_normalised_rms"},
        },
        "next_processing_steps": ["train model"],
    }

    write_report(report, tmp_path / "dataset.json", tmp_path / "dataset.md")

    markdown = (tmp_path / "dataset.md").read_text(encoding="utf-8")
    assert "MAST EFM Neural-Equilibrium Supervised Dataset" in markdown
    assert "train=2, validation=2, test=2" in markdown
    assert "Maximum LCFS points: 5" in markdown
    assert "predictive claims remain blocked" in markdown
    assert "Fallback features: none" in markdown
    assert "`ffprime_scale` from `ffprime_rms_T_rad`" in markdown
