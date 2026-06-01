# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural equilibrium reference converter tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from validation.convert_mast_efm_neural_equilibrium_reference import (
    CANDIDATE_SCHEMA,
    convert_campaign,
    extract_reference_arrays,
)


class FakeArray:
    def __init__(self, values: object, dims: tuple[str, ...]) -> None:
        self.values = np.asarray(values)
        self.dims = dims


class FakeDataset:
    def __init__(self, values: dict[str, FakeArray]) -> None:
        self._values = values
        self.variables = values
        self.sizes = {"time": int(values["status"].values.size)}

    def __getitem__(self, key: str) -> FakeArray:
        return self._values[key]


def _sample_dataset() -> FakeDataset:
    time = np.array([0.1, 0.2, 0.3])
    return FakeDataset(
        {
            "time": FakeArray(time, ("time",)),
            "profile_r": FakeArray(np.array([0.4, 0.5]), ("profile_r",)),
            "profile_z": FakeArray(np.array([-0.1, 0.1]), ("profile_z",)),
            "status": FakeArray([1, -1, 1], ("time",)),
            "cnvrgd_times": FakeArray([1, 1, 1], ("time",)),
            "psirz": FakeArray(
                np.array([[[1.0, np.nan], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]], [[8.0, 9.0], [10.0, 11.0]]]),
                ("time", "profile_z", "profile_r"),
            ),
            "psi_axis": FakeArray([0.1, 0.2, 0.3], ("time",)),
            "psi_boundary": FakeArray([1.1, 1.2, 1.3], ("time",)),
            "pprime": FakeArray(np.ones((3, 2)), ("time", "psi_norm")),
            "qpsi_c": FakeArray(np.full((3, 2), 2.0), ("time", "psi_norm")),
            "lcfs_r": FakeArray(np.full((3, 4), 0.8), ("time", "lcfs_coords")),
            "lcfs_z": FakeArray(np.full((3, 4), 0.1), ("time", "lcfs_coords")),
            "magnetic_axis_r": FakeArray([0.7, 0.71, 0.72], ("time",)),
            "magnetic_axis_z": FakeArray([0.01, 0.02, 0.03], ("time",)),
        }
    )


def test_extract_reference_arrays_keeps_only_converged_time_slices() -> None:
    arrays = extract_reference_arrays(_sample_dataset(), shot_id=30419)

    assert arrays["time_s"].tolist() == [0.1, 0.3]
    assert arrays["shot_id"].tolist() == [30419, 30419]
    assert arrays["psirz_Wb_per_rad"].shape == (2, 2, 2)
    assert arrays["psirz_valid_mask"].shape == (2, 2, 2)
    assert arrays["psirz_valid_mask"][0].tolist() == [[True, False], [True, True]]
    assert arrays["psi_axis_Wb_per_rad"].tolist() == [0.1, 0.3]
    assert arrays["psi_boundary_Wb_per_rad"].tolist() == [1.1, 1.3]
    assert arrays["lcfs_r_m"].shape == arrays["lcfs_z_m"].shape
    assert arrays["r_grid_m"].tolist() == [0.4, 0.5]
    assert arrays["z_grid_m"].tolist() == [-0.1, 0.1]


def test_extract_reference_arrays_rejects_missing_exact_coordinate_grid() -> None:
    ds = _sample_dataset()
    del ds.variables["profile_r"]

    try:
        extract_reference_arrays(ds, shot_id=30419)
    except ValueError as exc:
        assert "profile_r" in str(exc)
    else:
        raise AssertionError("missing profile_r was not rejected")


def test_extract_reference_arrays_rejects_missing_required_variable() -> None:
    ds = _sample_dataset()
    del ds.variables["psirz"]

    try:
        extract_reference_arrays(ds, shot_id=30419)
    except ValueError as exc:
        assert "psirz" in str(exc)
    else:
        raise AssertionError("missing psirz was not rejected")


def test_convert_campaign_reports_blocked_reference_candidate(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "converted"
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {"shots": [{"status": "acquired", "shot_id": 30419, "local_path": "mast/level1/shot_30419/efm.zarr"}]}
        ),
        encoding="utf-8",
    )

    def sample_convert_shot_zarr(**kwargs: Any) -> object:
        arrays = extract_reference_arrays(_sample_dataset(), shot_id=kwargs["shot_id"])
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(kwargs["output_path"], **arrays)
        from validation.convert_mast_efm_neural_equilibrium_reference import _converted_summary

        return _converted_summary(
            shot_id=kwargs["shot_id"],
            source_path=kwargs["zarr_path"],
            output_path=kwargs["output_path"],
            arrays=arrays,
        )

    monkeypatch.setattr(
        "validation.convert_mast_efm_neural_equilibrium_reference.convert_shot_zarr",
        sample_convert_shot_zarr,
    )

    report = convert_campaign(dataset_root=dataset_root, campaign_manifest=manifest, output_root=output_root)

    assert report["schema_version"] == CANDIDATE_SCHEMA
    assert report["status"] == "pass"
    assert report["admission_ready"] is False
    assert report["reference_equilibria_count"] == 2
    assert report["shots"][0]["sha256"]
    assert Path(report["shots"][0]["output_path"]).exists()
    assert "exact-model predictions" in report["blocked_reason"]
    assert len(report["payload_sha256"]) == 64
