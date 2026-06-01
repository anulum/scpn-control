# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural equilibrium evaluation tests

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_control.core.neural_equilibrium import NeuralEqConfig, NeuralEquilibriumAccelerator
from validation.evaluate_mast_efm_neural_equilibrium import (
    EVALUATION_SCHEMA,
    build_feature_projection,
    evaluate_flux_geometry,
    evaluate_reference_bundle,
    masked_rmse,
)


def _write_reference_bundle(path: Path, *, n: int = 2, grid_shape: tuple[int, int] = (5, 7)) -> None:
    z, r = grid_shape
    psirz = np.arange(n * z * r, dtype=np.float64).reshape(n, z, r) / 100.0
    mask = np.ones_like(psirz, dtype=bool)
    mask[0, 0, 0] = False
    np.savez_compressed(
        path,
        time_s=np.linspace(0.1, 0.2, n),
        psirz_Wb_per_rad=psirz,
        psirz_valid_mask=mask,
        psi_axis_Wb_per_rad=np.linspace(0.01, 0.02, n),
        psi_boundary_Wb_per_rad=np.linspace(1.01, 1.02, n),
        pprime_Pa_per_Wb_rad=np.ones((n, 3)),
        pprime_valid_mask=np.ones((n, 3), dtype=bool),
        q_profile=np.full((n, 3), 2.5),
        q_profile_valid_mask=np.ones((n, 3), dtype=bool),
        lcfs_r_m=np.array([[0.5, 0.7, 0.9, 0.7], [0.51, 0.71, 0.91, 0.71]]),
        lcfs_z_m=np.array([[0.0, 0.3, 0.0, -0.3], [0.0, 0.31, 0.0, -0.31]]),
        lcfs_valid_mask=np.ones((n, 4), dtype=bool),
        magnetic_axis_r_m=np.array([0.7, 0.71]),
        magnetic_axis_z_m=np.array([0.01, 0.02]),
        shot_id=np.full(n, 30419),
    )


def _write_weights(path: Path, *, grid_shape: tuple[int, int] = (5, 7)) -> None:
    acc = NeuralEquilibriumAccelerator(
        NeuralEqConfig(n_components=4, hidden_sizes=(), n_input_features=12, grid_shape=grid_shape)
    )
    acc.pretrain_from_synthetic_equilibria(80, seed=11, save_path=path)


def test_masked_rmse_uses_only_valid_reference_points() -> None:
    observed = np.array([[1.0, 100.0], [3.0, 5.0]])
    predicted = np.array([[2.0, -100.0], [1.0, 1.0]])
    mask = np.array([[True, False], [True, False]])

    assert masked_rmse(predicted, observed, mask) == np.sqrt((1.0 + 4.0) / 2.0)


def test_build_feature_projection_records_source_boundaries(tmp_path: Path) -> None:
    bundle = tmp_path / "reference.npz"
    _write_reference_bundle(bundle)

    with np.load(bundle, allow_pickle=False) as data:
        projection = build_feature_projection(data)

    assert projection.features.shape == (2, 12)
    assert projection.feature_names[0] == "Ip_MA"
    assert projection.mapping_notes["Ip_MA"].startswith("fallback")
    assert projection.mapping_notes["R_axis_m"] == "source: magnetic_axis_r_m"
    assert np.all(np.isfinite(projection.features))


def test_evaluate_flux_geometry_recovers_axis_and_lcfs_on_explicit_grid() -> None:
    r_grid = np.linspace(0.4, 1.0, 61)
    z_grid = np.linspace(-0.3, 0.3, 61)
    rr, zz = np.meshgrid(r_grid, z_grid)
    axis_r = 0.7
    axis_z = 0.0
    boundary_radius = 0.18
    psi = ((rr - axis_r) ** 2 + (zz - axis_z) ** 2)[None, :, :]
    theta = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    reference = {
        "r_grid_m": r_grid,
        "z_grid_m": z_grid,
        "psi_axis_Wb_per_rad": np.array([0.0]),
        "psi_boundary_Wb_per_rad": np.array([boundary_radius**2]),
        "magnetic_axis_r_m": np.array([axis_r]),
        "magnetic_axis_z_m": np.array([axis_z]),
        "lcfs_r_m": axis_r + boundary_radius * np.cos(theta)[None, :],
        "lcfs_z_m": axis_z + boundary_radius * np.sin(theta)[None, :],
        "lcfs_valid_mask": np.ones((1, theta.size), dtype=bool),
    }

    metrics, arrays = evaluate_flux_geometry(psi, reference)

    assert metrics["coordinate_grid_provenance"] == "source: r_grid_m and z_grid_m"
    assert metrics["magnetic_axis_rmse_m"] <= 1.0e-12
    assert metrics["boundary_mean_distance_m"] < 0.02
    assert metrics["boundary_p95_distance_m"] < 0.04
    assert arrays["derived_magnetic_axis_r_m"].shape == (1,)
    assert arrays["derived_lcfs_point_count"][0] > 20


def test_evaluate_reference_bundle_writes_predictions_and_blocks_admission(tmp_path: Path) -> None:
    bundle = tmp_path / "reference.npz"
    weights = tmp_path / "weights.npz"
    predictions = tmp_path / "predictions.npz"
    _write_reference_bundle(bundle)
    _write_weights(weights)

    report = evaluate_reference_bundle(reference_path=bundle, weights_path=weights, prediction_path=predictions)

    assert report["schema_version"] == EVALUATION_SCHEMA
    assert report["status"] == "pass"
    assert report["admission_ready"] is False
    assert report["strict_artifact_emitted"] is False
    assert report["reference_equilibria_count"] == 2
    assert report["metrics"]["psi_rmse_Wb"] >= 0.0
    assert report["metrics"]["magnetic_axis_rmse_m"] is not None
    assert report["metrics"]["boundary_mean_distance_m"] is not None
    assert report["metrics"]["pressure_rmse_Pa"] is None
    assert report["required_follow_up"]
    assert predictions.exists()
    with np.load(predictions, allow_pickle=False) as data:
        assert data["psi_prediction_Wb_per_rad"].shape == (2, 5, 7)
        assert data["feature_projection"].shape == (2, 12)
        assert data["derived_magnetic_axis_r_m"].shape == (2,)
        assert data["derived_lcfs_point_count"].shape == (2,)
