# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural equilibrium evaluation tests

"""Public MAST evaluator regression tests using real archives, models and geometry."""

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
    """Persist converted-shaped two-row reference arrays for real accelerator inference and diagnostic metrics."""
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
    """Exercise existing bounded synthetic pretraining through the accelerator API for the original inference regression."""
    acc = NeuralEquilibriumAccelerator(
        NeuralEqConfig(n_components=4, hidden_sizes=(), n_input_features=12, grid_shape=grid_shape)
    )
    acc.pretrain_from_synthetic_equilibria(80, seed=11, save_path=path)


def test_masked_rmse_uses_only_valid_reference_points() -> None:
    """Verify masked observations exclude unrelated residuals from the public metric."""
    observed = np.array([[1.0, 100.0], [3.0, 5.0]])
    predicted = np.array([[2.0, -100.0], [1.0, 1.0]])
    mask = np.array([[True, False], [True, False]])

    assert masked_rmse(predicted, observed, mask) == np.sqrt((1.0 + 4.0) / 2.0)


def test_build_feature_projection_records_source_boundaries(tmp_path: Path) -> None:
    """Read a real NPZ mapping and retain explicit fallback provenance for absent diagnostics."""
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
    """Recover analytic circular geometry through the public flux evaluator on exact source grids."""
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
    """Run actual accelerator inference, persist predictions and keep incomplete predictive admission false."""
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


def test_evaluation_single_row_matches_accelerator_batch_contract(tmp_path: Path) -> None:
    """The real accelerator's squeezed single-row result remains a one-row diagnostic artefact."""
    reference = tmp_path / "reference.npz"
    weights = tmp_path / "weights.npz"
    prediction = tmp_path / "prediction.without_npz_suffix"
    _write_reference_bundle(reference)
    with np.load(reference, allow_pickle=False) as data:
        arrays = {key: data[key][:1] for key in data.files}
    from scpn_control._npz import save_npz_arrays

    save_npz_arrays(reference, arrays)
    _write_weights(weights)
    report = evaluate_reference_bundle(reference, weights, prediction)
    assert report["reference_equilibria_count"] == 1
    assert prediction.is_file()
    assert not prediction.with_suffix(".without_npz_suffix.npz").exists()
    with np.load(prediction, allow_pickle=False) as payload:
        assert payload["psi_prediction_Wb_per_rad"].shape == (1, 5, 7)
    assert sorted(p.name for p in tmp_path.iterdir()) == [prediction.name, reference.name, weights.name]


def test_metric_refusals_and_representable_large_residuals() -> None:
    """Public RMSE refuses invalid selections and avoids square overflow without changing units."""
    import pytest

    with pytest.raises(ValueError, match="identical shapes"):
        masked_rmse(np.ones(2), np.ones(3), np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="mask and observed"):
        masked_rmse(np.ones(2), np.ones(2), np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="finite masked point"):
        masked_rmse(np.ones(2), np.ones(2), np.zeros(2, dtype=bool))
    with pytest.raises(ValueError, match="representable"):
        masked_rmse(np.array([1e308]), np.array([-1e308]), np.ones(1, dtype=bool))
    assert masked_rmse(np.ones(2), np.ones(2), np.ones(2, dtype=bool)) == 0.0
    assert masked_rmse(np.array([1e200]), np.zeros(1), np.ones(1, dtype=bool)) == 1e200


def test_reference_loader_and_report_outputs_preserve_inputs(tmp_path: Path) -> None:
    """Real archive/report writes refuse resolved input aliases and retain selected source bytes."""
    import pytest

    from validation.evaluate_mast_efm_neural_equilibrium import load_reference_bundle, write_report

    reference = tmp_path / "reference.npz"
    weights = tmp_path / "weights.npz"
    prediction = tmp_path / "prediction.npz"
    _write_reference_bundle(reference)
    _write_weights(weights)
    original = reference.read_bytes()
    arrays = load_reference_bundle(reference)
    assert arrays["psirz_Wb_per_rad"].shape == (2, 5, 7)
    for alias in (reference, weights):
        with pytest.raises(ValueError, match="must be distinct"):
            evaluate_reference_bundle(reference, weights, alias)
    report = evaluate_reference_bundle(reference, weights, prediction)
    for alias in (reference, weights, prediction):
        with pytest.raises(ValueError, match="must be distinct"):
            write_report(report, alias, None)
    assert reference.read_bytes() == original
    output = tmp_path / "evaluation.json"
    markdown = tmp_path / "evaluation.md"
    write_report(report, output, markdown)
    assert '"admission_ready": false' in output.read_text()
    assert "Admission ready: False" in markdown.read_text()
    write_report(report, None, None)


def test_real_model_grid_mismatch_refuses_before_prediction_write(tmp_path: Path) -> None:
    """A loaded real accelerator with the wrong spatial dimensions cannot emit matched-grid evidence."""
    import pytest

    reference = tmp_path / "reference.npz"
    weights = tmp_path / "weights.npz"
    prediction = tmp_path / "prediction.npz"
    _write_reference_bundle(reference)
    _write_weights(weights, grid_shape=(4, 4))
    with pytest.raises(ValueError, match="does not match"):
        evaluate_reference_bundle(reference, weights, prediction)
    assert not prediction.exists()


def test_cli_success_and_authored_refusal_use_real_inputs(tmp_path: Path) -> None:
    """Run the public CLI boundary, preserving false admission and refusing aliases before any inference."""
    import pytest

    from validation.evaluate_mast_efm_neural_equilibrium import main, parse_args

    reference = tmp_path / "reference.npz"
    weights = tmp_path / "weights.npz"
    prediction = tmp_path / "prediction.npz"
    output = tmp_path / "evaluation.json"
    markdown = tmp_path / "evaluation.md"
    _write_reference_bundle(reference)
    _write_weights(weights)
    args = [
        "--reference-path",
        str(reference),
        "--weights-path",
        str(weights),
        "--prediction-path",
        str(prediction),
        "--json-out",
        str(output),
        "--report-out",
        str(markdown),
        "--ffprime-reference",
        "1.0",
    ]
    assert main(args) == 0
    assert output.is_file() and markdown.is_file()
    args[args.index("--prediction-path") + 1] = str(reference)
    assert main(args) == 1
    args[args.index("--reference-path") + 1] = str(tmp_path / "missing-reference.npz")
    args[args.index("--prediction-path") + 1] = str(tmp_path / "missing-prediction.npz")
    assert main(args) == 1
    with pytest.raises(SystemExit) as help_exit:
        parse_args(["--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as usage_exit:
        parse_args([])
    assert usage_exit.value.code == 2
