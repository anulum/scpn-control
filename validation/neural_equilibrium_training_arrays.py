# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium trainer
"""Inspect MAST EFM tensors and execute the PCA/ridge baseline.

Tensor inspection requires all schema targets, real float64-representable finite features, supported
labels and declared counts, shot-held-out membership, increasing grids, literal
boolean masks and contiguous LCFS padding. Scalar/grid and observed target values
are checked after float64 conversion. Masked target values may be nonfinite;
they remain unobserved and are filled from training rows only for fitting.
NPZ decoding consumes the same captured bytes whose declared SHA was verified.

The numerical path keeps training-only normalisation/PCA and ridge heads.
The unpenalised intercept count is assigned exactly, avoiding cancellation at
large finite alpha. Nonfinite fits refuse before weight persistence. Format
regressions exercise this path through the public trainer; they are not
authenticated MAST data or scientific admission evidence.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from validation.neural_equilibrium_dataset_features import validate_feature_matrix
from validation.neural_equilibrium_dataset_tensors import load_verified_npz
from validation.neural_equilibrium_training_inputs import (
    FEATURE_NAMES,
    SPLITS,
    TARGET_KEYS,
    TRAINING_SCHEMA,
    TrainingInputs,
    _sha256_file,
)


def _load_dataset(path: Path, expected_sha256: str) -> dict[str, NDArray[Any]]:
    """Decode only the same captured supervised NPZ bytes verified against the dataset declaration, without pickle."""
    try:
        return load_verified_npz(
            path, expected_sha256, mismatch_message="dataset payload SHA-256 does not match the dataset report"
        )
    except ValueError as exc:
        raise ValueError(f"cannot load supervised NPZ dataset: {exc}") from exc


def _require_keys(data: dict[str, NDArray[Any]], keys: tuple[str, ...] | list[str]) -> None:
    """Require every consumed feature, grid, identity and target key before array conversion."""
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"dataset is missing required keys: {', '.join(missing)}")


def _finite_array(data: dict[str, NDArray[Any]], key: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Require exact real scalar/grid shape and finite values after conversion to the computation dtype float64."""
    raw = data[key]
    if raw.dtype.kind not in "ifu" or raw.shape != shape:
        raise ValueError(f"{key} must be finite real values with shape {shape}")
    values = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must be finite real values with shape {shape}")
    return values


def _masked_target(data: dict[str, NDArray[Any]], key: str, mask_key: str, shape: tuple[int, ...]) -> None:
    """Require real target/boolean mask shape and finite float64-representable declared valid observations."""
    raw, mask = data[key], data[mask_key]
    if raw.dtype.kind not in "ifu" or raw.shape != shape or mask.dtype.kind != "b" or mask.shape != shape:
        raise ValueError(f"{key} and {mask_key} must have target/boolean mask shape {shape}")
    if not np.all(np.isfinite(np.asarray(raw[mask], dtype=np.float64))):
        raise ValueError(f"{key} must be finite on valid mask entries")


def _dataset_metadata(data: dict[str, NDArray[Any]], dataset_report: dict[str, Any]) -> dict[str, Any]:
    """Validate actual row counts, declared grid, target masks and shot-held-out membership.

    Shapes, named features, strict boolean masks, finite float64-representable
    observed targets and grids, integer shot/LCFS counts and consistent padding are verified before
    fitting. This does not authenticate physical source provenance or units.
    """
    _require_keys(
        data,
        [
            "features",
            "feature_names",
            "split",
            "shot_id",
            "time_s",
            "lcfs_point_count",
            "r_grid_m",
            "z_grid_m",
            *TARGET_KEYS,
        ],
    )
    count = dataset_report["equilibria_count"]
    features = validate_feature_matrix(data["features"], row_count=count)
    names = data["feature_names"]
    if (
        names.dtype.kind not in "US"
        or names.shape != (len(FEATURE_NAMES),)
        or tuple(names.astype(str)) != FEATURE_NAMES
    ):
        raise ValueError("dataset feature_names do not match the declared training contract")
    split = data["split"]
    if split.dtype.kind not in "US" or split.shape != (count,):
        raise ValueError("split labels must have one string value per feature row")
    labels = split.astype(str)
    if not np.all(np.isin(labels, SPLITS)):
        raise ValueError("split labels must use train, validation or test")
    split_counts = {name: int(np.count_nonzero(labels == name)) for name in SPLITS}
    if split_counts != dataset_report["split_counts"]:
        raise ValueError("dataset split_counts do not match the dataset report")
    shots = data["shot_id"]
    if shots.dtype.kind not in "iu" or shots.shape != (count,) or np.any(shots <= 0):
        raise ValueError("shot_id must be positive integer per-equilibrium values")
    for shot in np.unique(shots):
        if len(set(labels[shots == shot])) != 1:
            raise ValueError("shot-held-out split must not leak one shot across train/validation/test")
    time_s = _finite_array(data, "time_s", (count,))
    if np.any(time_s < 0):
        raise ValueError("time_s must be nonnegative")
    nz, nr = dataset_report["grid_shape"]
    for key, length in [("z_grid_m", nz), ("r_grid_m", nr)]:
        grid = _finite_array(data, key, (length,))
        if np.any(np.diff(grid) <= 0):
            raise ValueError(f"{key} must be strictly increasing")
    _masked_target(data, "psirz_Wb_per_rad", "psirz_valid_mask", (count, nz, nr))
    for key in ("psi_axis_Wb_per_rad", "psi_boundary_Wb_per_rad", "magnetic_axis_r_m", "magnetic_axis_z_m"):
        _finite_array(data, key, (count,))
    for key, mask in [("pprime_Pa_per_Wb_rad", "pprime_valid_mask"), ("q_profile", "q_profile_valid_mask")]:
        raw = data[key]
        if raw.ndim != 2 or raw.shape[1] < 1:
            raise ValueError(f"{key} must have nonempty per-equilibrium profile columns")
        _masked_target(data, key, mask, (count, raw.shape[1]))
    width = dataset_report["ragged_target_policy"]["max_lcfs_points"]
    for key in ("lcfs_r_m", "lcfs_z_m"):
        _masked_target(data, key, "lcfs_valid_mask", (count, width))
    lcfs_count = data["lcfs_point_count"]
    if lcfs_count.dtype.kind not in "iu" or lcfs_count.shape != (count,) or np.any(lcfs_count < 1):
        raise ValueError("lcfs_point_count must be positive integer per-equilibrium values")
    lcfs_mask = data["lcfs_valid_mask"]
    if not np.array_equal(lcfs_count, lcfs_mask.sum(axis=1)) or not np.array_equal(
        lcfs_mask, np.arange(width)[None, :] < lcfs_count[:, None]
    ):
        raise ValueError("LCFS point counts must match contiguous valid entries and padded tail")
    return {
        "equilibria_count": count,
        "feature_count": int(features.shape[1]),
        "grid_shape": [nz, nr],
        "split_counts": split_counts,
        "target_keys": list(TARGET_KEYS),
        "max_lcfs_points": int(np.max(lcfs_count)),
    }


def _standardise_train(
    features: NDArray[np.float64], train_mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Standardise using training rows only, replacing near-zero training-column standard deviations with one."""
    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0)
    std[std < 1.0e-12] = 1.0
    return (features - mean) / std, mean, std


def _ridge_fit(x: NDArray[np.float64], y: NDArray[np.float64], ridge_alpha: float) -> NDArray[np.float64]:
    """Fit ridge normal equations with the exact unpenalised intercept count, avoiding large-alpha cancellation."""
    x_aug = np.column_stack([x, np.ones(x.shape[0])])
    gram = x_aug.T @ x_aug + ridge_alpha * np.eye(x_aug.shape[1])
    gram[-1, -1] = x_aug.shape[0]
    return cast(NDArray[np.float64], np.linalg.solve(gram, x_aug.T @ y))


def _ridge_predict(x: NDArray[np.float64], coeff: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply fitted ridge coefficients to augmented feature rows without refitting holdout observations."""
    x_aug = np.column_stack([x, np.ones(x.shape[0])])
    return np.asarray(x_aug @ coeff, dtype=np.float64)


def _pca_fit(
    y_train: NDArray[np.float64], n_components: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Fit centred training-row SVD components and return coefficients plus bounded explained variance."""
    mean = y_train.mean(axis=0)
    centred = y_train - mean
    _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)
    usable = min(max(1, int(n_components)), vt.shape[0])
    components = vt[:usable]
    coeffs = centred @ components.T
    total = float(np.sum(singular_values**2))
    explained = float(np.sum(singular_values[:usable] ** 2) / max(total, 1.0e-30))
    return mean, components, coeffs, explained


def _masked_rmse(
    predicted: NDArray[np.float64], observed: NDArray[np.float64], mask: NDArray[np.bool_]
) -> float | None:
    """Return RMS error on jointly finite declared valid observations, or None when none exist."""
    valid = mask & np.isfinite(predicted) & np.isfinite(observed)
    if not np.any(valid):
        return None
    residual = predicted[valid] - observed[valid]
    return float(np.sqrt(np.mean(residual**2)))


def _fill_masked_columns(
    values: NDArray[np.float64], mask: NDArray[np.bool_], train_mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Fill invalid/masked columns from valid training means, using zero only where no training observation exists."""
    filled = np.asarray(values, dtype=np.float64).copy()
    valid_train = mask[train_mask] & np.isfinite(filled[train_mask])
    defaults = np.zeros(filled.shape[1], dtype=np.float64)
    for col in range(filled.shape[1]):
        column_valid = valid_train[:, col]
        if np.any(column_valid):
            defaults[col] = float(np.mean(filled[train_mask][column_valid, col]))
    invalid = ~mask | ~np.isfinite(filled)
    rows, cols = np.nonzero(invalid)
    filled[rows, cols] = defaults[cols]
    return filled


def _fill_masked_flux(
    values: NDArray[np.float64], mask: NDArray[np.bool_], train_mask: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Flatten the flux grid for training-only masked-column filling without changing original target shape metadata."""
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    return _fill_masked_columns(flat_values, flat_mask, train_mask)


def _split_metrics(
    data: dict[str, NDArray[Any]],
    labels: NDArray[np.str_],
    predictions: dict[str, NDArray[np.float64]],
) -> dict[str, dict[str, float | None]]:
    """Compute actual per-split masked profile/geometry errors and combined magnetic-axis RMS error."""
    metrics: dict[str, dict[str, float | None]] = {}
    psirz = np.asarray(data["psirz_Wb_per_rad"], dtype=np.float64)
    pprime = np.asarray(data["pprime_Pa_per_Wb_rad"], dtype=np.float64)
    q_profile = np.asarray(data["q_profile"], dtype=np.float64)
    lcfs_r = np.asarray(data["lcfs_r_m"], dtype=np.float64)
    lcfs_z = np.asarray(data["lcfs_z_m"], dtype=np.float64)
    axis = np.column_stack(
        [
            np.asarray(data["magnetic_axis_r_m"], dtype=np.float64),
            np.asarray(data["magnetic_axis_z_m"], dtype=np.float64),
        ]
    )
    for split in SPLITS:
        split_mask = labels == split
        axis_error = np.linalg.norm(predictions["axis"][split_mask] - axis[split_mask], axis=1)
        metrics[split] = {
            "psi_rmse_Wb_per_rad": _masked_rmse(
                predictions["psirz"][split_mask],
                psirz[split_mask],
                np.asarray(data["psirz_valid_mask"], dtype=bool)[split_mask],
            ),
            "pprime_rmse_Pa_per_Wb_rad": _masked_rmse(
                predictions["pprime"][split_mask],
                pprime[split_mask],
                np.asarray(data["pprime_valid_mask"], dtype=bool)[split_mask],
            ),
            "q_profile_rmse": _masked_rmse(
                predictions["q_profile"][split_mask],
                q_profile[split_mask],
                np.asarray(data["q_profile_valid_mask"], dtype=bool)[split_mask],
            ),
            "lcfs_r_rmse_m": _masked_rmse(
                predictions["lcfs_r"][split_mask],
                lcfs_r[split_mask],
                np.asarray(data["lcfs_valid_mask"], dtype=bool)[split_mask],
            ),
            "lcfs_z_rmse_m": _masked_rmse(
                predictions["lcfs_z"][split_mask],
                lcfs_z[split_mask],
                np.asarray(data["lcfs_valid_mask"], dtype=bool)[split_mask],
            ),
            "magnetic_axis_rmse_m": float(np.sqrt(np.mean(axis_error**2))) if axis_error.size else None,
        }
    return metrics


def _execute_training(
    data: dict[str, NDArray[Any]],
    inputs: TrainingInputs,
) -> tuple[dict[str, Any], str]:
    """Fit the existing NumPy PCA/ridge heads and persist actual NPZ coefficients.

    At least two training rows are required. Feature normalisation, missing-target
    fill and PCA use training rows only; holdout rows do not fit these transforms.
    Coefficient/prediction nonfinite values refuse before persistence. The caller
    has already validated tensor shape, shot split and source/compute admission.
    Numerical/IO failures propagate; execution does not grant predictive admission.
    """
    start = time.perf_counter()
    features = np.asarray(data["features"], dtype=np.float64)
    labels = np.asarray(data["split"]).astype(str)
    train_mask = labels == "train"
    if int(np.count_nonzero(train_mask)) < 2:
        raise ValueError("at least two training equilibria are required")
    x, x_mean, x_std = _standardise_train(features, train_mask)
    psirz = np.asarray(data["psirz_Wb_per_rad"], dtype=np.float64)
    y_flux = _fill_masked_flux(psirz, np.asarray(data["psirz_valid_mask"], dtype=bool), train_mask)
    flux_mean, flux_components, flux_train_coeffs, flux_explained = _pca_fit(
        y_flux[train_mask],
        inputs.max_flux_components,
    )
    flux_regression = _ridge_fit(x[train_mask], flux_train_coeffs, inputs.ridge_alpha)
    flux_coeffs = _ridge_predict(x, flux_regression)
    flux_pred = (flux_coeffs @ flux_components + flux_mean).reshape(psirz.shape)

    pprime = np.asarray(data["pprime_Pa_per_Wb_rad"], dtype=np.float64)
    q_profile = np.asarray(data["q_profile"], dtype=np.float64)
    lcfs_r = np.asarray(data["lcfs_r_m"], dtype=np.float64)
    lcfs_z = np.asarray(data["lcfs_z_m"], dtype=np.float64)
    pprime_filled = _fill_masked_columns(pprime, np.asarray(data["pprime_valid_mask"], dtype=bool), train_mask)
    q_filled = _fill_masked_columns(q_profile, np.asarray(data["q_profile_valid_mask"], dtype=bool), train_mask)
    lcfs_mask = np.asarray(data["lcfs_valid_mask"], dtype=bool)
    lcfs_r_filled = _fill_masked_columns(lcfs_r, lcfs_mask, train_mask)
    lcfs_z_filled = _fill_masked_columns(lcfs_z, lcfs_mask, train_mask)
    axis = np.column_stack(
        [
            np.asarray(data["magnetic_axis_r_m"], dtype=np.float64),
            np.asarray(data["magnetic_axis_z_m"], dtype=np.float64),
        ]
    )
    regressions = {
        "pprime": _ridge_fit(x[train_mask], pprime_filled[train_mask], inputs.ridge_alpha),
        "q_profile": _ridge_fit(x[train_mask], q_filled[train_mask], inputs.ridge_alpha),
        "lcfs_r": _ridge_fit(x[train_mask], lcfs_r_filled[train_mask], inputs.ridge_alpha),
        "lcfs_z": _ridge_fit(x[train_mask], lcfs_z_filled[train_mask], inputs.ridge_alpha),
        "axis": _ridge_fit(x[train_mask], axis[train_mask], inputs.ridge_alpha),
    }
    predictions = {
        "psirz": flux_pred,
        "pprime": _ridge_predict(x, regressions["pprime"]),
        "q_profile": _ridge_predict(x, regressions["q_profile"]),
        "lcfs_r": _ridge_predict(x, regressions["lcfs_r"]),
        "lcfs_z": _ridge_predict(x, regressions["lcfs_z"]),
        "axis": _ridge_predict(x, regressions["axis"]),
    }
    if any(
        not np.all(np.isfinite(values))
        for values in [
            x_mean,
            x_std,
            flux_mean,
            flux_components,
            flux_regression,
            *regressions.values(),
            *predictions.values(),
        ]
    ):
        raise ValueError("baseline fit must produce finite coefficients and predictions before weights persistence")
    metrics = _split_metrics(data, labels, predictions)
    inputs.weights_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        inputs.weights_out,
        schema_version=np.asarray([TRAINING_SCHEMA]),
        feature_names=np.asarray(FEATURE_NAMES),
        x_mean=x_mean,
        x_std=x_std,
        ridge_alpha=np.asarray([inputs.ridge_alpha]),
        flux_mean=flux_mean,
        flux_components=flux_components,
        flux_regression=flux_regression,
        flux_explained_variance=np.asarray([flux_explained]),
        pprime_regression=regressions["pprime"],
        q_profile_regression=regressions["q_profile"],
        lcfs_r_regression=regressions["lcfs_r"],
        lcfs_z_regression=regressions["lcfs_z"],
        axis_regression=regressions["axis"],
        lcfs_point_count=np.asarray(data["lcfs_point_count"], dtype=np.int64),
        grid_shape=np.asarray(psirz.shape[1:], dtype=np.int64),
    )
    return (
        {
            "execution_mode": "execute",
            "weights_path": str(inputs.weights_out),
            "weights_sha256": _sha256_file(inputs.weights_out),
            "flux_components": int(flux_components.shape[0]),
            "flux_explained_variance": flux_explained,
            "ridge_alpha": float(inputs.ridge_alpha),
            "train_time_s": time.perf_counter() - start,
            "holdout_metrics": metrics,
        },
        _sha256_file(inputs.weights_out),
    )
