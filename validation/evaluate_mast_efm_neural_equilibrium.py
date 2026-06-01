# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-CONTROL — MAST EFM neural-equilibrium evaluation

"""Evaluate current neural-equilibrium predictions against converted MAST EFM bundles.

This script emits fail-closed prediction evidence only. The strict predictive
EFIT/P-EFIT admission artefact remains blocked until the model path supplies the
full public-reference contract: flux, pressure, q-profile, boundary, axis, exact
reference lineage, tolerances, and independently reviewable payload hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.neural_equilibrium import (
    NEURAL_EQ_FEATURE_NAMES,
    NeuralEquilibriumAccelerator,
)

EVALUATION_SCHEMA = "scpn-control.mast-efm-neural-equilibrium-evaluation.v1"


@dataclass(frozen=True)
class FeatureProjection:
    """Feature matrix and source notes for current neural-equilibrium inference."""

    features: NDArray[np.float64]
    feature_names: tuple[str, ...]
    mapping_notes: dict[str, str]


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for an artefact path."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: dict[str, Any]) -> str:
    """Return the SHA-256 digest for a canonical JSON payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def masked_rmse(
    predicted: NDArray[np.floating[Any]] | NDArray[np.integer[Any]],
    observed: NDArray[np.floating[Any]] | NDArray[np.integer[Any]],
    mask: NDArray[np.bool_] | NDArray[np.integer[Any]],
) -> float:
    """Compute RMSE on finite points selected by a reference mask."""

    predicted_arr = np.asarray(predicted, dtype=np.float64)
    observed_arr = np.asarray(observed, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    if predicted_arr.shape != observed_arr.shape:
        raise ValueError("predicted and observed arrays must have identical shapes")
    if mask_arr.shape != observed_arr.shape:
        raise ValueError("mask and observed arrays must have identical shapes")
    valid = mask_arr & np.isfinite(predicted_arr) & np.isfinite(observed_arr)
    if not np.any(valid):
        raise ValueError("at least one finite masked point is required")
    residual = predicted_arr[valid] - observed_arr[valid]
    return float(np.sqrt(np.mean(residual * residual)))


def _first_axis_length(data: dict[str, NDArray[Any]]) -> int:
    psi = np.asarray(data["psirz_Wb_per_rad"])
    if psi.ndim != 3:
        raise ValueError("psirz_Wb_per_rad must have shape (n_equilibria, nz, nr)")
    return int(psi.shape[0])


def _per_equilibrium_array(
    data: dict[str, NDArray[Any]],
    key: str,
    count: int,
    fallback: float,
) -> NDArray[np.float64]:
    raw = data.get(key)
    if raw is None:
        return np.full(count, fallback, dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 0:
        values = np.full(count, float(arr), dtype=np.float64)
    else:
        flat = arr.reshape(-1)
        if flat.size != count:
            raise ValueError(f"{key} must contain {count} per-equilibrium values")
        values = flat.astype(np.float64)
    values[~np.isfinite(values)] = fallback
    return values


def _last_finite_profile_value(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    valid = mask & np.isfinite(values)
    if not np.any(valid):
        return 4.0
    indices = np.flatnonzero(valid)
    return float(values[indices[-1]])


def _profile_last_values(
    data: dict[str, NDArray[Any]],
    value_key: str,
    mask_key: str,
    count: int,
    fallback: float,
) -> NDArray[np.float64]:
    raw = data.get(value_key)
    if raw is None:
        return np.full(count, fallback, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 1:
        values = np.tile(values.reshape(1, -1), (count, 1))
    if values.shape[0] != count:
        raise ValueError(f"{value_key} row count must match psirz_Wb_per_rad")
    raw_mask = data.get(mask_key)
    if raw_mask is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.ndim == 1:
            mask = np.tile(mask.reshape(1, -1), (count, 1))
        if mask.shape != values.shape:
            raise ValueError(f"{mask_key} shape must match {value_key}")
    return np.asarray(
        [_last_finite_profile_value(values[row], mask[row]) for row in range(count)],
        dtype=np.float64,
    )


def _pprime_scales(data: dict[str, NDArray[Any]], count: int) -> NDArray[np.float64]:
    raw = data.get("pprime_Pa_per_Wb_rad")
    if raw is None:
        return np.ones(count, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 1:
        values = np.tile(values.reshape(1, -1), (count, 1))
    if values.shape[0] != count:
        raise ValueError("pprime_Pa_per_Wb_rad row count must match psirz_Wb_per_rad")
    raw_mask = data.get("pprime_valid_mask")
    if raw_mask is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.ndim == 1:
            mask = np.tile(mask.reshape(1, -1), (count, 1))
        if mask.shape != values.shape:
            raise ValueError("pprime_valid_mask shape must match pprime_Pa_per_Wb_rad")
    magnitudes = np.ones(count, dtype=np.float64)
    for row in range(count):
        valid = mask[row] & np.isfinite(values[row])
        if np.any(valid):
            magnitudes[row] = float(np.nanmean(np.abs(values[row, valid])))
    positive = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]
    reference = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(reference) or reference <= 0.0:
        reference = 1.0
    return np.clip(magnitudes / reference, 0.25, 4.0).astype(np.float64)


def _lcfs_geometry_features(
    data: dict[str, NDArray[Any]],
    axis_r: NDArray[np.float64],
    axis_z: NDArray[np.float64],
    count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    raw_r = data.get("lcfs_r_m")
    raw_z = data.get("lcfs_z_m")
    if raw_r is None or raw_z is None:
        return (
            np.full(count, 1.7, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
        )
    r_values = np.asarray(raw_r, dtype=np.float64)
    z_values = np.asarray(raw_z, dtype=np.float64)
    if r_values.ndim == 1:
        r_values = np.tile(r_values.reshape(1, -1), (count, 1))
    if z_values.ndim == 1:
        z_values = np.tile(z_values.reshape(1, -1), (count, 1))
    if r_values.shape[0] != count:
        raise ValueError("lcfs_r_m row count must match psirz_Wb_per_rad")
    if z_values.shape != r_values.shape:
        raise ValueError("lcfs_z_m shape must match lcfs_r_m")
    raw_mask = data.get("lcfs_valid_mask")
    if raw_mask is None:
        mask = np.ones(r_values.shape, dtype=bool)
    else:
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.ndim == 1:
            mask = np.tile(mask.reshape(1, -1), (count, 1))
        if mask.shape != r_values.shape:
            raise ValueError("lcfs_valid_mask shape must match lcfs_r_m")

    kappa = np.full(count, 1.7, dtype=np.float64)
    delta_upper = np.zeros(count, dtype=np.float64)
    delta_lower = np.zeros(count, dtype=np.float64)
    for row in range(count):
        valid = mask[row] & np.isfinite(r_values[row]) & np.isfinite(z_values[row])
        if np.count_nonzero(valid) < 3:
            continue
        r_row = r_values[row, valid]
        z_row = z_values[row, valid]
        radial_minor = float(np.max(np.abs(r_row - axis_r[row])))
        if radial_minor <= 1e-9 or not np.isfinite(radial_minor):
            continue
        z_relative = z_row - axis_z[row]
        vertical_minor = float(max(np.max(z_relative), -np.min(z_relative)))
        if vertical_minor > 0.0 and np.isfinite(vertical_minor):
            kappa[row] = float(np.clip(vertical_minor / radial_minor, 0.5, 3.5))
        upper_index = int(np.argmax(z_relative))
        lower_index = int(np.argmin(z_relative))
        delta_upper[row] = float(np.clip((axis_r[row] - r_row[upper_index]) / radial_minor, -1.0, 1.0))
        delta_lower[row] = float(np.clip((axis_r[row] - r_row[lower_index]) / radial_minor, -1.0, 1.0))
    return kappa, delta_upper, delta_lower


def build_feature_projection(data: dict[str, NDArray[Any]]) -> FeatureProjection:
    """Project converted MAST EFM fields into the current neural-model feature order."""

    count = _first_axis_length(data)
    axis_r = _per_equilibrium_array(data, "magnetic_axis_r_m", count, 1.0)
    axis_z = _per_equilibrium_array(data, "magnetic_axis_z_m", count, 0.0)
    psi_axis = _per_equilibrium_array(data, "psi_axis_Wb_per_rad", count, 0.0)
    psi_boundary = _per_equilibrium_array(data, "psi_boundary_Wb_per_rad", count, 1.0)
    pprime_scale = _pprime_scales(data, count)
    q95 = _profile_last_values(data, "q_profile", "q_profile_valid_mask", count, 4.0)
    kappa, delta_upper, delta_lower = _lcfs_geometry_features(data, axis_r, axis_z, count)

    columns = {
        "Ip_MA": np.full(count, 8.0, dtype=np.float64),
        "Bt_T": np.full(count, 5.0, dtype=np.float64),
        "R_axis_m": axis_r,
        "Z_axis_m": axis_z,
        "pprime_scale": pprime_scale,
        "ffprime_scale": np.ones(count, dtype=np.float64),
        "simag_Wb": psi_axis,
        "sibry_Wb": psi_boundary,
        "kappa": kappa,
        "delta_upper": delta_upper,
        "delta_lower": delta_lower,
        "q95": q95,
    }
    features = np.column_stack([columns[name] for name in NEURAL_EQ_FEATURE_NAMES]).astype(np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("feature projection contains non-finite values")
    notes = {
        "Ip_MA": "fallback: unavailable in converted EFM bundle; synthetic-domain centre used",
        "Bt_T": "fallback: unavailable in converted EFM bundle; synthetic-domain centre used",
        "R_axis_m": "source: magnetic_axis_r_m",
        "Z_axis_m": "source: magnetic_axis_z_m",
        "pprime_scale": "source: converted EFM pprime magnitude normalised within the bundle",
        "ffprime_scale": "fallback: unavailable in converted EFM bundle; neutral scale used",
        "simag_Wb": "source: converted EFM psi_axis_Wb_per_rad",
        "sibry_Wb": "source: converted EFM psi_boundary_Wb_per_rad",
        "kappa": "derived: finite LCFS vertical minor radius relative to magnetic axis",
        "delta_upper": "derived: finite LCFS upper triangularity relative to magnetic axis",
        "delta_lower": "derived: finite LCFS lower triangularity relative to magnetic axis",
        "q95": "source: last finite converted EFM q_profile value",
    }
    return FeatureProjection(features=features, feature_names=NEURAL_EQ_FEATURE_NAMES, mapping_notes=notes)


def _one_dimensional_grid(data: dict[str, NDArray[Any]], key: str) -> NDArray[np.float64] | None:
    raw = data.get(key)
    if raw is None:
        return None
    grid = np.asarray(raw, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError(f"{key} must be a finite one-dimensional coordinate grid")
    if not (np.all(np.diff(grid) > 0.0) or np.all(np.diff(grid) < 0.0)):
        raise ValueError(f"{key} must be strictly monotonic")
    return grid


def _coordinate_grid(
    data: dict[str, NDArray[Any]],
    grid_shape: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], str]:
    explicit_r = _one_dimensional_grid(data, "r_grid_m")
    explicit_z = _one_dimensional_grid(data, "z_grid_m")
    nz, nr = grid_shape
    if explicit_r is not None or explicit_z is not None:
        if explicit_r is None or explicit_z is None:
            raise ValueError("r_grid_m and z_grid_m must be supplied together")
        if explicit_r.size != nr or explicit_z.size != nz:
            raise ValueError("coordinate grid lengths must match predicted flux grid")
        return explicit_r, explicit_z, "source: r_grid_m and z_grid_m"

    r_values = np.asarray(data.get("lcfs_r_m"), dtype=np.float64)
    z_values = np.asarray(data.get("lcfs_z_m"), dtype=np.float64)
    axis_r = np.asarray(data.get("magnetic_axis_r_m"), dtype=np.float64)
    axis_z = np.asarray(data.get("magnetic_axis_z_m"), dtype=np.float64)
    finite_r = r_values[np.isfinite(r_values)]
    finite_z = z_values[np.isfinite(z_values)]
    finite_axis_r = axis_r[np.isfinite(axis_r)]
    finite_axis_z = axis_z[np.isfinite(axis_z)]
    if finite_r.size < 2 or finite_z.size < 2 or finite_axis_r.size == 0 or finite_axis_z.size == 0:
        raise ValueError("coordinate inference requires finite LCFS and magnetic-axis references")
    r_min = float(min(np.min(finite_r), np.min(finite_axis_r)))
    r_max = float(max(np.max(finite_r), np.max(finite_axis_r)))
    z_min = float(min(np.min(finite_z), np.min(finite_axis_z)))
    z_max = float(max(np.max(finite_z), np.max(finite_axis_z)))
    r_padding = max(0.05, 0.1 * max(r_max - r_min, 1.0e-9))
    z_padding = max(0.05, 0.1 * max(z_max - z_min, 1.0e-9))
    return (
        np.linspace(r_min - r_padding, r_max + r_padding, nr, dtype=np.float64),
        np.linspace(z_min - z_padding, z_max + z_padding, nz, dtype=np.float64),
        "inferred: reference LCFS envelope",
    )


def _interpolate_crossing(
    point_a: tuple[float, float],
    value_a: float,
    point_b: tuple[float, float],
    value_b: float,
    level: float,
) -> tuple[float, float] | None:
    if not (np.isfinite(value_a) and np.isfinite(value_b)):
        return None
    delta = value_b - value_a
    if abs(delta) <= 1.0e-15:
        return None
    fraction = (level - value_a) / delta
    if fraction < 0.0 or fraction > 1.0:
        return None
    return (
        float(point_a[0] + fraction * (point_b[0] - point_a[0])),
        float(point_a[1] + fraction * (point_b[1] - point_a[1])),
    )


def _contour_points(
    psi: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    level: float,
) -> NDArray[np.float64]:
    points: list[tuple[float, float]] = []
    for z_index in range(psi.shape[0] - 1):
        for r_index in range(psi.shape[1] - 1):
            r_left = float(r_grid[r_index])
            r_right = float(r_grid[r_index + 1])
            z_low = float(z_grid[z_index])
            z_high = float(z_grid[z_index + 1])
            top_left = float(psi[z_index, r_index])
            top_right = float(psi[z_index, r_index + 1])
            bottom_left = float(psi[z_index + 1, r_index])
            bottom_right = float(psi[z_index + 1, r_index + 1])
            edges = (
                ((r_left, z_low), top_left, (r_right, z_low), top_right),
                ((r_right, z_low), top_right, (r_right, z_high), bottom_right),
                ((r_left, z_high), bottom_left, (r_right, z_high), bottom_right),
                ((r_left, z_low), top_left, (r_left, z_high), bottom_left),
            )
            for point_a, value_a, point_b, value_b in edges:
                if (value_a - level) * (value_b - level) <= 0.0:
                    point = _interpolate_crossing(point_a, value_a, point_b, value_b, level)
                    if point is not None:
                        points.append(point)
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    rounded = np.unique(np.round(np.asarray(points, dtype=np.float64), decimals=12), axis=0)
    return rounded.astype(np.float64)


def _reference_lcfs_points(data: dict[str, NDArray[Any]], row: int) -> NDArray[np.float64]:
    r_values = np.asarray(data["lcfs_r_m"], dtype=np.float64)
    z_values = np.asarray(data["lcfs_z_m"], dtype=np.float64)
    mask = np.asarray(data.get("lcfs_valid_mask", np.ones(r_values.shape, dtype=bool)), dtype=bool)
    if r_values.ndim == 1:
        r_row = r_values
        z_row = z_values
        mask_row = mask
    else:
        r_row = r_values[row]
        z_row = z_values[row]
        mask_row = mask[row]
    valid = mask_row & np.isfinite(r_row) & np.isfinite(z_row)
    if np.count_nonzero(valid) < 3:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack((r_row[valid], z_row[valid])).astype(np.float64)


def _nearest_distance_statistics(
    predicted_points: NDArray[np.float64],
    reference_points: NDArray[np.float64],
) -> tuple[float | None, float | None]:
    if predicted_points.size == 0 or reference_points.size == 0:
        return None, None
    diff = predicted_points[:, None, :] - reference_points[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    nearest = np.min(distances, axis=1)
    return float(np.mean(nearest)), float(np.percentile(nearest, 95.0))


def evaluate_flux_geometry(
    prediction: NDArray[np.floating[Any]],
    data: dict[str, NDArray[Any]],
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64] | NDArray[np.int64]]]:
    """Derive axis and LCFS geometry metrics from predicted poloidal flux."""

    predicted_flux = np.asarray(prediction, dtype=np.float64)
    if predicted_flux.ndim != 3:
        raise ValueError("prediction must have shape (n_equilibria, nz, nr)")
    count, nz, nr = predicted_flux.shape
    r_grid, z_grid, provenance = _coordinate_grid(data, (nz, nr))
    axis_r_reference = _per_equilibrium_array(data, "magnetic_axis_r_m", count, np.nan)
    axis_z_reference = _per_equilibrium_array(data, "magnetic_axis_z_m", count, np.nan)
    psi_axis = _per_equilibrium_array(data, "psi_axis_Wb_per_rad", count, 0.0)
    psi_boundary = _per_equilibrium_array(data, "psi_boundary_Wb_per_rad", count, 1.0)

    axis_r = np.full(count, np.nan, dtype=np.float64)
    axis_z = np.full(count, np.nan, dtype=np.float64)
    boundary_point_count = np.zeros(count, dtype=np.int64)
    boundary_mean = np.full(count, np.nan, dtype=np.float64)
    boundary_p95 = np.full(count, np.nan, dtype=np.float64)
    for row in range(count):
        flux = predicted_flux[row]
        if not np.all(np.isfinite(flux)):
            raise ValueError("predicted flux contains non-finite values")
        axis_index = int(np.argmin(flux) if psi_axis[row] <= psi_boundary[row] else np.argmax(flux))
        z_index, r_index = np.unravel_index(axis_index, flux.shape)
        axis_r[row] = float(r_grid[r_index])
        axis_z[row] = float(z_grid[z_index])
        predicted_lcfs = _contour_points(flux, r_grid, z_grid, float(psi_boundary[row]))
        reference_lcfs = _reference_lcfs_points(data, row)
        boundary_point_count[row] = int(predicted_lcfs.shape[0])
        mean_distance, p95_distance = _nearest_distance_statistics(predicted_lcfs, reference_lcfs)
        if mean_distance is not None:
            boundary_mean[row] = mean_distance
        if p95_distance is not None:
            boundary_p95[row] = p95_distance

    axis_valid = np.isfinite(axis_r_reference) & np.isfinite(axis_z_reference)
    axis_residual = np.sqrt((axis_r[axis_valid] - axis_r_reference[axis_valid]) ** 2 + (axis_z[axis_valid] - axis_z_reference[axis_valid]) ** 2)
    finite_boundary_mean = boundary_mean[np.isfinite(boundary_mean)]
    finite_boundary_p95 = boundary_p95[np.isfinite(boundary_p95)]
    metrics: dict[str, Any] = {
        "coordinate_grid_provenance": provenance,
        "magnetic_axis_rmse_m": float(np.sqrt(np.mean(axis_residual * axis_residual))) if axis_residual.size else None,
        "boundary_mean_distance_m": float(np.mean(finite_boundary_mean)) if finite_boundary_mean.size else None,
        "boundary_p95_distance_m": float(np.percentile(finite_boundary_p95, 95.0)) if finite_boundary_p95.size else None,
        "derived_lcfs_success_count": int(np.count_nonzero(boundary_point_count > 0)),
    }
    arrays: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {
        "derived_magnetic_axis_r_m": axis_r,
        "derived_magnetic_axis_z_m": axis_z,
        "derived_lcfs_point_count": boundary_point_count,
        "derived_lcfs_mean_distance_m": boundary_mean,
        "derived_lcfs_p95_distance_m": boundary_p95,
    }
    return metrics, arrays


def load_reference_bundle(path: str | Path) -> dict[str, NDArray[Any]]:
    """Load a converted reference bundle into memory."""

    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def evaluate_reference_bundle(
    reference_path: str | Path,
    weights_path: str | Path,
    prediction_path: str | Path,
) -> dict[str, Any]:
    """Evaluate one converted MAST EFM reference bundle with current model weights."""

    reference = load_reference_bundle(reference_path)
    projection = build_feature_projection(reference)
    psi_reference = np.asarray(reference["psirz_Wb_per_rad"], dtype=np.float64)
    psi_mask = np.asarray(reference["psirz_valid_mask"], dtype=bool)

    accelerator = NeuralEquilibriumAccelerator()
    accelerator.load_weights(weights_path)
    prediction = np.asarray(accelerator.predict(projection.features), dtype=np.float64)
    if prediction.shape != psi_reference.shape:
        raise ValueError(
            "model prediction grid does not match reference grid: "
            f"prediction={prediction.shape}, reference={psi_reference.shape}"
        )
    geometry_metrics, geometry_arrays = evaluate_flux_geometry(prediction, reference)

    prediction_file = Path(prediction_path)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_file,
        psi_prediction_Wb_per_rad=prediction,
        psi_reference_Wb_per_rad=psi_reference,
        psi_valid_mask=psi_mask,
        feature_projection=projection.features,
        feature_names=np.asarray(projection.feature_names),
        reference_path=np.asarray(str(Path(reference_path))),
        weights_sha256=np.asarray(sha256_file(weights_path)),
        coordinate_grid_provenance=np.asarray(geometry_metrics["coordinate_grid_provenance"]),
        **geometry_arrays,
    )

    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "schema_version": EVALUATION_SCHEMA,
        "status": "pass",
        "reference_path": str(Path(reference_path)),
        "weights_path": str(Path(weights_path)),
        "prediction_path": str(prediction_file),
        "reference_artifact_sha256": sha256_file(reference_path),
        "weights_sha256": sha256_file(weights_path),
        "prediction_artifact_sha256": sha256_file(prediction_file),
        "reference_equilibria_count": int(psi_reference.shape[0]),
        "grid_shape": [int(psi_reference.shape[1]), int(psi_reference.shape[2])],
        "feature_names": list(projection.feature_names),
        "feature_mapping_notes": projection.mapping_notes,
        "metrics": {
            "psi_rmse_Wb_per_rad": masked_rmse(prediction, psi_reference, psi_mask),
            "psi_rmse_Wb": masked_rmse(prediction, psi_reference, psi_mask),
            "pressure_rmse_Pa": None,
            "q_profile_rmse": None,
            "boundary_mean_distance_m": geometry_metrics["boundary_mean_distance_m"],
            "boundary_p95_distance_m": geometry_metrics["boundary_p95_distance_m"],
            "boundary_rmse_m": geometry_metrics["boundary_mean_distance_m"],
            "magnetic_axis_rmse_m": geometry_metrics["magnetic_axis_rmse_m"],
        },
        "geometry_evidence": {
            "coordinate_grid_provenance": geometry_metrics["coordinate_grid_provenance"],
            "derived_lcfs_success_count": geometry_metrics["derived_lcfs_success_count"],
            "reference_lcfs_contract": "nearest-distance residual from predicted psi_boundary contour to converted reference LCFS points",
            "admission_use": "diagnostic evidence only; strict predictive admission remains blocked",
        },
        "admission_ready": False,
        "strict_artifact_emitted": False,
        "blocked_reason": (
            "Current model path predicts poloidal flux only and the converted public EFM bundle "
            "does not supply complete exact inputs for strict predictive EFIT/P-EFIT admission."
        ),
        "required_follow_up": [
            "Add exact-model pressure, q-profile, LCFS, and magnetic-axis prediction surfaces.",
            "Replace synthetic-domain fallback features with acquired diagnostic inputs or documented public-reference artefacts.",
            "Define admission tolerances against matched public P-EFIT or documented public reference artefacts.",
        ],
    }
    report["payload_sha256"] = sha256_json(report)
    return report


def write_report(report: dict[str, Any], json_out: str | Path | None, markdown_out: str | Path | None) -> None:
    """Write JSON and Markdown evaluator reports."""

    if json_out is not None:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_out is not None:
        markdown_path = Path(markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = report["metrics"]
        lines = [
            "# MAST EFM Neural-Equilibrium Evaluation",
            "",
            f"Schema: `{report['schema']}`",
            f"Reference equilibria: {report['reference_equilibria_count']}",
            f"Grid shape: {report['grid_shape'][0]} x {report['grid_shape'][1]}",
            f"Flux RMSE: {metrics['psi_rmse_Wb_per_rad']:.12g} Wb/rad",
            f"Magnetic-axis RMSE: {metrics['magnetic_axis_rmse_m']} m",
            f"Boundary mean distance: {metrics['boundary_mean_distance_m']} m",
            f"Coordinate grid: {report['geometry_evidence']['coordinate_grid_provenance']}",
            f"Admission ready: {report['admission_ready']}",
            f"Strict artefact emitted: {report['strict_artifact_emitted']}",
            "",
            "## Blocked reason",
            "",
            report["blocked_reason"],
            "",
            "## Required follow-up",
            "",
        ]
        lines.extend(f"- {item}" for item in report["required_follow_up"])
        lines.append("")
        markdown_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", required=True, type=Path)
    parser.add_argument("--weights-path", required=True, type=Path)
    parser.add_argument("--prediction-path", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Run one converted-bundle evaluation from the command line."""

    args = parse_args()
    report = evaluate_reference_bundle(args.reference_path, args.weights_path, args.prediction_path)
    write_report(report, args.json_out, args.report_out)


if __name__ == "__main__":
    main()
