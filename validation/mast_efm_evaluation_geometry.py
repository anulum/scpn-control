# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Evaluate mast efm neural equilibrium.

"""Diagnostic flux-derived axis and boundary geometry on source or inferred metre grids.

Nearest-point distances are directed; they do not establish a closed, connected
LCFS or a predictive EFIT tolerance. Input arrays remain caller-owned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _per_equilibrium_array(
    data: dict[str, NDArray[Any]],
    key: str,
    count: int,
    fallback: float,
) -> NDArray[np.float64]:
    """Broadcast a scalar or require N values; absent or nonfinite entries use the selected diagnostic fallback."""
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


def _one_dimensional_grid(data: dict[str, NDArray[Any]], key: str) -> NDArray[np.float64] | None:
    """Require a finite strictly monotonic metre coordinate vector of at least two points; an absent key returns None."""
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
    """Use paired source coordinate vectors, or infer a padded LCFS/axis envelope for diagnostic geometry only."""
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
    """Interpolate a caller-validated finite bracketing edge; numerically flat edges return None.

    The public evaluator rejects nonfinite flux before contour traversal, and
    the contour collector calls this only when the level lies between endpoints.
    Scale the flux values if their finite endpoint difference would overflow.
    """
    delta = value_b - value_a
    if abs(delta) <= 1.0e-15:
        return None
    if np.isfinite(delta):
        fraction = (level - value_a) / delta
    else:
        scale = max(abs(value_a), abs(value_b), abs(level))
        fraction = (level / scale - value_a / scale) / (value_b / scale - value_a / scale)
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
    """Collect unique edge intersections with the supplied flux level; rounding to twelve decimals is diagnostic only."""
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
                if min(value_a, value_b) <= level <= max(value_a, value_b):
                    point = _interpolate_crossing(point_a, value_a, point_b, value_b, level)
                    if point is not None:
                        points.append(point)
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    rounded = np.unique(np.round(np.asarray(points, dtype=np.float64), decimals=12), axis=0)
    return rounded.astype(np.float64)


def _reference_lcfs_points(data: dict[str, NDArray[Any]], row: int) -> NDArray[np.float64]:
    """Select jointly finite masked LCFS coordinates for one row; fewer than three points yield empty geometry."""
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
    """Measure directed predicted-to-reference nearest-point mean and p95; missing point clouds yield None."""
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
    """Derive diagnostic axis and directed boundary residuals on paired metre coordinate grids.

    Prediction shape is (N, nz, nr) and every flux value must be finite. Paired
    finite monotonic source grids match nz/nr; absent grids infer a padded
    reference envelope. Axis location uses the grid minimum or maximum according
    to reference axis/boundary flux ordering. Missing observed axes yield null
    residuals. LCFS output is a rounded edge-crossing point cloud, not a proof of
    connectedness/enclosure. Distances are predicted-to-reference mean/p95 in
    metres and missing geometry yields null metrics. Arrays remain caller-owned.
    Invalid grids, flux shape/finiteness or per-row alignment raise ValueError.

    >>> evaluate_flux_geometry(np.zeros((2, 3)), {})
    Traceback (most recent call last):
        ...
    ValueError: prediction must have shape (n_equilibria, nz, nr)
    """
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
    axis_residual = np.sqrt(
        (axis_r[axis_valid] - axis_r_reference[axis_valid]) ** 2
        + (axis_z[axis_valid] - axis_z_reference[axis_valid]) ** 2
    )
    finite_boundary_mean = boundary_mean[np.isfinite(boundary_mean)]
    finite_boundary_p95 = boundary_p95[np.isfinite(boundary_p95)]
    metrics: dict[str, Any] = {
        "coordinate_grid_provenance": provenance,
        "magnetic_axis_rmse_m": float(np.sqrt(np.mean(axis_residual * axis_residual))) if axis_residual.size else None,
        "boundary_mean_distance_m": float(np.mean(finite_boundary_mean)) if finite_boundary_mean.size else None,
        "boundary_p95_distance_m": float(np.percentile(finite_boundary_p95, 95.0))
        if finite_boundary_p95.size
        else None,
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
