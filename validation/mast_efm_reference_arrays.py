# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural equilibrium reference converter
"""Extract exact time-aligned MAST EFM observations from real xarray datasets.

Missing time or convergence evidence is refused. Finite profile RMS preserves
ordinary reduction bytes and scales extreme values; masked target points remain
unobserved. These engineering array
checks do not authenticate measurement provenance or admit predictive claims.
"""

from __future__ import annotations

from typing import Any, Hashable, Protocol

import numpy as np
from numpy.typing import NDArray

from validation.mast_efm_reference_contracts import REQUIRED_EFM_VARIABLES, positive_int


class DatasetLike(Protocol):
    """The public dataset interface consumed by the array extractor."""

    @property
    def variables(self) -> Any:
        """Return the named labelled source variables."""

    def __getitem__(self, key: Hashable) -> Any:
        """Return a named array with values, dimensions and attributes."""


def _real(data: Any, *, name: Hashable) -> NDArray[np.float64]:
    """Reject boolean, text and complex coercion before converting real observations."""
    values = np.asarray(data.values)
    if values.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric observations")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(values, dtype=np.float64)
    if np.any(np.isfinite(values) & ~np.isfinite(converted)):
        raise ValueError(f"{name} must be representable in float64")
    return converted


def _time_array(ds: DatasetLike, name: str, rank: int, n_time: int) -> NDArray[np.float64]:
    """Require an explicit time dimension and move that axis to the leading position."""
    data = ds[name]
    dims = tuple(data.dims)
    values = _real(data, name=name)
    if len(dims) != rank or len(set(dims)) != rank or values.ndim != rank or "time" not in dims:
        raise ValueError(f"{name} must expose a rank-{rank} array with an explicit time dimension")
    values = np.moveaxis(values, dims.index("time"), 0)
    if values.shape[0] != n_time or any(size == 0 for size in values.shape):
        raise ValueError(f"{name} must share the nonempty exact time dimension")
    return values


def _coordinate(ds: DatasetLike, name: Hashable) -> NDArray[np.float64]:
    """Require the exact finite monotonic nonempty one-dimensional coordinate."""
    if name not in ds.variables:
        raise ValueError(f"MAST EFM dataset missing exact coordinate grid for {name}")
    values = _real(ds[name], name=name)
    if tuple(ds[name].dims) != (name,) or values.ndim != 1 or not values.size:
        raise ValueError(f"coordinate grid {name} must be a nonempty one-dimensional coordinate")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"coordinate grid {name} contains non-finite values")
    if not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)):
        raise ValueError(f"coordinate grid {name} must be strictly monotonic")
    return values


def _status_mask(ds: DatasetLike, n_time: int) -> NDArray[np.bool_]:
    """Apply an observed scalar shot-status guard or exact per-time status flags.

    Acquired MAST EFM stores expose the scheduler status as a scalar. Its success
    condition applies to the shot; it does not fabricate convergence times.
    Non-scalar status must still expose the exact time dimension, and malformed
    values or unsuccessful flags never permit a reconstruction row.
    """
    values = _real(ds["status"], name="status")
    if not ds["status"].dims and values.ndim == 0:
        return np.full(n_time, bool(np.isfinite(values) & ((values == 0) | (values == 1))), dtype=np.bool_)
    values = _time_array(ds, "status", 1, n_time)
    return np.asarray(np.isfinite(values) & ((values == 0) | (values == 1)), dtype=np.bool_)


def _profile_rms(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Preserve finite ordinary RMS reductions and scale extreme observations safely.

    The ordinary float64 reduction retains exact existing converted-reference
    identity. A conservative upper bound keeps its sum below overflow; values
    below the normal square range use scaling to retain positive tiny RMS.
    This changes arithmetic selection, not the strict source/reference check.
    """
    result = np.empty(values.shape[0], dtype=np.float64)
    for row, profile in enumerate(values):
        valid = np.abs(profile[np.isfinite(profile)])
        if not valid.size:
            raise ValueError("profile RMS cannot be computed from an all-non-finite row")
        scale = float(np.max(valid))
        if scale <= 0:
            raise ValueError("ffprime_rms_T_rad must be positive for selected equilibria")
        safe_max = 0.5 * float(np.sqrt(np.finfo(np.float64).max / valid.size))
        safe_min = float(np.sqrt(np.finfo(np.float64).tiny))
        if safe_min <= scale <= safe_max:
            result[row] = float(np.sqrt(np.mean(np.square(valid))))
        else:
            result[row] = scale * float(np.sqrt(np.mean(np.square(valid / scale))))
    return result


def extract_reference_arrays(ds: DatasetLike, *, shot_id: int, max_times: int | None = None) -> dict[str, NDArray[Any]]:
    """Extract full observed reference arrays without fabricating time, convergence or predictions.

    The source must be an actual xarray Dataset, whose dimension registry enforces
    consistent coordinate and variable lengths. Status may be an observed scalar
    shot guard; all other required variables expose the exact source time
    dimension. Rejected
    equilibrium scalar rows are filtered as before; all selected target rows need
    valid observations, and feature channels need finite values. Source current,
    axis field and FF-prime units must explicitly be A, T and T-rad respectively.
    Original spatial order is retained, including descending coordinate grids.

    >>> import xarray as xr
    >>> extract_reference_arrays(xr.Dataset(), shot_id=0)
    Traceback (most recent call last):
        ...
    ValueError: shot_id must be a positive integer
    """
    import xarray as xr

    if not isinstance(ds, xr.Dataset):
        raise ValueError("MAST EFM source must be a real xarray Dataset")
    positive_int(shot_id, field="shot_id")
    if max_times is not None:
        positive_int(max_times, field="max_times")
    missing = [name for name in REQUIRED_EFM_VARIABLES if name not in ds.variables]
    if missing:
        raise ValueError(f"MAST EFM dataset missing required variables: {', '.join(missing)}")
    time = _coordinate(ds, "time")
    if np.any(np.diff(time) <= 0):
        raise ValueError("reference time_s must be nonnegative and strictly increasing")
    n = time.size
    rank = {"psirz": 3, "ffprime": 2, "pprime": 2, "qpsi_c": 2, "lcfs_r": 2, "lcfs_z": 2}
    source = {name: _time_array(ds, name, rank.get(name, 1), n) for name in REQUIRED_EFM_VARIABLES if name != "status"}
    for name, units in (("plasma_current_x", "A"), ("bphi_rmag", "T"), ("ffprime", "T-rad")):
        if ds[name].attrs.get("units") != units:
            raise ValueError(f"{name} must declare units {units}")
    converged = source["cnvrgd_times"]
    selected = np.flatnonzero(_status_mask(ds, n) & np.isfinite(converged) & (converged > 0))
    if not selected.size:
        raise ValueError("MAST EFM dataset has no converged time slices")
    if np.any(time[selected] < 0):
        raise ValueError("reference time_s must be nonnegative and strictly increasing")
    axis, boundary = source["psi_axis"], source["psi_boundary"]
    scalar_mask = (
        np.isfinite(axis)
        & np.isfinite(boundary)
        & (axis != boundary)
        & np.isfinite(source["magnetic_axis_r"])
        & np.isfinite(source["magnetic_axis_z"])
    )
    selected = selected[scalar_mask[selected]]
    if max_times is not None:
        selected = selected[:max_times]
    if not selected.size:
        raise ValueError("MAST EFM dataset has no finite converged equilibrium slices")
    spatial = tuple(dim for dim in ds["psirz"].dims if dim != "time")
    z_grid, r_grid = _coordinate(ds, spatial[0]), _coordinate(ds, spatial[1])
    arrays: dict[str, NDArray[Any]] = {
        "time_s": time[selected],
        "r_grid_m": r_grid,
        "z_grid_m": z_grid,
        "shot_id": np.full(selected.shape, shot_id, dtype=np.int64),
        "Ip_MA": source["plasma_current_x"][selected] / 1e6,
        "Bt_T": source["bphi_rmag"][selected],
        "ffprime_rms_T_rad": _profile_rms(source["ffprime"][selected]),
    }
    for src, key in (
        ("psi_axis", "psi_axis_Wb_per_rad"),
        ("psi_boundary", "psi_boundary_Wb_per_rad"),
        ("magnetic_axis_r", "magnetic_axis_r_m"),
        ("magnetic_axis_z", "magnetic_axis_z_m"),
    ):
        arrays[key] = source[src][selected]
    for key in ("Ip_MA", "Bt_T"):
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"converted array {key} contains non-finite values")
    for src, key, mask_key in (
        ("psirz", "psirz_Wb_per_rad", "psirz_valid_mask"),
        ("pprime", "pprime_Pa_per_Wb_rad", "pprime_valid_mask"),
        ("qpsi_c", "q_profile", "q_profile_valid_mask"),
    ):
        values = source[src][selected]
        mask = np.isfinite(values)
        if np.any(~np.any(mask.reshape(selected.size, -1), axis=1)):
            raise ValueError(f"each converted {key} row must have finite valid points")
        arrays[key], arrays[mask_key] = values, mask
    lcfs_r, lcfs_z = source["lcfs_r"][selected], source["lcfs_z"][selected]
    if lcfs_r.shape != lcfs_z.shape:
        raise ValueError("LCFS R/Z arrays must have matching shapes")
    mask = np.isfinite(lcfs_r) & np.isfinite(lcfs_z)
    if np.any(~np.any(mask, axis=1)):
        raise ValueError("each converted LCFS row must contain a valid point")
    arrays["lcfs_r_m"], arrays["lcfs_z_m"], arrays["lcfs_valid_mask"] = lcfs_r, lcfs_z, mask
    return arrays
