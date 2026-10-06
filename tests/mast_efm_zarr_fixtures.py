# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural equilibrium reference converter tests
"""Real xarray/Zarr engineering fixtures retaining original converter observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr


def sample_dataset() -> xr.Dataset:
    """Return the exact retained three-time engineering arrays as real labelled data."""
    time = np.array([0.1, 0.2, 0.3])
    ds = xr.Dataset(
        {
            "time": xr.DataArray(time, dims=("time",)),
            "profile_r": xr.DataArray(np.array([0.4, 0.5]), dims=("profile_r",)),
            "profile_z": xr.DataArray(np.array([-0.1, 0.1]), dims=("profile_z",)),
            "status": xr.DataArray([1, -1, 1], dims=("time",)),
            "cnvrgd_times": xr.DataArray([1, 1, 1], dims=("time",)),
            "psirz": xr.DataArray(
                np.array([[[1.0, np.nan], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]], [[8.0, 9.0], [10.0, 11.0]]]),
                dims=("time", "profile_z", "profile_r"),
            ),
            "psi_axis": xr.DataArray([0.1, 0.2, 0.3], dims=("time",)),
            "psi_boundary": xr.DataArray([1.1, 1.2, 1.3], dims=("time",)),
            "plasma_current_x": xr.DataArray([8.0e6, 8.1e6, 8.2e6], dims=("time",)),
            "bphi_rmag": xr.DataArray([5.0, 5.1, 5.2], dims=("time",)),
            "ffprime": xr.DataArray(np.array([[2.0, 4.0], [3.0, 5.0], [6.0, 8.0]]), dims=("time", "psi_norm")),
            "pprime": xr.DataArray(np.ones((3, 2)), dims=("time", "psi_norm")),
            "qpsi_c": xr.DataArray(np.full((3, 2), 2.0), dims=("time", "psi_norm")),
            "lcfs_r": xr.DataArray(np.full((3, 4), 0.8), dims=("time", "lcfs_coords")),
            "lcfs_z": xr.DataArray(np.full((3, 4), 0.1), dims=("time", "lcfs_coords")),
            "magnetic_axis_r": xr.DataArray([0.7, 0.71, 0.72], dims=("time",)),
            "magnetic_axis_z": xr.DataArray([0.01, 0.02, 0.03], dims=("time",)),
        }
    )

    for name, units in (("plasma_current_x", "A"), ("bphi_rmag", "T"), ("ffprime", "T-rad")):
        ds[name].attrs["units"] = units
    return ds


def write_zarr(path: Path, ds: xr.Dataset | None = None) -> Path:
    """Write a genuine consolidated v2 store using the declared optional runtime.

    Zarr-Python 3 writes format 3 unless told otherwise, so format 2 is named
    there. Zarr-Python 2 writes only format 2 and does not know the argument.
    """
    values = sample_dataset() if ds is None else ds
    path.parent.mkdir(parents=True, exist_ok=True)
    if int(zarr.__version__.split(".", 1)[0]) >= 3:
        values.to_zarr(path, mode="w", consolidated=True, zarr_format=2)
    else:
        values.to_zarr(path, mode="w", consolidated=True)
    return path


def dataset_from_reference(reference: dict[str, Any]) -> xr.Dataset:
    """Represent engineering reference observations as real original-variable source arrays.

    This fixture is deliberately an explicit local format round trip, not an
    acquired measurement. Scalar and profile dimensions stay labelled separately;
    target masks become unobserved NaNs in their source fields.
    """
    times = np.asarray(reference["time_s"])
    n = times.size
    scalars = {
        "psi_axis": "psi_axis_Wb_per_rad",
        "psi_boundary": "psi_boundary_Wb_per_rad",
        "magnetic_axis_r": "magnetic_axis_r_m",
        "magnetic_axis_z": "magnetic_axis_z_m",
        "bphi_rmag": "Bt_T",
    }
    data: dict[str, Any] = {name: xr.DataArray(reference[key], dims=("time",)) for name, key in scalars.items()}
    data.update(
        status=xr.DataArray(np.ones(n), dims=("time",)),
        cnvrgd_times=xr.DataArray(np.ones(n), dims=("time",)),
        plasma_current_x=xr.DataArray(np.asarray(reference["Ip_MA"]) * 1e6, dims=("time",), attrs={"units": "A"}),
        ffprime=xr.DataArray(
            np.asarray(reference["ffprime_rms_T_rad"]).reshape(n, 1),
            dims=("time", "psi_norm"),
            attrs={"units": "T-rad"},
        ),
    )
    data["bphi_rmag"].attrs["units"] = "T"
    for name, key, mask, dims in (
        ("psirz", "psirz_Wb_per_rad", "psirz_valid_mask", ("time", "profile_z", "profile_r")),
        ("pprime", "pprime_Pa_per_Wb_rad", "pprime_valid_mask", ("time", "pprime_norm")),
        ("qpsi_c", "q_profile", "q_profile_valid_mask", ("time", "q_norm")),
        ("lcfs_r", "lcfs_r_m", "lcfs_valid_mask", ("time", "lcfs_coords")),
        ("lcfs_z", "lcfs_z_m", "lcfs_valid_mask", ("time", "lcfs_coords")),
    ):
        data[name] = xr.DataArray(np.where(reference[mask], reference[key], np.nan), dims=dims)
    return xr.Dataset(
        data, coords={"time": times, "profile_r": reference["r_grid_m"], "profile_z": reference["z_grid_m"]}
    )
