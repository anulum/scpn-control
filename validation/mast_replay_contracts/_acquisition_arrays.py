# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST native arrays and source metadata
"""Collect native sample arrays and JSON-safe metadata from FAIR-MAST groups."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from validation.mast_replay_contracts._acquisition_source import BUCKET
from validation.mast_replay_contracts._inputs import shot_identity

GroupOpener = Callable[[Any, int, str], Any]


GROUP_VARIABLES: dict[str, tuple[str, ...]] = {
    "summary": ("time", "ip", "line_average_n_e", "greenwald_density"),
    "equilibrium": (
        "time",
        "q95",
        "q_axis",
        "beta_tor_normal",
        "beta_tor",
        "bphi_rmag",
        "bvac_rmag",
        "minor_radius",
        "magnetic_axis_r",
        "magnetic_axis_z",
        "z",
        "x_point_z",
        "wmhd",
        "volume",
        "triangularity_upper",
        "triangularity_lower",
        "vloop_dynamic",
    ),
    "interferometer": ("time", "n_e_line"),
    "magnetics": (
        "time_saddle",
        "time_mirnov",
        "b_field_tor_probe_saddle_field",
        "b_field_tor_probe_saddle_m_phi",
        "b_field_tor_probe_saddle_u_phi",
        "b_field_tor_probe_saddle_l_phi",
        "b_field_tor_probe_cc_field",
        "b_field_tor_probe_cc_phi",
        "b_field_pol_probe_cc_field",
        "b_field_pol_probe_cc_phi",
        "b_field_pol_probe_cc_r",
        "b_field_pol_probe_cc_z",
    ),
}


def _open_group(fs: Any, shot_id: int, group: str) -> Any:
    """Open a consolidated group, bridging synchronous simplecache explicitly.

    The async wrapper calls the cache's synchronous methods on worker threads,
    preserving the S3 client's own I/O loop and whole-object cache behaviour.
    Other caller-supplied filesystem protocols retain their mapper behaviour.
    """
    import xarray as xr

    if getattr(fs, "protocol", None) == "simplecache":
        import fsspec

        fs = fsspec.filesystem("asyncwrapper", fs=fs, asynchronous=True)
    store = fs.get_mapper(f"s3://{BUCKET}/level2/shots/{shot_id}.zarr")
    return xr.open_zarr(store, group=group, consolidated=True)


def mirror_shot(
    fs: Any,
    shot_id: int,
    *,
    open_group: GroupOpener = _open_group,
    metadata_out: dict[str, dict[str, Any]] | None = None,
) -> dict[str, NDArray[Any]]:
    """Collect selected native sample arrays without resampling or calibration.

    Parameters
    ----------
    fs : filesystem
        Native mapper used by the default xarray opener, or passed unchanged to
        the caller's group opener. This routine owns no connection lifecycle.
    shot_id : int
        Identity supplied to each opener. Acquisition validates it separately.
    open_group : callable
        Opener returning a dataset with variables and array values/dimensions/
        attributes. The default uses xarray consolidated Zarr groups; supplied
        adapters are caller declarations and do not authenticate FAIR-MAST data.
    metadata_out : dict or None
        Caller mapping populated with available source dimensions, units, time
        dimension names, attributes and chunk sizes. Failures may leave entries.

    Returns
    -------
    dict of numpy.ndarray
        Present selected arrays keyed ``group.variable``, preserving source dtype,
        shape, sample values and resolution. Arrays may alias provider buffers.
        Native clocks are not aligned or checked for monotonicity here.

    Raises
    ------
    ValueError
        Any selected array has object dtype, or the required toroidal saddle
        array is missing or lacks nonempty channel/sample axes.
    TypeError
        Source attributes cannot be represented deterministically in JSON.
    Exception
        Native opener/array/metadata I/O and conversion errors propagate. No
        archive is written here. Dataset closure is the opener's responsibility.
    """
    shot_id = shot_identity(shot_id)
    payload: dict[str, NDArray[Any]] = {}
    for group, variables in GROUP_VARIABLES.items():
        dataset = open_group(fs, shot_id, group)
        for variable in variables:
            if variable in dataset.variables:
                archive_key = f"{group}.{variable}"
                source_array = dataset[variable]
                values = np.asarray(source_array.values)
                if values.dtype.hasobject:
                    raise ValueError("selected source arrays must not contain object dtype")
                payload[archive_key] = values
                if metadata_out is not None:
                    metadata_out[archive_key] = _source_array_metadata(source_array)
    if not any(key.startswith("magnetics.b_field_tor_probe_saddle_field") for key in payload):
        raise ValueError(f"shot {shot_id}: no toroidal saddle array present.")
    saddle = payload["magnetics.b_field_tor_probe_saddle_field"]
    if saddle.ndim != 2 or 0 in saddle.shape:
        raise ValueError("toroidal saddle values must have nonempty channel and sample axes")
    return payload


def _json_metadata_value(value: Any) -> Any:
    """Convert source metadata to deterministic JSON without silent stringification."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"non_finite_float": str(value)}
    if isinstance(value, np.generic):
        converted = value.item()
        if isinstance(converted, np.generic):
            raise TypeError("source metadata scalar has no Python JSON equivalent")
        return _json_metadata_value(converted)
    if isinstance(value, np.ndarray):
        return _json_metadata_value(value.tolist())
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Mapping):
        if len({str(key) for key in value}) != len(value):
            raise TypeError("source metadata keys collide after JSON string conversion")
        return {
            str(key): _json_metadata_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_metadata_value(item) for item in value]
    raise TypeError(f"unsupported source metadata type: {type(value).__module__}.{type(value).__qualname__}")


def _source_array_metadata(source_array: Any) -> dict[str, Any]:
    """Capture xarray structure without inventing absent physical metadata."""
    dimensions = [str(dimension) for dimension in getattr(source_array, "dims", ())]
    attributes = _json_metadata_value(dict(getattr(source_array, "attrs", {})))
    units = attributes.get("units") if isinstance(attributes.get("units"), str) else None
    time_dimensions = [dimension for dimension in dimensions if "time" in dimension.casefold()]
    chunks = getattr(source_array, "chunks", None)
    return {
        "dimensions": dimensions,
        "units": units,
        "timebase": {"kind": "source_dimension", "dimensions": time_dimensions} if time_dimensions else None,
        "source_attributes": attributes,
        "source_chunks": [list(chunk_sizes) for chunk_sizes in chunks] if chunks is not None else None,
        "metadata_status": "source_xarray",
    }


def _shot_summary(payload: dict[str, NDArray[Any]]) -> dict[str, Any]:
    """Summarize native Ip and saddle shape without changing array bytes."""
    ip = payload.get("summary.ip")
    ip_max_ka = None
    if ip is not None:
        values = np.abs(np.asarray(ip, dtype=np.float64))
        if values.size and np.isfinite(values).any():
            maximum = float(np.nanmax(values) / 1e3)
            ip_max_ka = maximum if math.isfinite(maximum) else None
    saddle = payload["magnetics.b_field_tor_probe_saddle_field"]
    return {
        "ip_max_ka": ip_max_ka,
        "saddle_channels": int(np.asarray(saddle).shape[0]),
        "saddle_samples": int(np.asarray(saddle).shape[1]),
        "variables": sorted(payload),
    }
