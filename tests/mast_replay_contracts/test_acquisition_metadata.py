# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST native xarray metadata acquisition contracts
"""Exercise source metadata through actual xarray/file group adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from mast_replay_contracts.test_acquisition_native_files import _source
from validation import acquire_mast_disruption_shots as acquisition


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (False, False),
        (np.float32(1.5), 1.5),
        (np.asarray([2, 1]), [2, 1]),
        (np.asarray(2.5), 2.5),
        (b"\x00\xff", {"bytes_hex": "00ff"}),
        (float("inf"), {"non_finite_float": "inf"}),
        ({2: "two", "a": 1}, {"2": "two", "a": 1}),
        ((2, "x"), [2, "x"]),
    ],
)
def test_mirror_shot_keeps_native_attribute_meaning(tmp_path: Path, value: object, expected: object) -> None:
    """Retain structured native metadata without changing selected array bytes."""
    source = _source(tmp_path / "native")

    def attributed_group(fs: Any, shot_id: int, group: str) -> xr.Dataset:
        """Attach declared xarray attributes to the actual decoded native group."""
        dataset: xr.Dataset = source.open_group(fs, shot_id, group)
        if group == "summary":
            dataset["ip"].attrs.update(declared=value, units=7)
        return dataset

    metadata: dict[str, dict[str, Any]] = {}
    arrays = acquisition.mirror_shot(
        source.make_fs(tmp_path), 30421, open_group=attributed_group, metadata_out=metadata
    )
    assert arrays["summary.ip"].tobytes() == np.asarray([-4e5, 4e5, 0.0], dtype=np.float64).tobytes()
    entry = metadata["summary.ip"]
    assert entry["source_attributes"]["declared"] == expected
    assert entry["units"] is None and entry["timebase"] == {"kind": "source_dimension", "dimensions": ["time"]}
    assert entry["source_chunks"] is None
    assert metadata["magnetics.b_field_tor_probe_saddle_field"]["dimensions"] == ["channel", "time_saddle"]


@pytest.mark.parametrize("value", [{1: "numeric", "1": "text"}, {1, 2}, np.complex128(1 + 2j), np.longdouble(1.5)])
def test_mirror_shot_refuses_ambiguous_native_metadata(tmp_path: Path, value: object) -> None:
    """Refuse attribute key collisions and unsupported values on real xarray data."""
    source = _source(tmp_path / "native")

    def attributed_group(fs: Any, shot_id: int, group: str) -> xr.Dataset:
        """Expose an actual decoded group's unsupported caller-declared attribute."""
        dataset: xr.Dataset = source.open_group(fs, shot_id, group)
        if group == "summary":
            dataset["ip"].attrs["declared"] = value
        return dataset

    with pytest.raises(TypeError):
        acquisition.mirror_shot(source.make_fs(tmp_path), 30421, open_group=attributed_group, metadata_out={})


def test_acquire_refuses_actual_native_object_array(tmp_path: Path) -> None:
    """Refuse a decoded NetCDF string/object variable before exporting an NPZ."""
    source = _source(tmp_path / "native")
    xr.Dataset({"ip": ("time", np.asarray(["first", "second", "third"], dtype=object))}).to_netcdf(
        source.root / "summary.nc", engine="scipy"
    )
    dataset = source.open_group(source.make_fs(tmp_path), 30421, "summary")
    assert dataset["ip"].values.dtype.hasobject
    material = tmp_path / "material"
    manifest = acquisition.acquire(
        [30421],
        out_dir=material,
        cache_dir=tmp_path / "cache",
        generated_at="fixed",
        retrieved_at="fixed",
        make_fs=source.make_fs,
        open_group=source.open_group,
        read_generation=source.read_generation,
    )
    assert manifest["status"] == "empty"
    assert manifest["shots"][0]["error"] == "Could not acquire the requested MAST shot."
    assert not list(material.iterdir())
