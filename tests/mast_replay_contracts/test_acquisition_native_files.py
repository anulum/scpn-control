# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST acquisition through native local file adapters
"""Exercise real xarray/file/NPZ transport with manufactured, unauthenticated data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pytest
import xarray as xr

from validation import acquire_mast_disruption_shots as acquisition
from validation.mast_source_object_manifest import validate_source_object_manifest


@dataclass
class _NativeFileSource:
    """Load real NetCDF groups through the acquisition's caller-adapter protocol."""

    root: Path

    def make_fs(self, namespace: Path) -> Any:
        """Create an actual local fsspec filesystem after cache reservation."""
        assert namespace.is_dir()
        return fsspec.filesystem("file")

    def open_group(self, fs: Any, shot_id: int, group: str) -> Any:
        """Decode a real NetCDF file and close its descriptor after materialization."""
        assert shot_id == 30421
        with fs.open(str(self.root / (group + ".nc")), "rb") as handle:
            return xr.load_dataset(handle, engine="scipy")

    def read_generation(self, shot_id: int) -> acquisition.SourceGenerationPin:
        """Bind actual local declaration bytes; no remote origin is authenticated."""
        data = (self.root / "root.json").read_bytes()
        return acquisition.SourceGenerationPin(
            f"s3://mast/level2/shots/{shot_id}.zarr", hashlib.sha256(data).hexdigest(), len(data), None, None
        )


def _source(root: Path) -> _NativeFileSource:
    """Write small native files for the public acquisition protocol."""
    root.mkdir()
    clock = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    groups = {
        "summary": xr.Dataset(
            {"time": ("time", clock), "ip": ("time", np.asarray([-4e5, 4e5, 0.0], dtype=np.float64))}
        ),
        "equilibrium": xr.Dataset(
            {"time": ("time", clock), "q95": ("time", np.asarray([3.0, 3.1, 3.2], dtype=np.float64))}
        ),
        "interferometer": xr.Dataset({"time": ("time", clock)}),
        "magnetics": xr.Dataset(
            {
                "time_saddle": ("time_saddle", clock),
                "b_field_tor_probe_saddle_field": (
                    ("channel", "time_saddle"),
                    np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
                ),
            }
        ),
    }
    groups["summary"]["ip"].attrs["units"] = "A"
    groups["magnetics"]["b_field_tor_probe_saddle_field"].attrs["units"] = "T"
    for group, dataset in groups.items():
        dataset.to_netcdf(root / (group + ".nc"), engine="scipy")
    (root / "root.json").write_text(
        json.dumps({"zarr_format": 3, "consolidated_metadata": {"kind": "inline", "metadata": {}}})
    )
    return _NativeFileSource(root)


def test_acquire_reads_native_files_and_checks_exact_export_bytes(tmp_path: Path) -> None:
    """Transport real local arrays into checked NPZ and source-object declarations."""
    source = _source(tmp_path / "native")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.root.iterdir()}
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
    assert manifest["status"] == "complete" and manifest["n_acquired"] == 1
    validate_source_object_manifest(manifest, artifact_root=material)
    artifact = manifest["shots"][0]["artifacts"][0]
    assert artifact["sha256"] == hashlib.sha256((material / "shot_30421.npz").read_bytes()).hexdigest()
    with np.load(material / "shot_30421.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["summary.ip"], [-4e5, 4e5, 0.0])
        np.testing.assert_array_equal(
            arrays["magnetics.b_field_tor_probe_saddle_field"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
    assert manifest["shots"][0]["summary"]["ip_max_ka"] == 400.0
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.root.iterdir()}


def test_acquire_refuses_actual_generation_file_drift_before_archive(tmp_path: Path) -> None:
    """Observe a real source-declaration change during native file collection."""
    source = _source(tmp_path / "native")

    def changing_group(fs: Any, shot_id: int, group: str) -> Any:
        """Read the native group while changing actual last-group declaration bytes."""
        dataset = source.open_group(fs, shot_id, group)
        if group == "magnetics":
            with (source.root / "root.json").open("a") as handle:
                handle.write(" ")
        return dataset

    material = tmp_path / "material"
    manifest = acquisition.acquire(
        [30421],
        out_dir=material,
        cache_dir=tmp_path / "cache",
        generated_at="fixed",
        retrieved_at="fixed",
        make_fs=source.make_fs,
        open_group=changing_group,
        read_generation=source.read_generation,
    )
    assert manifest["status"] == "empty" and manifest["n_acquired"] == 0
    assert "root metadata changed" in manifest["shots"][0]["error"]
    assert not list(material.iterdir())


def test_acquire_records_fixed_error_for_actual_missing_native_file(tmp_path: Path) -> None:
    """Keep native filesystem exception text and source paths out of failure records."""
    source = _source(tmp_path / "native")
    (source.root / "equilibrium.nc").unlink()
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
    assert manifest["status"] == "empty" and manifest["n_acquired"] == 0
    assert manifest["shots"][0]["error"] == "Could not acquire the requested MAST shot."
    assert str(source.root) not in json.dumps(manifest)
    assert not list(material.iterdir())


def test_acquire_refuses_reusing_actual_reserved_cache(tmp_path: Path) -> None:
    """Retain an earlier checked archive when the same cache declaration repeats."""
    source = _source(tmp_path / "native")
    material, cache = tmp_path / "material", tmp_path / "cache"
    acquire_selected = partial(
        acquisition.acquire,
        [30421],
        out_dir=material,
        cache_dir=cache,
        generated_at="fixed",
        retrieved_at="fixed",
        make_fs=source.make_fs,
        open_group=source.open_group,
        read_generation=source.read_generation,
    )
    first = acquire_selected()
    assert first["status"] == "complete"
    before = hashlib.sha256((material / "shot_30421.npz").read_bytes()).hexdigest()
    second = acquire_selected()
    assert second["status"] == "empty" and "refusing cross-run cache reuse" in second["shots"][0]["error"]
    assert hashlib.sha256((material / "shot_30421.npz").read_bytes()).hexdigest() == before
    assert len(list((cache / "runs").iterdir())) == 1


def test_acquire_refuses_a_different_declared_shot_before_cache(tmp_path: Path) -> None:
    """Bind the requested shot before reading any native group or reserving cache."""
    source = _source(tmp_path / "native")

    def different_shot(shot_id: int) -> acquisition.SourceGenerationPin:
        """Declare the actual captured local root bytes for a different identity."""
        return source.read_generation(shot_id + 1)

    material, cache = tmp_path / "material", tmp_path / "cache"
    manifest = acquisition.acquire(
        [30421],
        out_dir=material,
        cache_dir=cache,
        generated_at="fixed",
        retrieved_at="fixed",
        make_fs=source.make_fs,
        open_group=source.open_group,
        read_generation=different_shot,
    )
    assert manifest["status"] == "empty"
    assert manifest["shots"][0]["error"] == "source generation does not identify the requested shot"
    assert not cache.exists() and not list(material.iterdir())


@pytest.mark.parametrize("shape", [None, (3,), (0, 3), (2, 0)])
def test_acquire_refuses_invalid_actual_native_saddle_shape(tmp_path: Path, shape: tuple[int, ...] | None) -> None:
    """Refuse missing, one-dimensional or empty real NetCDF saddle arrays."""
    source = _source(tmp_path / "native")
    dataset = xr.Dataset()
    if shape is not None:
        dimensions = tuple(f"axis_{index}" for index in range(len(shape)))
        dataset["b_field_tor_probe_saddle_field"] = xr.DataArray(np.zeros(shape), dims=dimensions)
    dataset.to_netcdf(source.root / "magnetics.nc", engine="scipy")
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


@pytest.mark.parametrize("values", [None, [], [float("nan"), float("nan")], [float("inf"), 1.0]])
def test_acquire_preserves_native_ip_with_no_finite_summary(tmp_path: Path, values: list[float] | None) -> None:
    """Keep absent/empty/nonfinite samples while emitting a finite JSON summary."""
    source = _source(tmp_path / "native")
    dataset = xr.Dataset()
    if values is not None:
        dataset["ip"] = xr.DataArray(np.asarray(values, dtype=np.float64), dims=("time",))
    dataset.to_netcdf(source.root / "summary.nc", engine="scipy")
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
    assert manifest["status"] == "complete" and manifest["shots"][0]["summary"]["ip_max_ka"] is None
    with np.load(material / "shot_30421.npz", allow_pickle=False) as arrays:
        if values is None:
            assert "summary.ip" not in arrays.files
        else:
            assert arrays["summary.ip"].tobytes() == np.asarray(values, dtype=np.float64).tobytes()
