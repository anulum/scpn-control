# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Original MAST Zarr byte-custody tests

"""Verify real store capture, scalar shot status and malformed observation custody."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from mast_efm_zarr_fixtures import sample_dataset, write_zarr

from validation.convert_mast_efm_neural_equilibrium_reference import read_reference_zarr
from validation.mast_efm_zarr_store import (
    CapturedZarrStore,
    capture_zarr_store,
    consolidated_metadata,
    read_reference_store,
)


@pytest.mark.parametrize(
    ("times", "message"),
    [([0.1, 0.1, 0.3], "strictly monotonic"), ([0.3, 0.2, 0.1], "strictly increasing")],
)
def test_real_source_clock_refuses_duplicate_observation_times(
    tmp_path: Path, times: list[float], message: str
) -> None:
    """A real store with a duplicate or descending clock cannot produce ordered observations."""
    ds = sample_dataset().assign_coords(time=times)
    store = write_zarr(tmp_path / "invalid-clock.zarr", ds)
    with pytest.raises(ValueError, match=message):
        read_reference_zarr(zarr_path=store, shot_id=30419)


def test_real_store_refuses_nonarray_flux_dimension_metadata(tmp_path: Path) -> None:
    """Malformed consolidated dimension labels cannot be interpreted as physical flux axes."""
    store = write_zarr(tmp_path / "invalid-dims.zarr")
    path = store / ".zmetadata"
    metadata = json.loads(path.read_text())
    metadata["metadata"]["psirz/.zattrs"]["_ARRAY_DIMENSIONS"] = None
    path.write_text(json.dumps(metadata))
    with pytest.raises(TypeError, match="iterable"):
        read_reference_zarr(zarr_path=store, shot_id=30419)


def test_captured_store_hashes_and_decodes_the_same_private_bytes(tmp_path: Path) -> None:
    """A later source rewrite cannot change already captured observations or their manifest."""
    store = write_zarr(tmp_path / "original.zarr")
    snapshot = capture_zarr_store(store)
    files = dict(snapshot.files)
    copied = CapturedZarrStore(files)
    original_manifest = snapshot.manifest()
    files[".zmetadata"] = b"invalid later caller mutation"
    ds = sample_dataset()
    ds["pprime"].values[:] += 0.125
    write_zarr(store, ds)
    np.testing.assert_array_equal(read_reference_store(copied, shot_id=30419)["pprime_Pa_per_Wb_rad"], [[1, 1], [1, 1]])
    np.testing.assert_array_equal(
        read_reference_zarr(zarr_path=store, shot_id=30419)["pprime_Pa_per_Wb_rad"], [[1.125, 1.125], [1.125, 1.125]]
    )
    assert copied.manifest() == snapshot.manifest() == original_manifest
    assert capture_zarr_store(store).manifest()["snapshot_sha256"] != original_manifest["snapshot_sha256"]


@pytest.mark.parametrize(
    "payload",
    [
        b"[]",
        b'{"zarr_consolidated_format":true,"metadata":{}}',
        b'{"zarr_consolidated_format":1.0,"metadata":{}}',
        b'{"zarr_consolidated_format":2,"metadata":{}}',
        b'{"zarr_consolidated_format":1,"metadata":[]}',
        b'{"zarr_consolidated_format":1,"metadata":{".zgroup":{"zarr_format":2.0}}}',
        b'{"zarr_consolidated_format":1,"metadata":{".zgroup":{"zarr_format":true}}}',
        b'{"zarr_consolidated_format":1,"metadata":{".zgroup":{"zarr_format":3}}}',
        b'{"zarr_consolidated_format":1,"zarr_consolidated_format":1,"metadata":{}}',
        b'{"zarr_consolidated_format":1,"metadata":{".zgroup":{"zarr_format":2},"attr":NaN}}',
        b'{"zarr_consolidated_format":1,"metadata":{".zgroup":{"zarr_format":2},"attr":1e999}}',
        b"invalid JSON",
        bytes([255]),
    ],
)
def test_consolidated_source_metadata_refuses_ambiguous_or_nonfinite_contracts(payload: bytes) -> None:
    """Malformed encodings, duplicate keys and numeric format aliases never become valid source metadata."""
    with pytest.raises(ValueError):
        consolidated_metadata(payload)


@pytest.mark.parametrize("kind", ["symlink_file", "symlink_directory", "fifo"])
def test_source_capture_refuses_nonregular_or_linked_entries(tmp_path: Path, kind: str) -> None:
    """Source byte capture must not follow linked payloads or block on a nonregular FIFO."""
    store = write_zarr(tmp_path / "original.zarr")
    if kind == "symlink_file":
        target = tmp_path / "outside"
        target.write_bytes(b"external bytes")
        (store / "linked").symlink_to(target)
    elif kind == "symlink_directory":
        target = tmp_path / "outside"
        target.mkdir()
        (store / "linked").symlink_to(target, target_is_directory=True)
    else:
        os.mkfifo(store / "pipe")
    with pytest.raises(ValueError, match="symbolic links|regular files"):
        capture_zarr_store(store)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("declaration", []),
        ("shape", None),
        ("chunks", None),
        ("shape", [3, 1]),
        ("shape", [-1]),
        ("shape", [True]),
        ("chunks", [0]),
        ("chunks", [True]),
        ("dimension_separator", ":"),
    ],
)
def test_required_chunk_declarations_refuse_malformed_shapes(tmp_path: Path, field: str, value: Any) -> None:
    """Actual public decoding rejects malformed observation metadata before admitting fill values."""
    store = write_zarr(tmp_path / "original.zarr")
    path = store / ".zmetadata"
    declaration = json.loads(path.read_text())
    if field == "declaration":
        declaration["metadata"]["status/.zarray"] = value
    else:
        declaration["metadata"]["status/.zarray"][field] = value
    path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="required Zarr"):
        read_reference_zarr(zarr_path=store, shot_id=30419)


@pytest.mark.parametrize("name", ["status", "ffprime", "pprime", "profile_r", "time"])
def test_required_observations_cannot_be_supplied_by_missing_chunk_fill(tmp_path: Path, name: str) -> None:
    """Deleting a physical observation chunk cannot grant readiness from its metadata/fill descriptor."""
    store = write_zarr(tmp_path / "original.zarr")
    chunk = next(p for p in (store / name).iterdir() if not p.name.startswith("."))
    chunk.unlink()
    with pytest.raises(ValueError, match="required Zarr dimension chunk is missing"):
        read_reference_zarr(zarr_path=store, shot_id=30419)


@pytest.mark.parametrize("status", [0.0, 1.0])
def test_observed_scalar_scheduler_status_keeps_exact_convergence_times(tmp_path: Path, status: float) -> None:
    """Real MAST scalar scheduler status applies as a shot guard without manufacturing converged rows."""
    ds = sample_dataset()
    ds["status"] = xr.DataArray(status)
    ds["cnvrgd_times"].values[:] = [1, 0, 1]
    store = write_zarr(tmp_path / "original.zarr", ds)
    arrays = read_reference_zarr(zarr_path=store, shot_id=30419)
    np.testing.assert_array_equal(arrays["time_s"], [0.1, 0.3])
    (store / "status/0").unlink()
    with pytest.raises(ValueError, match="status/0"):
        read_reference_zarr(zarr_path=store, shot_id=30419)


@pytest.mark.parametrize("status", [-1.0, 2.0, float("nan")])
def test_failed_scalar_scheduler_status_never_admits_time_rows(tmp_path: Path, status: float) -> None:
    """Observed unsuccessful or nonfinite whole-shot status refuses even with positive convergence times."""
    ds = sample_dataset()
    ds["status"] = xr.DataArray(status)
    with pytest.raises(ValueError, match="no converged time slices"):
        read_reference_zarr(zarr_path=write_zarr(tmp_path / "original.zarr", ds), shot_id=30419)


def test_observed_pretrigger_times_are_preserved_without_admitting_unconverged_rows(tmp_path: Path) -> None:
    """Acquired negative pretrigger times remain observed; only actual positive-convergence rows are selected."""
    ds = sample_dataset().assign_coords(time=[-0.05, 0.0, 0.1])
    ds["status"] = xr.DataArray(1.0)
    ds["cnvrgd_times"] = xr.DataArray([-0.05, 0.0, 0.1], dims=("time",))
    arrays = read_reference_zarr(zarr_path=write_zarr(tmp_path / "original.zarr", ds), shot_id=30419)
    np.testing.assert_array_equal(arrays["time_s"], [0.1])
    assert arrays["shot_id"].tolist() == [30419]


@pytest.mark.parametrize("magnitude", [1e-300, 1e154, 1e308])
def test_extreme_profile_rms_remains_positive_and_finite_on_actual_stores(tmp_path: Path, magnitude: float) -> None:
    """Tiny squares and overflowing sums retain the exact constant-profile RMS through scaled arithmetic."""
    ds = sample_dataset()
    ds["ffprime"].values[:] = magnitude
    arrays = read_reference_zarr(zarr_path=write_zarr(tmp_path / "original.zarr", ds), shot_id=30419)
    np.testing.assert_array_equal(arrays["ffprime_rms_T_rad"], [magnitude, magnitude])
