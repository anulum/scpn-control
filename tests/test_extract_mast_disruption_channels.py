# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the FAIR-MAST disruption channel extractor
"""Offline tests for :mod:`validation.extract_mast_disruption_channels`."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from validation.build_mast_disruption_dataset import MEASURED_CHANNELS, _load_shots, build_dataset
from validation.extract_mast_disruption_channels import (
    ToroidalFieldConfig,
    build_channels_npz,
    convert_shot_zarr,
    derive_channels,
    main,
    read_shot_signals,
)

_ANGLES = 2.0 * np.pi * np.arange(12, dtype=np.float64) / 12.0
_N = 40
_TF = ToroidalFieldConfig(n_turns=100, r_geo_m=0.7)
_FIXED_TS = "2026-07-10T00:00:00+00:00"


class _FakeArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _FakeDataset:
    """Minimal xarray-like store for the extractor tests."""

    def __init__(self, variables: dict[str, np.ndarray]) -> None:
        self.variables = variables
        self.closed = False

    def __getitem__(self, key: str) -> _FakeArray:
        return _FakeArray(self.variables[key])

    def close(self) -> None:
        self.closed = True


def _store(**overrides: np.ndarray) -> dict[str, np.ndarray]:
    saddle = np.outer(np.ones(_N), np.cos(_ANGLES))  # pure locked n=1
    store: dict[str, np.ndarray] = {
        "time": np.linspace(0.0, 0.04, _N),
        "ip": np.full(_N, 5.0e5),
        "line_average_n_e": np.full(_N, 3.0e19),
        "beta_normal": np.full(_N, 1.5),
        "q": np.tile(np.array([1.0, 1.5, 2.0, 3.0, 4.0]), (_N, 1)),
        "psi_norm": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        "saddle": saddle,
        "saddle_angle": _ANGLES,
        "poloidal_probe": np.linspace(0.0, 0.1, _N),
        "magnetic_axis_z": np.zeros(_N),
        "tf_current": np.full(_N, 1.0e6),
    }
    store.update(overrides)
    return store


# --------------------------------------------------------------------------- #
# read_shot_signals
# --------------------------------------------------------------------------- #
def test_read_shot_signals_resolves_bt_from_tf_current() -> None:
    raw = read_shot_signals(_FakeDataset(_store()), tf_config=_TF)
    assert raw["time_s"].shape == (_N,)
    assert np.all(raw["bt_t_tesla"] > 0.0)


def test_read_shot_signals_prefers_direct_toroidal_field() -> None:
    store = _store(toroidal_field=np.full(_N, 0.585))
    raw = read_shot_signals(_FakeDataset(store), tf_config=_TF)
    assert np.allclose(raw["bt_t_tesla"], 0.585)


def test_read_shot_signals_rejects_missing_variables() -> None:
    store = _store()
    del store["ip"]
    with pytest.raises(ValueError, match="missing required variables"):
        read_shot_signals(_FakeDataset(store), tf_config=_TF)


def test_read_shot_signals_blocks_when_bt_unresolved() -> None:
    store = _store()
    del store["tf_current"]
    with pytest.raises(RuntimeError, match="lookup_needed"):
        read_shot_signals(_FakeDataset(store), tf_config=_TF)


# --------------------------------------------------------------------------- #
# derive_channels
# --------------------------------------------------------------------------- #
def test_derive_channels_produces_measured_schema() -> None:
    raw = read_shot_signals(_FakeDataset(_store()), tf_config=_TF)
    channels = derive_channels(raw, locked_window=12)
    assert tuple(channels) == MEASURED_CHANNELS
    assert all(channels[name].shape == (_N,) for name in MEASURED_CHANNELS)
    assert np.allclose(channels["Ip_MA"], 0.5)
    assert np.allclose(channels["q95"], 3.8)  # interp of [1,1.5,2,3,4] at psi_n=0.95


def test_derive_channels_rejects_shape_mismatch() -> None:
    raw = read_shot_signals(_FakeDataset(_store()), tf_config=_TF)
    raw["q_profile"] = np.array([[1.0, 1.5, 2.0, 3.0, 4.0]])  # single-sample profile
    with pytest.raises(ValueError, match="must be 1-D with"):
        derive_channels(raw, locked_window=12)


def test_derive_channels_rejects_non_finite() -> None:
    raw = read_shot_signals(_FakeDataset(_store()), tf_config=_TF)
    raw["poloidal_probe_tesla"] = raw["poloidal_probe_tesla"].copy()
    raw["poloidal_probe_tesla"][3] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        derive_channels(raw, locked_window=12)


# --------------------------------------------------------------------------- #
# convert_shot_zarr
# --------------------------------------------------------------------------- #
def test_convert_shot_zarr_uses_injected_opener_and_closes() -> None:
    ds = _FakeDataset(_store())
    result = convert_shot_zarr(11766, Path("unused.zarr"), tf_config=_TF, locked_window=12, open_dataset=lambda _p: ds)
    assert result["shot_id"] == 11766
    assert result["n_samples"] == _N
    assert tuple(result["channels"]) == MEASURED_CHANNELS
    assert ds.closed is True


# --------------------------------------------------------------------------- #
# build_channels_npz — pipeline + Stage-2 round trip
# --------------------------------------------------------------------------- #
def _write_manifest(path: Path) -> None:
    manifest = {
        "shots": [
            {"shot_id": 11767, "local_path": "b.zarr", "status": "planned"},
            {"shot_id": 11766, "local_path": "a.zarr", "status": "acquired"},
        ]
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_build_channels_npz_extracts_acquired_only(tmp_path: Path) -> None:
    manifest_path = tmp_path / "campaign.json"
    _write_manifest(manifest_path)
    report = build_channels_npz(
        manifest_path,
        dataset_root=tmp_path,
        out_dir=tmp_path / "out",
        tf_config=_TF,
        locked_window=12,
        generated_at=_FIXED_TS,
        open_dataset=lambda _p: _FakeDataset(_store()),
    )
    assert report["status"] == "blocked"
    assert report["admission_ready"] is False
    assert report["n_shots_extracted"] == 1
    statuses = {rec["shot_id"]: rec["status"] for rec in report["shots"]}
    assert statuses == {11766: "extracted", 11767: "skipped"}

    npz_path = tmp_path / "out" / "channels.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        assert list(data["shot_ids"]) == [11766]
        assert set(f"11766:{name}" for name in MEASURED_CHANNELS).issubset(set(data.files))

    # Round trip: the emitted channels.npz feeds the Stage-2 dataset builder directly.
    shots = _load_shots(npz_path)
    dataset = build_dataset(
        shots,
        dataset_id="mast-disruption-roundtrip",
        out_dir=tmp_path / "ds",
        retrieved_at=_FIXED_TS,
        generated_at=_FIXED_TS,
    )
    assert dataset["n_shots"] == 1
    assert dataset["status"] == "blocked"


def test_build_channels_npz_rejects_empty_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "empty.json"
    manifest_path.write_text(json.dumps({"shots": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty shots list"):
        build_channels_npz(
            manifest_path,
            dataset_root=tmp_path,
            out_dir=tmp_path / "out",
            tf_config=_TF,
            locked_window=12,
            generated_at=_FIXED_TS,
            open_dataset=lambda _p: _FakeDataset(_store()),
        )


# --------------------------------------------------------------------------- #
# main (default opener resolved dynamically for monkeypatching)
# --------------------------------------------------------------------------- #
def test_main_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "campaign.json"
    _write_manifest(manifest_path)
    monkeypatch.setattr(
        "validation.extract_mast_disruption_channels._default_open_dataset",
        lambda _p: _FakeDataset(_store()),
    )
    json_out = tmp_path / "report.json"
    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--dataset-root",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(json_out),
            "--n-turns",
            "100",
            "--r-geo-m",
            "0.7",
            "--locked-window",
            "12",
            "--generated-at",
            _FIXED_TS,
        ]
    )
    assert exit_code == 0
    report = json.loads(json_out.read_text(encoding="utf-8"))
    assert report["n_shots_extracted"] == 1
    assert report["status"] == "blocked"


# --------------------------------------------------------------------------- #
# _default_open_dataset — xarray binding via a stubbed module
# --------------------------------------------------------------------------- #
def test_default_open_dataset_uses_xarray(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from validation import extract_mast_disruption_channels as extractor

    opened: dict[str, Any] = {}

    def _open_zarr(path: Path, consolidated: bool) -> _FakeDataset:
        opened["path"] = path
        opened["consolidated"] = consolidated
        return _FakeDataset(_store())

    fake_xarray = types.ModuleType("xarray")
    fake_xarray.open_zarr = _open_zarr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xarray", fake_xarray)
    zarr_dir = tmp_path / "shot.zarr"
    zarr_dir.mkdir()
    ds = extractor._default_open_dataset(zarr_dir)
    assert isinstance(ds, _FakeDataset)
    assert opened == {"path": zarr_dir, "consolidated": True}


def test_default_open_dataset_missing_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from validation import extract_mast_disruption_channels as extractor

    fake_xarray = types.ModuleType("xarray")
    fake_xarray.open_zarr = lambda path, consolidated: _FakeDataset(_store())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xarray", fake_xarray)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        extractor._default_open_dataset(tmp_path / "absent.zarr")
