# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the FAIR-MAST level2 high-fidelity acquisition
"""Offline tests for :mod:`validation.acquire_mast_disruption_shots`."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from validation import acquire_mast_disruption_shots as acq

_ANGLES = np.tile((np.arange(12, dtype=np.float64) * 30.0).reshape(-1, 1), (1, 28))


class _FakeVar:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _FakeDataset:
    def __init__(self, variables: dict[str, np.ndarray]) -> None:
        self.variables = variables

    def __getitem__(self, key: str) -> _FakeVar:
        return _FakeVar(self.variables[key])


def _group_vars(group: str, *, with_saddle: bool = True) -> dict[str, np.ndarray]:
    if group == "summary":
        return {"time": np.linspace(0, 0.06, 60), "ip": np.full(60, 5.0e5), "line_average_n_e": np.full(60, 3.0e19)}
    if group == "equilibrium":
        return {"time": np.linspace(0, 0.06, 20), "q95": np.full(20, 3.8), "beta_tor_normal": np.full(20, 1.5)}
    if group == "interferometer":
        return {"time": np.linspace(0, 0.06, 100), "n_e_line": np.full(100, 3.0e19)}
    if group == "magnetics":
        base: dict[str, np.ndarray] = {"time_saddle": np.linspace(0, 0.06, 500)}
        if with_saddle:
            base["b_field_tor_probe_saddle_field"] = np.ones((12, 500))
            base["b_field_tor_probe_saddle_m_phi"] = _ANGLES
        return base
    return {}


def _open_group(with_saddle: bool = True) -> Any:
    def opener(_fs: Any, _shot_id: int, group: str) -> _FakeDataset:
        return _FakeDataset(_group_vars(group, with_saddle=with_saddle))

    return opener


# --------------------------------------------------------------------------- #
# mirror_shot
# --------------------------------------------------------------------------- #
def test_mirror_shot_collects_present_variables() -> None:
    payload = acq.mirror_shot(object(), 30421, open_group=_open_group())
    assert "magnetics.b_field_tor_probe_saddle_field" in payload
    assert "summary.ip" in payload
    assert payload["equilibrium.q95"].shape == (20,)


def test_mirror_shot_requires_saddle_array() -> None:
    with pytest.raises(ValueError, match="no toroidal saddle array"):
        acq.mirror_shot(object(), 30421, open_group=_open_group(with_saddle=False))


# --------------------------------------------------------------------------- #
# acquire (directory + manifest)
# --------------------------------------------------------------------------- #
def test_acquire_writes_mirrors_and_manifest(tmp_path: Path) -> None:
    manifest = acq.acquire(
        [30421, 30424],
        out_dir=tmp_path / "material",
        cache_dir=tmp_path / "cache",
        generated_at="2026-07-10T00:00:00Z",
        retrieved_at="2026-07-10T00:00:00Z",
        make_fs=lambda _c: object(),
        open_group=_open_group(),
    )
    assert manifest["schema_version"] == "scpn-control.mast-disruption-material.v1"
    assert manifest["n_acquired"] == 2
    assert manifest["synthetic"] is False
    assert manifest["consumers"] == ["SCPN-CONTROL", "SCPN-FUSION-CORE", "MIF-CORE"]
    assert manifest["licence"] == "CC-BY-SA-4.0"
    assert manifest["licence_url"] == "https://creativecommons.org/licenses/by-sa/4.0/"
    assert len(manifest["citations"]) == 2
    assert "10.1109/TPS.2025.3583419" in manifest["citation"]
    assert manifest["source_policy_url"] == "https://mastapp.site/"
    assert manifest["payload_sha256"] == acq._sha256_json({**manifest, "payload_sha256": None})
    record = next(r for r in manifest["shots"] if r["shot_id"] == 30421)
    assert record["saddle_channels"] == 12 and record["saddle_samples"] == 500
    assert len(record["checksum_sha256"]) == 64
    loaded = np.load(tmp_path / "material" / "shot_30421.npz")
    assert "magnetics.b_field_tor_probe_saddle_field" in loaded.files


def test_acquire_records_failed_shot(tmp_path: Path) -> None:
    manifest = acq.acquire(
        [30421],
        out_dir=tmp_path / "material",
        cache_dir=tmp_path / "cache",
        generated_at="2026-07-10T00:00:00Z",
        retrieved_at="2026-07-10T00:00:00Z",
        make_fs=lambda _c: object(),
        open_group=_open_group(with_saddle=False),
    )
    assert manifest["n_acquired"] == 0
    assert manifest["status"] == "empty"
    assert manifest["shots"][0]["status"] == "failed"


# --------------------------------------------------------------------------- #
# make_filesystem / _open_group (S3 seam via stubbed modules)
# --------------------------------------------------------------------------- #
def test_make_filesystem_uses_fsspec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    fake_fsspec = types.ModuleType("fsspec")
    fake_fsspec.filesystem = lambda protocol, **kw: calls.update({"protocol": protocol, **kw}) or "FS"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fsspec", fake_fsspec)
    fs = acq.make_filesystem(tmp_path / "cache")
    assert fs == "FS"
    assert calls["protocol"] == "simplecache"
    assert calls["target_options"]["anon"] is True


def test_open_group_uses_xarray(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def _open_zarr(store: Any, group: str, consolidated: bool) -> str:
        seen.update({"store": store, "group": group, "consolidated": consolidated})
        return "DS"

    fake_xr = types.ModuleType("xarray")
    fake_xr.open_zarr = _open_zarr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xarray", fake_xr)

    class _FS:
        def get_mapper(self, url: str) -> str:
            seen["url"] = url
            return "MAP"

    assert acq._open_group(_FS(), 30421, "magnetics") == "DS"
    assert seen["group"] == "magnetics" and seen["consolidated"] is True
    assert seen["url"].endswith("level2/shots/30421.zarr")


# --------------------------------------------------------------------------- #
# main (default seams resolved dynamically for monkeypatching)
# --------------------------------------------------------------------------- #
def test_main_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acq, "make_filesystem", lambda _c: object())
    monkeypatch.setattr(acq, "_open_group", lambda _fs, _s, g: _FakeDataset(_group_vars(g)))
    manifest_out = tmp_path / "manifest.json"
    code = acq.main(
        [
            "--shots",
            "30419-30420 30424",
            "--out-dir",
            str(tmp_path / "material"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--manifest-out",
            str(manifest_out),
            "--generated-at",
            "2026-07-10T00:00:00Z",
            "--retrieved-at",
            "2026-07-10T00:00:00Z",
        ]
    )
    assert code == 0
    assert json.loads(manifest_out.read_text(encoding="utf-8"))["n_acquired"] == 3


def test_parse_shots_ranges_and_ids() -> None:
    assert acq._parse_shots("11766 30419-30421, 29876") == [11766, 30419, 30420, 30421, 29876]
