# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for the disruption replay-channel derivation
"""Offline tests for :mod:`validation.build_disruption_replay_channels`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation.build_disruption_replay_channels import (
    _MEASURED,
    _interp,
    _peak_to_grid,
    build_channels,
    derive_replay_channels,
    main,
)
from validation.build_mast_disruption_dataset import MEASURED_CHANNELS, _load_shots, build_dataset

_FIXED_TS = "2026-07-10T00:00:00+00:00"
_ANGLES_DEG = np.arange(12, dtype=np.float64) * 30.0  # 12-channel toroidal saddle, 30 deg apart


def _mirror(n_grid: int = 60, n_eq: int = 20, n_fast: int = 4000, *, z_len: int | None = None) -> dict[str, np.ndarray]:
    grid = np.linspace(0.0, 0.06, n_grid)
    t_eq = np.linspace(0.0, 0.06, n_eq)
    t_fast = np.linspace(0.0, 0.06, n_fast)
    ip = np.concatenate([np.linspace(0.0, 6.0e5, n_grid // 3), np.full(n_grid - n_grid // 3, 6.0e5)])
    saddle = np.outer(np.cos(np.deg2rad(_ANGLES_DEG)), 0.01 * np.sin(2.0 * np.pi * 50.0 * t_fast))  # (12, n_fast)
    phi = np.tile(np.deg2rad(_ANGLES_DEG).reshape(-1, 1), (1, 28))
    z_axis = np.zeros(z_len if z_len is not None else n_eq)
    return {
        "summary.time": grid,
        "summary.ip": ip,
        "summary.line_average_n_e": np.full(n_grid, 3.0e19),
        "equilibrium.time": t_eq,
        "equilibrium.q95": np.full(n_eq, 3.8),
        "equilibrium.beta_tor_normal": np.full(n_eq, 1.5),
        "equilibrium.bvac_rmag": np.full(n_eq, 0.58),
        "equilibrium.z": z_axis,
        "magnetics.time_saddle": t_fast,
        "magnetics.time_mirnov": t_fast,
        "magnetics.b_field_tor_probe_saddle_field": saddle,
        "magnetics.b_field_tor_probe_saddle_m_phi": phi,
        "magnetics.b_field_pol_probe_cc_field": np.tile(0.02 * t_fast, (5, 1)),
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_interp_aligned() -> None:
    grid = np.linspace(0.0, 1.0, 10)
    out = _interp(np.array([0.0, 2.0]), np.array([0.0, 1.0]), grid)
    assert np.allclose(out, 2.0 * grid)


def test_interp_length_mismatch_uses_uniform_window() -> None:
    # value on an unknown timebase (length != src) -> uniform over the grid window
    grid = np.linspace(0.0, 1.0, 5)
    out = _interp(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]), grid)
    assert out.shape == (5,)
    assert np.isclose(out[0], 0.0) and np.isclose(out[-1], 2.0)


def test_interp_rejects_all_non_finite() -> None:
    with pytest.raises(ValueError, match="no finite samples"):
        _interp(np.array([np.nan, np.nan]), np.array([0.0, 1.0]), np.linspace(0, 1, 3))


def test_peak_to_grid_preserves_burst() -> None:
    fast_t = np.linspace(0.0, 1.0, 1000)
    fast = np.zeros_like(fast_t)
    fast[500] = 9.0  # a single burst
    grid = np.linspace(0.0, 1.0, 10)
    out = _peak_to_grid(fast, fast_t, grid)
    assert np.isclose(out.max(), 9.0)  # the burst survives the reduction


def test_peak_to_grid_empty_bin_falls_back() -> None:
    # a grid finer than the fast series forces empty bins -> interpolation fallback
    out = _peak_to_grid(np.array([1.0, 3.0]), np.array([0.0, 1.0]), np.linspace(0.0, 1.0, 20))
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# derive_replay_channels
# --------------------------------------------------------------------------- #
def test_derive_replay_channels_full_schema() -> None:
    channels = derive_replay_channels(_mirror())
    assert tuple(channels) == _MEASURED == MEASURED_CHANNELS
    assert all(channels[name].shape == (60,) for name in _MEASURED)
    assert np.all(np.isfinite(channels["n1_amp"]))
    assert np.allclose(channels["q95"], 3.8)
    assert np.allclose(channels["Ip_MA"].max(), 0.6, atol=1e-6)


def test_derive_replay_channels_handles_z_timebase_mismatch() -> None:
    # magnetic-axis Z on a coarser 8-sample timebase must not crash the shot
    channels = derive_replay_channels(_mirror(z_len=8))
    assert np.all(np.isfinite(channels["vertical_position_m"]))


# --------------------------------------------------------------------------- #
# build_channels (directory) + Stage-2 round trip
# --------------------------------------------------------------------------- #
def test_build_channels_and_stage2_round_trip(tmp_path: Path) -> None:
    material = tmp_path / "material"
    material.mkdir()
    for shot_id in (11766, 11767):
        np.savez_compressed(material / f"shot_{shot_id}.npz", **_mirror())
    out_dir = tmp_path / "out"
    report = build_channels(material, out_dir=out_dir, generated_at=_FIXED_TS)
    assert report["n_derived"] == 2
    assert report["synthetic"] is False

    shots = _load_shots(out_dir / "channels.npz")
    dataset = build_dataset(
        shots, dataset_id="rt", out_dir=tmp_path / "ds", retrieved_at=_FIXED_TS, generated_at=_FIXED_TS
    )
    assert dataset["n_shots"] == 2
    assert dataset["status"] == "blocked"


def test_build_channels_records_malformed_mirror(tmp_path: Path) -> None:
    material = tmp_path / "material"
    material.mkdir()
    np.savez_compressed(material / "shot_999.npz", **{"summary.time": np.linspace(0, 1, 5)})  # missing vars
    report = build_channels(material, out_dir=tmp_path / "out", generated_at=_FIXED_TS)
    assert report["n_derived"] == 0
    assert report["shots"][0]["status"] == "failed"


def test_main_writes_report(tmp_path: Path) -> None:
    material = tmp_path / "material"
    material.mkdir()
    np.savez_compressed(material / "shot_11766.npz", **_mirror())
    json_out = tmp_path / "report.json"
    code = main(
        [
            "--material-dir",
            str(material),
            "--out-dir",
            str(tmp_path / "out"),
            "--json-out",
            str(json_out),
            "--generated-at",
            _FIXED_TS,
        ]
    )
    assert code == 0
    assert json.loads(json_out.read_text(encoding="utf-8"))["n_derived"] == 1
