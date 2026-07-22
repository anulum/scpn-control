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
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import validation.build_disruption_replay_channels as replay_builder
from validation.build_disruption_replay_channels import (
    _MEASURED,
    REPORT_SCHEMA,
    _interp,
    _peak_to_grid,
    build_channels,
    derive_replay_channels,
    inspect_replay_archive,
    inspect_replay_archive_bytes,
    main,
)
from validation.build_mast_disruption_dataset import MEASURED_CHANNELS, _load_shots, build_dataset

_FIXED_TS = "2026-07-10T00:00:00+00:00"
_ANGLES_DEG = np.arange(12, dtype=np.float64) * 30.0  # 12-channel toroidal saddle, 30 deg apart


def _mirror(
    n_grid: int = 60,
    n_eq: int = 20,
    n_fast: int = 4000,
    *,
    z_len: int | None = None,
) -> dict[str, NDArray[Any]]:
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
    """Interpolate a finite aligned signal onto the requested grid."""
    grid = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    out = _interp(np.asarray([0.0, 2.0], dtype=np.float64), np.asarray([0.0, 1.0], dtype=np.float64), grid)
    assert np.allclose(out, 2.0 * grid)


def test_interp_length_mismatch_uses_uniform_window() -> None:
    """Use a uniform source window when value and time lengths differ."""
    # value on an unknown timebase (length != src) -> uniform over the grid window
    grid = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    out = _interp(
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        grid,
    )
    assert out.shape == (5,)
    assert np.isclose(out[0], 0.0) and np.isclose(out[-1], 2.0)


def test_interp_rejects_all_non_finite() -> None:
    """Reject a source signal with no finite samples."""
    with pytest.raises(ValueError, match="no finite samples"):
        _interp(
            np.asarray([np.nan, np.nan], dtype=np.float64),
            np.asarray([0.0, 1.0], dtype=np.float64),
            np.linspace(0.0, 1.0, 3, dtype=np.float64),
        )


def test_peak_to_grid_preserves_burst() -> None:
    """Preserve a short warning burst through peak reduction."""
    fast_t = np.linspace(0.0, 1.0, 1000, dtype=np.float64)
    fast = np.zeros_like(fast_t)
    fast[500] = 9.0  # a single burst
    grid = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    out = _peak_to_grid(fast, fast_t, grid)
    assert np.isclose(out.max(), 9.0)  # the burst survives the reduction


def test_peak_to_grid_empty_bin_falls_back() -> None:
    """Fill empty peak bins through finite interpolation."""
    # a grid finer than the fast series forces empty bins -> interpolation fallback
    out = _peak_to_grid(
        np.asarray([1.0, 3.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.linspace(0.0, 1.0, 20, dtype=np.float64),
    )
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# derive_replay_channels
# --------------------------------------------------------------------------- #
def test_derive_replay_channels_full_schema() -> None:
    """Derive the complete finite eleven-channel replay schema."""
    channels = derive_replay_channels(_mirror())
    assert tuple(channels) == _MEASURED == MEASURED_CHANNELS
    assert all(channels[name].shape == (60,) for name in _MEASURED)
    assert np.all(np.isfinite(channels["n1_amp"]))
    assert np.allclose(channels["q95"], 3.8)
    assert np.allclose(channels["Ip_MA"].max(), 0.6, atol=1e-6)


def test_derive_replay_channels_handles_z_timebase_mismatch() -> None:
    """Handle a magnetic-axis signal with a distinct sample count."""
    # magnetic-axis Z on a coarser 8-sample timebase must not crash the shot
    channels = derive_replay_channels(_mirror(z_len=8))
    assert np.all(np.isfinite(channels["vertical_position_m"]))


# --------------------------------------------------------------------------- #
# build_channels (directory) + Stage-2 round trip
# --------------------------------------------------------------------------- #
def test_build_channels_and_stage2_round_trip(tmp_path: Path) -> None:
    """Persist a digest-bound replay archive consumable by Stage 2."""
    material = tmp_path / "material"
    material.mkdir()
    for shot_id in (11766, 11767):
        np.savez_compressed(
            material / f"shot_{shot_id}.npz",
            **_mirror(),  # type: ignore[arg-type]  # numpy savez **kwds stub limitation
        )
    out_dir = tmp_path / "out"
    report = build_channels(material, out_dir=out_dir, generated_at=_FIXED_TS)
    assert report["n_derived"] == 2
    assert report["synthetic"] is False
    assert report["schema_version"] == REPORT_SCHEMA
    binding = report["channels_archive"]
    archive_path = out_dir / "channels.npz"
    assert binding["file_sha256"] == sha256(archive_path.read_bytes()).hexdigest()
    assert binding["bytes"] == archive_path.stat().st_size
    assert binding["shot_count"] == 2
    assert [member["shot_id"] for member in binding["shot_members"]] == [11766, 11767]
    assert all(len(member["sha256"]) == 64 for member in binding["shot_members"])

    shots = _load_shots(out_dir / "channels.npz")
    dataset = build_dataset(
        shots, dataset_id="rt", out_dir=tmp_path / "ds", retrieved_at=_FIXED_TS, generated_at=_FIXED_TS
    )
    assert dataset["n_shots"] == 2
    assert dataset["status"] == "blocked"


def test_build_channels_orders_shots_by_numeric_identity(tmp_path: Path) -> None:
    """Keep producer inventory canonical when filename widths differ."""
    material = tmp_path / "material"
    material.mkdir()
    for shot_id in (10, 2):
        np.savez_compressed(
            material / f"shot_{shot_id}.npz",
            **_mirror(),  # type: ignore[arg-type]  # numpy savez **kwds stub limitation
        )
    out = tmp_path / "out"
    report = build_channels(material, out_dir=out, generated_at=_FIXED_TS)
    assert [record["shot_id"] for record in report["shots"]] == [2, 10]
    assert report["channels_archive"]["shot_count"] == 2


def test_build_channels_is_byte_reproducible(tmp_path: Path) -> None:
    """Reproduce identical archive and report bytes from fixed inputs and time."""
    material = tmp_path / "material"
    material.mkdir()
    np.savez_compressed(
        material / "shot_11766.npz",
        **_mirror(),  # type: ignore[arg-type]  # numpy savez **kwds stub limitation
    )
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    first = build_channels(material, out_dir=first_out, generated_at=_FIXED_TS)
    second = build_channels(material, out_dir=second_out, generated_at=_FIXED_TS)
    assert first == second
    assert (first_out / "channels.npz").read_bytes() == (second_out / "channels.npz").read_bytes()


def test_build_channels_records_malformed_mirror(tmp_path: Path) -> None:
    """Record malformed source mirrors without promoting them."""
    material = tmp_path / "material"
    material.mkdir()
    np.savez_compressed(
        material / "shot_999.npz",
        **{"summary.time": np.linspace(0, 1, 5)},  # type: ignore[arg-type]  # numpy savez **kwds stub limitation
    )  # missing vars
    report = build_channels(material, out_dir=tmp_path / "out", generated_at=_FIXED_TS)
    assert report["n_derived"] == 0
    assert report["shots"][0]["status"] == "failed"
    assert report["channels_archive"]["shot_count"] == 0


def test_build_channels_refuses_invalid_metadata_and_overwrite(tmp_path: Path) -> None:
    """Require explicit time/window metadata and a fresh archive destination."""
    material = tmp_path / "material"
    material.mkdir()
    with pytest.raises(ValueError, match="generated_at"):
        build_channels(material, out_dir=tmp_path / "out", generated_at="")
    for locked_window in (0, 2):
        with pytest.raises(ValueError, match="positive odd"):
            build_channels(material, out_dir=tmp_path / "out", generated_at=_FIXED_TS, locked_window=locked_window)
    out = tmp_path / "existing"
    out.mkdir()
    (out / "channels.npz").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_channels(material, out_dir=out, generated_at=_FIXED_TS)
    assert (out / "channels.npz").read_bytes() == b"preserve"


def test_build_channels_requires_source_and_confines_output(tmp_path: Path) -> None:
    """Reject a missing source or any output nested in immutable material."""
    with pytest.raises(ValueError, match="material_dir is not a directory"):
        build_channels(tmp_path / "missing", out_dir=tmp_path / "out", generated_at=_FIXED_TS)
    material = tmp_path / "material"
    material.mkdir()
    with pytest.raises(ValueError, match="outside the immutable material_dir"):
        build_channels(material, out_dir=material / "derived", generated_at=_FIXED_TS)
    assert list(material.iterdir()) == []


def test_build_channels_removes_temporary_archive_on_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave no output or temporary bytes when persisted validation fails."""
    material = tmp_path / "material"
    material.mkdir()
    np.savez_compressed(
        material / "shot_11766.npz",
        **_mirror(),  # type: ignore[arg-type]  # numpy savez **kwds stub limitation
    )
    out = tmp_path / "out"

    def _fail_inspection(path: Path, *, expected_shot_ids: object = None) -> dict[str, Any]:
        raise ValueError("fixture validation failure")

    monkeypatch.setattr(replay_builder, "inspect_replay_archive", _fail_inspection)
    with pytest.raises(ValueError, match="fixture validation failure"):
        build_channels(material, out_dir=out, generated_at=_FIXED_TS)
    assert not (out / "channels.npz").exists()
    assert list(out.iterdir()) == []


def test_build_channels_rejects_publish_race_without_leaking_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed if another writer wins the final archive path race."""
    material = tmp_path / "material"
    material.mkdir()
    out = tmp_path / "out"

    def _race(_source: Path, _target: Path) -> None:
        raise FileExistsError("fixture race")

    monkeypatch.setattr("validation.build_disruption_replay_channels.os.link", _race)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_channels(material, out_dir=out, generated_at=_FIXED_TS)
    assert not (out / "channels.npz").exists()
    assert list(out.iterdir()) == []


def test_inspect_replay_archive_rejects_missing_and_inventory_drift(tmp_path: Path) -> None:
    """Reopen persisted bytes and reject missing, extra, or mismatched members."""
    with pytest.raises(ValueError, match="does not exist"):
        inspect_replay_archive(tmp_path / "missing.npz")
    archive = tmp_path / "archive.npz"
    np.savez(archive, extra=np.asarray([1.0]))
    with pytest.raises(ValueError, match="contain shot_ids"):
        inspect_replay_archive(archive)
    arrays: dict[str, NDArray[Any]] = {f"101:{name}": np.arange(3, dtype=np.float64) for name in _MEASURED}
    np.savez(archive, shot_ids=np.asarray([101]), extra=np.asarray([1.0]), **arrays)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="member inventory"):
        inspect_replay_archive(archive)
    np.savez(archive, shot_ids=np.asarray([101]), **arrays)  # type: ignore[arg-type]
    assert inspect_replay_archive_bytes(archive.read_bytes(), path_name=archive.name) == inspect_replay_archive(archive)
    with pytest.raises(ValueError, match="producer inventory"):
        inspect_replay_archive(archive, expected_shot_ids=[102])
    with pytest.raises(ValueError, match="path_name"):
        inspect_replay_archive_bytes(archive.read_bytes(), path_name="")


def test_inspect_replay_archive_normalises_read_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalise a persisted-byte read failure after path admission."""
    archive = tmp_path / "archive.npz"
    archive.write_bytes(b"present")

    def _fail_read(_path: Path) -> bytes:
        raise OSError("fixture read failure")

    monkeypatch.setattr(Path, "read_bytes", _fail_read)
    with pytest.raises(ValueError, match="cannot read replay archive"):
        inspect_replay_archive(archive)


@pytest.mark.parametrize(
    ("shot_ids", "mutate", "message"),
    [
        (np.asarray([101.0]), None, "integer vector"),
        (np.asarray([101, 101]), None, "unique, positive, and sorted"),
        (np.asarray([101]), lambda p: p.update({"101:Ip_MA": np.asarray([1], dtype=np.int64)}), "finite float"),
        (np.asarray([101]), lambda p: p.update({"101:Ip_MA": np.asarray([np.nan])}), "finite float"),
        (np.asarray([101]), lambda p: p.update({"101:Ip_MA": np.arange(2, dtype=np.float64)}), "lengths differ"),
        (
            np.asarray([101]),
            lambda p: p.update({name: np.asarray([], dtype=np.float64) for name in p}),
            "contain samples",
        ),
    ],
)
def test_inspect_replay_archive_rejects_malformed_vectors(
    tmp_path: Path,
    shot_ids: NDArray[Any],
    mutate: Callable[[dict[str, NDArray[Any]]], None] | None,
    message: str,
) -> None:
    """Reject malformed identities, channel dtypes, values, lengths, and empties."""
    arrays: dict[str, NDArray[Any]] = {f"101:{name}": np.arange(3, dtype=np.float64) for name in _MEASURED}
    if mutate is not None:
        mutate(arrays)
    archive = tmp_path / "archive.npz"
    np.savez(archive, shot_ids=shot_ids, **arrays)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        inspect_replay_archive(archive)


def test_main_writes_report(tmp_path: Path) -> None:
    """Write the v2 producer report through the public CLI path."""
    material = tmp_path / "material"
    material.mkdir()
    np.savez_compressed(
        material / "shot_11766.npz",
        **_mirror(),  # type: ignore[arg-type]  # numpy savez **kwds stub limitation
    )
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


def test_main_refuses_unsafe_or_existing_report_paths(tmp_path: Path) -> None:
    """Keep the report outside source material and preserve existing bytes."""
    material = tmp_path / "material"
    material.mkdir()
    with pytest.raises(ValueError, match="outside the immutable material_dir"):
        main(
            [
                "--material-dir",
                str(material),
                "--out-dir",
                str(tmp_path / "out"),
                "--json-out",
                str(material / "report.json"),
                "--generated-at",
                _FIXED_TS,
            ]
        )
    archive_path = tmp_path / "out" / "channels.npz"
    with pytest.raises(ValueError, match="must differ"):
        main(
            [
                "--material-dir",
                str(material),
                "--out-dir",
                str(tmp_path / "out"),
                "--json-out",
                str(archive_path),
                "--generated-at",
                _FIXED_TS,
            ]
        )
    report_path = tmp_path / "existing.json"
    report_path.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        main(
            [
                "--material-dir",
                str(material),
                "--out-dir",
                str(tmp_path / "out"),
                "--json-out",
                str(report_path),
                "--generated-at",
                _FIXED_TS,
            ]
        )
    assert report_path.read_text(encoding="utf-8") == "preserve"
    assert not archive_path.exists()

    archive_path.parent.mkdir()
    archive_path.write_bytes(b"preserve archive")
    fresh_report = tmp_path / "fresh.json"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        main(
            [
                "--material-dir",
                str(material),
                "--out-dir",
                str(tmp_path / "out"),
                "--json-out",
                str(fresh_report),
                "--generated-at",
                _FIXED_TS,
            ]
        )
    assert archive_path.read_bytes() == b"preserve archive"
    assert not fresh_report.exists()


def test_main_rolls_back_reserved_report_and_archive_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove both outputs if report serialization fails after archive publish."""
    material = tmp_path / "material"
    material.mkdir()
    out = tmp_path / "out"
    report_path = tmp_path / "report.json"

    def _fail_dump(*_args: object, **_kwargs: object) -> None:
        raise OSError("fixture report write failure")

    monkeypatch.setattr("validation.build_disruption_replay_channels.json.dump", _fail_dump)
    with pytest.raises(OSError, match="fixture report write failure"):
        main(
            [
                "--material-dir",
                str(material),
                "--out-dir",
                str(out),
                "--json-out",
                str(report_path),
                "--generated-at",
                _FIXED_TS,
            ]
        )
    assert not report_path.exists()
    assert not (out / "channels.npz").exists()
