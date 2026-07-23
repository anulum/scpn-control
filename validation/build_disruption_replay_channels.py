#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Derive disruption replay channels from the shared MAST material
"""Derive the run_real_shot_replay channels from the shared MAST material set.

This consumes the native-resolution per-shot mirrors written by
:mod:`validation.acquire_mast_disruption_shots` and produces the combined
``channels.npz`` that :mod:`validation.build_mast_disruption_dataset` labels. It
is the SCPN-CONTROL view of the shared material: it reads local mirrors (no
re-download), derives the eleven measured channels on the summary timebase, and
writes them in the exact Stage-2 compatibility schema. Producer report v2 binds
the reopened archive bytes and per-shot candidate channel values before an
exclusive publish; the source material remains immutable. The report does not
make every candidate a canonical physical binding.

Fluctuation channels (toroidal n=1/n=2 mode amplitudes and the locked-mode
envelope from the 12-channel saddle array, and dB/dt from a poloidal probe) are
computed on their native fast timebase and reduced to the common grid by a
per-bin peak, preserving warning-relevant bursts a plain interpolation would
alias away; the equilibrium and summary scalar channels are linearly
interpolated. The modal candidates retain the historical missing-row
zero-replacement recipe but remain inadmissible under the L2F-12c authority
gate. The locked-mode envelope is additionally blocked by the L2F-12d
stationary-estimator authority gate. The poloidal-probe candidate is blocked by
the dB/dt source-quantity gate, which prevents either a missed derivative or a
second derivative until the ``T`` versus ``Tesla/sec`` conflict is attested.
Labels are added by the dataset builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from validation.disruption_channel_recipes import (
    amperes_to_megamperes,
    dbdt_gauss_per_s,
    locked_mode_envelope,
    n_mode_amplitude,
    per_1e19,
)
from validation.mast_dbdt_authority import (
    GEOMETRY_KEYS as DBDT_GEOMETRY_KEYS,
)
from validation.mast_dbdt_authority import (
    mast_dbdt_authority_spec,
)
from validation.mast_locked_mode_authority import mast_locked_mode_authority_spec
from validation.mast_saddle_modal_authority import GEOMETRY_KEYS, mast_saddle_modal_authority_spec
from validation.mast_source_object_manifest import array_value_sha256, canonical_json_sha256

REPORT_SCHEMA = "scpn-control.mast-disruption-replay-channels.v2.0.0"
REPLAY_MEMBER_DIGEST_KIND = "canonical-channel-values-sha256-v1"

_MEASURED = (
    "time_s",
    "Ip_MA",
    "BT_T",
    "beta_N",
    "q95",
    "ne_1e19",
    "n1_amp",
    "n2_amp",
    "locked_mode_amp",
    "dBdt_gauss_per_s",
    "vertical_position_m",
)


def _interp(values: NDArray[np.float64], src: NDArray[np.float64], grid: NDArray[np.float64]) -> NDArray[np.float64]:
    value = np.asarray(values, dtype=np.float64).ravel()
    time: NDArray[np.float64] = np.asarray(src, dtype=np.float64).ravel()
    # Some equilibrium channels (e.g. the magnetic-axis Z) carry their own coarser
    # timebase; when the length does not match the group timebase, assume uniform
    # sampling over the grid window (the values are real, only the alignment is
    # approximate, and this channel is not consumed by the scoring core).
    if value.shape[0] != time.shape[0]:
        time = np.linspace(float(grid[0]), float(grid[-1]), value.shape[0]).astype(np.float64)
    finite = np.isfinite(value) & np.isfinite(time)
    if not bool(np.any(finite)):
        raise ValueError("no finite samples to interpolate.")
    order = np.argsort(time[finite])
    interpolated: NDArray[np.float64] = np.interp(grid, time[finite][order], value[finite][order]).astype(np.float64)
    return interpolated


def _peak_to_grid(
    fast: NDArray[np.float64], fast_time: NDArray[np.float64], grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Reduce a fast channel to the grid by the per-bin peak magnitude."""
    magnitude = np.abs(np.asarray(fast, dtype=np.float64))
    time = np.asarray(fast_time, dtype=np.float64)
    finite = np.isfinite(magnitude) & np.isfinite(time)
    magnitude, time = magnitude[finite], time[finite]
    order = np.argsort(time)
    magnitude, time = magnitude[order], time[order]
    edges = np.empty(grid.shape[0] + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (grid[1:] + grid[:-1])
    edges[0] = grid[0] - 0.5 * (grid[1] - grid[0])
    edges[-1] = grid[-1] + 0.5 * (grid[-1] - grid[-2])
    idx = np.searchsorted(time, edges)
    out = np.empty(grid.shape[0], dtype=np.float64)
    for i in range(grid.shape[0]):
        lo, hi = int(idx[i]), int(idx[i + 1])
        out[i] = float(magnitude[lo:hi].max()) if hi > lo else float("nan")
    empty = ~np.isfinite(out)
    if bool(np.any(empty)):
        out[empty] = np.interp(grid[empty], time, magnitude)
    return out


def derive_replay_channels(
    mirror: dict[str, NDArray[Any]], *, locked_window: int = 201
) -> dict[str, NDArray[np.float64]]:
    """Derive eleven compatibility channels from one native-resolution mirror."""
    grid = np.asarray(mirror["summary.time"], dtype=np.float64)
    t_eq = np.asarray(mirror["equilibrium.time"], dtype=np.float64)
    t_saddle = np.asarray(mirror["magnetics.time_saddle"], dtype=np.float64)
    bphi_rmag = np.asarray(mirror["equilibrium.bphi_rmag"], dtype=np.float64)
    magnetic_axis_r = np.asarray(mirror["equilibrium.magnetic_axis_r"], dtype=np.float64)
    if (
        bphi_rmag.ndim != 1
        or magnetic_axis_r.ndim != 1
        or t_eq.ndim != 1
        or not (bphi_rmag.shape == magnetic_axis_r.shape == t_eq.shape)
    ):
        raise ValueError("bphi_rmag, magnetic_axis_r, and equilibrium.time must be aligned one-dimensional arrays")
    joint_field_radius = np.isfinite(bphi_rmag) & np.isfinite(magnetic_axis_r) & (magnetic_axis_r > 0.0)
    if not bool(np.any(joint_field_radius)):
        raise ValueError("bphi_rmag has no finite sample at a finite positive magnetic_axis_r")

    saddle = np.nan_to_num(np.asarray(mirror["magnetics.b_field_tor_probe_saddle_field"], dtype=np.float64))  # (12, T)
    phi = np.deg2rad(
        np.nanmean(np.asarray(mirror["magnetics.b_field_tor_probe_saddle_m_phi"], dtype=np.float64), axis=1)
    )
    n1 = n_mode_amplitude(saddle.T, phi, 1)
    n2 = n_mode_amplitude(saddle.T, phi, 2)
    locked = locked_mode_envelope(saddle.T, phi, window=locked_window)

    pol = np.nan_to_num(np.asarray(mirror["magnetics.b_field_pol_probe_cc_field"], dtype=np.float64))
    pol_trace = pol[0] if pol.ndim == 2 else pol
    t_pol = np.asarray(mirror["magnetics.time_mirnov"], dtype=np.float64)[: pol_trace.shape[0]]
    dbdt = dbdt_gauss_per_s(pol_trace, t_pol)

    ip = np.nan_to_num(np.asarray(mirror["summary.ip"], dtype=np.float64))
    ne = np.nan_to_num(np.asarray(mirror["summary.line_average_n_e"], dtype=np.float64))
    channels: dict[str, NDArray[np.float64]] = {
        "time_s": grid,
        "Ip_MA": amperes_to_megamperes(ip),
        "BT_T": _interp(bphi_rmag, t_eq, grid),
        "beta_N": _interp(mirror["equilibrium.beta_tor_normal"], t_eq, grid),
        "q95": _interp(mirror["equilibrium.q95"], t_eq, grid),
        "ne_1e19": per_1e19(ne),
        "n1_amp": _peak_to_grid(n1, t_saddle, grid),
        "n2_amp": _peak_to_grid(n2, t_saddle, grid),
        "locked_mode_amp": _peak_to_grid(locked, t_saddle, grid),
        "dBdt_gauss_per_s": _peak_to_grid(dbdt, t_pol, grid),
        "vertical_position_m": _interp(mirror["equilibrium.z"], t_eq, grid),
    }
    for name, array in channels.items():
        channels[name] = np.nan_to_num(np.asarray(array, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return channels


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _is_within(path: Path, directory: Path) -> bool:
    """Return whether ``path`` resolves inside or exactly at ``directory``."""
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def inspect_replay_archive(
    path: Path,
    *,
    expected_shot_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Read once, reopen, and digest a replay archive through its byte surface."""
    if not path.is_file():
        raise ValueError(f"replay archive does not exist: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read replay archive: {exc}") from exc
    return inspect_replay_archive_bytes(raw, path_name=path.name, expected_shot_ids=expected_shot_ids)


def inspect_replay_archive_bytes(
    raw: bytes,
    *,
    path_name: str,
    expected_shot_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate and digest one immutable replay-archive byte snapshot."""
    if not path_name:
        raise ValueError("replay archive path_name must be non-empty")
    try:
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            if "shot_ids" not in archive.files:
                raise ValueError("replay archive must contain shot_ids")
            identifiers = np.asarray(archive["shot_ids"])
            if identifiers.ndim != 1 or identifiers.dtype.kind not in "iu":
                raise ValueError("replay archive shot_ids must be a one-dimensional integer vector")
            shot_ids = [int(value) for value in identifiers]
            if any(shot_id <= 0 for shot_id in shot_ids) or shot_ids != sorted(set(shot_ids)):
                raise ValueError("replay archive shot_ids must be unique, positive, and sorted")
            if expected_shot_ids is not None and shot_ids != list(expected_shot_ids):
                raise ValueError("replay archive shot_ids do not match the producer inventory")
            expected_members = {"shot_ids"} | {f"{shot_id}:{channel}" for shot_id in shot_ids for channel in _MEASURED}
            if set(archive.files) != expected_members:
                raise ValueError("replay archive member inventory does not match its shot/channel schema")
            shot_members: list[dict[str, Any]] = []
            for shot_id in shot_ids:
                channels: list[dict[str, str]] = []
                sample_count: int | None = None
                for channel in _MEASURED:
                    array = np.asarray(archive[f"{shot_id}:{channel}"])
                    if array.ndim != 1 or array.dtype.kind != "f" or not bool(np.all(np.isfinite(array))):
                        raise ValueError(
                            f"replay archive shot {shot_id} channel {channel} must be a finite float vector"
                        )
                    if sample_count is None:
                        sample_count = int(array.shape[0])
                    elif array.shape[0] != sample_count:
                        raise ValueError(f"replay archive shot {shot_id} channel lengths differ")
                    channels.append({"name": channel, "value_sha256": array_value_sha256(array)})
                if sample_count is None or sample_count <= 0:
                    raise ValueError(f"replay archive shot {shot_id} must contain samples")
                shot_members.append(
                    {
                        "shot_id": shot_id,
                        "n_samples": sample_count,
                        "sha256": canonical_json_sha256({"shot_id": shot_id, "channels": channels}),
                    }
                )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot validate replay archive: {exc}") from exc
    return {
        "path": path_name,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "shot_count": len(shot_members),
        "member_digest_kind": REPLAY_MEMBER_DIGEST_KIND,
        "shot_members": shot_members,
    }


def build_channels(material_dir: Path, *, out_dir: Path, generated_at: str, locked_window: int = 201) -> dict[str, Any]:
    """Derive replay channels for every mirror in ``material_dir`` into channels.npz."""
    if not generated_at:
        raise ValueError("generated_at must be non-empty")
    if locked_window <= 0 or locked_window % 2 == 0:
        raise ValueError("locked_window must be a positive odd integer")
    if not material_dir.is_dir():
        raise ValueError(f"material_dir is not a directory: {material_dir}")
    if _is_within(out_dir, material_dir):
        raise ValueError("out_dir must be outside the immutable material_dir")
    payload: dict[str, NDArray[Any]] = {}
    shot_ids: list[int] = []
    records: list[dict[str, Any]] = []
    shot_paths = sorted(material_dir.glob("shot_*.npz"), key=lambda path: int(path.stem.split("_")[1]))
    for shot_path in shot_paths:
        shot_id = int(shot_path.stem.split("_")[1])
        try:
            with np.load(shot_path, allow_pickle=False) as mirror:
                channels = derive_replay_channels({k: mirror[k] for k in mirror.files}, locked_window=locked_window)
        except Exception as exc:  # noqa: BLE001 - record and continue over malformed mirrors
            records.append({"shot_id": shot_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            continue
        shot_ids.append(shot_id)
        for name in _MEASURED:
            payload[f"{shot_id}:{name}"] = channels[name]
        records.append({"shot_id": shot_id, "status": "derived", "n_samples": int(channels["time_s"].shape[0])})

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "channels.npz"
    if npz_path.exists():
        raise ValueError("refusing to overwrite an existing replay archive")
    payload["shot_ids"] = np.asarray(shot_ids, dtype=np.int64)
    with tempfile.NamedTemporaryFile(prefix=".channels.", suffix=".npz", dir=out_dir, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez(
            temporary_path,
            **payload,  # type: ignore[arg-type]  # numpy savez stub: **kwds ArrayLike splat vs allow_pickle bool
        )
        archive_binding = inspect_replay_archive(temporary_path, expected_shot_ids=shot_ids)
        try:
            os.link(temporary_path, npz_path)
        except FileExistsError as exc:
            raise ValueError("refusing to overwrite an existing replay archive") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    archive_binding["path"] = npz_path.name

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "synthetic": False,
        "material_dir": material_dir.name,
        "channels_npz": npz_path.name,
        "channels_archive": archive_binding,
        "channel_schema": list(_MEASURED),
        "channel_authority": {
            "BT_T": {
                "source_key": "equilibrium.bphi_rmag",
                "reference_radius_key": "equilibrium.magnetic_axis_r",
                "canonical_binding_admissible": False,
                "blocker": "toroidal_field_authority_incomplete",
            },
            "beta_N": {
                "source_key": "equilibrium.beta_tor_normal",
                "canonical_binding_admissible": False,
                "blocker": "normalised_beta_authority_incomplete",
            },
            "n1_amp": {
                "source_key": "magnetics.b_field_tor_probe_saddle_field",
                "geometry_keys": list(GEOMETRY_KEYS),
                "authority_spec_sha256": mast_saddle_modal_authority_spec()["payload_sha256"],
                "canonical_binding_admissible": False,
                "blocker": "saddle_modal_authority_incomplete",
            },
            "n2_amp": {
                "source_key": "magnetics.b_field_tor_probe_saddle_field",
                "geometry_keys": list(GEOMETRY_KEYS),
                "authority_spec_sha256": mast_saddle_modal_authority_spec()["payload_sha256"],
                "canonical_binding_admissible": False,
                "blocker": "saddle_modal_authority_incomplete",
            },
            "locked_mode_amp": {
                "source_key": "magnetics.b_field_tor_probe_saddle_field",
                "geometry_keys": list(GEOMETRY_KEYS),
                "authority_spec_sha256": mast_locked_mode_authority_spec()["payload_sha256"],
                "canonical_binding_admissible": False,
                "blocker": "locked_mode_authority_incomplete",
            },
            "dBdt_gauss_per_s": {
                "source_key": "magnetics.b_field_pol_probe_cc_field",
                "geometry_keys": list(DBDT_GEOMETRY_KEYS),
                "authority_spec_sha256": mast_dbdt_authority_spec()["payload_sha256"],
                "canonical_binding_admissible": False,
                "blocker": "dbdt_authority_incomplete",
            },
        },
        "claim_boundary": {
            "scientific_validation": False,
            "training_admission": False,
            "facility_prediction": False,
            "control_admission": False,
        },
        "locked_window": locked_window,
        "n_derived": len(shot_ids),
        "shots": records,
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = _sha256_json(report)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-dir", type=Path, required=True, help="Directory of shot_<id>.npz mirrors.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for channels.npz.")
    parser.add_argument("--json-out", type=Path, required=True, help="Report JSON output path.")
    parser.add_argument("--generated-at", type=str, default="", help="Fixed UTC timestamp label.")
    parser.add_argument("--locked-window", type=int, default=201, help="Locked-mode envelope window (saddle samples).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: derive replay channels from the material set."""
    args = _parse_args(argv)
    archive_path = args.out_dir / "channels.npz"
    if _is_within(args.json_out, args.material_dir):
        raise ValueError("json_out must be outside the immutable material_dir")
    if args.json_out.resolve() == archive_path.resolve():
        raise ValueError("json_out must differ from the replay archive path")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    archive_published = False
    try:
        with args.json_out.open("x", encoding="utf-8") as report_handle:
            try:
                report = build_channels(
                    args.material_dir,
                    out_dir=args.out_dir,
                    generated_at=args.generated_at,
                    locked_window=args.locked_window,
                )
                archive_published = True
                json.dump(report, report_handle, indent=2, sort_keys=True)
                report_handle.write("\n")
                report_handle.flush()
                os.fsync(report_handle.fileno())
            except Exception:
                if archive_published:
                    archive_path.unlink(missing_ok=True)
                raise
    except FileExistsError as exc:
        raise ValueError("refusing to overwrite an existing replay report") from exc
    except Exception:
        args.json_out.unlink(missing_ok=True)
        raise
    print(f"derived {report['n_derived']} shots -> {report['channels_npz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
