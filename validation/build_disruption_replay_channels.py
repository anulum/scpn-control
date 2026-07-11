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
writes them in the exact Stage-2 schema.

Fluctuation channels (toroidal n=1/n=2 mode amplitudes and the locked-mode
envelope from the 12-channel saddle array, and dB/dt from a poloidal probe) are
computed on their native fast timebase and reduced to the common grid by a
per-bin peak, preserving warning-relevant bursts a plain interpolation would
alias away; the equilibrium and summary scalar channels are linearly
interpolated. All channels come from real level2 signals; labels are added by the
dataset builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

REPORT_SCHEMA = "scpn-control.mast-disruption-replay-channels.v1"

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
    """Derive the eleven measured channels from one native-resolution shot mirror."""
    grid = np.asarray(mirror["summary.time"], dtype=np.float64)
    t_eq = np.asarray(mirror["equilibrium.time"], dtype=np.float64)
    t_saddle = np.asarray(mirror["magnetics.time_saddle"], dtype=np.float64)

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
        "BT_T": _interp(mirror["equilibrium.bvac_rmag"], t_eq, grid),
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


def build_channels(material_dir: Path, *, out_dir: Path, generated_at: str, locked_window: int = 201) -> dict[str, Any]:
    """Derive replay channels for every mirror in ``material_dir`` into channels.npz."""
    payload: dict[str, NDArray[Any]] = {}
    shot_ids: list[int] = []
    records: list[dict[str, Any]] = []
    for shot_path in sorted(material_dir.glob("shot_*.npz")):
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
    payload["shot_ids"] = np.asarray(shot_ids, dtype=np.int64)
    np.savez(npz_path, **payload)  # type: ignore[arg-type]  # numpy savez stub: **kwds ArrayLike splat vs allow_pickle bool

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "synthetic": False,
        "material_dir": material_dir.name,
        "channels_npz": npz_path.name,
        "channel_schema": list(_MEASURED),
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
    report = build_channels(
        args.material_dir, out_dir=args.out_dir, generated_at=args.generated_at, locked_window=args.locked_window
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"derived {report['n_derived']} shots -> {report['channels_npz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
