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
zero-replacement recipe but remain inadmissible under the saddle-modal authority
gate. The locked-mode envelope is additionally blocked by the locked-mode
stationary-estimator authority gate. The poloidal-probe candidate is blocked by
the dB/dt source-quantity gate, which prevents either a missed derivative or a
second derivative until the ``T`` versus ``Tesla/sec`` conflict is attested.
Labels are added by the dataset builder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_control._npz import save_npz_arrays
from validation.disruption_channel_recipes import (
    amperes_to_megamperes,
    dbdt_gauss_per_s,
    locked_mode_envelope,
    n_mode_amplitude,
    per_1e19,
)
from validation.mast_replay_contracts._archive import inspect_replay_archive as _inspect_archive
from validation.mast_replay_contracts._archive import inspect_replay_archive_bytes as _inspect_archive_bytes
from validation.mast_replay_contracts._inputs import MEASURED_CHANNELS, channel_vectors, shot_identity, time_axis
from validation.mast_replay_contracts._report import REPORT_SCHEMA as REPORT_SCHEMA
from validation.mast_replay_contracts._report import _sha256_json as _sha256_json
from validation.mast_replay_contracts._report import replay_report

REPLAY_MEMBER_DIGEST_KIND = "canonical-channel-values-sha256-v1"

_MEASURED = MEASURED_CHANNELS


def _interp(values: NDArray[np.float64], src: NDArray[np.float64], grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolate finite sorted samples; preserve the historical length fallback."""
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
    """Derive eleven candidate vectors on the mirror's summary timebase.

    ``mirror`` uses the acquisition's dotted NPZ keys. Summary current/density
    use A and m^-3, equilibrium field/radius use T and m, and saddle fields
    use T with coil angles in degrees. Summary, equilibrium and saddle clocks
    must be finite, one-dimensional and strictly increasing; the summary
    clock needs at least two samples. The Mirnov clock is truncated to the
    selected first probe's length and checked by the derivative recipe.

    Scalar equilibrium channels use linear interpolation with endpoint
    clamping; a mismatched value/time length retains the historical uniform
    time-grid assumption. Fast modal and derivative magnitudes use per-bin
    peaks with interpolation across empty bins. The positive integer
    ``locked_window`` defaults to 201 native saddle samples; the pure envelope
    also permits even windows and refuses windows longer than the trace.

    Return fresh aligned float64 vectors named by ``MEASURED_CHANNELS`` in
    the dataset's ``CHANNEL_UNITS``. Inputs are unchanged and no files are
    read or written. Missing keys raise ``KeyError``; invalid clocks, shapes,
    recipes or final channel alignment raise ``ValueError``. Native array
    conversion/extreme arithmetic errors propagate.

    Historical missing-row replacement and final nonfinite-to-zero handling
    are retained. Field/radius checks require a valid paired observation but
    do not authenticate sign, geometry, timing or source quantity. This
    routine grants no canonical physical, training or facility admission.
    """
    grid = time_axis(mirror["summary.time"], name="summary.time", minimum=2)
    t_eq = time_axis(mirror["equilibrium.time"], name="equilibrium.time")
    t_saddle = time_axis(mirror["magnetics.time_saddle"], name="magnetics.time_saddle")
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
    channel_vectors(channels, shot_id=1)
    return channels


def _is_within(path: Path, directory: Path) -> bool:
    """Return whether ``path`` resolves inside or exactly at ``directory``."""
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def inspect_replay_archive(path: Path, *, expected_shot_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """Read one strict replay snapshot and return its original byte/value schema.

    Positive sorted integer identities, exact unique members and finite aligned
    floating vectors with increasing nonempty times are required. Optional
    expected IDs compare directly. Authored ``ValueError`` refuses invalid/read
    inputs. Symlinks are followed; no producer authentication or size bound.
    """
    return _inspect_archive(path, expected_shot_ids=expected_shot_ids)


def inspect_replay_archive_bytes(
    raw: bytes, *, path_name: str, expected_shot_ids: Sequence[int] | None = None
) -> dict[str, Any]:
    """Inspect immutable bytes with the same strict integer/clock schema.

    ``path_name`` is a nonempty label, not a freshness/containment proof. Return
    bindings cover raw bytes and canonical values, not physical provenance.
    Authored ``ValueError`` refuses malformed/non-pickle-safe archives.
    """
    return _inspect_archive_bytes(raw, path_name=path_name, expected_shot_ids=expected_shot_ids)


def build_channels(material_dir: Path, *, out_dir: Path, generated_at: str, locked_window: int = 201) -> dict[str, Any]:
    """Derive channels from local mirrors and publish an uncompressed archive.

    Parameters
    ----------
    material_dir
        Immutable directory containing ``shot_<integer>.npz`` mirrors. Shots
        are visited in numeric identity order and loaded without pickling.
    out_dir
        Destination outside ``material_dir``. A temporary archive is reopened
        and bound to its candidate values before exclusive publication as
        ``channels.npz``; an existing archive is refused.
    generated_at
        Nonempty reproducibility label copied to the returned report. It is
        not parsed as a clock reading or evidence of source freshness.
    locked_window
        Positive odd envelope-window length in native saddle samples.

    Returns
    -------
    dict
        Schema-versioned report with per-shot outcomes, archive byte/value
        digests and explicit channel-authority blockers. Archive members use
        ``<shot_id>:<channel>`` plus an int64 ``shot_ids`` vector. Failed mirror
        reads or derivations are recorded and the remaining shots continue.
        Scientific, training, facility and control admission remain false.

    Raises
    ------
    ValueError
        If metadata, directories, window or reopened archive are invalid, or
        the archive already exists. Temporary files are removed on failure.
    OSError
        If file creation or exclusive publication fails.
    """
    if not generated_at:
        raise ValueError("generated_at must be non-empty")
    if (
        not isinstance(locked_window, int)
        or isinstance(locked_window, bool)
        or locked_window <= 0
        or locked_window % 2 == 0
    ):
        raise ValueError("locked_window must be a positive odd integer")
    if not material_dir.is_dir():
        raise ValueError(f"material_dir is not a directory: {material_dir}")
    if _is_within(out_dir, material_dir):
        raise ValueError("out_dir must be outside the immutable material_dir")
    payload: dict[str, NDArray[Any]] = {}
    shot_ids: list[int] = []
    records: list[dict[str, Any]] = []
    try:
        shot_paths = sorted(material_dir.glob("shot_*.npz"), key=lambda path: int(path.stem.split("_")[1]))
        ids = [shot_identity(int(path.stem.split("_")[1])) for path in shot_paths]
    except (ValueError, IndexError):
        raise ValueError("material shot filenames must contain positive integer identities") from None
    if len(ids) != len(set(ids)):
        raise ValueError("material shot filenames must have unique numeric identities")
    for shot_path in shot_paths:
        shot_id = int(shot_path.stem.split("_")[1])
        try:
            with np.load(shot_path, allow_pickle=False) as mirror:
                channels = derive_replay_channels({k: mirror[k] for k in mirror.files}, locked_window=locked_window)
        except Exception:  # noqa: BLE001 - record and continue over malformed mirrors
            records.append({"shot_id": shot_id, "status": "failed", "error": "mirror could not be read or derived"})
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
        save_npz_arrays(temporary_path, payload, allow_pickle=True)
        archive_binding = inspect_replay_archive(temporary_path, expected_shot_ids=shot_ids)
        try:
            os.link(temporary_path, npz_path)
        except FileExistsError as exc:
            raise ValueError("refusing to overwrite an existing replay archive") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    archive_binding["path"] = npz_path.name

    return replay_report(
        archive_binding,
        records,
        shot_ids,
        material_name=material_dir.name,
        archive_name=npz_path.name,
        generated_at=generated_at,
        locked_window=locked_window,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the source-tree replay-channel command without reading material."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-dir", type=Path, required=True, help="Directory of shot_<id>.npz mirrors.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for channels.npz.")
    parser.add_argument("--json-out", type=Path, required=True, help="Report JSON output path.")
    parser.add_argument("--generated-at", type=str, default="", help="Fixed UTC timestamp label.")
    parser.add_argument("--locked-window", type=int, default=201, help="Locked-mode envelope window (saddle samples).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the source-tree replay command and write an exclusive v2 report.

    ``argv=None`` reads process arguments. Required material/output/report
    paths and a nonempty generated-at label are parsed with argparse; help
    raises ``SystemExit(0)`` and parser errors ``SystemExit(2)``. The report
    must lie outside the immutable material directory and differ from the
    archive path. Existing reports and archives are refused.

    Return ``0`` after archive and report publication, including an empty candidate
    or per-shot failures; this is completion of candidate assembly, not
    scientific admission. A report-write failure removes this command's
    published archive and report when possible. Directory creation, reads,
    publication and cleanup are sequential filesystem operations, not a
    transaction or an authenticated snapshot. Concurrent path replacement
    and cleanup failures are not controlled by this API.

    Calling ``main`` directly preserves authored ``ValueError`` and native
    I/O/conversion errors. The module command maps caught value/I/O/type/key
    errors to a fixed stderr sentence and exit ``2`` without interpreter text.
    """
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
    try:
        raise SystemExit(main())
    except (ValueError, OSError, TypeError, KeyError):
        print("Could not build MAST replay channels.", file=sys.stderr)
        raise SystemExit(2) from None
