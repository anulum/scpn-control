#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST disruption channel extraction (level2 signals to replay channels)
"""Extract the run_real_shot_replay channels from acquired FAIR-MAST level2 shots.

This is the out-of-band bridge between an acquired MAST level2 Zarr store and the
disruption dataset builder: it reads the raw level2 signals, applies the
:mod:`validation.disruption_channel_recipes` derivations, and writes the eleven
measured channels for each shot into the combined ``channels.npz`` that
:mod:`validation.build_mast_disruption_dataset` consumes.

The ``level2`` variable names are the acquisition-resolved binding of the
feature-source audit's channel map; the reader fails closed by listing any
missing variables, and the ``BT_T`` channel stays ``lookup_needed`` — it is
resolved only from a direct toroidal-field signal or, failing that, from the
TF-coil current with machine constants confirmed at acquisition. Zarr access is
lazy and injectable so the pipeline is exercised offline with a stubbed store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from validation.build_mast_disruption_dataset import MEASURED_CHANNELS
from validation.disruption_channel_recipes import (
    amperes_to_megamperes,
    dbdt_gauss_per_s,
    locked_mode_envelope,
    n_mode_amplitude,
    per_1e19,
    q_at_psi_norm,
    vacuum_toroidal_field,
)

EXTRACTION_SCHEMA = "scpn-control.mast-disruption-channel-extraction.v1"

#: Timebase variable name in an acquired level2 shot store.
TIME_VARIABLE = "time"
#: Optional direct toroidal-field signal (preferred BT_T source when present).
BT_DIRECT_VARIABLE = "toroidal_field"
#: TF-coil current used to derive the vacuum toroidal field when no direct signal exists.
TF_CURRENT_VARIABLE = "tf_current"

# Expected level2 variable name for each raw signal the recipes need. These are the
# concrete binding of the feature-source audit's level2_source groups; acquisition
# confirms or remaps them against the real level2 catalogue, and a mismatch fails
# closed via the missing-variable check rather than producing a silent channel.
LEVEL2_VARIABLES: dict[str, str] = {
    "ip_amperes": "ip",
    "ne_per_m3": "line_average_n_e",
    "beta_normal": "beta_normal",
    "q_profile": "q",
    "psi_norm_grid": "psi_norm",
    "saddle_tesla": "saddle",
    "saddle_angles_rad": "saddle_angle",
    "poloidal_probe_tesla": "poloidal_probe",
    "axis_z_m": "magnetic_axis_z",
}


class DatasetLike(Protocol):
    """Minimal xarray-like interface used by the extractor core."""

    variables: Any

    def __getitem__(self, key: str) -> Any: ...

    def close(self) -> None: ...


DatasetOpener = Callable[[Path], DatasetLike]


@dataclass(frozen=True)
class ToroidalFieldConfig:
    """MAST TF geometry used to derive ``BT_T`` from the TF-coil current.

    ``n_turns`` and ``r_geo_m`` are machine constants confirmed from the MAST
    machine description at acquisition; they are only used when no direct
    toroidal-field signal is present in the acquired store.
    """

    n_turns: int
    r_geo_m: float


def _values(ds: DatasetLike, name: str) -> NDArray[np.float64]:
    """Read a level2 variable into a float array."""
    return np.asarray(ds[name].values, dtype=np.float64)


def read_shot_signals(ds: DatasetLike, *, tf_config: ToroidalFieldConfig) -> dict[str, NDArray[np.float64]]:
    """Read the raw semantic signal arrays the recipes need from a level2 store.

    Raises :class:`ValueError` listing any missing required variables, and
    :class:`RuntimeError` when ``BT_T`` cannot be resolved (still ``lookup_needed``).
    """
    required = (TIME_VARIABLE, *LEVEL2_VARIABLES.values())
    missing = [name for name in required if name not in ds.variables]
    if missing:
        raise ValueError(f"level2 store is missing required variables: {missing}")
    raw: dict[str, NDArray[np.float64]] = {"time_s": _values(ds, TIME_VARIABLE)}
    for semantic, variable in LEVEL2_VARIABLES.items():
        raw[semantic] = _values(ds, variable)
    if BT_DIRECT_VARIABLE in ds.variables:
        raw["bt_t_tesla"] = _values(ds, BT_DIRECT_VARIABLE)
    elif TF_CURRENT_VARIABLE in ds.variables:
        raw["bt_t_tesla"] = vacuum_toroidal_field(
            _values(ds, TF_CURRENT_VARIABLE), tf_config.r_geo_m, n_turns=tf_config.n_turns
        )
    else:
        raise RuntimeError(
            "BT_T level2 source unresolved (lookup_needed): the acquired store has "
            f"neither a {BT_DIRECT_VARIABLE!r} signal nor a {TF_CURRENT_VARIABLE!r} current."
        )
    return raw


def _validate_channels(channels: dict[str, NDArray[np.float64]], n_samples: int) -> None:
    for name, array in channels.items():
        values = np.asarray(array)
        if values.ndim != 1 or values.shape[0] != n_samples:
            raise ValueError(f"channel {name!r} must be 1-D with {n_samples} samples.")
        if not bool(np.all(np.isfinite(values))):
            raise ValueError(f"channel {name!r} must be finite.")


def derive_channels(raw: dict[str, NDArray[np.float64]], *, locked_window: int) -> dict[str, NDArray[np.float64]]:
    """Apply the recipes to raw signals and return the eleven measured channels."""
    time_s = np.asarray(raw["time_s"], dtype=np.float64)
    saddle = raw["saddle_tesla"]
    angles = raw["saddle_angles_rad"]
    channels: dict[str, NDArray[np.float64]] = {
        "time_s": time_s,
        "Ip_MA": amperes_to_megamperes(raw["ip_amperes"]),
        "BT_T": np.asarray(raw["bt_t_tesla"], dtype=np.float64),
        "beta_N": np.asarray(raw["beta_normal"], dtype=np.float64),
        "q95": q_at_psi_norm(raw["q_profile"], raw["psi_norm_grid"]),
        "ne_1e19": per_1e19(raw["ne_per_m3"]),
        "n1_amp": n_mode_amplitude(saddle, angles, 1),
        "n2_amp": n_mode_amplitude(saddle, angles, 2),
        "locked_mode_amp": locked_mode_envelope(saddle, angles, window=locked_window),
        "dBdt_gauss_per_s": dbdt_gauss_per_s(raw["poloidal_probe_tesla"], time_s),
        "vertical_position_m": np.asarray(raw["axis_z_m"], dtype=np.float64),
    }
    _validate_channels(channels, int(time_s.shape[0]))
    return {name: channels[name] for name in MEASURED_CHANNELS}


def _default_open_dataset(zarr_path: Path) -> DatasetLike:
    """Open an acquired level2 Zarr store with a lazy xarray import."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray is required to open acquired MAST level2 Zarr stores") from exc
    if not zarr_path.exists():
        raise FileNotFoundError(f"acquired level2 Zarr path does not exist: {zarr_path}")
    dataset: DatasetLike = xr.open_zarr(zarr_path, consolidated=True)
    return dataset


def convert_shot_zarr(
    shot_id: int,
    zarr_path: Path,
    *,
    tf_config: ToroidalFieldConfig,
    locked_window: int,
    open_dataset: DatasetOpener | None = None,
) -> dict[str, Any]:
    """Open one acquired shot store and return its derived measured channels."""
    opener = open_dataset if open_dataset is not None else _default_open_dataset
    ds = opener(zarr_path)
    try:
        raw = read_shot_signals(ds, tf_config=tf_config)
        channels = derive_channels(raw, locked_window=locked_window)
    finally:
        ds.close()
    return {"shot_id": shot_id, "channels": channels, "n_samples": int(channels["time_s"].shape[0])}


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_channels_npz(
    manifest_path: Path,
    *,
    dataset_root: Path,
    out_dir: Path,
    tf_config: ToroidalFieldConfig,
    locked_window: int,
    generated_at: str,
    open_dataset: DatasetOpener | None = None,
) -> dict[str, Any]:
    """Extract every acquired shot in a manifest into the combined channels NPZ.

    ``manifest_path`` is an acquisition manifest whose ``shots`` list carries a
    ``shot_id``, ``local_path`` and ``status``; only ``status == "acquired"``
    shots are extracted. Writes ``channels.npz`` (``shot_ids`` plus
    ``"<shot_id>:<channel>"`` arrays) into ``out_dir`` and returns a
    schema-versioned, ``status:"blocked"`` extraction report.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shots = manifest.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("acquisition manifest must contain a non-empty shots list.")
    payload: dict[str, NDArray[Any]] = {}
    shot_ids: list[int] = []
    records: list[dict[str, Any]] = []
    for shot in sorted(shots, key=lambda item: int(item["shot_id"])):
        shot_id = int(shot["shot_id"])
        if shot.get("status") != "acquired":
            records.append({"shot_id": shot_id, "status": "skipped", "reason": "not acquired"})
            continue
        zarr_path = dataset_root / str(shot["local_path"])
        result = convert_shot_zarr(
            shot_id,
            zarr_path,
            tf_config=tf_config,
            locked_window=locked_window,
            open_dataset=open_dataset,
        )
        shot_ids.append(shot_id)
        for name, array in result["channels"].items():
            payload[f"{shot_id}:{name}"] = array
        records.append({"shot_id": shot_id, "status": "extracted", "n_samples": result["n_samples"]})

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "channels.npz"
    payload["shot_ids"] = np.asarray(shot_ids, dtype=np.int64)
    np.savez(npz_path, **payload)  # type: ignore[arg-type]  # numpy savez stub: **kwds ArrayLike splat vs allow_pickle bool

    report: dict[str, Any] = {
        "schema_version": EXTRACTION_SCHEMA,
        "status": "blocked",
        "admission_ready": False,
        "blocked_reason": (
            "channels are extracted from acquired level2 signals with derived recipes "
            "(toroidal mode decomposition, locked-mode envelope, EFIT q95, vacuum "
            "toroidal field); the derived labels and BT_T resolution are bounded, not "
            "facility-validated, so the dataset admission gate stays closed."
        ),
        "channels_npz": npz_path.name,
        "channels_sha256": _sha256_file(npz_path),
        "channel_schema": list(MEASURED_CHANNELS),
        "locked_window": locked_window,
        "toroidal_field_config": {"n_turns": tf_config.n_turns, "r_geo_m": tf_config.r_geo_m},
        "n_shots_extracted": len(shot_ids),
        "shots": records,
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    report["payload_sha256"] = _sha256_json(report)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Acquisition manifest with acquired shots.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root holding the acquired level2 stores.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for channels.npz.")
    parser.add_argument("--json-out", type=Path, required=True, help="Extraction report JSON output path.")
    parser.add_argument("--n-turns", type=int, required=True, help="TF-coil turns (MAST machine constant).")
    parser.add_argument("--r-geo-m", type=float, required=True, help="Geometric-axis major radius (m).")
    parser.add_argument("--locked-window", type=int, default=11, help="Locked-mode envelope window (samples).")
    parser.add_argument("--generated-at", type=str, default="", help="Fixed UTC timestamp label.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: extract acquired shots into channels.npz and write the report."""
    args = _parse_args(argv)
    report = build_channels_npz(
        args.manifest,
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        tf_config=ToroidalFieldConfig(n_turns=args.n_turns, r_geo_m=args.r_geo_m),
        locked_window=args.locked_window,
        generated_at=args.generated_at,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"extraction: {report['n_shots_extracted']} shot(s) -> {report['channels_npz']} (status={report['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
