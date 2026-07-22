#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST level2 disruption-shot high-fidelity acquisition
"""Mirror FAIR-MAST level2 disruption-relevant signals at native resolution.

This produces a shared, high-fidelity MAST disruption material set consumed by
SCPN-CONTROL, SCPN-FUSION-CORE and MIF-CORE. For each shot it reads the public
FAIR-MAST level2 Zarr v3 store (anonymous S3, ``s3.echo.stfc.ac.uk``) and writes
one ``shot_<id>.npz`` holding the disruption-relevant variables at their native
timebases — the fast toroidal saddle array (12 channels, 20 us), the Mirnov and
poloidal probe arrays, plus the equilibrium, summary and interferometer scalar
channels — with no downsampling or derived reduction, so every consumer can build
its own view without re-downloading. A schema-versioned manifest records the
per-shot files, SHA-256 checksums, the variable inventory, timebases, units and
the official FAIR-MAST licence and citations.

Labels are deliberately not assigned here (the DEFUSE HDF5 labels return HTTP 403
and the DEFUSE shot ids do not intersect the level2 shot range); consumers derive
the Ip current-quench label. Heavy arrays stay off any code repository under a
shared datasets root. Requires the optional FAIR-MAST stack (``zarr``, ``s3fs``,
``xarray``, ``fsspec``) and network access; it is an out-of-band tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from validation.fair_mast_source_policy import fair_mast_provenance

# Injection seams so the S3/Zarr I/O can be stubbed in offline tests.
FilesystemFactory = Callable[[Path], Any]
GroupOpener = Callable[[Any, int, str], Any]

MANIFEST_SCHEMA = "scpn-control.mast-disruption-material.v1"
ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"
BUCKET = "mast"

# Native-resolution variables mirrored per group. Geometry (phi/r/z) accompanies
# each probe array so consumers can perform toroidal-mode decomposition. Units are
# recorded in the manifest and follow the FAIR-MAST level2 convention.
GROUP_VARIABLES: dict[str, tuple[str, ...]] = {
    "summary": ("time", "ip", "line_average_n_e", "greenwald_density"),
    "equilibrium": (
        "time",
        "q95",
        "q_axis",
        "beta_tor_normal",
        "beta_tor",
        "bvac_rmag",
        "z",
        "x_point_z",
        "wmhd",
        "volume",
        "triangularity_upper",
        "triangularity_lower",
        "vloop_dynamic",
    ),
    "interferometer": ("time", "n_e_line"),
    "magnetics": (
        "time_saddle",
        "time_mirnov",
        "b_field_tor_probe_saddle_field",
        "b_field_tor_probe_saddle_m_phi",
        "b_field_tor_probe_saddle_u_phi",
        "b_field_tor_probe_saddle_l_phi",
        "b_field_tor_probe_cc_field",
        "b_field_tor_probe_cc_phi",
        "b_field_pol_probe_cc_field",
        "b_field_pol_probe_cc_phi",
        "b_field_pol_probe_cc_r",
        "b_field_pol_probe_cc_z",
    ),
}


def make_filesystem(cache_dir: Path) -> Any:
    """Build the anonymous FAIR-MAST S3 cache filesystem (proven access pattern)."""
    import fsspec

    cache_dir.mkdir(parents=True, exist_ok=True)
    return fsspec.filesystem(
        "simplecache",
        cache_storage=str(cache_dir),
        target_protocol="s3",
        target_options={"anon": True, "endpoint_url": ENDPOINT_URL, "skip_instance_cache": True},
    )


def _open_group(fs: Any, shot_id: int, group: str) -> Any:
    import xarray as xr

    store = fs.get_mapper(f"s3://{BUCKET}/level2/shots/{shot_id}.zarr")
    return xr.open_zarr(store, group=group, consolidated=True)


def mirror_shot(fs: Any, shot_id: int, *, open_group: GroupOpener = _open_group) -> dict[str, NDArray[Any]]:
    """Read the native-resolution disruption-relevant variables for one shot."""
    payload: dict[str, NDArray[Any]] = {}
    for group, variables in GROUP_VARIABLES.items():
        dataset = open_group(fs, shot_id, group)
        for variable in variables:
            if variable in dataset.variables:
                payload[f"{group}.{variable}"] = np.asarray(dataset[variable].values)
    if not any(key.startswith("magnetics.b_field_tor_probe_saddle_field") for key in payload):
        raise ValueError(f"shot {shot_id}: no toroidal saddle array present.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _shot_summary(payload: dict[str, NDArray[Any]]) -> dict[str, Any]:
    ip = payload.get("summary.ip")
    ip_max_ka = float(np.nanmax(np.abs(np.asarray(ip, dtype=np.float64))) / 1e3) if ip is not None else None
    saddle = payload["magnetics.b_field_tor_probe_saddle_field"]
    return {
        "ip_max_ka": ip_max_ka,
        "saddle_channels": int(np.asarray(saddle).shape[0]),
        "saddle_samples": int(np.asarray(saddle).shape[1]),
        "variables": sorted(payload),
    }


def acquire(
    shot_ids: list[int],
    *,
    out_dir: Path,
    cache_dir: Path,
    generated_at: str,
    retrieved_at: str,
    make_fs: FilesystemFactory | None = None,
    open_group: GroupOpener | None = None,
) -> dict[str, Any]:
    """Mirror every shot to ``out_dir`` and return a schema-versioned manifest."""
    make_fs = make_fs if make_fs is not None else make_filesystem
    open_group = open_group if open_group is not None else _open_group
    fs = make_fs(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for shot_id in shot_ids:
        try:
            payload = mirror_shot(fs, shot_id, open_group=open_group)
        except Exception as exc:  # noqa: BLE001 - record and continue over unavailable shots
            records.append({"shot_id": shot_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            continue
        shot_path = out_dir / f"shot_{shot_id}.npz"
        np.savez_compressed(shot_path, **payload)  # type: ignore[arg-type]  # numpy savez stub: **kwds ArrayLike splat vs allow_pickle bool
        record = {
            "shot_id": shot_id,
            "status": "acquired",
            "npz": shot_path.name,
            "checksum_sha256": _sha256_file(shot_path),
            "bytes": int(shot_path.stat().st_size),
        }
        record.update(_shot_summary(payload))
        records.append(record)

    acquired = [r for r in records if r["status"] == "acquired"]
    failed = [r for r in records if r["status"] == "failed"]
    status = "empty" if not acquired else ("partial" if failed else "complete")
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "status": status,
        "synthetic": False,
        "consumers": ["SCPN-CONTROL", "SCPN-FUSION-CORE", "MIF-CORE"],
        "source": {
            "bucket": f"s3://{BUCKET}",
            "endpoint": ENDPOINT_URL,
            "access": "anonymous",
            "format": "zarr_v3_level2",
            "path_template": f"s3://{BUCKET}/level2/shots/{{shot_id}}.zarr",
        },
        **fair_mast_provenance(),
        "fidelity": "native_resolution_no_downsampling",
        "group_variables": {group: list(variables) for group, variables in GROUP_VARIABLES.items()},
        "label_policy": (
            "labels not assigned; DEFUSE HDF5 labels are HTTP 403 and its shot ids do "
            "not intersect the level2 range, so consumers derive the Ip current-quench label"
        ),
        "retrieved_at": retrieved_at,
        "n_acquired": len(acquired),
        "n_requested": len(shot_ids),
        "total_bytes": sum(int(r["bytes"]) for r in acquired),
        "shots": records,
        "generated_at": generated_at,
        "payload_sha256": None,
    }
    manifest["payload_sha256"] = _sha256_json(manifest)
    return manifest


def _parse_shots(text: str) -> list[int]:
    out: list[int] = []
    for token in text.replace(",", " ").split():
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=str, required=True, help="Shot ids/ranges, e.g. '30419-30424 29876'.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Shared datasets directory for shot_<id>.npz.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Local S3 cache directory (off-repo).")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Manifest JSON output path.")
    parser.add_argument("--generated-at", type=str, default="", help="Fixed UTC timestamp label.")
    parser.add_argument("--retrieved-at", type=str, default="", help="Acquisition timestamp (ISO 8601).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: mirror the shot list and write the material manifest."""
    args = _parse_args(argv)
    manifest = acquire(
        _parse_shots(args.shots),
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        generated_at=args.generated_at,
        retrieved_at=args.retrieved_at,
    )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mb = manifest["total_bytes"] / 1e6
    print(f"acquired {manifest['n_acquired']}/{manifest['n_requested']} shots ({mb:.1f} MB) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
