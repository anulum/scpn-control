#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST level2 disruption-shot high-fidelity acquisition
"""Cache selected FAIR-MAST level2 disruption signals without resampling.

This produces a shared, high-fidelity MAST disruption material set consumed by
SCPN-CONTROL, SCPN-FUSION-CORE and MIF-CORE. For each shot it reads the public
FAIR-MAST level2 Zarr v3 store (anonymous S3, ``s3.echo.stfc.ac.uk``) and writes
one derived ``shot_<id>.npz`` cache holding selected values at their source sample
resolution.  The remote Zarr remains the native source: NPZ does not preserve its
chunking, hierarchy, or attributes.  A SourceObjectManifest v2 therefore records
the source hierarchy and available xarray metadata separately, binds each array's
exact values, binds the derived-file bytes, and declares the lossy container
boundary instead of calling NPZ a raw native object.

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
import math
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from validation.fair_mast_source_policy import FAIR_MAST_LICENCE, fair_mast_provenance
from validation.mast_source_object_manifest import (
    SOURCE_GENERATION_DIGEST_KIND,
    SOURCE_GENERATION_SCHEMA,
    SOURCE_OBJECT_MANIFEST_SCHEMA,
    build_derived_npz_artifact,
    canonical_json_sha256,
    finalise_source_object_manifest,
    validate_source_object_manifest,
)

# Injection seams so the S3/Zarr I/O can be stubbed in offline tests.
FilesystemFactory = Callable[[Path], Any]
GroupOpener = Callable[[Any, int, str], Any]

MANIFEST_SCHEMA = SOURCE_OBJECT_MANIFEST_SCHEMA
ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"
BUCKET = "mast"
CACHE_GENERATION_SCHEMA = "scpn-control.fair-mast-cache-generation.v1.0.0"
_MAX_ROOT_METADATA_BYTES = 16 << 20
_SOURCE_METADATA_TIMEOUT_S = 30.0


class SourceGenerationError(ValueError):
    """Raised when a FAIR-MAST source generation cannot be pinned safely."""


@dataclass(frozen=True)
class SourceGenerationPin:
    """Exact upstream root-metadata identity for one FAIR-MAST shot."""

    source_uri: str
    sha256: str
    byte_count: int
    etag: str | None
    last_modified: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest-bound source-generation record."""
        return {
            "schema_version": SOURCE_GENERATION_SCHEMA,
            "digest_kind": SOURCE_GENERATION_DIGEST_KIND,
            "source_uri": self.source_uri,
            "metadata_path": "zarr.json",
            "sha256": self.sha256,
            "bytes": self.byte_count,
            "zarr_format": 3,
            "consolidated_metadata_kind": "inline",
            "etag": self.etag,
            "last_modified": self.last_modified,
        }


GenerationReader = Callable[[int], SourceGenerationPin]


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceGenerationError(f"duplicate JSON key {key!r} in root metadata")
        result[key] = value
    return result


def read_source_generation(shot_id: int) -> SourceGenerationPin:
    """Read exact root ``zarr.json`` bytes outside simplecache and pin them."""
    if not isinstance(shot_id, int) or isinstance(shot_id, bool) or shot_id <= 0:
        raise SourceGenerationError("shot_id must be a positive integer")
    source_uri = f"s3://{BUCKET}/level2/shots/{shot_id}.zarr"
    url = f"{ENDPOINT_URL}/{BUCKET}/level2/shots/{shot_id}.zarr/zarr.json"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "SCPN-CONTROL-FAIR-MAST-acquisition/1"},
    )
    try:
        # The URL has a fixed HTTPS origin and a validated positive-integer path component.
        with urllib.request.urlopen(  # nosec B310
            request, timeout=_SOURCE_METADATA_TIMEOUT_S
        ) as response:
            raw = response.read(_MAX_ROOT_METADATA_BYTES + 1)
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
    except (OSError, urllib.error.URLError) as exc:
        raise SourceGenerationError(f"cannot read upstream root metadata for shot {shot_id}: {exc}") from exc
    if len(raw) > _MAX_ROOT_METADATA_BYTES:
        raise SourceGenerationError(
            f"upstream root metadata for shot {shot_id} exceeds {_MAX_ROOT_METADATA_BYTES} bytes"
        )
    try:
        metadata = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceGenerationError(f"invalid upstream root metadata for shot {shot_id}: {exc}") from exc
    if not isinstance(metadata, Mapping) or metadata.get("zarr_format") != 3:
        raise SourceGenerationError(f"shot {shot_id} root metadata is not Zarr format 3")
    consolidated = metadata.get("consolidated_metadata")
    if not isinstance(consolidated, Mapping) or consolidated.get("kind") != "inline":
        raise SourceGenerationError(f"shot {shot_id} root metadata is not inline consolidated metadata")
    return SourceGenerationPin(
        source_uri=source_uri,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
        etag=etag,
        last_modified=last_modified,
    )


def _new_cache_namespace(
    cache_dir: Path,
    *,
    shot_id: int,
    generated_at: str,
    retrieved_at: str,
    source_generation: SourceGenerationPin,
) -> tuple[Path, dict[str, Any]]:
    descriptor: dict[str, Any] = {
        "schema_version": CACHE_GENERATION_SCHEMA,
        "shot_id": shot_id,
        "generated_at": generated_at,
        "retrieved_at": retrieved_at,
        "source_generation_sha256": source_generation.sha256,
    }
    namespace_id = canonical_json_sha256(descriptor)
    relative_path = Path("runs") / namespace_id
    namespace = cache_dir / relative_path
    try:
        namespace.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SourceGenerationError(
            f"isolated cache namespace {relative_path.as_posix()!r} already exists; refusing cross-run cache reuse"
        ) from exc
    return namespace, {
        **descriptor,
        "namespace_id": namespace_id,
        "relative_path": relative_path.as_posix(),
        "existing_cache_reused": False,
        "pre_and_post_source_generation_match": True,
    }


def _same_source_generation(left: SourceGenerationPin, right: SourceGenerationPin) -> bool:
    """Compare immutable content identity, excluding advisory HTTP headers."""
    return left.source_uri == right.source_uri and left.sha256 == right.sha256 and left.byte_count == right.byte_count


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
        "bphi_rmag",
        "bvac_rmag",
        "minor_radius",
        "magnetic_axis_r",
        "magnetic_axis_z",
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


def mirror_shot(
    fs: Any,
    shot_id: int,
    *,
    open_group: GroupOpener = _open_group,
    metadata_out: dict[str, dict[str, Any]] | None = None,
) -> dict[str, NDArray[Any]]:
    """Read selected source-resolution values and optional source metadata."""
    payload: dict[str, NDArray[Any]] = {}
    for group, variables in GROUP_VARIABLES.items():
        dataset = open_group(fs, shot_id, group)
        for variable in variables:
            if variable in dataset.variables:
                archive_key = f"{group}.{variable}"
                source_array = dataset[variable]
                payload[archive_key] = np.asarray(source_array.values)
                if metadata_out is not None:
                    metadata_out[archive_key] = _source_array_metadata(source_array)
    if not any(key.startswith("magnetics.b_field_tor_probe_saddle_field") for key in payload):
        raise ValueError(f"shot {shot_id}: no toroidal saddle array present.")
    return payload


def _json_metadata_value(value: Any) -> Any:
    """Convert source metadata to deterministic JSON without silent stringification."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"non_finite_float": str(value)}
    if isinstance(value, np.generic):
        return _json_metadata_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_metadata_value(item) for item in value.tolist()]
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _json_metadata_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_metadata_value(item) for item in value]
    raise TypeError(f"unsupported source metadata type: {type(value).__module__}.{type(value).__qualname__}")


def _source_array_metadata(source_array: Any) -> dict[str, Any]:
    """Capture xarray structure without inventing absent physical metadata."""
    dimensions = [str(dimension) for dimension in getattr(source_array, "dims", ())]
    attributes = _json_metadata_value(dict(getattr(source_array, "attrs", {})))
    units = attributes.get("units") if isinstance(attributes.get("units"), str) else None
    time_dimensions = [dimension for dimension in dimensions if "time" in dimension.casefold()]
    chunks = getattr(source_array, "chunks", None)
    return {
        "dimensions": dimensions,
        "units": units,
        "timebase": {"kind": "source_dimension", "dimensions": time_dimensions} if time_dimensions else None,
        "source_attributes": attributes,
        "source_chunks": [list(chunk_sizes) for chunk_sizes in chunks] if chunks is not None else None,
        "metadata_status": "source_xarray",
    }


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
    read_generation: GenerationReader | None = None,
) -> dict[str, Any]:
    """Mirror every shot to ``out_dir`` and return a schema-versioned manifest."""
    if not generated_at.strip() or not retrieved_at.strip():
        raise ValueError("generated_at and retrieved_at must be non-empty reproducibility labels")
    make_fs = make_fs if make_fs is not None else make_filesystem
    open_group = open_group if open_group is not None else _open_group
    read_generation = read_generation if read_generation is not None else read_source_generation
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for shot_id in shot_ids:
        try:
            generation_before = read_generation(shot_id)
            namespace, cache_generation = _new_cache_namespace(
                cache_dir,
                shot_id=shot_id,
                generated_at=generated_at,
                retrieved_at=retrieved_at,
                source_generation=generation_before,
            )
            fs = make_fs(namespace)
            source_metadata: dict[str, dict[str, Any]] = {}
            payload = mirror_shot(fs, shot_id, open_group=open_group, metadata_out=source_metadata)
            generation_after = read_generation(shot_id)
            if not _same_source_generation(generation_before, generation_after):
                raise SourceGenerationError(f"upstream root metadata changed while shot {shot_id} was being acquired")
        except Exception as exc:  # noqa: BLE001 - record and continue over unavailable shots
            records.append(
                {
                    "shot_id": shot_id,
                    "status": "failed",
                    "programme_class": "unknown",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        shot_path = out_dir / f"shot_{shot_id}.npz"
        np.savez_compressed(shot_path, **payload)  # type: ignore[arg-type]  # numpy savez stub: **kwds ArrayLike splat vs allow_pickle bool
        artifact = build_derived_npz_artifact(
            local_path=shot_path.name,
            artifact_path=shot_path,
            source_uri=f"s3://{BUCKET}/level2/shots/{shot_id}.zarr",
            arrays=payload,
            source_metadata=source_metadata,
            source_generation=generation_before.to_dict(),
        )
        record: dict[str, Any] = {
            "shot_id": shot_id,
            "status": "acquired",
            "programme_class": "unknown",
            "artifacts": [artifact],
            "cache_generation": cache_generation,
            "summary": _shot_summary(payload),
        }
        records.append(record)

    acquired = [r for r in records if r["status"] == "acquired"]
    failed = [r for r in records if r["status"] == "failed"]
    status = "empty" if not acquired else ("partial" if failed else "complete")
    manifest: dict[str, Any] = {
        "schema_version": SOURCE_OBJECT_MANIFEST_SCHEMA,
        "manifest_kind": "source_object_inventory",
        "machine": "MAST",
        "campaign": "FAIR-MAST level2 disruption material",
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
        "licence_spdx": FAIR_MAST_LICENCE,
        **fair_mast_provenance(),
        "fidelity": {
            "sample_values": "selected source-resolution values; no resampling",
            "native_source": "remote FAIR-MAST Zarr v3",
            "local_cache": "derived NPZ; not a native/raw object",
            "source_hierarchy": "preserved in manifest, flattened in NPZ archive keys",
            "source_metadata": "preserved in manifest when exposed by xarray",
            "source_chunking": "recorded when exposed; not preserved in NPZ",
            "source_generation": "exact root zarr.json bytes checked before and after acquisition",
        },
        "cache_policy": {
            "schema_version": CACHE_GENERATION_SCHEMA,
            "strategy": "unique empty namespace per shot and acquisition label",
            "persistent_cross_run_reuse": False,
            "generation_identity": SOURCE_GENERATION_DIGEST_KIND,
            "pre_and_post_generation_check": True,
        },
        "group_variables": {group: list(variables) for group, variables in GROUP_VARIABLES.items()},
        "label_policy": (
            "labels not assigned; DEFUSE HDF5 labels are HTTP 403 and its shot ids do "
            "not intersect the level2 range, so consumers derive the Ip current-quench label"
        ),
        "retrieved_at": retrieved_at,
        "n_acquired": len(acquired),
        "n_requested": len(shot_ids),
        "total_bytes": sum(int(r["artifacts"][0]["bytes"]) for r in acquired),
        "shots": records,
        "generated_at": generated_at,
    }
    finalised = finalise_source_object_manifest(manifest)
    validate_source_object_manifest(finalised, artifact_root=out_dir)
    return finalised


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
    parser.add_argument("--generated-at", type=str, required=True, help="Fixed UTC timestamp label.")
    parser.add_argument("--retrieved-at", type=str, required=True, help="Acquisition timestamp (ISO 8601).")
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
