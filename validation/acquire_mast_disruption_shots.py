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

Labels are deliberately not assigned here. The manifest retains the historical
DEFUSE access/shot-range policy; this routine does not establish current DEFUSE
availability or independent labels. Consumers derive the Ip current-quench label.
Heavy arrays stay off any code repository under a
shared datasets root. Requires the optional FAIR-MAST stack (``zarr``, ``s3fs``,
``xarray``, ``fsspec``) and network access; it is an out-of-band tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from scpn_control._npz import save_npz_arrays
from validation.fair_mast_source_policy import FAIR_MAST_LICENCE as FAIR_MAST_LICENCE
from validation.fair_mast_source_policy import fair_mast_provenance as fair_mast_provenance
from validation.mast_replay_contracts import (
    _acquisition_arrays,
    _acquisition_report,
    _acquisition_selection,
    _acquisition_source,
)
from validation.mast_replay_contracts._acquisition_arrays import GROUP_VARIABLES as GROUP_VARIABLES
from validation.mast_replay_contracts._acquisition_arrays import _json_metadata_value as _json_metadata_value
from validation.mast_replay_contracts._acquisition_arrays import _open_group as _open_group
from validation.mast_replay_contracts._acquisition_arrays import _shot_summary as _shot_summary
from validation.mast_replay_contracts._acquisition_arrays import _source_array_metadata as _source_array_metadata
from validation.mast_replay_contracts._acquisition_arrays import mirror_shot as mirror_shot
from validation.mast_replay_contracts._acquisition_report import acquisition_report
from validation.mast_replay_contracts._acquisition_selection import MAX_REQUESTED_SHOTS as MAX_REQUESTED_SHOTS
from validation.mast_replay_contracts._acquisition_selection import parse_shots as parse_shots
from validation.mast_replay_contracts._acquisition_source import BUCKET as BUCKET
from validation.mast_replay_contracts._acquisition_source import CACHE_GENERATION_SCHEMA as CACHE_GENERATION_SCHEMA
from validation.mast_replay_contracts._acquisition_source import ENDPOINT_URL as ENDPOINT_URL
from validation.mast_replay_contracts._acquisition_source import SourceGenerationError as SourceGenerationError
from validation.mast_replay_contracts._acquisition_source import SourceGenerationPin as SourceGenerationPin
from validation.mast_replay_contracts._acquisition_source import _new_cache_namespace as _new_cache_namespace
from validation.mast_replay_contracts._acquisition_source import _same_source_generation as _same_source_generation
from validation.mast_replay_contracts._acquisition_source import decode_source_generation as decode_source_generation
from validation.mast_replay_contracts._acquisition_source import read_source_generation as read_source_generation
from validation.mast_replay_contracts._inputs import shot_identity
from validation.mast_source_object_manifest import SOURCE_GENERATION_DIGEST_KIND as SOURCE_GENERATION_DIGEST_KIND
from validation.mast_source_object_manifest import SOURCE_GENERATION_SCHEMA as SOURCE_GENERATION_SCHEMA
from validation.mast_source_object_manifest import (
    SOURCE_OBJECT_MANIFEST_SCHEMA,
    build_derived_npz_artifact,
    finalise_source_object_manifest,
    validate_source_object_manifest,
)
from validation.report_output_paths import checked_report_destination

FilesystemFactory = Callable[[Path], Any]
GroupOpener = Callable[[Any, int, str], Any]

MANIFEST_SCHEMA = SOURCE_OBJECT_MANIFEST_SCHEMA


GenerationReader = Callable[[int], SourceGenerationPin]


def make_filesystem(cache_dir: Path) -> Any:
    """Create a simplecache filesystem targeting anonymous FAIR-MAST S3.

    Parameters
    ----------
    cache_dir : Path
        Directory created as needed; fsspec stores downloaded cache objects here.

    Returns
    -------
    fsspec filesystem
        S3-backed simplecache with the fixed HTTPS endpoint and no shared S3
        instance cache. Filesystem/connection lifecycle follows fsspec.

    Raises
    ------
    ImportError
        fsspec or its optional S3 provider is unavailable.
    OSError, ValueError
        Cache creation or native filesystem configuration fails.

    Notes
    -----
    This API configures a filesystem; it does not establish source availability,
    Zarr-v3 support, metadata validity or original acquisition provenance.
    """
    import fsspec

    cache_dir.mkdir(parents=True, exist_ok=True)
    return fsspec.filesystem(
        "simplecache",
        cache_storage=str(cache_dir),
        target_protocol="s3",
        target_options={"anon": True, "endpoint_url": ENDPOINT_URL, "skip_instance_cache": True},
    )


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
    """Acquire source-resolution signals into compressed per-shot NPZ mirrors.

    Parameters
    ----------
    shot_ids
        Nonempty sequence of at most 100000 unique positive int64 identities,
        copied/checked in the supplied order before filesystem mutation. This
        routine assigns no disruption labels or programme classifications.
    out_dir
        Output directory, created as needed. Successfully acquired shots write
        ``shot_<id>.npz`` directly and overwrite an existing same-shot archive.
    cache_dir
        Parent of unique empty per-shot cache namespaces; namespaces may not
        be reused across acquisition labels.
    generated_at, retrieved_at
        Nonempty reproducibility labels retained in cache and manifest records,
        not live-clock or independent source-validity evidence.
    make_fs
        Optional filesystem factory for each cache namespace. The default
        uses anonymous FAIR-MAST S3 access through the native cache filesystem.
    open_group
        Optional opener for the selected xarray groups. The default opens
        consolidated FAIR-MAST Zarr groups on the filesystem.
    read_generation
        Optional source-generation reader. Exact root-metadata identity is
        compared before and after each shot; the default reads uncached remote
        ``zarr.json`` bytes.

    Returns
    -------
    dict
        Finalised source-object manifest with complete, partial or empty status,
        per-shot source/value/file bindings and retained source metadata. Arrays
        keep native sample resolution and flattened ``<group>.<variable>`` names;
        the compressed NPZ is a derived container, not the native Zarr source.

    Raises
    ------
    ValueError
        If shot selection, reproducibility labels or final artifact/manifest bindings are
        invalid. Source-generation/read failures are recorded per shot and
        acquisition continues; post-read export/manifest failures propagate.
    OSError
        If output creation or writing fails; direct publication may leave a
        partial archive.

    Notes
    -----
    Root identity is checked before/after each shot; root equality is not a
    chunk snapshot or independent provenance proof. Supplied native adapters are
    caller-controlled declarations. Selected arrays reject object dtype and
    require a nonempty 2D saddle array before NPZ publication; no sample resampling
    or numerical calibration is performed. A selected archive cannot alias these
    acquisition source owners; path checks do not prevent concurrent
    replacement or protect every dependency. Existing unrelated shot files are
    overwritten directly. Failed-shot records retain authored source-generation
    refusals; other caught failures use a fixed sentence without exception text.
    Failed-shot records and isolated caches remain; later
    export/manifest failures can leave earlier files and namespaces in place.
    """
    if (
        not isinstance(generated_at, str)
        or not isinstance(retrieved_at, str)
        or not generated_at.strip()
        or not retrieved_at.strip()
    ):
        raise ValueError("generated_at and retrieved_at must be non-empty reproducibility labels")
    requested = [shot_identity(value) for value in shot_ids]
    if not requested or len(requested) > MAX_REQUESTED_SHOTS:
        raise ValueError("shot_ids must contain between 1 and 100000 shot identities")
    if len(set(requested)) != len(requested):
        raise ValueError("shot_ids must be unique")
    source_files = [
        Path(__file__),
        Path(_acquisition_arrays.__file__),
        Path(_acquisition_source.__file__),
        Path(_acquisition_report.__file__),
        Path(_acquisition_selection.__file__),
    ]
    for shot_id in requested:
        checked_report_destination(out_dir / f"shot_{shot_id}.npz", inputs=source_files)
    make_fs = make_fs if make_fs is not None else make_filesystem
    open_group = open_group if open_group is not None else _open_group
    read_generation = read_generation if read_generation is not None else read_source_generation
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for shot_id in requested:
        try:
            generation_before = read_generation(shot_id)
            if generation_before.source_uri != f"s3://{BUCKET}/level2/shots/{shot_id}.zarr":
                raise SourceGenerationError("source generation does not identify the requested shot")
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
            summary = _shot_summary(payload)
            generation_after = read_generation(shot_id)
            if not _same_source_generation(generation_before, generation_after):
                raise SourceGenerationError(f"upstream root metadata changed while shot {shot_id} was being acquired")
        except Exception as exc:  # noqa: BLE001 - record and continue over unavailable shots
            records.append(
                {
                    "shot_id": shot_id,
                    "status": "failed",
                    "programme_class": "unknown",
                    "error": (
                        str(exc)
                        if isinstance(exc, SourceGenerationError)
                        else "Could not acquire the requested MAST shot."
                    ),
                }
            )
            continue
        shot_path = out_dir / f"shot_{shot_id}.npz"
        save_npz_arrays(shot_path, payload, compressed=True, allow_pickle=False)
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
            "summary": summary,
        }
        records.append(record)

    manifest = acquisition_report(
        records, requested_count=len(requested), generated_at=generated_at, retrieved_at=retrieved_at
    )
    finalised = finalise_source_object_manifest(manifest)
    validate_source_object_manifest(finalised, artifact_root=out_dir)
    return finalised


_parse_shots = parse_shots


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse required shot selection, destinations and reproducibility labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=str, required=True, help="Shot ids/ranges, e.g. '30419-30424 29876'.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Shared datasets directory for shot_<id>.npz.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Local S3 cache directory (off-repo).")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Manifest JSON output path.")
    parser.add_argument("--generated-at", type=str, required=True, help="Fixed UTC timestamp label.")
    parser.add_argument("--retrieved-at", type=str, required=True, help="Acquisition timestamp (ISO 8601).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Acquire selected shots and write a source-object declaration report.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments, or the process arguments when None. Shots, destinations
        and both nonempty reproducibility labels are required.

    Returns
    -------
    int
        Zero for a complete acquisition report; one for a partial/empty report.
        Completion here does not admit data for scientific training.

    Raises
    ------
    SystemExit
        Native argparse help or argument handling, with status0 or2.
    ValueError, OSError, TypeError
        Invalid selection/labels/output aliases or propagated I/O/manifest errors.

    Notes
    -----
    Report output cannot alias these acquisition owners or a selected shot NPZ.
    Checks precede acquisition and are sequential, without a path lock. Report
    write is direct and may leave partial output; successful shot archives/caches
    are retained on subsequent errors. The source module command maps caught
    value/I/O/type errors to authored stderr and exit2 without interpreter text.
    A report's fixed synthetic=False field is a declaration, not authentication.
    """
    args = _parse_args(argv)
    shots = parse_shots(args.shots)
    source_files = [
        Path(__file__),
        Path(_acquisition_arrays.__file__),
        Path(_acquisition_source.__file__),
        Path(_acquisition_report.__file__),
        Path(_acquisition_selection.__file__),
    ]
    checked_report_destination(
        args.manifest_out,
        inputs=[*source_files, *[args.out_dir / f"shot_{shot_id}.npz" for shot_id in shots]],
    )
    manifest = acquire(
        shots,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        generated_at=args.generated_at,
        retrieved_at=args.retrieved_at,
    )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mb = manifest["total_bytes"] / 1e6
    print(f"acquired {manifest['n_acquired']}/{manifest['n_requested']} shots ({mb:.1f} MB) -> {args.out_dir}")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, TypeError):
        print("Could not acquire MAST disruption shots.", file=sys.stderr)
        raise SystemExit(2) from None
