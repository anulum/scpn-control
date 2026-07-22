#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Exact-object FAIR-MAST pilot replay gate
"""Replay a preserved FAIR-MAST Zarr pilot through the current CONTROL boundary.

The gate verifies every native object against the tracked SCPN-FUSION-CORE
provenance manifest, materialises only the currently selected arrays into an
ephemeral SourceObjectManifest-v2 NPZ, and runs the production binding and
mask-preserving alignment gate. Raw arrays are never emitted in the report.

Historical pilot downloads may contain only a subset of Level-2 groups. Their
absence is an input fact: the gate records it and requires affected bindings to
remain blocked. It never fetches, imputes, resamples, or substitutes a channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from validation.acquire_mast_disruption_shots import GROUP_VARIABLES, _source_array_metadata
from validation.fair_mast_source_policy import FAIR_MAST_LICENCE, fair_mast_provenance
from validation.mast_disruption_signal_binding import mast_level2_signal_binding_spec
from validation.mast_source_object_manifest import (
    SOURCE_OBJECT_MANIFEST_SCHEMA,
    build_derived_npz_artifact,
    canonical_json_sha256,
    file_sha256,
    finalise_source_object_manifest,
    validate_source_object_manifest,
)
from validation.verify_mast_real_object_alignment import verify_real_object_alignment

PILOT_PROVENANCE_SCHEMA = "scpn-fusion-open-disruption-data-provenance.v1"
PILOT_REPLAY_SCHEMA = "scpn-control.mast-pilot-replay.v1.0.0"
EXPECTED_MISSING_GROUPS = ("equilibrium", "interferometer")
_SOURCE_URL_PREFIX = "https://s3.echo.stfc.ac.uk/mast/level2/shots"


class MastPilotReplayError(ValueError):
    """Raised when pilot identity, object integrity, or replay truth is invalid."""


@dataclass(frozen=True)
class PilotObjectPin:
    """Exact native-object identity recovered from a FUSION pilot manifest."""

    shot_id: int
    provenance_sha256: str
    aggregate_sha256: str
    object_count: int
    total_size_bytes: int
    downloaded_groups: tuple[str, ...]
    retrieved_at: str


GroupOpener = Callable[[Path], Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise MastPilotReplayError(f"duplicate JSON key {key!r} in pilot provenance")
        payload[key] = value
    return payload


def _load_pinned_provenance(path: Path, *, expected_sha256: str) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MastPilotReplayError(f"cannot read pilot provenance: {exc}") from exc
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise MastPilotReplayError("pilot provenance SHA-256 does not match the pinned digest")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except MastPilotReplayError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MastPilotReplayError(f"cannot decode pilot provenance: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise MastPilotReplayError("pilot provenance root must be an object")
    return payload, actual_sha256


def _object_path(object_root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise MastPilotReplayError(f"unsafe pilot object path {relative_path!r}")
    root = object_root.resolve(strict=True)
    try:
        candidate = (root / Path(*pure.parts)).resolve(strict=True)
    except OSError as exc:
        raise MastPilotReplayError(f"pilot object does not resolve: {relative_path!r}") from exc
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise MastPilotReplayError(f"pilot object escapes its root or is not a file: {relative_path!r}")
    return candidate


def _hash_file_once(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise MastPilotReplayError(f"cannot hash pilot object {path}: {exc}") from exc
    return digest.hexdigest(), size


def _verify_native_objects(
    provenance: Mapping[str, Any],
    *,
    provenance_sha256: str,
    object_root: Path,
    shot_id: int,
    expected_aggregate_sha256: str,
) -> PilotObjectPin:
    if provenance.get("schema") != PILOT_PROVENANCE_SCHEMA:
        raise MastPilotReplayError(f"pilot provenance schema must equal {PILOT_PROVENANCE_SCHEMA!r}")
    dataset = provenance.get("dataset")
    if not isinstance(dataset, Mapping):
        raise MastPilotReplayError("pilot provenance dataset must be an object")
    if dataset.get("device") != "MAST" or dataset.get("shot_id") != shot_id:
        raise MastPilotReplayError("pilot dataset does not identify the requested MAST shot")
    declared_aggregate = dataset.get("download_manifest_sha256")
    if declared_aggregate != expected_aggregate_sha256:
        raise MastPilotReplayError("pilot aggregate SHA-256 does not match the pinned digest")
    files = dataset.get("files")
    if not isinstance(files, list) or not files:
        raise MastPilotReplayError("pilot dataset files must be a non-empty list")
    if dataset.get("object_count") != len(files):
        raise MastPilotReplayError("pilot object_count does not match files length")
    raw_groups = dataset.get("downloaded_groups")
    if not isinstance(raw_groups, list) or not raw_groups or not all(isinstance(group, str) for group in raw_groups):
        raise MastPilotReplayError("pilot downloaded_groups must be a non-empty string list")
    groups = tuple(cast(list[str], raw_groups))
    if len(set(groups)) != len(groups) or any(group not in GROUP_VARIABLES for group in groups):
        raise MastPilotReplayError("pilot downloaded_groups contains duplicates or unsupported groups")

    seen: set[str] = set()
    records: list[tuple[str, int, str]] = []
    total_size = 0
    for index, raw_entry in enumerate(files):
        if not isinstance(raw_entry, Mapping):
            raise MastPilotReplayError(f"pilot files[{index}] must be an object")
        relative_path = raw_entry.get("path")
        expected_sha = raw_entry.get("sha256")
        expected_size = raw_entry.get("size_bytes")
        source_url = raw_entry.get("source_url")
        if not isinstance(relative_path, str) or relative_path in seen:
            raise MastPilotReplayError(f"pilot files[{index}].path must be a unique string")
        seen.add(relative_path)
        path_parts = PurePosixPath(relative_path).parts
        expected_prefix = ("raw", f"{shot_id}.zarr")
        if path_parts[:2] != expected_prefix or len(path_parts) < 4 or path_parts[2] not in groups:
            raise MastPilotReplayError(f"pilot object path is outside its declared shot/groups: {relative_path!r}")
        remote_suffix = "/".join(path_parts[2:])
        expected_url = f"{_SOURCE_URL_PREFIX}/{shot_id}.zarr/{remote_suffix}"
        if source_url != expected_url:
            raise MastPilotReplayError(f"pilot source_url mismatch for {relative_path!r}")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise MastPilotReplayError(f"pilot files[{index}].sha256 must be a SHA-256 digest")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            raise MastPilotReplayError(f"pilot files[{index}].size_bytes must be a non-negative integer")
        actual_sha, actual_size = _hash_file_once(_object_path(object_root, relative_path))
        if actual_sha != expected_sha or actual_size != expected_size:
            raise MastPilotReplayError(f"pilot object integrity mismatch for {relative_path!r}")
        records.append((actual_sha, actual_size, relative_path))
        total_size += actual_size

    aggregate = hashlib.sha256(
        "".join(f"{sha256}:{size}:{path}\n" for sha256, size, path in sorted(records, key=lambda item: item[2])).encode(
            "utf-8"
        )
    ).hexdigest()
    if aggregate != declared_aggregate:
        raise MastPilotReplayError("recomputed pilot aggregate SHA-256 does not match provenance")
    if dataset.get("total_size_bytes") != total_size:
        raise MastPilotReplayError("pilot total_size_bytes does not match verified objects")
    retrieved_at = provenance.get("retrieved_at_utc")
    if not isinstance(retrieved_at, str) or not retrieved_at:
        raise MastPilotReplayError("pilot retrieved_at_utc must be a non-empty string")
    return PilotObjectPin(
        shot_id=shot_id,
        provenance_sha256=provenance_sha256,
        aggregate_sha256=aggregate,
        object_count=len(records),
        total_size_bytes=total_size,
        downloaded_groups=groups,
        retrieved_at=retrieved_at,
    )


def _open_local_group(path: Path) -> Any:
    import xarray as xr

    return xr.open_zarr(path, consolidated=False)


def _read_selected_arrays(
    zarr_root: Path,
    *,
    downloaded_groups: Sequence[str],
    open_group: GroupOpener,
) -> tuple[dict[str, NDArray[Any]], dict[str, dict[str, Any]], tuple[str, ...]]:
    arrays: dict[str, NDArray[Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    missing_groups = tuple(group for group in GROUP_VARIABLES if group not in downloaded_groups)
    for group in downloaded_groups:
        group_path = zarr_root / group
        if not group_path.is_dir():
            raise MastPilotReplayError(f"declared pilot group does not resolve to a directory: {group!r}")
        try:
            dataset = open_group(group_path)
        except Exception as exc:  # noqa: BLE001 - optional xarray/zarr boundary is normalised here
            raise MastPilotReplayError(f"cannot open preserved pilot group {group!r}: {exc}") from exc
        try:
            for variable in GROUP_VARIABLES[group]:
                if variable not in dataset.variables:
                    continue
                source_array = dataset[variable]
                archive_key = f"{group}.{variable}"
                arrays[archive_key] = np.asarray(source_array.values)
                metadata[archive_key] = _source_array_metadata(source_array)
        finally:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
    if "magnetics.b_field_tor_probe_saddle_field" not in arrays:
        raise MastPilotReplayError("pilot selection has no toroidal saddle array")
    return arrays, metadata, missing_groups


def _save_named_arrays(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    writer = cast(Callable[..., None], np.savez_compressed)
    writer(path, **arrays)


def _build_ephemeral_manifest(
    directory: Path,
    *,
    pin: PilotObjectPin,
    arrays: Mapping[str, NDArray[Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    missing_groups: tuple[str, ...],
) -> tuple[dict[str, Any], Path, Path]:
    artifact_path = directory / f"shot_{pin.shot_id}.npz"
    _save_named_arrays(artifact_path, arrays)
    source_uri = f"s3://mast/level2/shots/{pin.shot_id}.zarr"
    artifact = build_derived_npz_artifact(
        local_path=artifact_path.name,
        artifact_path=artifact_path,
        source_uri=source_uri,
        arrays=arrays,
        source_metadata=metadata,
        source_generation=None,
    )
    manifest = finalise_source_object_manifest(
        {
            "schema_version": SOURCE_OBJECT_MANIFEST_SCHEMA,
            "manifest_kind": "source_object_inventory",
            "machine": "MAST",
            "campaign": f"FAIR-MAST shot {pin.shot_id} forced-VDE pilot replay",
            "status": "complete",
            "synthetic": False,
            "consumers": ["SCPN-CONTROL", "SCPN-FUSION-CORE"],
            "source": {
                "bucket": "s3://mast",
                "endpoint": "https://s3.echo.stfc.ac.uk",
                "access": "anonymous",
                "format": "zarr_v3_level2",
                "path_template": "s3://mast/level2/shots/{shot_id}.zarr",
            },
            "licence_spdx": FAIR_MAST_LICENCE,
            **fair_mast_provenance(),
            "fidelity": {
                "sample_values": "selected source-resolution values; no resampling",
                "native_source": "preserved local FAIR-MAST Zarr-v3 group objects",
                "local_cache": "ephemeral derived NPZ; not a native/raw object",
                "source_hierarchy": "preserved in manifest, flattened in NPZ archive keys",
                "source_metadata": "preserved in manifest when exposed by xarray",
                "source_chunking": "recorded when exposed; not preserved in NPZ",
                "source_generation": (
                    "historical pilot lacks root zarr.json; native object-manifest digest verified separately"
                ),
            },
            "pilot_object_manifest": {
                "provenance_sha256": pin.provenance_sha256,
                "aggregate_sha256": pin.aggregate_sha256,
                "object_count": pin.object_count,
                "total_size_bytes": pin.total_size_bytes,
                "downloaded_groups": list(pin.downloaded_groups),
                "missing_groups": list(missing_groups),
            },
            "group_variables": {group: list(variables) for group, variables in GROUP_VARIABLES.items()},
            "retrieved_at": pin.retrieved_at,
            "n_acquired": 1,
            "n_requested": 1,
            "total_bytes": artifact["bytes"],
            "shots": [
                {
                    "shot_id": pin.shot_id,
                    "status": "acquired",
                    "programme_class": "forced_vde",
                    "missing_source_groups": list(missing_groups),
                    "artifacts": [artifact],
                }
            ],
            "generated_at": pin.retrieved_at,
        }
    )
    validate_source_object_manifest(manifest, artifact_root=directory)
    manifest_path = directory / "source-object-manifest-v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, manifest_path, artifact_path


def _missing_group_outcomes(
    missing_groups: tuple[str, ...], alignment_channels: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_channel = {str(channel.get("channel")): channel for channel in alignment_channels}
    bindings = mast_level2_signal_binding_spec().bindings
    outcomes: list[dict[str, Any]] = []
    for group in missing_groups:
        consumers = [
            binding.channel
            for binding in bindings
            if binding.source_key is not None and binding.source_key.startswith(f"{group}.")
        ]
        channel_outcomes = [
            {
                "channel": channel,
                "status": by_channel[channel].get("status"),
                "reason_code": by_channel[channel].get("reason_code"),
            }
            for channel in consumers
            if channel in by_channel
        ]
        if any(outcome["status"] != "not_aligned" for outcome in channel_outcomes):
            raise MastPilotReplayError(f"missing group {group!r} unexpectedly produced an aligned channel")
        outcomes.append(
            {
                "group": group,
                "status": "fail_closed" if channel_outcomes else "absent_no_current_binding_consumer",
                "channel_outcomes": channel_outcomes,
            }
        )
    return outcomes


def verify_mast_pilot_replay(
    provenance_path: Path,
    *,
    object_root: Path,
    shot_id: int,
    expected_provenance_sha256: str,
    expected_aggregate_sha256: str,
    expected_missing_groups: tuple[str, ...] = EXPECTED_MISSING_GROUPS,
    open_group: GroupOpener = _open_local_group,
) -> dict[str, Any]:
    """Verify and replay one exact preserved pilot without emitting raw arrays."""
    if not isinstance(shot_id, int) or isinstance(shot_id, bool) or shot_id <= 0:
        raise MastPilotReplayError("shot_id must be a positive integer")
    if len(set(expected_missing_groups)) != len(expected_missing_groups):
        raise MastPilotReplayError("expected_missing_groups must be unique")
    provenance, provenance_sha256 = _load_pinned_provenance(provenance_path, expected_sha256=expected_provenance_sha256)
    pin_before = _verify_native_objects(
        provenance,
        provenance_sha256=provenance_sha256,
        object_root=object_root,
        shot_id=shot_id,
        expected_aggregate_sha256=expected_aggregate_sha256,
    )
    zarr_root = object_root / "raw" / f"{shot_id}.zarr"
    arrays, metadata, missing_groups = _read_selected_arrays(
        zarr_root,
        downloaded_groups=pin_before.downloaded_groups,
        open_group=open_group,
    )
    if missing_groups != expected_missing_groups:
        raise MastPilotReplayError(
            f"missing groups {missing_groups!r} do not match expected {expected_missing_groups!r}"
        )

    with tempfile.TemporaryDirectory(prefix="scpn-control-l2f13-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        manifest, manifest_path, artifact_path = _build_ephemeral_manifest(
            temporary_root,
            pin=pin_before,
            arrays=arrays,
            metadata=metadata,
            missing_groups=missing_groups,
        )
        alignment = verify_real_object_alignment(
            manifest_path,
            manifest_format="source-object-v2",
            artifact_root=temporary_root,
            shot_id=shot_id,
            expected_manifest_sha256=file_sha256(manifest_path),
            expected_artifact_sha256=file_sha256(artifact_path),
        )
        artifact = cast(Mapping[str, Any], cast(list[Any], cast(list[Any], manifest["shots"])[0]["artifacts"])[0])

    pin_after = _verify_native_objects(
        provenance,
        provenance_sha256=provenance_sha256,
        object_root=object_root,
        shot_id=shot_id,
        expected_aggregate_sha256=expected_aggregate_sha256,
    )
    if pin_after != pin_before:
        raise MastPilotReplayError("pilot native-object identity changed during replay")

    channels = cast(list[Mapping[str, Any]], alignment["channels"])
    aligned = [channel for channel in channels if channel.get("status") == "aligned_with_validity_mask"]
    if any(not isinstance(channel.get("valid_mask_sha256"), str) for channel in aligned):
        raise MastPilotReplayError("an aligned pilot channel lacks a validity-mask digest")
    missing_outcomes = _missing_group_outcomes(missing_groups, channels)
    report: dict[str, Any] = {
        "schema_version": PILOT_REPLAY_SCHEMA,
        "status": "expected_missing_groups_fail_closed",
        "shot_id": shot_id,
        "programme_class": "forced_vde",
        "source_object_integrity_verified": True,
        "transport_interoperability_verified": alignment["transport_interoperability_verified"],
        "bound_scalar_alignment_complete": alignment["bound_scalar_alignment_complete"],
        "full_canonical_extraction_admissible": alignment["full_canonical_extraction_admissible"],
        "scientific_validity_claim_admissible": False,
        "cohort_claim_admissible": False,
        "model_training_admissible": False,
        "facility_claim_admissible": False,
        "control_admission_admissible": False,
        "pilot_provenance_sha256": pin_before.provenance_sha256,
        "pilot_download_manifest_sha256": pin_before.aggregate_sha256,
        "pilot_object_count": pin_before.object_count,
        "pilot_total_size_bytes": pin_before.total_size_bytes,
        "downloaded_groups": list(pin_before.downloaded_groups),
        "missing_groups": list(missing_groups),
        "missing_group_outcomes": missing_outcomes,
        "selected_array_count": len(arrays),
        "selected_archive_keys": sorted(arrays),
        "source_generation_pinned": False,
        "source_generation_blocker": "historical_pilot_root_zarr_json_not_preserved",
        "gate_implementation_sha256": file_sha256(Path(__file__)),
        "source_manifest_payload_sha256": manifest["payload_sha256"],
        "artifact_sha256": artifact["sha256"],
        "parent_digest": artifact["parent_digest"],
        "transform_digest": artifact["transform_digest"],
        "binding_readiness_sha256": alignment["binding_readiness_sha256"],
        "alignment_report_sha256": alignment["alignment_report_sha256"],
        "binding_spec_sha256": alignment["binding_spec_sha256"],
        "alignment_spec_sha256": alignment["alignment_spec_sha256"],
        "target_time_sha256": alignment["target_time_sha256"],
        "target_samples": alignment["target_samples"],
        "n_bound_channels_aligned": alignment["n_bound_channels_aligned"],
        "n_channels_not_aligned": alignment["n_channels_not_aligned"],
        "channels": channels,
        "payload_sha256": None,
    }
    report["payload_sha256"] = canonical_json_sha256(report)
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True, help="Tracked FUSION pilot provenance JSON.")
    parser.add_argument("--object-root", type=Path, required=True, help="Root beneath which files[].path resolves.")
    parser.add_argument("--shot-id", type=int, required=True, help="Pinned MAST shot id.")
    parser.add_argument("--expected-provenance-sha256", required=True, help="Pinned provenance-file SHA-256.")
    parser.add_argument("--expected-aggregate-sha256", required=True, help="Pinned native-object aggregate SHA-256.")
    parser.add_argument("--json-out", type=Path, required=True, help="Destination for the raw-data-free replay report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the exact-object pilot replay and write its bounded evidence report."""
    args = _parse_args(argv)
    report = verify_mast_pilot_replay(
        args.provenance,
        object_root=args.object_root,
        shot_id=args.shot_id,
        expected_provenance_sha256=args.expected_provenance_sha256,
        expected_aggregate_sha256=args.expected_aggregate_sha256,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "MAST pilot replay: "
        f"shot {report['shot_id']} native objects verified; "
        f"{report['n_bound_channels_aligned']}/11 channels aligned with masks; missing groups fail closed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
