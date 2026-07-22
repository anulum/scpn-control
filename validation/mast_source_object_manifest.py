#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST source-object manifest contract
"""Build, validate, and migrate FAIR-MAST source-object manifests.

Version 2 separates a remote native object, an exact selected-array snapshot,
and a derived local cache.  A digest of array values and source metadata binds
the remote selection without pretending to be a byte digest of the entire Zarr
prefix.  Derived NPZ files additionally carry their own byte digest and a
digest of the declared transform.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from validation.fair_mast_source_policy import (
    FAIR_MAST_CATALOG_URL,
    FAIR_MAST_LICENCE,
)

SOURCE_OBJECT_MANIFEST_SCHEMA = "scpn-control.source-object-manifest.v2.0.0"
LEGACY_MATERIAL_MANIFEST_SCHEMA = "scpn-control.mast-disruption-material.v1"
DERIVED_NPZ_ARTIFACT_KIND = "derived_npz_cache"
SOURCE_GENERATION_SCHEMA = "scpn-control.fair-mast-source-generation.v1.0.0"
SOURCE_GENERATION_DIGEST_KIND = "zarr-v3-root-metadata-sha256"
_SHA256_HEX_LENGTH = 64


class SourceObjectManifestError(ValueError):
    """Raised when a source-object manifest or referenced artefact is invalid."""


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a deterministic compact JSON representation."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_value_sha256(array: NDArray[Any]) -> str:
    """Hash an array's dtype, shape, and exact C-order value bytes."""
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise SourceObjectManifestError("object-dtype arrays are forbidden in source-object manifests")
    header = {
        "dtype": value.dtype.str,
        "order": "C",
        "shape": list(value.shape),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file without loading it wholly into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_array_inventory(
    arrays: Mapping[str, NDArray[Any]],
    *,
    source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Describe selected source arrays with exact structure and value digests."""
    if not arrays:
        raise SourceObjectManifestError("an acquired artefact must contain at least one array")
    inventory: list[dict[str, Any]] = []
    for archive_key in sorted(arrays):
        group, separator, array_name = archive_key.partition(".")
        if not separator or not group or not array_name:
            raise SourceObjectManifestError(f"array key {archive_key!r} must have '<group>.<array_name>' form")
        value = np.asarray(arrays[archive_key])
        if value.dtype.hasobject:
            raise SourceObjectManifestError(f"array {archive_key!r} has forbidden object dtype")
        metadata = dict(source_metadata.get(archive_key, {})) if source_metadata is not None else {}
        metadata_status = metadata.pop("metadata_status", "values_only")
        if metadata_status not in {"source_xarray", "values_only"}:
            raise SourceObjectManifestError(f"array {archive_key!r} has unsupported metadata_status")
        inventory.append(
            {
                "archive_key": archive_key,
                "group": group,
                "array_name": array_name,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "nbytes": int(value.nbytes),
                "value_sha256": array_value_sha256(value),
                "dimensions": metadata.pop("dimensions", None),
                "units": metadata.pop("units", None),
                "timebase": metadata.pop("timebase", None),
                "source_attributes": metadata.pop("source_attributes", None),
                "source_chunks": metadata.pop("source_chunks", None),
                "metadata_status": metadata_status,
                "unrepresented_metadata": metadata,
            }
        )
    return inventory


def source_hierarchy(arrays: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return the deterministic group-to-array hierarchy for an inventory."""
    groups: dict[str, list[str]] = {}
    for array in arrays:
        group = str(array["group"])
        groups.setdefault(group, []).append(str(array["array_name"]))
    return {group: sorted(names) for group, names in sorted(groups.items())}


def build_derived_npz_artifact(
    *,
    local_path: str,
    artifact_path: Path,
    source_uri: str,
    arrays: Mapping[str, NDArray[Any]],
    source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    source_generation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance-bound descriptor for one derived NPZ cache file."""
    _validate_relative_path(local_path, field="local_path")
    if not artifact_path.is_file():
        raise SourceObjectManifestError(f"derived artefact does not exist: {artifact_path}")
    generation: dict[str, Any] | None = None
    if source_generation is not None:
        _validate_source_generation(source_generation, source_uri=source_uri, field="source_generation")
        generation = dict(source_generation)
    inventory = build_array_inventory(arrays, source_metadata=source_metadata)
    snapshot_descriptor: dict[str, Any] = {
        "artifact_kind": "remote_zarr_v3_selected_arrays",
        "source_uri": source_uri,
        "arrays": inventory,
    }
    if generation is not None:
        snapshot_descriptor["source_generation"] = generation
    parent_digest = canonical_json_sha256(snapshot_descriptor)
    transform_descriptor: dict[str, Any] = {
        "name": "fair-mast-zarr-selection-to-npz",
        "version": "1.0.0",
        "operation": "numpy.savez_compressed",
        "archive_key_template": "{group}.{array_name}",
        "preserves_sample_values": True,
        "preserves_source_hierarchy_in_npz": False,
        "preserves_source_metadata_in_npz": False,
        "preserves_source_chunking_in_npz": False,
    }
    transform_digest = canonical_json_sha256(transform_descriptor)
    return {
        "artifact_kind": DERIVED_NPZ_ARTIFACT_KIND,
        "local_path": local_path,
        "media_type": "application/x-npz",
        "encoding": "numpy-npz-deflate",
        "sha256": file_sha256(artifact_path),
        "bytes": int(artifact_path.stat().st_size),
        "source_uri": source_uri,
        "source_generation": generation,
        "source_hierarchy": source_hierarchy(inventory),
        "arrays": inventory,
        "parent": {
            "digest_kind": "selected-array-snapshot-sha256",
            "sha256": parent_digest,
            "descriptor": snapshot_descriptor,
        },
        "parent_digest": parent_digest,
        "transform": {**transform_descriptor, "sha256": transform_digest},
        "transform_digest": transform_digest,
        "fidelity": {
            "sample_values": "exact selected-array values; no resampling",
            "native_zarr_bytes_preserved": False,
            "source_hierarchy_preserved_in_manifest": True,
            "source_metadata_preserved_in_manifest": source_metadata is not None,
            "source_chunking_preserved_in_npz": False,
            "source_generation_pinned": generation is not None,
        },
    }


def finalise_source_object_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a manifest copy with a deterministic self-digest."""
    manifest = dict(payload)
    manifest["payload_sha256"] = None
    manifest["payload_sha256"] = canonical_json_sha256(manifest)
    return manifest


def require_source_object_manifest_v2(
    payload: Mapping[str, Any],
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Validate and return v2, explicitly rejecting legacy v1 input."""
    schema = payload.get("schema_version")
    if schema == LEGACY_MATERIAL_MANIFEST_SCHEMA:
        raise SourceObjectManifestError(
            "legacy material manifest v1 is not accepted directly; call "
            "validation.migrate_mast_source_object_manifest.migrate_material_manifest_v1(...) first"
        )
    validate_source_object_manifest(payload, artifact_root=artifact_root)
    return dict(payload)


def validate_source_object_manifest(
    payload: Mapping[str, Any],
    *,
    artifact_root: Path | None = None,
) -> None:
    """Validate v2 structure, lineage digests, and optional local artefacts."""
    if payload.get("schema_version") != SOURCE_OBJECT_MANIFEST_SCHEMA:
        raise SourceObjectManifestError(f"schema_version must equal {SOURCE_OBJECT_MANIFEST_SCHEMA!r}")
    for field in ("machine", "campaign", "retrieved_at"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SourceObjectManifestError(f"{field} must be a non-empty string")
    if payload.get("synthetic") is not False:
        raise SourceObjectManifestError("synthetic must be false for FAIR-MAST source objects")
    if payload.get("licence_spdx") != FAIR_MAST_LICENCE or payload.get("licence") != FAIR_MAST_LICENCE:
        raise SourceObjectManifestError(f"FAIR-MAST licence must be {FAIR_MAST_LICENCE}")
    if payload.get("source_policy_url") != FAIR_MAST_CATALOG_URL:
        raise SourceObjectManifestError(f"source_policy_url must be {FAIR_MAST_CATALOG_URL}")
    citations = payload.get("citations")
    if not isinstance(payload.get("citation"), str) or not isinstance(citations, list) or not citations:
        raise SourceObjectManifestError("FAIR-MAST citation and non-empty citations list are required")
    expected_payload_digest = payload.get("payload_sha256")
    if not _is_sha256(expected_payload_digest):
        raise SourceObjectManifestError("payload_sha256 must be a lowercase SHA-256 hex digest")
    digest_payload = dict(payload)
    digest_payload["payload_sha256"] = None
    if canonical_json_sha256(digest_payload) != expected_payload_digest:
        raise SourceObjectManifestError("payload_sha256 mismatch")
    shots = payload.get("shots")
    if not isinstance(shots, list):
        raise SourceObjectManifestError("shots must be a list")
    acquired_count = 0
    total_bytes = 0
    seen_shots: set[int] = set()
    for index, shot in enumerate(shots):
        if not isinstance(shot, Mapping):
            raise SourceObjectManifestError(f"shots[{index}] must be an object")
        shot_id = shot.get("shot_id")
        if not isinstance(shot_id, int) or isinstance(shot_id, bool) or shot_id <= 0:
            raise SourceObjectManifestError(f"shots[{index}].shot_id must be a positive integer")
        if shot_id in seen_shots:
            raise SourceObjectManifestError(f"duplicate shot_id {shot_id}")
        seen_shots.add(shot_id)
        programme_class = shot.get("programme_class")
        if programme_class not in {"forced_vde", "spontaneous", "control", "aborted", "unknown"}:
            raise SourceObjectManifestError(f"shots[{index}].programme_class is unsupported")
        status = shot.get("status")
        if status == "failed":
            if not isinstance(shot.get("error"), str) or not str(shot["error"]).strip():
                raise SourceObjectManifestError(f"shots[{index}].error must explain a failed record")
            continue
        if status != "acquired":
            raise SourceObjectManifestError(f"shots[{index}].status must be 'acquired' or 'failed'")
        acquired_count += 1
        artifacts = shot.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise SourceObjectManifestError(f"shots[{index}].artifacts must be a non-empty list")
        for artifact_index, artifact in enumerate(artifacts):
            field = f"shots[{index}].artifacts[{artifact_index}]"
            total_bytes += _validate_artifact(artifact, field=field, artifact_root=artifact_root)
    if payload.get("n_requested") != len(shots):
        raise SourceObjectManifestError("n_requested does not match shots length")
    if payload.get("n_acquired") != acquired_count:
        raise SourceObjectManifestError("n_acquired does not match acquired shot records")
    if payload.get("total_bytes") != total_bytes:
        raise SourceObjectManifestError("total_bytes does not match acquired artefacts")
    failed_count = len(shots) - acquired_count
    expected_status = "empty" if acquired_count == 0 else ("partial" if failed_count else "complete")
    if payload.get("status") != expected_status:
        raise SourceObjectManifestError("status does not match acquired and failed shot records")


def _validate_artifact(artifact: Any, *, field: str, artifact_root: Path | None) -> int:
    if not isinstance(artifact, Mapping):
        raise SourceObjectManifestError(f"{field} must be an object")
    if artifact.get("artifact_kind") != DERIVED_NPZ_ARTIFACT_KIND:
        raise SourceObjectManifestError(f"{field}.artifact_kind is unsupported")
    local_path = artifact.get("local_path")
    if not isinstance(local_path, str):
        raise SourceObjectManifestError(f"{field}.local_path must be a string")
    _validate_relative_path(local_path, field=f"{field}.local_path")
    source_uri = artifact.get("source_uri")
    if not isinstance(source_uri, str) or not source_uri.startswith("s3://mast/level2/shots/"):
        raise SourceObjectManifestError(f"{field}.source_uri must identify a FAIR-MAST level2 shot")
    sha256 = artifact.get("sha256")
    if not _is_sha256(sha256):
        raise SourceObjectManifestError(f"{field}.sha256 must be a lowercase SHA-256 hex digest")
    byte_count = artifact.get("bytes")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise SourceObjectManifestError(f"{field}.bytes must be a non-negative integer")
    arrays = artifact.get("arrays")
    if not isinstance(arrays, list) or not arrays:
        raise SourceObjectManifestError(f"{field}.arrays must be a non-empty list")
    _validate_array_inventory(arrays, field=f"{field}.arrays")
    if artifact.get("source_hierarchy") != source_hierarchy(arrays):
        raise SourceObjectManifestError(f"{field}.source_hierarchy does not match arrays")
    parent = artifact.get("parent")
    if not isinstance(parent, Mapping):
        raise SourceObjectManifestError(f"{field}.parent must be an object")
    descriptor = parent.get("descriptor")
    if not isinstance(descriptor, Mapping) or descriptor.get("arrays") != arrays:
        raise SourceObjectManifestError(f"{field}.parent descriptor must bind the exact array inventory")
    if descriptor.get("source_uri") != source_uri:
        raise SourceObjectManifestError(f"{field}.parent source_uri mismatch")
    source_generation = artifact.get("source_generation")
    if source_generation is not None:
        _validate_source_generation(source_generation, source_uri=source_uri, field=f"{field}.source_generation")
    if descriptor.get("source_generation") != source_generation:
        raise SourceObjectManifestError(f"{field}.parent source_generation mismatch")
    fidelity = artifact.get("fidelity")
    if fidelity is None:
        if source_generation is not None:
            raise SourceObjectManifestError(f"{field}.fidelity must attest the source-generation pin")
    else:
        if not isinstance(fidelity, Mapping):
            raise SourceObjectManifestError(f"{field}.fidelity must be an object")
        generation_pinned = fidelity.get("source_generation_pinned")
        expected_generation_pinned = source_generation is not None
        if expected_generation_pinned and generation_pinned is not True:
            raise SourceObjectManifestError(f"{field}.fidelity source_generation_pinned mismatch")
        if not expected_generation_pinned and generation_pinned is not None and generation_pinned is not False:
            raise SourceObjectManifestError(f"{field}.fidelity source_generation_pinned mismatch")
    if parent.get("digest_kind") != "selected-array-snapshot-sha256":
        raise SourceObjectManifestError(f"{field}.parent digest_kind is unsupported")
    parent_digest = canonical_json_sha256(descriptor)
    if parent.get("sha256") != parent_digest or artifact.get("parent_digest") != parent_digest:
        raise SourceObjectManifestError(f"{field}.parent_digest mismatch")
    transform = artifact.get("transform")
    if not isinstance(transform, Mapping):
        raise SourceObjectManifestError(f"{field}.transform must be an object")
    for transform_field in (
        "preserves_sample_values",
        "preserves_source_hierarchy_in_npz",
        "preserves_source_metadata_in_npz",
        "preserves_source_chunking_in_npz",
    ):
        expected = transform_field == "preserves_sample_values"
        if transform.get(transform_field) is not expected:
            raise SourceObjectManifestError(f"{field}.transform {transform_field} must be {expected}")
    transform_descriptor = {key: value for key, value in transform.items() if key != "sha256"}
    transform_digest = canonical_json_sha256(transform_descriptor)
    if transform.get("sha256") != transform_digest or artifact.get("transform_digest") != transform_digest:
        raise SourceObjectManifestError(f"{field}.transform_digest mismatch")
    if artifact_root is not None:
        path = _resolve_artifact_path(artifact_root, local_path, field=f"{field}.local_path")
        if path.stat().st_size != byte_count:
            raise SourceObjectManifestError(f"{field}.bytes does not match the local file")
        if file_sha256(path) != sha256:
            raise SourceObjectManifestError(f"{field}.sha256 does not match the local file")
        _validate_npz_arrays(path, arrays, field=field)
    return byte_count


def _validate_source_generation(
    generation: Any,
    *,
    source_uri: str,
    field: str,
) -> None:
    if not isinstance(generation, Mapping):
        raise SourceObjectManifestError(f"{field} must be an object or null")
    if generation.get("schema_version") != SOURCE_GENERATION_SCHEMA:
        raise SourceObjectManifestError(f"{field}.schema_version must equal {SOURCE_GENERATION_SCHEMA!r}")
    if generation.get("digest_kind") != SOURCE_GENERATION_DIGEST_KIND:
        raise SourceObjectManifestError(f"{field}.digest_kind must equal {SOURCE_GENERATION_DIGEST_KIND!r}")
    if generation.get("source_uri") != source_uri:
        raise SourceObjectManifestError(f"{field}.source_uri mismatch")
    if generation.get("metadata_path") != "zarr.json":
        raise SourceObjectManifestError(f"{field}.metadata_path must equal 'zarr.json'")
    if not _is_sha256(generation.get("sha256")):
        raise SourceObjectManifestError(f"{field}.sha256 must be a lowercase SHA-256 hex digest")
    byte_count = generation.get("bytes")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
        raise SourceObjectManifestError(f"{field}.bytes must be a positive integer")
    if generation.get("zarr_format") != 3:
        raise SourceObjectManifestError(f"{field}.zarr_format must equal 3")
    if generation.get("consolidated_metadata_kind") != "inline":
        raise SourceObjectManifestError(f"{field}.consolidated_metadata_kind must equal 'inline'")
    for header in ("etag", "last_modified"):
        value = generation.get(header)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise SourceObjectManifestError(f"{field}.{header} must be a non-empty string or null")


def _validate_array_inventory(arrays: list[Any], *, field: str) -> None:
    seen: set[str] = set()
    for index, array in enumerate(arrays):
        item = f"{field}[{index}]"
        if not isinstance(array, Mapping):
            raise SourceObjectManifestError(f"{item} must be an object")
        archive_key = array.get("archive_key")
        if not isinstance(archive_key, str) or archive_key in seen:
            raise SourceObjectManifestError(f"{item}.archive_key must be a unique string")
        seen.add(archive_key)
        if archive_key != f"{array.get('group')}.{array.get('array_name')}":
            raise SourceObjectManifestError(f"{item} hierarchy does not match archive_key")
        try:
            dtype = np.dtype(array.get("dtype"))
        except TypeError as exc:
            raise SourceObjectManifestError(f"{item}.dtype is invalid") from exc
        if dtype.hasobject:
            raise SourceObjectManifestError(f"{item}.dtype must not contain Python objects")
        shape = array.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape
        ):
            raise SourceObjectManifestError(f"{item}.shape must contain non-negative integers")
        if not isinstance(array.get("nbytes"), int) or int(array["nbytes"]) < 0:
            raise SourceObjectManifestError(f"{item}.nbytes must be a non-negative integer")
        expected_nbytes = int(dtype.itemsize * np.prod(shape, dtype=np.int64))
        if array["nbytes"] != expected_nbytes:
            raise SourceObjectManifestError(f"{item}.nbytes does not match dtype and shape")
        if not _is_sha256(array.get("value_sha256")):
            raise SourceObjectManifestError(f"{item}.value_sha256 must be a lowercase SHA-256 hex digest")
        metadata_status = array.get("metadata_status")
        if metadata_status not in {"source_xarray", "values_only"}:
            raise SourceObjectManifestError(f"{item}.metadata_status is unsupported")
        dimensions = array.get("dimensions")
        if metadata_status == "source_xarray":
            if not isinstance(dimensions, list) or any(not isinstance(dimension, str) for dimension in dimensions):
                raise SourceObjectManifestError(f"{item}.dimensions must preserve source dimension names")
            if len(dimensions) != len(shape):
                raise SourceObjectManifestError(f"{item}.dimensions must match array rank")
            if not isinstance(array.get("source_attributes"), Mapping):
                raise SourceObjectManifestError(f"{item}.source_attributes must be an object")
            units = array.get("units")
            if units is not None and not isinstance(units, str):
                raise SourceObjectManifestError(f"{item}.units must be a string or null")
            timebase = array.get("timebase")
            if timebase is not None:
                if not isinstance(timebase, Mapping) or timebase.get("kind") != "source_dimension":
                    raise SourceObjectManifestError(f"{item}.timebase must be a source-dimension reference")
                time_dimensions = timebase.get("dimensions")
                if not isinstance(time_dimensions, list) or any(
                    not isinstance(dimension, str) or dimension not in dimensions for dimension in time_dimensions
                ):
                    raise SourceObjectManifestError(f"{item}.timebase dimensions must exist in dimensions")
            chunks = array.get("source_chunks")
            if chunks is not None:
                if not isinstance(chunks, list) or len(chunks) != len(shape):
                    raise SourceObjectManifestError(f"{item}.source_chunks must match array rank")
                for axis, chunk_sizes in enumerate(chunks):
                    if (
                        not isinstance(chunk_sizes, list)
                        or not chunk_sizes
                        or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in chunk_sizes)
                        or sum(chunk_sizes) != shape[axis]
                    ):
                        raise SourceObjectManifestError(f"{item}.source_chunks[{axis}] does not tile its dimension")
        elif any(
            array.get(field) is not None
            for field in ("dimensions", "units", "timebase", "source_attributes", "source_chunks")
        ):
            raise SourceObjectManifestError(f"{item} values_only metadata fields must remain null")
        if not isinstance(array.get("unrepresented_metadata"), Mapping):
            raise SourceObjectManifestError(f"{item}.unrepresented_metadata must be an object")


def _validate_npz_arrays(path: Path, arrays: list[Any], *, field: str) -> None:
    expected = {str(array["archive_key"]): array for array in arrays}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise SourceObjectManifestError(f"{field}.arrays do not match NPZ members")
        for key, metadata in expected.items():
            value = np.asarray(archive[key])
            if value.dtype.str != metadata["dtype"]:
                raise SourceObjectManifestError(f"{field} array {key!r} dtype mismatch")
            if list(value.shape) != metadata["shape"]:
                raise SourceObjectManifestError(f"{field} array {key!r} shape mismatch")
            if array_value_sha256(value) != metadata["value_sha256"]:
                raise SourceObjectManifestError(f"{field} array {key!r} value digest mismatch")


def _validate_relative_path(local_path: str, *, field: str) -> None:
    if not local_path or "\\" in local_path:
        raise SourceObjectManifestError(f"{field} must be a non-empty portable POSIX-relative path")
    candidate = PurePosixPath(local_path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise SourceObjectManifestError(f"{field} must stay beneath the artefact root")


def _resolve_artifact_path(root: Path, local_path: str, *, field: str) -> Path:
    _validate_relative_path(local_path, field=field)
    resolved_root = root.resolve()
    resolved = (resolved_root / local_path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SourceObjectManifestError(f"{field} escapes the artefact root") from exc
    if not resolved.is_file():
        raise SourceObjectManifestError(f"{field} does not resolve to a file")
    return resolved


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )
