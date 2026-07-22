#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Verified FAIR-MAST source artefact reader
"""Open provenance-bound FAIR-MAST NPZ artefacts through manifest v2.

The reader accepts only a validated SourceObjectManifest v2 record and exactly
one artefact of the requested kind for an acquired shot. It verifies the full
manifest, local file bytes, archive membership, dtype, shape, and array-value
digests before exposing immutable group-aware arrays to downstream binding
logic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from validation.mast_source_object_manifest import (
    DERIVED_NPZ_ARTIFACT_KIND,
    SourceObjectManifestError,
    array_value_sha256,
    file_sha256,
    require_source_object_manifest_v2,
)


class SourceArtifactReaderError(ValueError):
    """Raised when a manifest cannot yield one verified source artefact."""


@dataclass(frozen=True)
class VerifiedSourceArtifact:
    """Immutable view of one verified derived source artefact.

    Parameters
    ----------
    shot_id:
        Positive MAST shot identifier.
    artifact_kind:
        Manifest artefact discriminator.
    local_path:
        Portable path relative to the caller-supplied artefact root.
    source_uri:
        Canonical remote source-object URI.
    manifest_sha256:
        Self-digest of the validated source manifest.
    artifact_sha256:
        Byte digest of the opened NPZ file.
    parent_digest:
        Digest of the selected remote-array snapshot.
    transform_digest:
        Digest of the Zarr-selection-to-NPZ transform descriptor.
    arrays:
        Read-only arrays keyed by exact ``<group>.<array_name>`` archive keys.
    metadata:
        Read-only manifest array descriptors keyed by the same archive keys.
    """

    shot_id: int
    artifact_kind: str
    local_path: str
    source_uri: str
    manifest_sha256: str
    artifact_sha256: str
    parent_digest: str
    transform_digest: str
    arrays: Mapping[str, NDArray[Any]]
    metadata: Mapping[str, Mapping[str, Any]]

    @property
    def archive_keys(self) -> tuple[str, ...]:
        """Return the exact archive keys in deterministic order."""
        return tuple(sorted(self.arrays))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise SourceArtifactReaderError(f"duplicate JSON key {key!r} in source-object manifest")
        payload[key] = value
    return payload


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _decode_source_manifest(manifest_bytes: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except SourceArtifactReaderError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SourceArtifactReaderError(f"cannot read source-object manifest: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SourceArtifactReaderError("source-object manifest root must be an object")
    return payload


def _read_source_manifest_bytes(manifest_path: Path) -> bytes:
    try:
        return manifest_path.read_bytes()
    except OSError as exc:
        raise SourceArtifactReaderError(f"cannot read source-object manifest: {exc}") from exc


def load_verified_source_manifest(manifest_path: Path, *, artifact_root: Path) -> dict[str, Any]:
    """Load and fully validate a SourceObjectManifest v2 document.

    Parameters
    ----------
    manifest_path:
        JSON manifest produced by the FAIR-MAST acquisition surface.
    artifact_root:
        Root beneath which every declared local artefact must resolve.

    Returns
    -------
    dict[str, Any]
        A validated manifest copy.

    Raises
    ------
    SourceArtifactReaderError
        If JSON loading, schema validation, or local artefact verification fails.
    """
    payload = _decode_source_manifest(_read_source_manifest_bytes(manifest_path))
    try:
        return require_source_object_manifest_v2(payload, artifact_root=artifact_root)
    except (OSError, SourceObjectManifestError) as exc:
        raise SourceArtifactReaderError(f"source-object manifest verification failed: {exc}") from exc


def load_pinned_source_manifest(
    manifest_path: Path,
    *,
    artifact_root: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Load one exact-byte SourceObjectManifest v2 document.

    Parameters
    ----------
    manifest_path:
        JSON manifest produced by the FAIR-MAST acquisition surface.
    artifact_root:
        Root beneath which every declared local artefact must resolve.
    expected_sha256:
        Exact SHA-256 digest of the manifest bytes.

    Returns
    -------
    tuple[dict[str, Any], str]
        Fully validated manifest and its verified byte digest.

    Raises
    ------
    SourceArtifactReaderError
        If the bytes do not match, JSON is ambiguous, or v2 validation fails.
    """
    manifest_bytes = _read_source_manifest_bytes(manifest_path)
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SourceArtifactReaderError("source-object manifest SHA-256 does not match the pinned digest")
    payload = _decode_source_manifest(manifest_bytes)
    try:
        verified = require_source_object_manifest_v2(payload, artifact_root=artifact_root)
    except (OSError, SourceObjectManifestError) as exc:
        raise SourceArtifactReaderError(f"source-object manifest verification failed: {exc}") from exc
    return verified, actual_sha256


def read_verified_npz_artifact(
    manifest: Mapping[str, Any],
    *,
    artifact_root: Path,
    shot_id: int,
    artifact_kind: str = DERIVED_NPZ_ARTIFACT_KIND,
) -> VerifiedSourceArtifact:
    """Resolve and open exactly one verified NPZ artefact for an acquired shot.

    The manifest and every referenced local artefact are revalidated on each
    call. The opened arrays are then checked again against the inventory before
    they are exposed, preventing downstream code from bypassing the v2 lineage
    and member contract.

    Parameters
    ----------
    manifest:
        SourceObjectManifest v2 mapping.
    artifact_root:
        Root beneath which the artefact path must resolve.
    shot_id:
        Positive shot identifier to resolve.
    artifact_kind:
        Exact artefact discriminator; defaults to the derived NPZ cache kind.

    Returns
    -------
    VerifiedSourceArtifact
        Immutable verified arrays and lineage metadata.

    Raises
    ------
    SourceArtifactReaderError
        If the shot or artefact cardinality is invalid, or verification fails.
    """
    try:
        verified = require_source_object_manifest_v2(manifest, artifact_root=artifact_root)
    except (OSError, SourceObjectManifestError) as exc:
        raise SourceArtifactReaderError(f"source-object manifest verification failed: {exc}") from exc
    shots = verified["shots"]
    matching_shots = [shot for shot in shots if shot["shot_id"] == shot_id]
    if len(matching_shots) != 1:
        raise SourceArtifactReaderError(f"shot {shot_id} must resolve to exactly one manifest record")
    shot = matching_shots[0]
    if shot["status"] != "acquired":
        raise SourceArtifactReaderError(f"shot {shot_id} is not acquired")
    artifacts = [artifact for artifact in shot["artifacts"] if artifact["artifact_kind"] == artifact_kind]
    if len(artifacts) != 1:
        raise SourceArtifactReaderError(
            f"shot {shot_id} must contain exactly one artefact of kind {artifact_kind!r}; found {len(artifacts)}"
        )
    artifact = artifacts[0]
    resolved_root = artifact_root.resolve()
    path = (resolved_root / artifact["local_path"]).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise SourceArtifactReaderError("verified artefact path changed to escape its root") from exc
    inventory = {item["archive_key"]: item for item in artifact["arrays"]}
    arrays: dict[str, NDArray[Any]] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(inventory):
                raise SourceArtifactReaderError("verified NPZ member set changed before opening")
            for key in sorted(inventory):
                value = np.asarray(archive[key]).copy()
                metadata = inventory[key]
                if (
                    value.dtype.str != metadata["dtype"]
                    or list(value.shape) != metadata["shape"]
                    or array_value_sha256(value) != metadata["value_sha256"]
                ):
                    raise SourceArtifactReaderError(f"verified NPZ member {key!r} changed before opening")
                value.setflags(write=False)
                arrays[key] = value
    except SourceArtifactReaderError:
        raise
    except (OSError, ValueError) as exc:
        raise SourceArtifactReaderError(f"cannot open verified NPZ artefact: {exc}") from exc
    try:
        if path.stat().st_size != artifact["bytes"] or file_sha256(path) != artifact["sha256"]:
            raise SourceArtifactReaderError("verified NPZ file changed while it was opened")
    except OSError as exc:
        raise SourceArtifactReaderError(f"cannot recheck verified NPZ artefact: {exc}") from exc
    metadata = {key: _freeze_metadata(inventory[key]) for key in sorted(inventory)}
    return VerifiedSourceArtifact(
        shot_id=shot_id,
        artifact_kind=artifact_kind,
        local_path=artifact["local_path"],
        source_uri=artifact["source_uri"],
        manifest_sha256=verified["payload_sha256"],
        artifact_sha256=artifact["sha256"],
        parent_digest=artifact["parent_digest"],
        transform_digest=artifact["transform_digest"],
        arrays=MappingProxyType(arrays),
        metadata=MappingProxyType(metadata),
    )
