# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — FAIR-MAST root metadata pins and isolated cache generations
"""Bind raw root metadata and reserve content-specific acquisition caches."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from validation.mast_replay_contracts._inputs import shot_identity
from validation.mast_source_object_manifest import (
    SOURCE_GENERATION_DIGEST_KIND,
    SOURCE_GENERATION_SCHEMA,
    canonical_json_sha256,
)

ENDPOINT_URL = "https://s3.echo.stfc.ac.uk"


BUCKET = "mast"


CACHE_GENERATION_SCHEMA = "scpn-control.fair-mast-cache-generation.v1.0.0"


_MAX_ROOT_METADATA_BYTES = 16 << 20


_SOURCE_METADATA_TIMEOUT_S = 30.0


class SourceGenerationError(ValueError):
    """Authored refusal when a FAIR-MAST source generation cannot be pinned."""


@dataclass(frozen=True)
class SourceGenerationPin:
    """Frozen declaration of a FAIR-MAST root-metadata content identity.

    Parameters
    ----------
    source_uri : str
        Exact ``s3://mast/level2/shots/<positive int64>.zarr`` spelling.
    sha256 : str
        Lowercase SHA-256 hex digest of the root metadata bytes.
    byte_count : int
        Positive Python integer byte count, at most 16 MiB; booleans refuse.
    etag, last_modified : str or None
        Advisory HTTP header values. They are not part of generation equality.

    Raises
    ------
    ValueError
        A declared identity field has an invalid spelling, type or domain.

    Notes
    -----
    Construction checks declarations and reads no source. A caller-created pin
    does not authenticate a remote object or establish that its hash is true.
    """

    source_uri: str
    sha256: str
    byte_count: int
    etag: str | None
    last_modified: str | None

    def __post_init__(self) -> None:
        """Check declared fields before they can reserve a cache namespace."""
        match = (
            re.fullmatch(r"s3://mast/level2/shots/([1-9][0-9]*)\.zarr", self.source_uri)
            if isinstance(self.source_uri, str)
            else None
        )
        if match is None:
            raise SourceGenerationError("source_uri must identify a positive FAIR-MAST shot")
        shot_identity(int(match.group(1)))
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise SourceGenerationError("sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.byte_count, int)
            or isinstance(self.byte_count, bool)
            or not 0 < self.byte_count <= _MAX_ROOT_METADATA_BYTES
        ):
            raise SourceGenerationError("byte_count must be a positive integer at most 16 MiB")
        if any(value is not None and not isinstance(value, str) for value in (self.etag, self.last_modified)):
            raise SourceGenerationError("advisory source headers must be strings or None")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh v1 declaration with byte count, SHA-256 and headers.

        Returns
        -------
        dict
            Manifest source-generation fields, fixed Zarr-v3 inline-metadata
            interpretation and advisory headers. No source I/O is performed.
        """
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


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decode a JSON object while refusing duplicate keys at every nesting level."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceGenerationError(f"duplicate JSON key {key!r} in root metadata")
        result[key] = value
    return result


def read_source_generation(shot_id: int) -> SourceGenerationPin:
    """Hash bounded uncached root metadata from the fixed public HTTPS origin.

    Parameters
    ----------
    shot_id : int
        Positive Python int64 shot identity; booleans and other types refuse.

    Returns
    -------
    SourceGenerationPin
        Exact raw-byte SHA-256/count and optional HTTP headers. Root metadata
        must be UTF-8 JSON with unique keys, integer Zarr version 3 and inline
        consolidated metadata. This is not validation of every Zarr node.

    Raises
    ------
    SourceGenerationError
        Identity, transport, 16 MiB bound, JSON or selected metadata checks fail.

    Notes
    -----
    Each call performs a new urllib HTTPS read with a 30-second socket timeout,
    outside simplecache. Redirect handling follows urllib. Two equal root hashes
    do not prove immutable chunks, original acquisition or source authentication.
    """
    if not isinstance(shot_id, int) or isinstance(shot_id, bool) or shot_id <= 0:
        raise SourceGenerationError("shot_id must be a positive integer")
    try:
        shot_identity(shot_id)
    except ValueError as exc:
        raise SourceGenerationError("shot_id must be a positive int64 identity") from exc
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
        raise SourceGenerationError(f"cannot read upstream root metadata for shot {shot_id}") from exc
    return decode_source_generation(shot_id, raw, etag=etag, last_modified=last_modified)


def decode_source_generation(
    shot_id: int, raw: bytes, *, etag: str | None = None, last_modified: str | None = None
) -> SourceGenerationPin:
    """Decode and hash a captured root-metadata byte snapshot without network I/O.

    Parameters
    ----------
    shot_id : int
        Positive Python int64 identity used for the declared FAIR-MAST URI.
    raw : bytes
        Exact UTF-8 root JSON snapshot, at most 16 MiB. The bytes are not rewritten.
    etag, last_modified : str or None
        Advisory source headers retained in the result, outside content equality.

    Returns
    -------
    SourceGenerationPin
        Checked raw-byte identity and selected Zarr-v3 inline-metadata fields.

    Raises
    ------
    SourceGenerationError
        Identity/byte type/bound, JSON or selected metadata checks fail.

    Notes
    -----
    This supports offline replay of captured root bytes and is the exact decoder
    called by the uncached HTTPS reader. Supplied bytes/headers are declarations;
    the function authenticates no source or chunk inventory.
    """
    if not isinstance(shot_id, int) or isinstance(shot_id, bool) or shot_id <= 0:
        raise SourceGenerationError("shot_id must be a positive integer")
    try:
        shot_identity(shot_id)
    except ValueError as exc:
        raise SourceGenerationError("shot_id must be a positive int64 identity") from exc
    if not isinstance(raw, bytes):
        raise SourceGenerationError("raw root metadata must be bytes")
    source_uri = f"s3://{BUCKET}/level2/shots/{shot_id}.zarr"
    if len(raw) > _MAX_ROOT_METADATA_BYTES:
        raise SourceGenerationError(
            f"upstream root metadata for shot {shot_id} exceeds {_MAX_ROOT_METADATA_BYTES} bytes"
        )
    try:
        metadata = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_nonfinite_root_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceGenerationError(f"invalid upstream root metadata for shot {shot_id}") from exc
    if (
        not isinstance(metadata, Mapping)
        or type(metadata.get("zarr_format")) is not int
        or metadata.get("zarr_format") != 3
    ):
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


def _nonfinite_root_constant(_value: str) -> NoReturn:
    """Refuse JSON extensions that cannot represent finite metadata numbers."""
    raise SourceGenerationError("nonfinite JSON numeric literals are forbidden in root metadata")


def _new_cache_namespace(
    cache_dir: Path,
    *,
    shot_id: int,
    generated_at: str,
    retrieved_at: str,
    source_generation: SourceGenerationPin,
) -> tuple[Path, dict[str, Any]]:
    """Reserve a fresh cache directory and return its content-bound declaration."""
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
