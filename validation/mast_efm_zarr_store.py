# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Captured original MAST Zarr byte custody.
"""Decode original Zarr from the same captured bytes that supply its digest.

Captures are sequential per-file reads, not a coherent directory transaction or
measurement authentication. Owners must control concurrent directory changes.
Required observation chunks must exist; uninitialised fill is not a measurement.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from numpy.typing import NDArray

from validation.mast_efm_reference_arrays import extract_reference_arrays
from validation.mast_efm_reference_contracts import REQUIRED_EFM_VARIABLES, json_sha256


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse duplicate metadata keys before interpretation."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate Zarr metadata key: {key}")
        result[key] = value
    return result


def consolidated_metadata(captured: bytes) -> dict[str, Any]:
    """Parse finite unique-key consolidated format1 metadata for a v2 group."""
    payload = json.loads(captured.decode("utf-8"), object_pairs_hook=_unique)
    json.dumps(payload, allow_nan=False)
    if (
        not isinstance(payload, dict)
        or type(payload.get("zarr_consolidated_format")) is not int
        or payload["zarr_consolidated_format"] != 1
    ):
        raise ValueError("original source must declare consolidated Zarr format1")
    metadata = payload.get("metadata")
    group = metadata.get(".zgroup") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or not isinstance(group, dict)
        or type(group.get("zarr_format")) is not int
        or group != {"zarr_format": 2}
    ):
        raise ValueError("original source must declare consolidated Zarr v2 group metadata")
    return metadata


@dataclass(frozen=True)
class CapturedZarrStore:
    """Retain private immutable bytes and derive every manifest entry from them."""

    files: Mapping[str, bytes]

    def __post_init__(self) -> None:
        """Copy the mapping and freeze its keys; bytes themselves are immutable."""
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))

    def manifest(self) -> dict[str, Any]:
        """Return sorted file hashes/sizes and their canonical manifest digest."""
        files = [
            {"path": key, "sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
            for key, value in sorted(self.files.items())
        ]
        return {
            "files": files,
            "file_count": len(files),
            "metadata_sha256": hashlib.sha256(self.files[".zmetadata"]).hexdigest(),
            "snapshot_sha256": json_sha256(files),
        }


def capture_zarr_store(path: Path) -> CapturedZarrStore:
    """Capture local group files within the resolved root without reopening them for decoding.

    Symbolic links and nonregular entries refuse; the source is an operator-owned
    directory. No directory consistency or path-race immunity is asserted.
    Only digests, sizes and relative names leave the private captured mapping.

    >>> capture_zarr_store(Path(__file__))
    Traceback (most recent call last):
        ...
    FileNotFoundError: consolidated Zarr metadata is missing: ...
    """
    if not (path / ".zmetadata").is_file():
        raise FileNotFoundError(f"consolidated Zarr metadata is missing: {path / '.zmetadata'}")
    root = path.resolve()
    files: dict[str, bytes] = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError("original Zarr source must not contain symbolic links")
        if item.is_dir():
            continue
        if not item.is_file() or not item.resolve().is_relative_to(root):
            raise ValueError("original Zarr source must contain only local regular files")
        files[item.relative_to(root).as_posix()] = item.read_bytes()
    consolidated_metadata(files[".zmetadata"])
    return CapturedZarrStore(files)


def _require_observation_chunks(snapshot: CapturedZarrStore, metadata: dict[str, Any]) -> None:
    """Require actual chunks for required variables and the flux/time coordinate axes."""
    attrs = metadata.get("psirz/.zattrs", {})
    dims = attrs.get("_ARRAY_DIMENSIONS", []) if isinstance(attrs, dict) else []
    names = set(REQUIRED_EFM_VARIABLES) | {"time"}
    if isinstance(dims, list):
        names.update(name for name in dims if isinstance(name, str))
    for name in sorted(names):
        declaration = metadata.get(name + "/.zarray")
        if declaration is None:
            continue
        if not isinstance(declaration, dict):
            raise ValueError(f"invalid required Zarr array declaration: {name}")
        shape, chunks = declaration.get("shape"), declaration.get("chunks")
        if (
            not isinstance(shape, list)
            or not isinstance(chunks, list)
            or len(shape) != len(chunks)
            or any(type(size) is not int or size < 0 for size in shape)
            or any(type(size) is not int or size <= 0 for size in chunks)
        ):
            raise ValueError(f"invalid required Zarr shape/chunks: {name}")
        separator = declaration.get("dimension_separator", ".")
        if separator not in {".", "/"}:
            raise ValueError(f"invalid required Zarr chunk separator: {name}")
        axes = [range((size + chunk - 1) // chunk) for size, chunk in zip(shape, chunks, strict=True)]
        for index in itertools.product(*axes):
            key = name + "/" + (separator.join(str(value) for value in index) if index else "0")
            if key not in snapshot.files:
                raise ValueError(f"required Zarr dimension chunk is missing: {key}")


def read_reference_store(
    snapshot: CapturedZarrStore, *, shot_id: int, max_times: int | None = None
) -> dict[str, NDArray[Any]]:
    """Decode and extract actual observations from the privately captured Zarr bytes.

    Dictionary storage is the actual Zarr backend, not a dataset substitute. It
    cannot reopen the original filesystem path after the manifest is computed.
    """
    import xarray as xr

    metadata = consolidated_metadata(snapshot.files[".zmetadata"])
    _require_observation_chunks(snapshot, metadata)
    with xr.open_zarr(dict(snapshot.files), consolidated=True, chunks=None) as ds:
        return extract_reference_arrays(ds, shot_id=shot_id, max_times=max_times)
