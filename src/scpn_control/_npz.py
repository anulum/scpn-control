# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Typed NPZ archive writer.
"""Typed helpers for NumPy ``.npz`` archives used by control artefacts."""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import ArrayLike

NpzPath = str | PathLike[str]

# A ``.npz`` is a zip archive; a small deflate-compressed member can expand to
# gigabytes (a decompression bomb). 512 MiB is generous for real shot arrays
# (per-shot channels are ~MB) while rejecting a bomb before it exhausts memory.
_MAX_NPZ_DECOMPRESSED_BYTES = 512 * 1024 * 1024


class NpzSizeError(ValueError):
    """Raised when an ``.npz`` archive's declared decompressed size exceeds the cap."""


def _member_name(name: str) -> str:
    """Return the zip member name for one NPZ array."""
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"invalid NPZ array name: {name!r}")
    return f"{name}.npy"


def save_npz_arrays(path: NpzPath, arrays: Mapping[str, ArrayLike]) -> None:
    """Write named numeric arrays to an uncompressed NumPy ``.npz`` archive.

    Parameters
    ----------
    path : str or PathLike[str]
        Destination archive path.
    arrays : Mapping[str, ArrayLike]
        Named arrays to store. Keys become member names in the archive and must
        be simple file-name components.

    Raises
    ------
    ValueError
        If an array name is empty, path-like, or an array requires pickling.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, value in arrays.items():
            with archive.open(_member_name(name), mode="w", force_zip64=True) as member:
                np.save(member, np.asanyarray(value), allow_pickle=False)


def load_npz_capped(path: NpzPath, *, max_decompressed_bytes: int = _MAX_NPZ_DECOMPRESSED_BYTES) -> NpzFile:
    """Load a ``.npz`` with ``allow_pickle=False`` after auditing its decompressed size.

    A ``.npz`` is a zip archive, so a small compressed file can expand to gigabytes
    (a decompression bomb). This reads the zip central-directory member sizes without
    decompressing anything and refuses if their total exceeds ``max_decompressed_bytes``
    before ``np.load`` materialises the archive, so a hostile file cannot exhaust memory.

    Parameters
    ----------
    path : str or PathLike[str]
        Archive to load.
    max_decompressed_bytes : int
        Total declared-uncompressed byte cap across all members.

    Returns
    -------
    numpy.lib.npyio.NpzFile
        The lazily-loaded archive; each member is decompressed on access, bounded by
        the audited cap.

    Raises
    ------
    NpzSizeError
        If the total declared decompressed size exceeds ``max_decompressed_bytes``.
    """
    total = 0
    with zipfile.ZipFile(Path(path)) as archive:
        for info in archive.infolist():
            total += int(info.file_size)
            if total > max_decompressed_bytes:
                raise NpzSizeError(f"NPZ decompressed size exceeds cap: {total} > {max_decompressed_bytes} bytes")
    return cast(NpzFile, np.load(path, allow_pickle=False))


__all__ = ["NpzSizeError", "load_npz_capped", "save_npz_arrays"]
