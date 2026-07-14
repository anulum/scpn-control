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

import numpy as np
from numpy.typing import ArrayLike

NpzPath = str | PathLike[str]


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


__all__ = ["save_npz_arrays"]
