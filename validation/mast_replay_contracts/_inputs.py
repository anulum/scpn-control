# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Shared MAST replay vector and identity contracts.
"""Check candidate channel structure without granting physical admission."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

MEASURED_CHANNELS = (
    "time_s",
    "Ip_MA",
    "BT_T",
    "beta_N",
    "q95",
    "ne_1e19",
    "n1_amp",
    "n2_amp",
    "locked_mode_amp",
    "dBdt_gauss_per_s",
    "vertical_position_m",
)


def finite_array(value: NDArray[Any], *, name: str) -> NDArray[np.float64]:
    """Convert a legacy NumPy input and refuse nonfinite values without mutation."""
    array = np.asarray(value, dtype=np.float64)
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    return array


def positive_integer(value: int, *, name: str) -> int:
    """Require a positive Python integer, excluding bool and lossy coercion."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be positive; expected a positive integer")
    return value


def shot_identity(value: object) -> int:
    """Require a positive Python shot identity representable by signed int64."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > np.iinfo(np.int64).max:
        raise ValueError("shot_id must be a positive signed-int64 integer")
    return value


def time_axis(value: NDArray[Any], *, name: str, minimum: int = 1) -> NDArray[np.float64]:
    """Require a finite 1-D, sufficiently long, strictly increasing time axis."""
    time = finite_array(value, name=name)
    if time.ndim != 1 or time.size < minimum:
        raise ValueError(f"{name} must be a 1-D time axis with at least {minimum} samples")
    if not bool(np.all(np.diff(time) > 0.0)):
        raise ValueError(f"{name} must be strictly increasing")
    return time


def channel_vectors(channels: Mapping[str, NDArray[Any]], *, shot_id: int) -> int:
    """Check eleven finite aligned vectors and a chronological nonempty time axis."""
    if any(name not in channels for name in MEASURED_CHANNELS):
        raise ValueError(f"shot {shot_id}: missing measured channels")
    time = time_axis(channels["time_s"], name="time_s")
    for name in MEASURED_CHANNELS:
        value = np.asarray(channels[name], dtype=np.float64)
        if value.ndim != 1 or value.shape != time.shape:
            raise ValueError(f"shot {shot_id}: channel {name} must be 1-D with matching samples")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"shot {shot_id}: channel {name} must be finite")
    return int(time.size)


def shot_identifiers(values: NDArray[Any], *, integral_float_compatibility: bool = False) -> list[int]:
    """Decode sorted unique positive int64 identities without truncation.

    The dataset's historical CLI also accepts a 1-D floating vector when each
    value is finite, integral and representable as a signed int64. The replay
    producer/inspector retains its stricter integer-dtype schema. Empty vectors
    are permitted for an explicitly empty candidate archive.
    """
    array = np.asarray(values)
    allowed = "iuf" if integral_float_compatibility else "iu"
    if array.ndim != 1 or array.dtype.kind not in allowed:
        raise ValueError("replay archive shot_ids must be a one-dimensional integer vector")
    identifiers = []
    for value in array:
        if not np.isfinite(value) or value != np.floor(value):
            raise ValueError("replay archive shot_ids must contain exact finite integers")
        integer = int(value)
        if integer <= 0 or integer > np.iinfo(np.int64).max:
            raise ValueError("replay archive shot_ids must be positive signed-int64 identities")
        identifiers.append(integer)
    if identifiers != sorted(set(identifiers)):
        raise ValueError("replay archive shot_ids must be unique, positive, and sorted")
    return identifiers
