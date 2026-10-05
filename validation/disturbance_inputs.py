# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disturbance inputs.

"""Validate the finite scalar and state domains of the reduced benchmark."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _finite(value: object, name: str) -> float:
    """Convert a nonboolean real scalar without accepting strings or infinity."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as a finite float") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, name: str) -> float:
    """Require a representable finite real scalar strictly above zero."""
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _count(value: object, name: str, minimum: int = 1) -> int:
    """Require an integer count without truncating floats or accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _state(value: object) -> FloatArray:
    """Copy a finite numeric two-vector of position metres and velocity metres/s."""
    array = np.asarray(value)
    if array.shape != (2,) or array.dtype.kind not in "fiu":
        raise ValueError("x0 must be a real numeric vector of shape (2,)")
    result = np.asarray(array, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("x0 must contain finite position and velocity")
    return result
