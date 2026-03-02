"""Shared input validators for public API boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def require_finite_float(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def require_positive_float(name: str, value: Any) -> float:
    out = require_finite_float(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return out


def require_int(name: str, value: Any, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        if minimum is None:
            raise ValueError(f"{name} must be an integer.")
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def require_non_negative_float(name: str, value: Any) -> float:
    out = require_finite_float(name, value)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return out


def require_fraction(name: str, value: Any) -> float:
    out = require_finite_float(name, value)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return out


def require_range(
    name: str,
    value: tuple[float, float],
    *,
    min_allowed: float = -np.inf,
) -> tuple[float, float]:
    low = require_finite_float(f"{name}[0]", value[0])
    high = require_finite_float(f"{name}[1]", value[1])
    if low < min_allowed:
        raise ValueError(f"{name}[0] must be >= {min_allowed}, got {low}")
    if high <= low:
        raise ValueError(f"{name} must satisfy low < high, got ({low}, {high})")
    return (low, high)


def require_1d_array(
    name: str,
    value: Any,
    *,
    minimum_size: int = 1,
    expected_size: int | None = None,
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size < minimum_size:
        raise ValueError(f"{name} must have at least {minimum_size} samples.")
    if expected_size is not None and arr.size != expected_size:
        raise ValueError(f"{name} must have {expected_size} samples (got {arr.size}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr
