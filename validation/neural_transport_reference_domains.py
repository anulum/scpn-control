#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural transport reference declaration domains


"""Retain lexical artifact/executable declarations, transport units and finite error/score domains."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_UNITS = {
    "chi_i": "m^2/s",
    "chi_e": "m^2/s",
    "D_e": "m^2/s",
    "input_gradients": "dimensionless",
}


def _artifact_uri_error(value: object) -> str | None:
    """Retain relative/http/https/doi/s3/gs lexical admission and NUL/absolute/parent refusal without fetching bytes."""
    if not isinstance(value, str) or not value.strip():
        return "artifact URI must be a non-empty string"
    ref = value.strip()
    if "\x00" in ref:
        return "artifact URI must not contain NUL bytes"
    if ref.startswith(("http://", "https://", "doi:", "s3://", "gs://")):
        return None
    path = Path(ref)
    if path.is_absolute():
        return "artifact URI must be relative or an admitted external reference URI"
    if any(part == ".." for part in path.parts):
        return "artifact URI must not contain traversal"
    return None


def _valid_units(value: object) -> bool:
    """Require chi_i, chi_e and D_e m^2/s and dimensionless input-gradient labels without conversions."""
    if not isinstance(value, dict):
        return False
    return all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require nonblank URL or DOI presence without resolving or authenticating a citation."""
    for field in ("reference_url", "reference_doi"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Admit nonboolean int/float whose binary64 conversion is finite, refusing overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Require admitted finite declared errors at least zero."""
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Require admitted finite maximum-error bounds strictly above zero."""
    return _is_finite_number(value) and float(value) > 0.0


def _is_unit_interval(value: object) -> TypeGuard[int | float]:
    """Require finite score or minimum-score declarations in the inclusive interval zero to one."""
    return _is_finite_number(value) and 0.0 <= float(value) <= 1.0
