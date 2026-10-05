#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural turbulence reference declaration domains


"""Retain campaign/public citation presence, gyroBohm units and finite error/score domains."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_UNITS = {
    "Q_i": "gyroBohm",
    "Q_e": "gyroBohm",
    "Gamma_e": "gyroBohm",
    "input_gradients": "dimensionless",
}


def _valid_units(value: object) -> bool:
    """Require Q_i, Q_e and Gamma_e gyroBohm and dimensionless input-gradient labels without conversions."""
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


def _has_gk_campaign_reference(payload: dict[str, object]) -> bool:
    """Require nonblank campaign artifact text without URI parsing, fetching or execution."""
    value = payload.get("campaign_artifact_uri")
    return isinstance(value, str) and bool(value.strip())
