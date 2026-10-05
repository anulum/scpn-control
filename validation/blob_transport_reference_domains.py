#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Blob transport declaration domains

"""Retain original lexical URI, campaign, coordinate, pair and finite numeric domains."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_UNITS = {
    "radius": "m",
    "time": "s",
    "velocity": "m/s",
    "density": "m^-3",
    "temperature": "eV",
    "magnetic_field": "T",
    "wall_flux": "m^-2 s^-1",
}


def _artifact_uri_error(value: object) -> str | None:
    """Retain original lexical NUL/absolute/parent refusal and relative/http/https/doi/s3/gs-prefix admission without parsing or fetching artifacts."""
    if not isinstance(value, str) or not value.strip():
        return "artifact URI must be a non-empty string"
    ref = value.strip()
    if "\x00" in ref:
        return "artifact URI must not contain NUL bytes"
    if ref.startswith(("http://", "https://", "doi:", "s3://", "gs://")):
        return None
    artifact_path = Path(ref)
    if artifact_path.is_absolute():
        return "artifact URI must be relative or an admitted external reference URI"
    if any(part == ".." for part in artifact_path.parts):
        return "artifact URI must not contain traversal"
    return None


def _valid_units(value: object) -> bool:
    """Require original m, s, m/s, m^-3, eV, T and wall-flux labels without conversion."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require original nonblank URL or DOI presence without citation authentication."""
    for field in ("reference_url", "reference_doi"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _has_measured_campaign(payload: dict[str, object]) -> bool:
    """Require nonblank machine and shot OR campaign identity without measured-producer verification."""
    machine = payload.get("machine")
    shot_id = payload.get("shot_id")
    campaign_id = payload.get("campaign_id")
    return (
        isinstance(machine, str)
        and bool(machine.strip())
        and any(isinstance(value, str) and bool(value.strip()) for value in (shot_id, campaign_id))
    )


def _strictly_increasing_nonnegative(value: object) -> bool:
    """Require at least two finite nonnegative strictly increasing values, refusing boolean/nonnumber/overflow conversions."""
    if not isinstance(value, list | tuple) or len(value) < 2:
        return False
    last = -math.inf
    for item in value:
        if not _is_finite_number(item):
            return False
        current = float(item)
        if current < 0.0 or current <= last:
            return False
        last = current
    return True


def _positive_ordered_pair(value: object) -> bool:
    """Require two finite ordered values with nonnegative lower and positive upper; original detector AND blob-size domains admit lower zero."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    lo, hi = value
    if not _is_nonnegative_finite(lo) or not _is_positive_finite(hi):
        return False
    return float(hi) > float(lo)


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Admit nonboolean int/float whose binary64 conversion is finite, refusing overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Require admitted finite numbers at least zero."""
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Require admitted finite numbers strictly above zero."""
    return _is_finite_number(value) and float(value) > 0.0
