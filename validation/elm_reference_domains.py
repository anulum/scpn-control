# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — ELM reference artifact validator

"""Retain original ELM lexical URI, provenance, grid, time-window and Type-I fraction domains."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_UNITS = {
    "time": "s",
    "energy": "J",
    "density": "m^-3",
    "temperature": "eV",
    "pressure": "Pa",
    "rmp_perturbation": "dimensionless",
    "heat_flux": "MW/m^2",
}


def _artifact_uri_error(value: object) -> str | None:
    """Inspect original nonblank URI text: refuse NUL, absolute paths and parent components; retain admitted external prefixes and relative text without fetching or parsing schemes."""
    if not isinstance(value, str) or not value.strip():
        return "artifact URI must be a non-empty string"
    ref = value.strip()
    if "\x00" in ref:
        return "artifact URI must not contain NUL bytes"
    if ref.startswith(("http://", "https://", "doi:", "s3://", "gs://")):
        return None
    # The declaration is a document, not a path on this machine: POSIX rules on every platform.
    artifact_path = PurePosixPath(ref)
    if artifact_path.is_absolute():
        return "artifact URI must be relative or an admitted external reference URI"
    if any(part == ".." for part in artifact_path.parts):
        return "artifact URI must not contain traversal"
    return None


def _valid_units(value: object) -> bool:
    """Require original seven ELM/RMP unit labels without converting quantities."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Require nonblank public URL or DOI text without citation syntax validation."""
    for field in ("reference_url", "reference_doi"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _has_measured_campaign(payload: dict[str, object]) -> bool:
    """Require nonblank machine and either shot or campaign text without measurement authentication."""
    machine = payload.get("machine")
    shot_id = payload.get("shot_id")
    campaign_id = payload.get("campaign_id")
    return (
        isinstance(machine, str)
        and bool(machine.strip())
        and any(isinstance(value, str) and bool(value.strip()) for value in (shot_id, campaign_id))
    )


def _strictly_increasing_unit_grid(value: object) -> bool:
    """Require at least two finite nonboolean scalar coordinates strictly increasing in the closed unit interval."""
    if not isinstance(value, list | tuple) or len(value) < 2:
        return False
    last = -math.inf
    for item in value:
        if not _is_finite_number(item):
            return False
        current = float(item)
        if current < 0.0 or current > 1.0 or current <= last:
            return False
        last = current
    return True


def _valid_elm_fraction_range(value: object) -> bool:
    """Require two positive finite energy fractions with 0.04 <= lower <= upper <= 0.15; equal endpoints remain admitted."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    lo, hi = value
    if not _is_positive_finite(lo) or not _is_positive_finite(hi):
        return False
    return 0.04 <= float(lo) <= float(hi) <= 0.15


def _positive_ordered_pair(value: object) -> bool:
    """Require two finite nonboolean times: lower at least zero and upper strictly greater than lower, without relating separate windows."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    lo, hi = value
    if not _is_nonnegative_finite(lo) or not _is_positive_finite(hi):
        return False
    return float(hi) > float(lo)


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Require an admitted finite declared error or coupling at least zero."""
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Require an admitted finite declared error bound strictly greater than zero."""
    return _is_finite_number(value) and float(value) > 0.0


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Admit nonboolean int/float whose binary64 conversion is finite, refusing overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False
