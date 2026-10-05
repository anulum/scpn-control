# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species finite scalar and drift contracts

"""GK species finite scalar and drift contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from validation.gk_species_reference_contracts import _ABS_TOLERANCE


def _numeric_scalar(value: object) -> float:
    """Return a finite nonboolean numeric scalar or raise a fixed domain refusal."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("field must be a finite numeric scalar")
    try:
        scalar = float(value)
    except OverflowError:
        raise ValueError("field must be a finite numeric scalar") from None
    if not np.isfinite(scalar):
        raise ValueError("field must be a finite numeric scalar")
    return scalar


def _object_fields_are_numeric(
    path: Path, index: int, payload: dict[object, object], fields: tuple[str, ...], errors: list[dict[str, object]]
) -> bool:
    """Check each required scalar without leaking conversion or interpreter exception text."""
    ok = True
    for field in fields:
        try:
            _numeric_scalar(payload.get(field))
        except ValueError:
            errors.append({"path": str(path), "index": index, "field": field, "error": "field must be finite numeric"})
            ok = False
    return ok


def _relative_error(actual: float, expected: float) -> float:
    """Compute original relative drift using the declared absolute tolerance floor."""
    return abs(actual - expected) / max(abs(expected), _ABS_TOLERANCE)
