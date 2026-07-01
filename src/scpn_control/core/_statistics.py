# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Small statistical helpers for runtime evidence paths.
"""Statistical helpers for runtime paths that avoid fragile optional reductions."""

from __future__ import annotations

import math
from collections.abc import Sequence


def linear_percentile(values: Sequence[float], percentile: float) -> float:
    """Return the linearly interpolated percentile of a finite sample.

    Parameters
    ----------
    values
        Non-empty finite scalar sample.
    percentile
        Percentile in the inclusive range ``[0, 100]``.

    Returns
    -------
    float
        The linearly interpolated percentile, matching NumPy's default
        percentile interpolation for one-dimensional samples.

    Raises
    ------
    ValueError
        If the sample is empty, contains a non-finite value, or the percentile
        lies outside ``[0, 100]``.

    """
    if not values:
        raise ValueError("percentile sample must not be empty.")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100].")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile sample values must be finite.")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
