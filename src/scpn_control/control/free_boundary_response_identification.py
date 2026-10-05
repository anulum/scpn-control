# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary response identification
"""Build a candidate coil-response matrix without publishing partial columns."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from scpn_control._typing import FloatArray


def identify_coil_response(
    *,
    original_currents: FloatArray,
    current_limits: FloatArray,
    perturbation: float,
    n_observations: int,
    set_currents: Callable[[FloatArray], None],
    solve: Callable[[], object],
    observe: Callable[[], FloatArray],
) -> FloatArray:
    """Construct finite-difference response columns in a private matrix.

    Parameters
    ----------
    original_currents
        Coil currents at the start of identification.
    current_limits
        Positive absolute current bounds, one per coil.
    perturbation
        Positive current perturbation used for both directions.
    n_observations
        Expected objective-vector width.
    set_currents, solve, observe
        Callbacks for applying a current vector, solving the plant, and reading
        the effective objective vector. The caller restores state afterward.

    Returns
    -------
    FloatArray
        Fully computed response matrix. No column is published on failure.
    """
    n_coils = int(original_currents.size)
    candidate = np.zeros((n_observations, n_coils), dtype=np.float64)
    for idx in range(n_coils):
        limit = float(current_limits[idx])
        plus = original_currents.copy()
        minus = original_currents.copy()
        with np.errstate(over="ignore"):
            plus[idx] = float(np.clip(original_currents[idx] + perturbation, -limit, limit))
            minus[idx] = float(np.clip(original_currents[idx] - perturbation, -limit, limit))
            denominator = float(plus[idx] - minus[idx])
        if not np.isfinite((plus[idx], minus[idx], denominator)).all():
            raise ValueError("coil perturbation produced a nonfinite current")
        if abs(denominator) < 1e-12:
            continue

        set_currents(plus)
        solve()
        positive = np.asarray(observe(), dtype=np.float64).reshape(-1)
        set_currents(minus)
        solve()
        negative = np.asarray(observe(), dtype=np.float64).reshape(-1)
        if positive.shape != (n_observations,) or negative.shape != (n_observations,):
            raise ValueError("response observation has an unexpected objective width")
        with np.errstate(over="ignore", invalid="ignore"):
            column = (positive - negative) / denominator
        if not np.isfinite(column).all():
            raise ValueError("response observation produced a nonfinite matrix column")
        candidate[:, idx] = column
    return candidate
