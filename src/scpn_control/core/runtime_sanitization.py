# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime State Sanitization

"""Runtime numerical hardening for the integrated transport solver.

Stateless helper extracted from the integrated transport solver: replace any
non-finite entries of a profile with a caller-supplied fallback and clamp the
result to optional physical lower/upper bounds. Used each transport step to keep
temperatures, densities, and diffusivities finite and bounded so a single bad
cell cannot propagate NaNs through the solve.
"""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

__all__ = ["sanitize_with_fallback"]


def sanitize_with_fallback(
    arr: AnyFloatArray,
    fallback: AnyFloatArray,
    *,
    floor: float | None = None,
    ceil: float | None = None,
) -> tuple[FloatArray, int]:
    """Replace non-finite entries with *fallback* and enforce optional bounds.

    Parameters
    ----------
    arr : array
        Profile to harden; not mutated (a copy is returned).
    fallback : array
        Replacement values, taken element-wise where *arr* is non-finite.
    floor : float, optional
        Lower clamp applied after replacement.
    ceil : float, optional
        Upper clamp applied after replacement.

    Returns
    -------
    out : array
        The hardened profile.
    recovered : int
        Number of non-finite entries that were replaced.
    """
    out = np.asarray(arr, dtype=np.float64).copy()
    fb = np.asarray(fallback, dtype=np.float64)
    bad = ~np.isfinite(out)
    recovered = int(np.count_nonzero(bad))
    if recovered > 0:
        out[bad] = fb[bad]
    if floor is not None:
        np.maximum(out, floor, out=out)
    if ceil is not None:
        np.minimum(out, ceil, out=out)
    return out, recovered
