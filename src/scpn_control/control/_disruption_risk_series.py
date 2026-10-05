# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption risk-series scoring
"""Per-window fixed-weight disruption-risk scoring for replay and ROC analysis."""

from __future__ import annotations

from math import hypot

import numpy as np
from numpy.typing import NDArray

from scpn_control.control.disruption_predictor import predict_disruption_risk

# Real-shot replay convention; n=3 is a bounded approximation, not a measured channel.
_TOROIDAL_AMP_CLIP = 10.0
_N3_FROM_N2 = 0.4


def score_risk_series(
    dbdt: NDArray[np.float64],
    n1_amp: NDArray[np.float64],
    n2_amp: NDArray[np.float64],
    *,
    window_size: int,
) -> NDArray[np.float64]:
    """Return fixed-weight risk for each full preceding signal window.

    The signal window ends before the scored sample. The leading
    ``window_size`` samples have no full history and retain zero risk.
    At least one later sample must be scoreable. Toroidal n=3 is approximated
    as ``0.4 * n2``. Nonfinite channels or predictor outputs refuse.
    """
    if isinstance(window_size, (bool, np.bool_)) or not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError("window_size must be >= 1 and an integer.")
    for name, arr in (("dbdt", dbdt), ("n1_amp", n1_amp), ("n2_amp", n2_amp)):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
    n = int(dbdt.shape[0])
    if window_size >= n:
        raise ValueError("window_size must leave at least one scored sample.")
    if not n1_amp.shape[0] == n2_amp.shape[0] == n:
        raise ValueError("dbdt, n1_amp and n2_amp must share the same length.")
    for name, arr in (("dbdt", dbdt), ("n1_amp", n1_amp), ("n2_amp", n2_amp)):
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError(f"{name} must be finite.")

    n3_amp = n2_amp * _N3_FROM_N2
    risk = np.zeros(n, dtype=np.float64)
    for t in range(window_size, n):
        window = dbdt[t - window_size : t]
        toroidal = {
            "toroidal_n1_amp": float(np.clip(n1_amp[t], 0.0, _TOROIDAL_AMP_CLIP)),
            "toroidal_n2_amp": float(np.clip(n2_amp[t], 0.0, _TOROIDAL_AMP_CLIP)),
            "toroidal_n3_amp": float(np.clip(n3_amp[t], 0.0, _TOROIDAL_AMP_CLIP)),
            "toroidal_asymmetry_index": hypot(float(n1_amp[t]), float(n2_amp[t]), float(n3_amp[t])),
            "toroidal_radial_spread": float(0.02 + 0.05 * n1_amp[t]),
        }
        score = predict_disruption_risk(window, toroidal)
        if not np.isfinite(score):
            raise ValueError("predictor risk must be finite.")
        risk[t] = float(np.clip(score, 0.0, 1.0))
    return risk
