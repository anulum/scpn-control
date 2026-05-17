# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Dashboard phase state
# © 1998–2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Streamlit-independent phase-sync display helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BridgeCouplingIndicators:
    """GK-to-UPDE dashboard coupling indicators."""

    p0_p1_turb_zonal: float
    p1_p4_zonal_barrier: float
    p3_p4_elm_barrier: float


def bridge_coupling_indicators(r_global: float) -> BridgeCouplingIndicators:
    """Compute dashboard GK-to-UPDE coupling indicators from global coherence."""
    if not np.isfinite(r_global):
        raise ValueError("R_global must be finite.")
    r_value = float(r_global)
    return BridgeCouplingIndicators(
        p0_p1_turb_zonal=0.5 * (1.0 + 0.5 * float(np.tanh(r_value / 0.2))),
        p1_p4_zonal_barrier=0.5 * (1.0 + 0.3 * float(np.clip(r_value, 0.0, 2.0))),
        p3_p4_elm_barrier=0.5 * (1.0 + 0.4 * (r_value - 0.5)),
    )


__all__ = ["BridgeCouplingIndicators", "bridge_coupling_indicators"]
