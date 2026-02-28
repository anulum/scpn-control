# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Control Package Init
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Control modules (heavy imports guarded).

Nengo, matplotlib, and torch are optional — use accessor functions
to defer their import until actually needed.
"""
from __future__ import annotations

import numpy as np


def normalize_bounds(bounds: tuple[float, float], name: str) -> tuple[float, float]:
    """Validate (lo, hi) float pair. Shared by MPC, optimal, and SOC modules."""
    lo = float(bounds[0])
    hi = float(bounds[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"{name} must be finite with lower < upper.")
    return lo, hi


def get_nengo_controller():
    """Lazy import of NengoSNNController (requires ``pip install scpn-control[nengo]``)."""
    from scpn_control.control.nengo_snn_wrapper import NengoSNNController
    return NengoSNNController


__all__ = ["get_nengo_controller"]
