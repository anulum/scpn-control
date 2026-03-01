# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Scaling Laws Edge Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for ipb98y2_tau_e invalid tau (line 240) and
ipb98y2_with_uncertainty non-finite variance (lines 323-324)."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core.scaling_laws import ipb98y2_tau_e, ipb98y2_with_uncertainty


class TestIPB98y2Invalid:
    def test_zero_power_raises(self):
        """Zero Ploss raises at validation (line 213)."""
        with pytest.raises(ValueError, match="Ploss"):
            ipb98y2_tau_e(
                Ip=15.0, BT=5.3, ne19=10.0, Ploss=0.0,
                R=6.2, kappa=1.7, epsilon=0.32,
            )

    def test_nan_power_raises(self):
        """NaN Ploss raises at validation."""
        with pytest.raises(ValueError, match="Ploss"):
            ipb98y2_tau_e(
                Ip=15.0, BT=5.3, ne19=10.0, Ploss=float("nan"),
                R=6.2, kappa=1.7, epsilon=0.32,
            )


class TestIPB98y2Uncertainty:
    def test_normal_uncertainty(self):
        """Normal case returns finite tau and sigma."""
        tau, sigma = ipb98y2_with_uncertainty(
            Ip=15.0, BT=5.3, ne19=10.0, Ploss=50.0,
            R=6.2, kappa=1.7, epsilon=0.32,
        )
        assert np.isfinite(tau) and tau > 0
        assert np.isfinite(sigma) and sigma >= 0
