# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Scaling Laws overflow guard tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for ipb98y2_tau_e overflow tau (line 240)."""

from __future__ import annotations

import pytest

from scpn_control.core.scaling_laws import ipb98y2_tau_e


class TestOverflowTau:
    def test_extreme_inputs_tau_overflow(self):
        """Extreme inputs trigger tau=inf → ValueError (line 240)."""
        with pytest.raises(ValueError, match="invalid"):
            ipb98y2_tau_e(
                Ip=1e150,
                BT=1e150,
                ne19=1e150,
                Ploss=1.0,
                R=1e150,
                kappa=1e150,
                epsilon=1e150,
            )
