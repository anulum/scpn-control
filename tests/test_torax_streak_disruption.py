# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Torax hybrid loop streak/disruption edge path tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for streak_high_risk >= 3 disruption break (213, 217-218)."""

from __future__ import annotations

from unittest.mock import patch

from scpn_control.control.torax_hybrid_loop import run_nstxu_torax_hybrid_campaign


class TestStreakHighRiskDisruption:
    def test_sustained_high_risk_triggers_disruption(self):
        """3 consecutive risk > 0.93 triggers disruption break (line 213, 216-218)."""
        with patch(
            "scpn_control.control.torax_hybrid_loop._predict_disruption_risk",
            return_value=0.99,
        ):
            result = run_nstxu_torax_hybrid_campaign(
                episodes=2,
                steps_per_episode=32,
                seed=42,
            )

        assert result.disruption_avoidance_rate < 1.0
