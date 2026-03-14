# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Torax hybrid loop streak/disruption edge path tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
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

    def test_low_risk_avoids_disruption(self):
        """Consistently low risk keeps avoidance rate at 100%."""
        with patch(
            "scpn_control.control.torax_hybrid_loop._predict_disruption_risk",
            return_value=0.10,
        ):
            result = run_nstxu_torax_hybrid_campaign(
                episodes=2,
                steps_per_episode=32,
                seed=99,
            )
        assert result.disruption_avoidance_rate == 1.0

    def test_moderate_risk_no_streak(self):
        """Risk above threshold but not sustained (alternating) avoids disruption."""
        call_count = {"n": 0}

        def _alternating_risk(*args, **kwargs):
            call_count["n"] += 1
            return 0.95 if call_count["n"] % 2 == 0 else 0.50

        with patch(
            "scpn_control.control.torax_hybrid_loop._predict_disruption_risk",
            side_effect=_alternating_risk,
        ):
            result = run_nstxu_torax_hybrid_campaign(
                episodes=1,
                steps_per_episode=32,
                seed=77,
            )
        assert result.disruption_avoidance_rate == 1.0
