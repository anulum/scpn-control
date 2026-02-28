# ──────────────────────────────────────────────────────────────────────
# SCPN Control — TORAX Hybrid Loop Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for torax_hybrid_loop: campaign runner, helpers, edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.torax_hybrid_loop import (
    ToraxHybridCampaignResult,
    ToraxPlasmaState,
    _build_hybrid_controller,
    _estimated_loop_latency_ms,
    _risk_signal,
    _torax_policy,
    _torax_step,
    run_nstxu_torax_hybrid_campaign,
)


class TestToraxPlasmaState:
    def test_frozen(self):
        s = ToraxPlasmaState(beta_n=1.85, q95=4.9, li3=0.95, w_thermal_mj=140.0)
        with pytest.raises(AttributeError):
            s.beta_n = 2.0  # type: ignore[misc]


class TestEstimatedLoopLatency:
    def test_baseline(self):
        assert _estimated_loop_latency_ms(0.0, 0.0) == pytest.approx(0.24)

    def test_increases_with_disturbance(self):
        assert _estimated_loop_latency_ms(1.0, 0.0) > _estimated_loop_latency_ms(0.0, 0.0)

    def test_increases_with_snn_corr(self):
        assert _estimated_loop_latency_ms(0.0, 1.0) > _estimated_loop_latency_ms(0.0, 0.0)


class TestRiskSignal:
    def test_low_risk_at_nominal(self):
        s = ToraxPlasmaState(beta_n=1.85, q95=4.9, li3=0.95, w_thermal_mj=140.0)
        assert _risk_signal(s, 0.0) == pytest.approx(0.40)

    def test_high_risk_at_limits(self):
        s = ToraxPlasmaState(beta_n=3.0, q95=3.0, li3=1.8, w_thermal_mj=140.0)
        assert _risk_signal(s, 1.0) > 1.0


class TestToraxPolicy:
    def test_returns_float(self):
        s = ToraxPlasmaState(beta_n=1.85, q95=4.9, li3=0.95, w_thermal_mj=140.0)
        cmd = _torax_policy(s)
        assert isinstance(cmd, float)
        assert -1.6 <= cmd <= 1.6


class TestToraxStep:
    def test_returns_state(self):
        rng = np.random.default_rng(0)
        s = ToraxPlasmaState(beta_n=1.85, q95=4.9, li3=0.95, w_thermal_mj=140.0)
        s2 = _torax_step(s, 0.0, 0.0, rng)
        assert isinstance(s2, ToraxPlasmaState)

    def test_clipped_ranges(self):
        rng = np.random.default_rng(0)
        s = ToraxPlasmaState(beta_n=1.85, q95=4.9, li3=0.95, w_thermal_mj=140.0)
        for _ in range(200):
            s = _torax_step(s, 2.0, 1.0, rng)
        assert 0.6 <= s.beta_n <= 3.2
        assert 2.8 <= s.q95 <= 7.5
        assert 0.45 <= s.li3 <= 1.8
        assert 50.0 <= s.w_thermal_mj <= 260.0


class TestBuildHybridController:
    def test_builds_successfully(self):
        ctrl = _build_hybrid_controller()
        obs = {"R_axis_m": 1.85, "Z_axis_m": 0.0}
        action = ctrl.step(obs, 0)
        assert "dI_PF3_A" in action


class TestRunCampaign:
    def test_deterministic(self):
        r1 = run_nstxu_torax_hybrid_campaign(seed=42, episodes=4, steps_per_episode=64)
        r2 = run_nstxu_torax_hybrid_campaign(seed=42, episodes=4, steps_per_episode=64)
        assert r1.disruption_avoidance_rate == r2.disruption_avoidance_rate
        assert r1.torax_parity_pct == r2.torax_parity_pct

    def test_result_fields(self):
        r = run_nstxu_torax_hybrid_campaign(seed=0, episodes=4, steps_per_episode=64)
        assert isinstance(r, ToraxHybridCampaignResult)
        assert r.episodes == 4
        assert r.steps_per_episode == 64
        assert 0.0 <= r.disruption_avoidance_rate <= 1.0
        assert 0.0 <= r.torax_parity_pct <= 100.0
        assert r.p95_loop_latency_ms > 0.0
        assert 0.0 <= r.mean_risk <= 2.0
        assert isinstance(r.passes_thresholds, bool)

    def test_bad_episodes_raises(self):
        with pytest.raises(ValueError, match="episodes"):
            run_nstxu_torax_hybrid_campaign(episodes=0)

    def test_bad_steps_raises(self):
        with pytest.raises(ValueError, match="steps_per_episode"):
            run_nstxu_torax_hybrid_campaign(steps_per_episode=10)

    def test_longer_campaign(self):
        r = run_nstxu_torax_hybrid_campaign(seed=42, episodes=8, steps_per_episode=120)
        assert r.episodes == 8
        assert r.p95_loop_latency_ms < 2.0
