# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption Episode Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for run_disruption_episode and its full mitigation pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_control.control.disruption_contracts import run_disruption_episode


class _MockDesignExplorer:
    """Minimal GlobalDesignExplorer stand-in returning a synthetic design dict."""

    def evaluate_design(self, r_maj: float, b_t: float, ip: float) -> dict:
        return {
            "Q": 12.0,
            "P_fusion_MW": 500.0,
            "neutron_wall_load_MW_m2": 1.2,
            "cost_index": 1.0,
        }


def _make_agent():
    return FusionAIAgent(n_states_turb=4, n_states_flow=4, n_actions=3)


class TestRunDisruptionEpisode:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        result = run_disruption_episode(
            rng=rng, rl_agent=_make_agent(),
            base_tbr=1.15, explorer=_MockDesignExplorer(),
        )
        for key in (
            "disturbance", "risk_before", "risk_after",
            "neon_quantity_mol", "argon_quantity_mol", "xenon_quantity_mol",
            "total_impurity_mol", "zeff",
            "halo_current_ma", "runaway_beam_ma",
            "wall_damage_index", "prevented", "objective_success",
        ):
            assert key in result, f"Missing key: {key}"

    def test_risk_bounded(self):
        rng = np.random.default_rng(99)
        result = run_disruption_episode(
            rng=rng, rl_agent=_make_agent(),
            base_tbr=1.10, explorer=_MockDesignExplorer(),
        )
        assert 0.0 <= result["risk_before"] <= 1.0
        assert 0.0 <= result["risk_after"] <= 1.0

    def test_wall_damage_bounded(self):
        rng = np.random.default_rng(7)
        result = run_disruption_episode(
            rng=rng, rl_agent=_make_agent(),
            base_tbr=1.20, explorer=_MockDesignExplorer(),
        )
        assert 0.0 <= result["wall_damage_index"] <= 3.0

    def test_deterministic_with_seed(self):
        def _run(seed):
            rng = np.random.default_rng(seed)
            return run_disruption_episode(
                rng=rng, rl_agent=_make_agent(),
                base_tbr=1.15, explorer=_MockDesignExplorer(),
            )
        r1 = _run(123)
        r2 = _run(123)
        assert r1["disturbance"] == r2["disturbance"]
        assert r1["risk_before"] == r2["risk_before"]

    def test_multiple_episodes_train_agent(self):
        rng = np.random.default_rng(0)
        agent = _make_agent()
        explorer = _MockDesignExplorer()
        for _ in range(5):
            result = run_disruption_episode(
                rng=rng, rl_agent=agent, base_tbr=1.15, explorer=explorer,
            )
        assert isinstance(result["prevented"], bool)
        assert isinstance(result["objective_success"], bool)
