# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Disruption Episode
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Disruption Episode Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for run_disruption_episode and its full mitigation pipeline."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from scpn_control.control.advanced_soc_fusion_learning import FusionAIAgent
from scpn_control.control.disruption_contracts import run_disruption_episode
from scpn_control.core.global_design_scanner import GlobalDesignExplorer

EpisodeResult: TypeAlias = dict[str, float | bool]


def _make_agent() -> FusionAIAgent:
    return FusionAIAgent(n_states_turb=4, n_states_flow=4, n_actions=3)


def _make_explorer() -> GlobalDesignExplorer:
    """Return the production design explorer used by disruption episodes."""

    return GlobalDesignExplorer("episode-test")


class TestRunDisruptionEpisode:
    def test_returns_expected_keys(self) -> None:
        rng = np.random.default_rng(42)
        result = run_disruption_episode(
            rng=rng,
            rl_agent=_make_agent(),
            base_tbr=1.15,
            explorer=_make_explorer(),
        )
        for key in (
            "disturbance",
            "risk_before",
            "risk_after",
            "neon_quantity_mol",
            "argon_quantity_mol",
            "xenon_quantity_mol",
            "total_impurity_mol",
            "zeff",
            "halo_current_ma",
            "runaway_beam_ma",
            "wall_damage_index",
            "prevented",
            "objective_success",
        ):
            assert key in result, f"Missing key: {key}"

    def test_risk_bounded(self) -> None:
        rng = np.random.default_rng(99)
        result = run_disruption_episode(
            rng=rng,
            rl_agent=_make_agent(),
            base_tbr=1.10,
            explorer=_make_explorer(),
        )
        assert 0.0 <= result["risk_before"] <= 1.0
        assert 0.0 <= result["risk_after"] <= 1.0

    def test_wall_damage_bounded(self) -> None:
        rng = np.random.default_rng(7)
        result = run_disruption_episode(
            rng=rng,
            rl_agent=_make_agent(),
            base_tbr=1.20,
            explorer=_make_explorer(),
        )
        assert 0.0 <= result["wall_damage_index"] <= 3.0

    def test_deterministic_with_seed(self) -> None:
        def _run(seed: int) -> EpisodeResult:
            rng = np.random.default_rng(seed)
            return run_disruption_episode(
                rng=rng,
                rl_agent=_make_agent(),
                base_tbr=1.15,
                explorer=_make_explorer(),
            )

        r1 = _run(123)
        r2 = _run(123)
        assert r1["disturbance"] == r2["disturbance"]
        assert r1["risk_before"] == r2["risk_before"]

    def test_multiple_episodes_train_agent(self) -> None:
        rng = np.random.default_rng(0)
        agent = _make_agent()
        explorer = _make_explorer()
        for _ in range(5):
            result = run_disruption_episode(
                rng=rng,
                rl_agent=agent,
                base_tbr=1.15,
                explorer=explorer,
            )
        assert isinstance(result["prevented"], bool)
        assert isinstance(result["objective_success"], bool)
