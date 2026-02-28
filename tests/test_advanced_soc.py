# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Tests for Advanced SOC Fusion Learning
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests for CoupledSandpileReactor, FusionAIAgent, and run_advanced_learning_sim."""

import numpy as np
import pytest

from scpn_control.control.advanced_soc_fusion_learning import (
    CoupledSandpileReactor,
    FusionAIAgent,
    FusionAI_Agent,
    run_advanced_learning_sim,
)


# ── CoupledSandpileReactor ───────────────────────────────────────────

class TestCoupledSandpileReactor:
    def test_initial_state_zero(self):
        r = CoupledSandpileReactor()
        assert np.all(r.Z == 0.0)
        assert r.flow == 0.0

    def test_drive_increases_z0(self):
        r = CoupledSandpileReactor()
        r.drive(5.0)
        assert r.Z[0] == 5.0

    def test_drive_negative_clamped(self):
        r = CoupledSandpileReactor()
        r.drive(-10.0)
        assert r.Z[0] == 0.0

    def test_step_physics_returns_triple(self):
        r = CoupledSandpileReactor()
        r.drive(10.0)
        topple, flow, shear = r.step_physics(0.0)
        assert isinstance(topple, int)
        assert isinstance(flow, float)
        assert isinstance(shear, float)

    def test_toppling_occurs_above_z_crit(self):
        r = CoupledSandpileReactor(z_crit_base=3.0)
        r.Z[0] = 10.0
        topple, _, _ = r.step_physics(0.0)
        assert topple > 0

    def test_no_topple_below_z_crit(self):
        r = CoupledSandpileReactor(z_crit_base=20.0)
        r.Z[0] = 5.0
        topple, _, _ = r.step_physics(0.0)
        assert topple == 0

    def test_flow_bounded(self):
        r = CoupledSandpileReactor(flow_bounds=(0.0, 2.0))
        r.Z[:] = 50.0
        for _ in range(20):
            r.step_physics(0.0)
        assert r.flow <= 2.0

    def test_get_profile_energy(self):
        r = CoupledSandpileReactor(size=10)
        r.Z = np.arange(10, dtype=np.float64)
        e = r.get_profile_energy()
        assert e > 0.0
        assert r.h.shape == (10,)

    def test_rejects_small_size(self):
        with pytest.raises(ValueError, match="size"):
            CoupledSandpileReactor(size=4)

    def test_rejects_negative_flow_generation(self):
        with pytest.raises(ValueError, match="flow_generation"):
            CoupledSandpileReactor(flow_generation=-1.0)

    def test_rejects_bad_flow_damping(self):
        with pytest.raises(ValueError, match="flow_damping"):
            CoupledSandpileReactor(flow_damping=1.0)

    def test_rejects_negative_shear_efficiency(self):
        with pytest.raises(ValueError, match="shear_efficiency"):
            CoupledSandpileReactor(shear_efficiency=-0.5)

    def test_rejects_bad_max_sub_steps(self):
        with pytest.raises(ValueError, match="max_sub_steps"):
            CoupledSandpileReactor(max_sub_steps=0)

    def test_shear_raises_z_crit(self):
        r = CoupledSandpileReactor(z_crit_base=5.0, shear_efficiency=2.0)
        r.Z[0] = 8.0
        topple_no_shear, _, _ = r.step_physics(0.0)
        r2 = CoupledSandpileReactor(z_crit_base=5.0, shear_efficiency=2.0)
        r2.Z[0] = 8.0
        topple_with_shear, _, _ = r2.step_physics(5.0)
        assert topple_with_shear <= topple_no_shear


# ── FusionAIAgent ────────────────────────────────────────────────────

class TestFusionAIAgent:
    def test_q_table_shape(self):
        agent = FusionAIAgent(n_states_turb=5, n_states_flow=5, n_actions=3)
        assert agent.q_table.shape == (5, 5, 3)

    def test_q_table_initial_zero(self):
        agent = FusionAIAgent()
        assert np.all(agent.q_table == 0.0)

    def test_discretize_state(self):
        agent = FusionAIAgent()
        s_t, s_f = agent.discretize_state(0.0, 0.0)
        assert s_t == 0
        assert s_f == 0

    def test_discretize_clamps_high(self):
        agent = FusionAIAgent(n_states_turb=3, n_states_flow=3)
        s_t, s_f = agent.discretize_state(1000.0, 100.0)
        assert s_t == 2
        assert s_f == 2

    def test_choose_action_range(self):
        agent = FusionAIAgent(n_actions=3)
        rng = np.random.default_rng(0)
        for _ in range(50):
            a = agent.choose_action((0, 0), rng)
            assert 0 <= a < 3

    def test_learn_updates_q(self):
        agent = FusionAIAgent(alpha=0.5)
        old_q = agent.q_table[0, 0, 0]
        agent.learn((0, 0), 0, (0, 0), 10.0)
        assert agent.q_table[0, 0, 0] != old_q

    def test_learn_accumulates_reward(self):
        agent = FusionAIAgent()
        agent.learn((0, 0), 0, (0, 0), 5.0)
        agent.learn((0, 0), 1, (1, 1), 3.0)
        assert agent.total_reward == pytest.approx(8.0)

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            FusionAIAgent(alpha=1.5)

    def test_rejects_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            FusionAIAgent(gamma=-0.1)

    def test_rejects_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            FusionAIAgent(epsilon=2.0)

    def test_rejects_bad_n_states(self):
        with pytest.raises(ValueError, match="n_states_turb"):
            FusionAIAgent(n_states_turb=0)

    def test_rejects_bad_n_actions(self):
        with pytest.raises(ValueError, match="n_actions"):
            FusionAIAgent(n_actions=0)


# ── Backward-compatible alias ────────────────────────────────────────

class TestFusionAIAgentAlias:
    def test_alias_is_subclass(self):
        agent = FusionAI_Agent()
        assert isinstance(agent, FusionAIAgent)


# ── run_advanced_learning_sim ────────────────────────────────────────

class TestRunAdvancedLearningSim:
    def test_returns_expected_keys(self):
        result = run_advanced_learning_sim(
            size=10, time_steps=50, save_plot=False, verbose=False,
        )
        assert "final_core_temp" in result
        assert "mean_turbulence" in result
        assert "total_reward" in result
        assert "q_table_mean" in result

    def test_deterministic(self):
        a = run_advanced_learning_sim(
            size=10, time_steps=100, seed=42, save_plot=False, verbose=False,
        )
        b = run_advanced_learning_sim(
            size=10, time_steps=100, seed=42, save_plot=False, verbose=False,
        )
        assert a["total_reward"] == b["total_reward"]
        assert a["final_core_temp"] == b["final_core_temp"]

    def test_steps_stored(self):
        r = run_advanced_learning_sim(
            size=10, time_steps=75, save_plot=False, verbose=False,
        )
        assert r["steps"] == 75

    def test_rejects_bad_steps(self):
        with pytest.raises(ValueError, match="time_steps"):
            run_advanced_learning_sim(time_steps=0, save_plot=False, verbose=False)

    def test_rejects_bad_shear_step(self):
        with pytest.raises(ValueError, match="shear_step"):
            run_advanced_learning_sim(
                time_steps=10, shear_step=-1.0, save_plot=False, verbose=False,
            )

    def test_rejects_bad_noise_prob(self):
        with pytest.raises(ValueError, match="noise_probability"):
            run_advanced_learning_sim(
                time_steps=10, noise_probability=2.0, save_plot=False, verbose=False,
            )

    def test_core_temp_positive(self):
        r = run_advanced_learning_sim(
            size=10, time_steps=200, save_plot=False, verbose=False,
        )
        assert r["final_core_temp"] > 0.0

    def test_verbose_prints_output(self, capsys):
        run_advanced_learning_sim(
            size=10, time_steps=20, save_plot=False, verbose=True,
        )
        captured = capsys.readouterr()
        assert "Predator-Prey" in captured.out
        assert "Step" in captured.out


class TestFusionAIAgentFlowValidation:
    def test_rejects_bad_n_states_flow(self):
        with pytest.raises(ValueError, match="n_states_flow"):
            FusionAIAgent(n_states_flow=0)
