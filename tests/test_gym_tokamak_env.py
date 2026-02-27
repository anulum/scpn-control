# ──────────────────────────────────────────────────────────────────────
# Tests for gymnasium-compatible TokamakEnv
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.gym_tokamak_env import TokamakEnv


class TestTokamakEnv:
    def test_reset_returns_obs_and_info(self):
        env = TokamakEnv()
        obs, info = env.reset()
        assert obs.shape == (6,)
        assert isinstance(info, dict)

    def test_step_returns_5_tuple(self):
        env = TokamakEnv()
        env.reset()
        action = np.array([0.0, 0.0])
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert bool(terminated) in (True, False)
        assert bool(truncated) in (True, False)

    def test_deterministic_with_seed(self):
        env1 = TokamakEnv(seed=123)
        obs1, _ = env1.reset(seed=123)
        env2 = TokamakEnv(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_episode_truncates_at_max_steps(self):
        env = TokamakEnv(max_steps=10)
        env.reset()
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
            if terminated:
                break
        if not terminated:
            assert truncated

    def test_action_clipping(self):
        env = TokamakEnv()
        env.reset()
        obs, _, _, _, _ = env.step(np.array([100.0, 100.0]))
        assert np.all(np.isfinite(obs))

    def test_disruption_terminates(self):
        env = TokamakEnv()
        env.reset()
        # Drive current to zero -> q95 diverges or beta_N goes high
        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(np.array([5.0, -1.0]))
            if terminated:
                assert info["disrupted"]
                break

    def test_obs_within_bounds(self):
        env = TokamakEnv()
        obs, _ = env.reset()
        assert np.all(obs >= env.observation_low - 1e-6)
        assert np.all(obs <= env.observation_high + 1e-6)

    def test_negative_reward_for_error(self):
        env = TokamakEnv(T_target=20.0)
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert reward < 0  # T_axis starts at ~10, target is 20

    def test_render_does_not_crash(self):
        env = TokamakEnv()
        env.reset()
        env.render()

    def test_spaces_properties(self):
        env = TokamakEnv()
        assert env.observation_space_shape == (6,)
        assert env.action_space_shape == (2,)
