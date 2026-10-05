# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RL tokamak tests

"""Tests: PPO agent loading, inference, Gymnasium wrapper, PID baseline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import stable_baselines3 as sb3

from tools.train_rl_tokamak import GymTokamakEnv, PIDController, evaluate_agent

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / "weights" / "ppo_tokamak.zip"
METRICS_PATH = REPO_ROOT / "weights" / "ppo_tokamak.metrics.json"


# ── Gymnasium wrapper ────────────────────────────────────────────────


class TestGymWrapper:
    """Exercise the actual declared Gymnasium spaces and finite model transitions."""

    def test_observation_space(self) -> None:
        """Expose the original six-feature float32 observation space."""
        env = GymTokamakEnv()
        assert env.observation_space.shape == (6,)

    def test_action_space(self) -> None:
        """Expose the original two-component float32 action space."""
        env = GymTokamakEnv()
        assert env.action_space.shape == (2,)

    def test_reset_returns_correct_types(self) -> None:
        """Reset the actual model with seed42 and return float32 observations."""
        env = GymTokamakEnv()
        obs, info = env.reset(seed=42)
        assert obs.dtype == np.float32
        assert obs.shape == (6,)

    def test_step_returns_correct_types(self) -> None:
        """Advance one actual sampled action and retain the Gym result shape."""
        env = GymTokamakEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)

    def test_survives_multiple_steps(self) -> None:
        """Execute fifty actual zero-action transitions without premature termination."""
        env = GymTokamakEnv()
        env.reset(seed=42)
        survived = 0
        for _ in range(50):
            obs, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
            if terminated or truncated:
                break
            survived += 1
        assert survived >= 10


# ── PID baseline ──────────────────────────────────────────────────────


class TestPIDController:
    """Exercise the unchanged proportional baseline action and sign."""

    def test_action_shape(self) -> None:
        """Return the original two-component float32 proportional action."""
        pid = PIDController()
        obs = np.array([10.0, 2.0, 1.5, 0.85, 3.0, 15.0], dtype=np.float32)
        action = pid.act(obs)
        assert action.shape == (2,)

    def test_positive_correction_below_target(self) -> None:
        """Increase heating when the actual observed axis temperature is below target."""
        pid = PIDController(T_target=20.0)
        obs = np.array([10.0, 2.0, 1.5, 0.85, 3.0, 15.0], dtype=np.float32)
        action = pid.act(obs)
        assert action[0] > 0  # P_aux_delta should be positive when T_ax < T_target

    def test_negative_correction_above_target(self) -> None:
        """Reduce heating when the actual observed axis temperature exceeds target."""
        pid = PIDController(T_target=5.0)
        obs = np.array([10.0, 2.0, 1.5, 0.85, 3.0, 15.0], dtype=np.float32)
        action = pid.act(obs)
        assert action[0] < 0  # should reduce power when T_ax > T_target


# ── Agent loading and inference ───────────────────────────────────────


class TestPPOAgent:
    """Load and infer with the actual retained canonical policy without learning."""

    @pytest.fixture()
    def agent(self) -> sb3.PPO:
        """Load the real retained default PPO ZIP on CPU without learning or saving."""
        assert AGENT_PATH.exists(), f"Missing {AGENT_PATH} — run tools/train_rl_tokamak.py first"
        return sb3.PPO.load(str(AGENT_PATH.with_suffix("")), device="cpu")

    def test_agent_loads(self, agent: sb3.PPO) -> None:
        """Load the actual existing policy through its public SDK entry point."""
        assert agent is not None

    def test_agent_predicts(self, agent: sb3.PPO) -> None:
        """Infer finite two-component actions from the retained trained policy."""
        obs = np.array([10.0, 2.0, 1.5, 0.85, 3.0, 15.0], dtype=np.float32)
        action, _ = agent.predict(obs, deterministic=True)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))

    def test_agent_deterministic(self, agent: sb3.PPO) -> None:
        """Repeat actual deterministic policy predictions on one fixed observation."""
        obs = np.array([10.0, 2.0, 1.5, 0.85, 3.0, 15.0], dtype=np.float32)
        a1, _ = agent.predict(obs, deterministic=True)
        a2, _ = agent.predict(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_agent_no_disruption(self, agent: sb3.PPO) -> None:
        """Execute three actual seeded reduced-order episodes with the stored policy."""
        env = GymTokamakEnv()
        stats = evaluate_agent(env, agent, n_episodes=3)
        assert stats["disruption_rate"] < 1.0

    def test_metrics_exist(self) -> None:
        """Require the actual retained canonical model metrics file."""
        assert METRICS_PATH.exists()


# ── Training script ──────────────────────────────────────────────────


class TestTrainingScript:
    """Retain the executable CI learning case for an authorized training profile."""

    def test_script_runs_ci_mode(self, tmp_path: Path) -> None:
        """Retain the original real learning subprocess test; run only with training authorization."""
        out = tmp_path / "test_agent.zip"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "train_rl_tokamak.py"),
                "--ci",
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
