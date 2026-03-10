# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Tests for RL agent and benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests: PPO agent loading, inference, Gymnasium wrapper, PID baseline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")

from tools.train_rl_tokamak import GymTokamakEnv, PIDController, evaluate_agent

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / "weights" / "ppo_tokamak.zip"
METRICS_PATH = REPO_ROOT / "weights" / "ppo_tokamak.metrics.json"


# ── Gymnasium wrapper ────────────────────────────────────────────────


class TestGymWrapper:
    def test_observation_space(self) -> None:
        env = GymTokamakEnv()
        assert env.observation_space.shape == (6,)

    def test_action_space(self) -> None:
        env = GymTokamakEnv()
        assert env.action_space.shape == (2,)

    def test_reset_returns_correct_types(self) -> None:
        env = GymTokamakEnv()
        obs, info = env.reset(seed=42)
        assert obs.dtype == np.float32
        assert obs.shape == (6,)

    def test_step_returns_correct_types(self) -> None:
        env = GymTokamakEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)

    def test_survives_multiple_steps(self) -> None:
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
    def test_action_shape(self) -> None:
        pid = PIDController()
        obs = np.array([10.0, 2.0, 1.5, 0.85, 3.0, 15.0], dtype=np.float32)
        action = pid.act(obs)
        assert action.shape == (2,)

    def test_positive_correction_below_target(self) -> None:
        pid = PIDController(T_target=20.0)
        obs = np.array([10.0, 2.0, 1.5, 0.85, 3.0, 15.0], dtype=np.float32)
        action = pid.act(obs)
        assert action[0] > 0  # P_aux_delta should be positive when T_ax < T_target

    def test_negative_correction_above_target(self) -> None:
        pid = PIDController(T_target=5.0)
        obs = np.array([10.0, 2.0, 1.5, 0.85, 3.0, 15.0], dtype=np.float32)
        action = pid.act(obs)
        assert action[0] < 0  # should reduce power when T_ax > T_target


# ── Agent loading and inference ───────────────────────────────────────


class TestPPOAgent:
    @pytest.fixture()
    def agent(self) -> sb3.PPO:
        assert AGENT_PATH.exists(), f"Missing {AGENT_PATH} — run tools/train_rl_tokamak.py first"
        return sb3.PPO.load(str(AGENT_PATH.with_suffix("")))

    def test_agent_loads(self, agent: sb3.PPO) -> None:
        assert agent is not None

    def test_agent_predicts(self, agent: sb3.PPO) -> None:
        obs = np.array([10.0, 2.0, 1.5, 0.85, 3.0, 15.0], dtype=np.float32)
        action, _ = agent.predict(obs, deterministic=True)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))

    def test_agent_deterministic(self, agent: sb3.PPO) -> None:
        obs = np.array([10.0, 2.0, 1.5, 0.85, 3.0, 15.0], dtype=np.float32)
        a1, _ = agent.predict(obs, deterministic=True)
        a2, _ = agent.predict(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_agent_no_disruption(self, agent: sb3.PPO) -> None:
        env = GymTokamakEnv()
        stats = evaluate_agent(env, agent, n_episodes=3)
        assert stats["disruption_rate"] < 1.0

    def test_metrics_exist(self) -> None:
        assert METRICS_PATH.exists()


# ── Training script ──────────────────────────────────────────────────


class TestTrainingScript:
    def test_script_runs_ci_mode(self, tmp_path: Path) -> None:
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
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
