#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — PPO Agent Training for Tokamak Control
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Train a PPO agent on the TokamakEnv Gymnasium environment.

Usage:
    python tools/train_rl_tokamak.py --timesteps 50000
    python tools/train_rl_tokamak.py --timesteps 5000 --ci  # fast CI mode

The trained agent is saved to weights/ppo_tokamak.zip.
A metrics file with episode stats is saved alongside.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "weights" / "ppo_tokamak.zip"
DEFAULT_METRICS = REPO_ROOT / "weights" / "ppo_tokamak.metrics.json"


class GymTokamakEnv(gym.Env):
    """Gymnasium wrapper around TokamakEnv for stable-baselines3 compatibility."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, dt: float = 1e-3, max_steps: int = 500, T_target: float = 20.0) -> None:
        super().__init__()
        from scpn_control.control.gym_tokamak_env import TokamakEnv

        self._env = TokamakEnv(dt=dt, max_steps=max_steps, T_target=T_target)

        self.observation_space = spaces.Box(
            low=self._env.observation_low.astype(np.float32),
            high=self._env.observation_high.astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=self._env.action_low.astype(np.float32),
            high=self._env.action_high.astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed)
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self) -> None:
        self._env.render()


class PIDController:
    """Baseline PID controller for comparison.

    Tracks T_axis → T_target via P_aux, and Ip via proportional control.
    """

    def __init__(self, T_target: float = 20.0, Kp_T: float = 0.5, Kp_Ip: float = 0.1) -> None:
        self.T_target = T_target
        self.Kp_T = Kp_T
        self.Kp_Ip = Kp_Ip
        self.Ip_target = 15.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        T_ax = float(obs[0])
        Ip = float(obs[5]) if len(obs) > 5 else 15.0
        P_delta = np.clip(self.Kp_T * (self.T_target - T_ax), -5.0, 5.0)
        Ip_delta = np.clip(self.Kp_Ip * (self.Ip_target - Ip), -1.0, 1.0)
        return np.array([P_delta, Ip_delta], dtype=np.float32)


def evaluate_agent(env: GymTokamakEnv, predict_fn: object, n_episodes: int = 20) -> dict:
    """Run agent for n_episodes and collect stats."""
    rewards = []
    lengths = []
    disruptions = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 1000)
        total_reward = 0.0
        for step in range(env._env.max_steps):
            if callable(predict_fn):
                action = predict_fn(obs)
            else:
                action, _ = predict_fn.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                disruptions += 1
                break
            if truncated:
                break
        rewards.append(total_reward)
        lengths.append(step + 1)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "disruption_rate": disruptions / n_episodes,
        "n_episodes": n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent on TokamakEnv")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ci", action="store_true", help="Fast CI mode (5000 steps)")
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    if args.ci:
        args.timesteps = min(args.timesteps, 5000)
        args.eval_episodes = min(args.eval_episodes, 5)

    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("stable-baselines3 required: pip install stable-baselines3")
        sys.exit(1)

    env = GymTokamakEnv()

    logger.info("Training PPO for %d timesteps", args.timesteps)
    t0 = time.perf_counter()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        seed=42,
    )
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.perf_counter() - t0
    logger.info("Training complete in %.1fs", elapsed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output.with_suffix("")))
    logger.info("Saved agent to %s", args.output)

    # Evaluate PPO
    logger.info("Evaluating PPO (%d episodes)", args.eval_episodes)
    ppo_stats = evaluate_agent(env, model, args.eval_episodes)

    # Evaluate PID baseline
    logger.info("Evaluating PID baseline (%d episodes)", args.eval_episodes)
    pid = PIDController()
    pid_stats = evaluate_agent(env, pid.act, args.eval_episodes)

    # Compare
    logger.info(
        "PPO: reward=%.1f±%.1f  disruption=%.0f%%  len=%.0f",
        ppo_stats["mean_reward"],
        ppo_stats["std_reward"],
        ppo_stats["disruption_rate"] * 100,
        ppo_stats["mean_length"],
    )
    logger.info(
        "PID: reward=%.1f±%.1f  disruption=%.0f%%  len=%.0f",
        pid_stats["mean_reward"],
        pid_stats["std_reward"],
        pid_stats["disruption_rate"] * 100,
        pid_stats["mean_length"],
    )

    metrics = {
        "ppo": ppo_stats,
        "pid": pid_stats,
        "timesteps": args.timesteps,
        "train_time_s": elapsed,
        "ppo_advantage": ppo_stats["mean_reward"] - pid_stats["mean_reward"],
    }

    metrics_path = args.output.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
