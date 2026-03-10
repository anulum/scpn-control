#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — RL vs Classical Controller Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Compare PPO, PID, and MPC controllers on TokamakEnv.

Usage:
    python benchmarks/rl_vs_classical.py
    python benchmarks/rl_vs_classical.py --episodes 50

Outputs a JSON report to benchmarks/rl_vs_classical.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import sys

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class SimpleMPC:
    """1-step lookahead MPC using the known dynamics model.

    Evaluates a grid of candidate actions and picks the one
    minimising predicted |T_axis - T_target|.
    """

    def __init__(self, T_target: float = 20.0, dt: float = 1e-3) -> None:
        self.T_target = T_target
        self.dt = dt
        self._grid = np.array(
            [[p, ip] for p in np.linspace(-5, 5, 11) for ip in np.linspace(-1, 1, 5)],
            dtype=np.float32,
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        T_ax = float(obs[0])
        T_edge = float(obs[1])
        Ip = float(obs[5]) if len(obs) > 5 else 15.0

        best_cost = float("inf")
        best_action = self._grid[0]

        for action in self._grid:
            P_delta, Ip_delta = float(action[0]), float(action[1])
            T_ax_next = T_ax + self.dt * (50.0 * P_delta - 3.0 * (T_ax - T_edge))
            cost = abs(T_ax_next - self.T_target)
            if cost < best_cost:
                best_cost = cost
                best_action = action

        return best_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("stable-baselines3 required")
        return

    from tools.train_rl_tokamak import GymTokamakEnv, PIDController, evaluate_agent

    env = GymTokamakEnv()

    controllers = {}

    # PPO
    agent_path = REPO_ROOT / "weights" / "ppo_tokamak"
    if agent_path.with_suffix(".zip").exists():
        ppo = PPO.load(str(agent_path))
        controllers["PPO"] = lambda obs, m=ppo: m.predict(obs, deterministic=True)[0]

    # PID
    pid = PIDController()
    controllers["PID"] = pid.act

    # MPC
    mpc = SimpleMPC()
    controllers["MPC"] = mpc.act

    results = {}
    for name, ctrl_fn in controllers.items():
        logger.info("Evaluating %s (%d episodes)", name, args.episodes)
        stats = evaluate_agent(env, ctrl_fn, args.episodes)
        results[name] = stats
        logger.info(
            "  %s: reward=%.1f±%.1f  disruption=%.0f%%",
            name,
            stats["mean_reward"],
            stats["std_reward"],
            stats["disruption_rate"] * 100,
        )

    out_path = Path(__file__).parent / "rl_vs_classical.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
