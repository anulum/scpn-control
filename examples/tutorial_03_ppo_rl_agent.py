#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Explicit reduced-order RL tutorial
"""Run six actual reduced-order RL tutorial sections with explicit demo training.

Default execution performs model rollouts and stored-policy inference. --train-demo
adds the original5000-step CPU learning demo; import performs no rollout or model
construction. Historical benchmark values are displayed as retained artifact
claims, with no independent-seed, physical or safety acceptance implied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
try:
    import numpy as np

    from scpn_control.control.gym_tokamak_env import TokamakEnv
    from tools.rl_training_config import positive_integer
    from tools.rl_training_results import load_benchmark
    from tools.train_rl_tokamak import GymTokamakEnv, evaluate_agent
except Exception:
    if __name__ != "__main__":
        raise
    print("RL tutorial dependencies are unavailable.", file=sys.stderr)
    raise SystemExit(1) from None


def _anatomy() -> None:
    """Display actual reduced-order spaces and a seed42 initial observation."""
    env = TokamakEnv()
    print("SECTION 1: TokamakEnv — Gymnasium Interface")
    print(f"Observation shape: {env.observation_space_shape}; action shape: {env.action_space_shape}")
    print(f"Observation bounds: {env.observation_low} to {env.observation_high}")
    print(f"Action bounds: {env.action_low} to {env.action_high}")
    obs, _ = env.reset(seed=42)
    print(f"Initial T_axis={obs[0]:.1f}keV, beta_N={obs[2]:.2f}, q95={obs[4]:.2f}")


def _pid_rollout() -> None:
    """Execute the original500-step PID demo with gains0.5/0.01/0.1."""
    print("SECTION 2: PID Baseline Controller")
    env = TokamakEnv()
    obs, _ = env.reset(seed=42)
    integral = 0.0
    previous = 0.0
    total = 0.0
    for step in range(500):
        error = 20.0 - obs[0]
        integral += error * 1e-3
        derivative = (error - previous) / 1e-3
        previous = error
        heating = np.clip(0.5 * error + 0.01 * integral + 0.1 * derivative, -5, 5)
        obs, reward, terminated, truncated, _ = env.step(np.array([heating, 0.0], dtype=np.float32))
        total += reward
        if terminated or truncated:
            break
    print(f"PID demo reward={total:.1f}, T_axis={obs[0]:.2f}, terminated={terminated}, steps={step + 1}")


def _random_rollout() -> None:
    """Execute the original seed42 uniform-action model baseline without learning."""
    print("SECTION 3: Random Policy Rollout")
    env = TokamakEnv()
    env.reset(seed=42)
    rng = np.random.default_rng(42)
    total = 0.0
    for _ in range(500):
        action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    print(f"Sample random-policy reward={total:.1f}, T_axis={obs[0]:.2f}")


def _train_demo() -> None:
    """Run the explicitly selected original5000-step PPO demo without saving weights."""
    from stable_baselines3 import PPO

    env = GymTokamakEnv()
    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            verbose=0,
            seed=42,
            device="cpu",
        )
        model.learn(total_timesteps=5000)
        stats = evaluate_agent(env, model, n_episodes=1)
        print(f"PPO demo reward={stats['mean_reward']:.1f}; this is a model demonstration")
    finally:
        env.close()


def _stored_policy(episodes: int) -> None:
    """Load the actual retained default ZIP and report paired-seed model inference."""
    from stable_baselines3 import PPO

    print("SECTION 5: Stored PPO Agent")
    agent = ROOT / "weights/ppo_tokamak.zip"
    if not agent.is_file():
        raise FileNotFoundError("stored policy is unavailable")
    model = PPO.load(str(agent), device="cpu")
    env = GymTokamakEnv()
    try:
        stats = evaluate_agent(env, model, n_episodes=episodes)
    finally:
        env.close()
    print(
        f"Stored-policy reward={stats['mean_reward']:.1f} +/- {stats['std_reward']:.1f}, disruptions={stats['disruption_rate'] * 100:.0f}%"
    )


def _artifact_comparison() -> None:
    """Render the retained actual report while distinguishing its historical provenance."""
    print("SECTION 6: PPO vs PID vs MPC retained artifact")
    for name, stats in load_benchmark(ROOT / "benchmarks/rl_vs_classical.json").items():
        print(
            f"{name:>12s} {stats['mean_reward']:8.1f} {stats['disruption_rate'] * 100:3.0f}% ({stats['n_episodes']} episodes)"
        )
    print(
        "Legacy seed-labelled artifacts predate explicit seed forwarding; they do not establish independent training."
    )
    print("These reduced-order results do not establish experimental or controller-safety acceptance.")


def main(argv: list[str] | None = None) -> int:
    """Run actual tutorial sections; learning requires explicit --train-demo selection.

    --episodes controls positive stored-policy evaluation count, default3. Import
    has no side effects. Default CLI advances models/loads existing weights but
    never learns/saves. Any section failure stops subsequent sections and returns
    one with fixed text, including missing dependencies/files. No hardcoded zero
    disruption claim is substituted for observed outcomes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-demo", action="store_true", help="run the5000-step CPU learning demonstration")
    parser.add_argument("--episodes", type=positive_integer, default=3)
    args = parser.parse_args(argv)
    try:
        _anatomy()
        _pid_rollout()
        _random_rollout()
        print("SECTION 4: PPO Training Demo")
        if args.train_demo:
            _train_demo()
        else:
            print("Learning demo requires --train-demo; no model is trained in this run.")
        _stored_policy(args.episodes)
        _artifact_comparison()
    except Exception:
        print("RL tutorial dependency, model or input verification failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
