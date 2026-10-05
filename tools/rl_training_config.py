# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — PPO recipe configuration
"""Resolve CPU PPO recipe arguments without importing or constructing a model.

Plans describe requested configuration, not executed training or authenticated
weights. SB3 may collect a complete rollout beyond the requested timestep count.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict


class PPOParameters(TypedDict):
    """The original CPU MLP recipe with its caller-selected training seed."""

    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    verbose: int
    seed: int
    device: Literal["cpu"]


def positive_integer(text: str) -> int:
    """Parse an argparse episode/timestep count or raise fixed ArgumentTypeError."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("count must be a positive integer") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("count must be a positive integer")
    return value


def seed_integer(text: str) -> int:
    """Parse a seed in NumPy/SB3's shared unsigned 32-bit range."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("seed must be an unsigned 32-bit integer") from None
    if not 0 <= value <= 2**32 - 1:
        raise argparse.ArgumentTypeError("seed must be an unsigned 32-bit integer")
    return value


@dataclass(frozen=True)
class TrainingConfig:
    """Validated requested counts, output ZIP, seed and plan-only selection.

    Parameters
    ----------
    timesteps : int
        Positive requested learning budget; SB3 can round up to a rollout.
    eval_episodes : int
        Positive paired evaluation count, seeded from 1000.
    output : Path
        ZIP destination. Actual training additionally requires fresh output
        and metrics paths; construction and plans do not create files.
    seed : int
        Training seed in the unsigned 32-bit range, default 42.
    dry_run : bool
        True selects configuration output without model construction/learning.
    """

    timesteps: int
    eval_episodes: int
    output: Path
    seed: int = 42
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Refuse invalid direct API counts, seed or non-ZIP destinations."""
        for value in (self.timesteps, self.eval_episodes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("counts must be positive integers")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if self.output.suffix != ".zip":
            raise ValueError("training output must have a .zip suffix")

    def ppo_parameters(self) -> PPOParameters:
        """Return the same concrete keyword map consumed by the PPO constructor."""
        return {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "verbose": 0,
            "seed": self.seed,
            "device": "cpu",
        }

    def plan(self) -> dict[str, object]:
        """Describe requested configuration without claiming execution readiness."""
        return {
            "mode": "plan",
            "timesteps": self.timesteps,
            "eval_episodes": self.eval_episodes,
            "output": str(self.output),
            "metrics": str(self.output.with_suffix(".metrics.json")),
            "policy": "MlpPolicy",
            "ppo_parameters": self.ppo_parameters(),
        }


def parse_training_config(argv: list[str] | None, default_output: Path) -> TrainingConfig:
    """Parse the public recipe CLI, preserving --ci caps and adding --seed/--dry-run.

    Help exits zero; malformed arguments exit two. CI caps requested counts at
    5000 steps and five evaluation episodes. Plans validate configuration only;
    dependency availability, writer permissions and successful learning are not
    established. Invalid output suffix raises ValueError for the caller boundary.
    """
    parser = argparse.ArgumentParser(description="Train CPU PPO on the reduced-order TokamakEnv")
    parser.add_argument("--timesteps", type=positive_integer, default=50000)
    parser.add_argument("--eval-episodes", type=positive_integer, default=20)
    parser.add_argument("--seed", type=seed_integer, default=42)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--ci", action="store_true", help="cap requested steps at 5000 and evaluation episodes at five")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the configuration; do not construct or train a model"
    )
    args = parser.parse_args(argv)
    steps = min(args.timesteps, 5000) if args.ci else args.timesteps
    episodes = min(args.eval_episodes, 5) if args.ci else args.eval_episodes
    return TrainingConfig(steps, episodes, args.output, args.seed, args.dry_run)
