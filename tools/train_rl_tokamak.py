#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Seeded CPU PPO training
"""Train the fixed CPU PPO recipe with an explicit seed and fresh candidate paths.

Importing exposes the original GymTokamakEnv/PIDController/evaluate_agent API
without constructing a PPO model. --dry-run prints the exact constructor keyword
map without learning, creating directories or writing weights. Real execution
can learn and write candidates; it does not authenticate physical performance.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]
try:
    from tools.rl_tokamak_evaluation import GymTokamakEnv, PIDController, evaluate_agent
    from tools.rl_training_config import parse_training_config
    from tools.rl_training_results import check_fresh_output
except Exception:
    if __name__ != "__main__":
        raise
    print("PPO training dependencies are unavailable.", file=sys.stderr)
    raise SystemExit(1) from None

__all__ = ["GymTokamakEnv", "PIDController", "evaluate_agent", "main"]
DEFAULT_OUTPUT = REPO_ROOT / "weights/ppo_tokamak.zip"
DEFAULT_METRICS = REPO_ROOT / "weights/ppo_tokamak.metrics.json"
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the explicit recipe or print a plan; return one on execution failure.

    Positive counts and uint32 seed are required. Actual training refuses existing
    weight/metrics files, directories and dangling links before PPO construction.
    The original CPU hyperparameters are unchanged. Record requested and actual
    timesteps, training seed, paired evaluation seeds, elapsed learning time and
    PPO/PID summaries. Save and metrics writes are not an atomic pair; failed
    evaluation/I/O can leave a partial candidate. Fixed caller text replaces
    caught exception detail; help/syntax retain argparse's zero/two exits.
    """
    env: GymTokamakEnv | None = None
    try:
        config = parse_training_config(argv, DEFAULT_OUTPUT)
        if config.dry_run:
            print(json.dumps(config.plan(), sort_keys=True))
            return 0
        metrics_path = config.output.with_suffix(".metrics.json")
        check_fresh_output(config.output)
        check_fresh_output(metrics_path)
        from stable_baselines3 import PPO

        env = GymTokamakEnv()
        model = PPO("MlpPolicy", env, **config.ppo_parameters())
        start = time.perf_counter()
        model.learn(total_timesteps=config.timesteps)
        elapsed = time.perf_counter() - start
        ppo_stats = evaluate_agent(env, model, config.eval_episodes)
        pid_stats = evaluate_agent(env, PIDController().act, config.eval_episodes)
        metrics = {
            "ppo": ppo_stats,
            "pid": pid_stats,
            "timesteps": config.timesteps,
            "actual_timesteps": model.num_timesteps,
            "seed": config.seed,
            "evaluation_seeds": list(range(1000, 1000 + config.eval_episodes)),
            "train_time_s": elapsed,
            "ppo_advantage": ppo_stats["mean_reward"] - pid_stats["mean_reward"],
        }
        config.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(config.output.with_suffix("")))
        with metrics_path.open("x", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2, allow_nan=False)
    except Exception:
        print("PPO training configuration, execution or candidate delivery failed.", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()
    logger.info("Saved PPO candidate and metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
