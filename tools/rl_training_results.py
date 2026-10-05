# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RL candidate results
"""Validate candidate RL metrics and select/report them without training.

These are schema and filesystem checks, not weight authentication, independent
physics acceptance or proof of distinct training. No function changes a model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]
from scpn_control.benchmark_records import require_recorded_campaign


class EvaluationStats(TypedDict):
    """Finite episode summary with population deviation and termination rate."""

    mean_reward: float
    std_reward: float
    mean_length: float
    disruption_rate: float
    n_episodes: int


def validate_stats(raw: object) -> EvaluationStats:
    """Require all finite numeric metrics, positive episode count and valid ranges.

    Additional keys are ignored. Booleans, nonfinite/overflowing numbers and
    wrong JSON shapes raise ValueError. Reward can be arbitrarily negative;
    deviation is nonnegative, length at least one and disruption rate in [0,1].
    """
    if not isinstance(raw, dict):
        raise ValueError("evaluation metrics must be an object")
    count: object = raw.get("n_episodes")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("evaluation count must be a positive integer")
    values: dict[str, float] = {}
    for key in ("mean_reward", "std_reward", "mean_length", "disruption_rate"):
        value: object = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("evaluation metrics must be finite numbers")
        try:
            number = float(value)
        except OverflowError:
            raise ValueError("evaluation metrics must be finite numbers") from None
        if not math.isfinite(number):
            raise ValueError("evaluation metrics must be finite numbers")
        values[key] = number
    if values["std_reward"] < 0 or values["mean_length"] < 1 or not 0 <= values["disruption_rate"] <= 1:
        raise ValueError("evaluation metric ranges are invalid")
    return {
        "mean_reward": values["mean_reward"],
        "std_reward": values["std_reward"],
        "mean_length": values["mean_length"],
        "disruption_rate": values["disruption_rate"],
        "n_episodes": count,
    }


def check_fresh_output(output: Path) -> None:
    """Refuse an existing file, directory or dangling symlink before model work."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("candidate output already exists")
    if output.parent.exists() and not output.parent.is_dir():
        raise NotADirectoryError("candidate parent is unavailable")


def preflight_directory(directory: Path) -> None:
    """Require fresh nine-file recipe targets and recorded remote-report custody.

    No directory or model is created. A missing directory is allowed; a present
    non-directory refuses. The campaign check uses the canonical benchmark
    contract before any training, even when delivery uses a candidate directory.
    Concurrent writers are not locked and these checks are not atomic publication.
    """
    if (directory.exists() or directory.is_symlink()) and not directory.is_dir():
        raise NotADirectoryError("candidate directory is unavailable")
    names = ["ppo_tokamak.zip", "ppo_tokamak.metrics.json", "rl_vs_classical.json"]
    names.extend(f"ppo_tokamak_seed{seed}{suffix}" for seed in (42, 123, 456) for suffix in (".zip", ".metrics.json"))
    for name in names:
        check_fresh_output(directory / name)
    require_recorded_campaign(REPO_ROOT / "benchmarks/rl_vs_classical.json", repository_root=REPO_ROOT)


def select_best(directory: Path, seeds: tuple[int, ...] = (42, 123, 456)) -> tuple[int, float]:
    """Select the greatest finite recorded PPO reward; equal rewards keep first.

    Require a nonempty unique uint32 seed sequence. Each actual JSON metrics
    file must identify the requested seed and contain valid PPO episode stats.
    Legacy files without seed identity refuse; filenames alone cannot prove
    distinct training. Selection is based on declarations, not authenticated
    training provenance, and does not copy or promote weights.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("selection seeds must be nonempty and unique")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("selection seeds must be unsigned 32-bit integers")
    best_seed = seeds[0]
    best_reward: float | None = None
    for seed in seeds:
        raw: object = json.loads((directory / f"ppo_tokamak_seed{seed}.metrics.json").read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or isinstance(raw.get("seed"), bool)
            or not isinstance(raw.get("seed"), int)
            or raw.get("seed") != seed
        ):
            raise ValueError("candidate metrics do not identify their training seed")
        stats = validate_stats(raw.get("ppo"))
        reward = stats["mean_reward"]
        if best_reward is None or reward > best_reward:
            best_seed, best_reward = seed, reward
    assert best_reward is not None
    return best_seed, best_reward


def load_benchmark(path: Path) -> dict[str, EvaluationStats]:
    """Read actual uppercase MPC/PID/PPO metrics with equal episode counts."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark must be an object")
    results = {name: validate_stats(raw.get(name)) for name in ("MPC", "PID", "PPO")}
    if len({stats["n_episodes"] for stats in results.values()}) != 1:
        raise ValueError("benchmark episode counts must agree")
    return results


def main(argv: list[str] | None = None) -> int:
    """Run preflight/select/summarize against real files with fixed failure text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("preflight", "select", "summarize"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == "preflight":
            preflight_directory(args.path)
        elif args.operation == "select":
            print(select_best(args.path)[0])
        else:
            for name, stats in load_benchmark(args.path).items():
                print(
                    f"{name}: reward={stats['mean_reward']:.1f} +/- {stats['std_reward']:.1f}, disruption={stats['disruption_rate'] * 100:.0f}%"
                )
    except Exception:
        print("RL candidate metrics or output custody could not be verified.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
