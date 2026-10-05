#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Stored-policy model comparison
"""Compare actual stored PPO, proportional baseline and one-step MPC candidates.

Fresh output and recorded canonical report custody are checked before inference.
The unchanged reduced-order model and paired reset seeds define these metrics;
no experimental truth, stability certificate or independent training claim follows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]
try:
    import numpy as np
    from numpy.typing import NDArray

    from scpn_control.benchmark_records import require_recorded_campaign
    from tools.rl_tokamak_evaluation import Action, Observation
    from tools.rl_training_config import positive_integer
    from tools.rl_training_results import check_fresh_output
    from tools.train_rl_tokamak import GymTokamakEnv, PIDController, evaluate_agent
except Exception:
    if __name__ != "__main__":
        raise
    print("RL benchmark dependencies are unavailable.", file=sys.stderr)
    raise SystemExit(1) from None


class SimpleMPC:
    """Retain the legacy one-step grid and fixed temperature-response surrogate.

    Parameters
    ----------
    T_target : float
        Axis-temperature target in keV, default 20.
    dt : float
        Surrogate timestep in seconds, default 1e-3.

    Notes
    -----
    Evaluate 11 heating corrections by five current corrections and choose
    the first minimum absolute predicted temperature error. The surrogate
    temperature cost ignores current correction, so ties retain the first
    listed current action. This is not the current environment's energy
    balance or a constrained multi-step optimisation.
    """

    def __init__(self, T_target: float = 20.0, dt: float = 1e-3) -> None:
        """Store the original target, timestep and 11x5 candidate action grid."""
        self.T_target = T_target
        self.dt = dt
        self._grid: NDArray[np.float32] = np.array(
            [[p, ip] for p in np.linspace(-5, 5, 11) for ip in np.linspace(-1, 1, 5)],
            dtype=np.float32,
        )

    def act(self, obs: Observation) -> Action:
        """Choose the first action minimizing the original one-step temperature cost."""
        T_ax = float(obs[0])
        T_edge = float(obs[1])
        _Ip = float(obs[5]) if len(obs) > 5 else 15.0

        best_cost = float("inf")
        best_action: Action = self._grid[0]

        for action in self._grid:
            P_delta, _Ip_delta = float(action[0]), float(action[1])
            T_ax_next = T_ax + self.dt * (50.0 * P_delta - 3.0 * (T_ax - T_edge))
            cost = abs(T_ax_next - self.T_target)
            if cost < best_cost:
                best_cost = cost
                best_action = action

        return best_action


def main(argv: list[str] | None = None) -> int:
    """Evaluate a real stored agent and write a new uppercase-controller report.

    --episodes requires a positive count; --agent is the existing model ZIP and
    --output a fresh report path. Guard canonical campaign/output before loading
    a model. Missing/unloadable agents, invalid inference or file delivery return
    one with fixed text. Actual PPO/PID/MPC share seeds1000..1000+N-1 and the
    original500-step model. Output is candidate model evidence, not an admission
    certificate. No model is trained or saved, and old reports are never accepted
    as newly produced. The exclusive JSON write does not provide bundle locking.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=positive_integer, default=20)
    parser.add_argument("--agent", type=Path, default=REPO_ROOT / "weights/ppo_tokamak.zip")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "benchmarks/rl_vs_classical.json")
    args = parser.parse_args(argv)
    env: GymTokamakEnv | None = None
    try:
        check_fresh_output(args.output)
        require_recorded_campaign(args.output, repository_root=REPO_ROOT)
        if not args.agent.is_file():
            raise FileNotFoundError("stored PPO agent is unavailable")
        from stable_baselines3 import PPO

        env = GymTokamakEnv()
        model = PPO.load(str(args.agent), device="cpu")
        results = {
            "PPO": evaluate_agent(env, model, args.episodes),
            "PID": evaluate_agent(env, PIDController().act, args.episodes),
            "MPC": evaluate_agent(env, SimpleMPC().act, args.episodes),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2, allow_nan=False)
    except Exception:
        print("RL benchmark input, inference or candidate delivery failed.", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
