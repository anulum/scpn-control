# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual RL public workflow tests
"""Exercise actual seed plans, native shell refusals and reduced-order evaluation."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from scpn_control.benchmark_records import CAMPAIGN_ENV
from scpn_control.control.gym_tokamak_env import TokamakEnv
from tools.rl_training_config import TrainingConfig, parse_training_config
from tools.train_rl_tokamak import GymTokamakEnv, PIDController, evaluate_agent, main

ROOT = Path(__file__).resolve().parents[1]


def _pins() -> dict[str, str]:
    """Capture actual retained model/metrics/report byte identities."""
    paths = [*sorted((ROOT / "weights").glob("ppo_tokamak*")), ROOT / "benchmarks/rl_vs_classical.json"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


_POSIX_SHELL = pytest.mark.skipif(
    sys.platform == "win32", reason="the campaign script is a POSIX shell script; the runner's bash is the WSL stub"
)


@_POSIX_SHELL
def test_actual_three_seed_api_and_native_shell_plans(tmp_path: Path) -> None:
    """Three native child CLIs accept distinct real SB3 constructor seed maps."""
    before = _pins()
    output = tmp_path / "candidate"
    env = {**os.environ, "SCPN_RL_PYTHON": sys.executable}
    process = subprocess.run(
        ["bash", str(ROOT / "tools/train_rl_upcloud.sh"), "--dry-run", "--output-dir", str(output), "321", "4"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    plans = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(plans) == 3
    for seed, plan in zip((42, 123, 456), plans, strict=True):
        config = parse_training_config(
            [
                "--seed",
                str(seed),
                "--timesteps",
                "321",
                "--eval-episodes",
                "4",
                "--output",
                str(output / f"ppo_tokamak_seed{seed}.zip"),
                "--dry-run",
            ],
            ROOT / "weights/ppo_tokamak.zip",
        )
        assert plan == config.plan()
        env_model = GymTokamakEnv(max_steps=2)
        try:
            inspect.signature(PPO).bind("MlpPolicy", env_model, **config.ppo_parameters())
        finally:
            env_model.close()
        assert config.ppo_parameters()["seed"] == seed
    assert not output.exists()
    assert _pins() == before


def test_actual_cli_invalid_counts_and_retained_output(tmp_path: Path) -> None:
    """Real parser/output refusals occur before learning and preserve old bytes."""
    before = _pins()
    for arguments in (
        ["--seed", "-1"],
        ["--seed", str(2**32)],
        ["--seed", "invalid"],
        ["--timesteps", "0"],
        ["--eval-episodes", "0"],
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/train_rl_tokamak.py"), *arguments],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 2 and "Traceback" not in result.stderr
    assert main(["--seed", "123"]) == 1
    assert main(["--output", str(tmp_path / "invalid.suffix"), "--dry-run"]) == 1
    config = parse_training_config(["--ci", "--seed", "0", "--dry-run"], tmp_path / "candidate.zip")
    assert config.timesteps == 5000 and config.eval_episodes == 5 and config.seed == 0
    assert main(["--seed", str(2**32 - 1), "--dry-run"]) == 0
    for invalid in (0, -1, True):
        with pytest.raises(ValueError):
            TrainingConfig(invalid, 1, tmp_path / "candidate.zip")
        with pytest.raises(ValueError):
            TrainingConfig(1, invalid, tmp_path / "candidate.zip")
    for invalid_seed in (-1, 2**32, True):
        with pytest.raises(ValueError):
            TrainingConfig(1, 1, tmp_path / "candidate.zip", invalid_seed)
    assert _pins() == before and not (tmp_path / "candidate.zip").exists()


def test_actual_evaluation_matches_independent_public_model_replay() -> None:
    """Two float32-boundary episodes reproduce independent actual model rollouts."""
    env = GymTokamakEnv(max_steps=4)
    policy = PIDController()
    stats = evaluate_agent(env, policy.act, n_episodes=2)
    rewards = []
    for seed in (1000, 1001):
        model = TokamakEnv(max_steps=4)
        obs, _ = model.reset(seed=seed)
        total = 0.0
        for _ in range(4):
            obs, reward, terminated, truncated, _ = model.step(policy.act(obs.astype(np.float32)))
            total += reward
        rewards.append(total)
        assert truncated and not terminated
    assert stats["mean_reward"] == float(np.mean(rewards))
    assert stats["std_reward"] == float(np.std(rewards))
    assert stats["mean_length"] == 4 and stats["n_episodes"] == 2 and stats["disruption_rate"] == 0
    actual, _ = env.reset(seed=42, options={"unused": 1})
    model = TokamakEnv(max_steps=4)
    expected, _ = model.reset(seed=42)
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    env.render()
    env.close()
    action = policy.act(np.array([10, 2], dtype=np.float32))
    np.testing.assert_array_equal(action, np.array([5, 0], dtype=np.float32))


def test_actual_termination_and_evaluation_input_refusals() -> None:
    """Real current ramp produces model termination; invalid counts/overflow refuse."""
    policy = PIDController(T_target=50, Kp_Ip=10)
    policy.Ip_target = 0
    env = GymTokamakEnv(dt=0.2, max_steps=100)
    stats = evaluate_agent(env, policy.act, n_episodes=1)
    assert stats["disruption_rate"] == 1 and 1 <= stats["mean_length"] < 100
    env.close()
    for count in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            evaluate_agent(GymTokamakEnv(max_steps=2), PIDController().act, count)
    for count in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            GymTokamakEnv(max_steps=count)
    with pytest.raises(ValueError, match="finite"):
        evaluate_agent(GymTokamakEnv(max_steps=3, T_target=1e308), PIDController().act, 1)


@_POSIX_SHELL
def test_actual_shell_preflight_without_campaign_preserves_custody(tmp_path: Path) -> None:
    """The actual execution entry refuses absent campaign before models/directories."""
    before = _pins()
    env = {**os.environ, "SCPN_RL_PYTHON": sys.executable}
    env.pop(CAMPAIGN_ENV, None)
    target = tmp_path / "candidate"
    result = subprocess.run(
        ["bash", str(ROOT / "tools/train_rl_upcloud.sh"), "--output-dir", str(target), "12", "1"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "could not be verified" in result.stderr and "Traceback" not in result.stderr
    assert not target.exists() and _pins() == before


@pytest.mark.parametrize(
    "entrypoint,refusal",
    [
        ("tools/train_rl_tokamak.py", "PPO training dependencies are unavailable."),
        ("benchmarks/rl_vs_classical.py", "RL benchmark dependencies are unavailable."),
        ("examples/tutorial_03_ppo_rl_agent.py", "RL tutorial dependencies are unavailable."),
    ],
)
def test_actual_cold_dependency_cli_refusal(tmp_path: Path, entrypoint: str, refusal: str) -> None:
    """Real Python without site-packages refuses before rollouts, models or output writes."""
    before = _pins()
    process = subprocess.run(
        [sys.executable, "-S", str(ROOT / entrypoint)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert process.returncode == 1 and process.stdout == ""
    assert process.stderr == refusal + "\n"
    assert _pins() == before
