# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual RL public workflow tests
"""Exercise actual stored-policy producer CLI and protected-output refusals."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from benchmarks.rl_vs_classical import SimpleMPC
from tools.rl_training_results import load_benchmark
from tools.train_rl_tokamak import GymTokamakEnv, PIDController, evaluate_agent

ROOT = Path(__file__).resolve().parents[1]


def test_actual_native_benchmark_and_independent_public_policies(tmp_path: Path) -> None:
    """Run the real CLI with retained weights and independently replay every policy."""
    source = ROOT / "weights/ppo_tokamak.zip"
    report = ROOT / "benchmarks/rl_vs_classical.json"
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, report)}
    output = tmp_path / "candidate.json"
    process = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks/rl_vs_classical.py"), "--episodes", "1", "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    actual = load_benchmark(output)
    model = PPO.load(str(source), device="cpu")
    env = GymTokamakEnv()
    try:
        expected = {
            "PPO": evaluate_agent(env, model, 1),
            "PID": evaluate_agent(env, PIDController().act, 1),
            "MPC": evaluate_agent(env, SimpleMPC().act, 1),
        }
    finally:
        env.close()
    assert actual == expected
    assert all(x["n_episodes"] == 1 for x in actual.values())
    short = SimpleMPC().act(np.array([10, 2], dtype=np.float32))
    assert short.shape == (2,) and short.dtype == np.float32
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before} == before


def test_actual_benchmark_refusals_preserve_old_reports(tmp_path: Path) -> None:
    """Real CLI refuses old output, missing weights and invalid episode count."""
    canonical = ROOT / "benchmarks/rl_vs_classical.json"
    before = canonical.read_bytes()
    for arguments, code in (
        ([], 1),
        (["--episodes", "0"], 2),
        (["--output", str(tmp_path / "fresh.json"), "--agent", str(tmp_path / "absent.zip")], 1),
    ):
        process = subprocess.run(
            [sys.executable, str(ROOT / "benchmarks/rl_vs_classical.py"), *arguments],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert process.returncode == code and "Traceback" not in process.stderr
    assert canonical.read_bytes() == before and not (tmp_path / "fresh.json").exists()
