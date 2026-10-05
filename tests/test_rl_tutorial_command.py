# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual RL public workflow tests
"""Exercise the real tutorial default CLI without its explicitly selected learning demo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.rl_training_results import load_benchmark

ROOT = Path(__file__).resolve().parents[1]


def test_actual_default_tutorial_retained_artifact_and_no_learning(tmp_path: Path) -> None:
    """Execute all default sections and match actual retained report rows."""
    before = {p: p.read_bytes() for p in (ROOT / "weights").glob("ppo_tokamak*") if p.is_file()}
    process = subprocess.run(
        [sys.executable, str(ROOT / "examples/tutorial_03_ppo_rl_agent.py"), "--episodes", "1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    for section in range(1, 7):
        assert f"SECTION {section}:" in process.stdout
    assert "no model is trained in this run" in process.stdout
    assert "do not establish independent training" in process.stdout
    for name, stats in load_benchmark(ROOT / "benchmarks/rl_vs_classical.json").items():
        expected = f"{name:>12s} {stats['mean_reward']:8.1f} {stats['disruption_rate'] * 100:3.0f}% ({stats['n_episodes']} episodes)"
        assert expected in process.stdout
    assert {p: p.read_bytes() for p in before} == before


def test_actual_tutorial_help_and_invalid_count(tmp_path: Path) -> None:
    """Real parser describes opt-in demo training and rejects invalid counts."""
    for arguments, code in ((["--help"], 0), (["--episodes", "0"], 2)):
        process = subprocess.run(
            [sys.executable, str(ROOT / "examples/tutorial_03_ppo_rl_agent.py"), *arguments],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert process.returncode == code and "Traceback" not in process.stderr
    assert "--train-demo" in process.stdout + process.stderr
