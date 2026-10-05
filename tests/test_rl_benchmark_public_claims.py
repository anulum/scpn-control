# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RL benchmark public-claim boundary tests.
"""Regression tests for RL benchmark claims on outward-facing surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from examples.tutorial_03_ppo_rl_agent import main as tutorial_main
from tools.rl_training_results import EvaluationStats, load_benchmark

ROOT: Final = Path(__file__).resolve().parents[1]
BENCHMARK_PATH: Final = ROOT / "benchmarks" / "rl_vs_classical.json"
PUBLIC_SURFACES: Final = (
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "changelog.md",
    ROOT / "ROADMAP.md",
    ROOT / "docs" / "competitive_analysis.md",
    ROOT / "docs" / "pitch.md",
    ROOT / "examples" / "tutorial_03_ppo_rl_agent.py",
)
STALE_REWARD_VALUES: Final = ("143.7", "58.1", "-912.3", "+-0.2")


def _benchmark() -> dict[str, EvaluationStats]:
    """Load the committed RL-vs-classical benchmark artifact."""
    return load_benchmark(BENCHMARK_PATH)


def _mean_reward(report: dict[str, EvaluationStats], controller: str) -> str:
    """Return the one-decimal mean reward for ``controller``."""
    value = report[controller]["mean_reward"]
    return f"{float(value):.1f}"


def _episode_count(report: dict[str, EvaluationStats]) -> int:
    """Return the benchmark episode count shared by all controllers."""
    counts = {int(metrics["n_episodes"]) for metrics in report.values()}
    assert counts == {50}
    return counts.pop()


def test_public_surfaces_do_not_repeat_stale_rl_reward_values() -> None:
    """Public surfaces must not reintroduce inflated historical PPO numbers."""
    for path in PUBLIC_SURFACES:
        text = path.read_text(encoding="utf-8")
        for stale_value in STALE_REWARD_VALUES:
            assert stale_value not in text, f"{path.relative_to(ROOT)} contains {stale_value}"


def test_retained_public_rl_claims_match_committed_benchmark_artifact(capsys: pytest.CaptureFixture[str]) -> None:
    """Retained benchmark claims match the artifact and stay out of the evidence matrix."""
    report = _benchmark()
    ppo = _mean_reward(report, "PPO")
    mpc = _mean_reward(report, "MPC")
    pid = _mean_reward(report, "PID")
    episodes = _episode_count(report)

    changelog_claim = (
        f"PPO reward={ppo} beats MPC ({mpc}) and PID ({pid}), 0% disruption rate over\n  {episodes} benchmark episodes"
    )
    roadmap_claim = f"reward={ppo} vs MPC={mpc} vs PID={pid} over\n  {episodes} episodes"
    competitive_claim = f"PPO 500K benchmark artifact records PPO {ppo} vs MPC {mpc} over {episodes} episodes"

    assert changelog_claim in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog_claim in (ROOT / "docs" / "changelog.md").read_text(encoding="utf-8")
    assert roadmap_claim in (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    competitive_analysis = (ROOT / "docs" / "competitive_analysis.md").read_text(encoding="utf-8")
    assert competitive_claim not in competitive_analysis
    assert "This page compares documented scope and evidence" in competitive_analysis

    assert tutorial_main(["--episodes", "1"]) == 0
    tutorial = capsys.readouterr().out
    for name, stats in report.items():
        expected = f"{name:>12s} {stats['mean_reward']:8.1f} {stats['disruption_rate'] * 100:3.0f}% ({stats['n_episodes']} episodes)"
        assert expected in tutorial
    assert "do not establish independent training" in tutorial
