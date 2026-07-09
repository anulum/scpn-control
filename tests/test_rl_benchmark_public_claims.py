# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RL benchmark public-claim boundary tests.
"""Regression tests for RL benchmark claims on outward-facing surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, cast

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


def _benchmark() -> dict[str, dict[str, float | int]]:
    """Load the committed RL-vs-classical benchmark artifact."""

    raw: Any = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    return cast(dict[str, dict[str, float | int]], raw)


def _mean_reward(report: dict[str, dict[str, float | int]], controller: str) -> str:
    """Return the one-decimal mean reward for ``controller``."""

    value = report[controller]["mean_reward"]
    return f"{float(value):.1f}"


def _episode_count(report: dict[str, dict[str, float | int]]) -> int:
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


def test_public_rl_claims_match_committed_benchmark_artifact() -> None:
    """Public benchmark claims must match ``rl_vs_classical.json``."""

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
    assert competitive_claim in (ROOT / "docs" / "competitive_analysis.md").read_text(encoding="utf-8")

    tutorial = (ROOT / "examples" / "tutorial_03_ppo_rl_agent.py").read_text(encoding="utf-8")
    assert f"{{'PPO':>12s}}  {{'{ppo}':>8s}}  {{'0%':>10s}}" in tutorial
    assert f"{{'MPC':>12s}}  {{'{mpc}':>8s}}  {{'0%':>10s}}" in tutorial
    assert f"{{'PID':>12s}}  {{'{pid}':>8s}}  {{'0%':>10s}}" in tutorial
