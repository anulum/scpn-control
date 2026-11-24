# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — PID and H-infinity comparison tables.
"""PID and H-infinity compared on the IsoFlux flight-simulator shot.

The comparison runs the controllers registered in
:data:`validation.stress_test_campaign.CONTROLLERS` through
:func:`validation.stress_test_campaign.run_campaign` and renders the
aggregates as Markdown and LaTeX tables: episodes, mean reward, P95 per-step
wall time, error-limit rate and actuator tracking. To compare another
controller, register its episode runner there.

The simulator shot is deterministic, so repeated episodes repeat the same
shot and only the wall time varies. An episode failure, such as a solved
equilibrium without an interior magnetic axis, stops the comparison with
:class:`~validation.stress_test_campaign.EpisodeFailure`. The tables carry
no claim beyond this simulator; the error limit is a repository heuristic,
not a physical disruption criterion.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Run as a command (a script or ``python -m``), put the checkout and its src on the import path; an import
# as ``validation.<module>`` leaves the caller's import path unchanged.
repo_root = Path(__file__).resolve().parents[1]
if __name__ == "__main__" or __package__ in {None, ""}:
    for _source_root in (repo_root, repo_root / "src"):
        if str(_source_root) not in sys.path:
            sys.path.insert(0, str(_source_root))

from validation.stress_test_campaign import (
    CONTROLLERS,
    ControllerMetrics,
    EpisodeFailure,
    KernelFactory,
    run_campaign,
)

__all__ = [
    "CONTROLLERS",
    "ControllerMetrics",
    "generate_comparison_table",
    "generate_latex_table",
    "main",
    "run_comparison",
]


def run_comparison(
    n_episodes: int = 100,
    shot_duration: int = 30,
    config_path: Any = None,
    *,
    kernel_factory: KernelFactory | None = None,
) -> dict[str, ControllerMetrics]:
    """Run the registered controllers on the same simulator shot and aggregate them.

    Parameters
    ----------
    n_episodes : int
        Episodes per controller; at least 1.
    shot_duration : int
        Simulator steps per shot.
    config_path : Path or str or None
        Reactor configuration; defaults to the checkout's ``iter_config.json``.
    kernel_factory : callable or None
        Equilibrium kernel factory; ``None`` keeps the simulator's ``FusionKernel``.

    Returns
    -------
    dict of str to ControllerMetrics
        One aggregate per registered controller.

    Raises
    ------
    ValueError
        ``n_episodes`` is below 1.
    EpisodeFailure
        An episode raised; no metrics are returned.
    """
    return run_campaign(
        n_episodes=n_episodes,
        shot_duration=shot_duration,
        config_path=config_path,
        kernel_factory=kernel_factory,
    )


def generate_comparison_table(results: dict[str, ControllerMetrics]) -> str:
    """Return the comparison as a Markdown table, one row per controller."""
    lines = [
        "| Controller | Episodes | Mean Reward | P95 Latency (us) | Error-Limit Rate | Actuator Tracking |",
        "|------------|----------|-------------|------------------|------------------|-------------------|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name:<10} | {m.n_episodes:>8} | {m.mean_reward:>11.4f} "
            f"| {m.p95_latency_us:>16.0f} | {m.error_limit_rate:>16.2%} "
            f"| {m.mean_actuator_tracking:>17.3f} |"
        )
    return "\n".join(lines)


def generate_latex_table(results: dict[str, ControllerMetrics]) -> str:
    """Return the comparison as a LaTeX ``booktabs`` table, one row per controller."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{PID and H-infinity on the IsoFlux flight-simulator shot}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Controller & Episodes & Mean Reward & P95 Lat. & Error Limit & Tracking \\",
        r"\midrule",
    ]
    for name, m in results.items():
        lines.append(
            f"  {name} & {m.n_episodes} & {m.mean_reward:.4f} "
            f"& {m.p95_latency_us:.0f} & {m.error_limit_rate:.2%} "
            f"& {m.mean_actuator_tracking:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, kernel_factory: KernelFactory | None = None) -> int:
    """Run the comparison from the command line and print the Markdown table.

    Parameters
    ----------
    argv : sequence of str or None
        Command-line arguments; ``None`` reads ``sys.argv``.
    kernel_factory : callable or None
        Equilibrium kernel factory for the simulator; ``None`` keeps ``FusionKernel``.

    Returns
    -------
    int
        0 after printing the table; 1 when an episode failed, after printing
        the refusal to standard error.
    """
    parser = argparse.ArgumentParser(description="PID and H-infinity flight-simulator comparison")
    parser.add_argument("--episodes", type=int, default=100, help="episodes per controller (default: 100)")
    parser.add_argument("--quick", action="store_true", help="5 episodes")
    parser.add_argument("--config", type=str, default=None, help="reactor configuration JSON")
    args = parser.parse_args(argv)
    if args.quick:
        args.episodes = 5
    try:
        results = run_comparison(n_episodes=args.episodes, config_path=args.config, kernel_factory=kernel_factory)
    except EpisodeFailure as exc:
        print(f"Controller comparison refused: {exc}", file=sys.stderr)
        return 1
    print(generate_comparison_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
