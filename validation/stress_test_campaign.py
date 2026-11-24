# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Repeated controller episodes on the flight-simulator shot.
"""Repeated PID and H-infinity episodes on the IsoFlux flight-simulator shot.

Every registered controller runs ``n_episodes`` episodes of the current-ramp
and divertor-formation shot of
:class:`~scpn_control.control.tokamak_flight_sim.IsoFluxController`. PID uses
the simulator's own radial and vertical loops. H-infinity replaces both loops
with two controllers from
:func:`~scpn_control.control.h_infinity_controller.get_flight_sim_controller`,
synthesised for the simulator's quasi-static axis error with a first-order
actuator lag.

Per controller the campaign reports the mean and standard deviation of the
reward ``-(R error + Z error)``, the mean radial error, the P50, P95 and P99
per-step wall time, the fraction of episodes whose mean radial or vertical
axis error exceeds :data:`MEAN_AXIS_ERROR_LIMIT_M`, and the mean actuator
tracking score ``1 / (1 + mean absolute radial actuator lag)``.

The harness has no stochastic input: repeated episodes repeat the same shot,
so reward and error statistics have zero spread and only the measured wall
time varies. No noise, ELM or ramp perturbation is injected. Any episode
failure, such as a solved equilibrium without an interior magnetic axis,
stops the campaign with :class:`EpisodeFailure`; no partial or zero-filled
metrics are returned.

Usage::

    python validation/stress_test_campaign.py --config iter_config.json --episodes 20
    python validation/stress_test_campaign.py --quick --output stress.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Run as a command (a script or ``python -m``), put the checkout's src on the import path once; an import
# as ``validation.<module>`` leaves the caller's import path unchanged.
repo_root = Path(__file__).resolve().parents[1]
if (__name__ == "__main__" or __package__ in {None, ""}) and str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from scpn_control.control.h_infinity_controller import get_flight_sim_controller
from scpn_control.control.pid_controller import PIDController
from scpn_control.control.tokamak_flight_sim import IsoFluxController

KernelFactory = Callable[[str], Any]
EpisodeRunner = Callable[[Any, int, "KernelFactory | None"], "EpisodeResult"]

MEAN_AXIS_ERROR_LIMIT_M = 0.5
"""Mean absolute axis error (m) above which an episode counts as exceeding the limit.

A repository heuristic for the flight-simulator shot, not a physical
disruption criterion.
"""


class EpisodeFailure(RuntimeError):
    """An episode raised, so the campaign returns no metrics.

    The message names the controller and the zero-based episode; the
    original exception is the ``__cause__``.
    """


@dataclass(frozen=True)
class EpisodeResult:
    """Measured outcome of one flight-simulator episode.

    Attributes
    ----------
    mean_abs_r_error, mean_abs_z_error : float
        Mean absolute radial and vertical magnetic-axis errors over the shot (m).
    reward : float
        ``-(mean_abs_r_error + mean_abs_z_error)``.
    latency_us : float
        Wall time of the shot divided by its step count (microseconds).
    exceeded_error_limit : bool
        Whether either mean error exceeds :data:`MEAN_AXIS_ERROR_LIMIT_M`.
    actuator_tracking : float
        ``1 / (1 + mean absolute radial actuator lag)``; 1 means the applied
        radial command followed the request exactly.
    """

    mean_abs_r_error: float
    mean_abs_z_error: float
    reward: float
    latency_us: float
    exceeded_error_limit: bool
    actuator_tracking: float


@dataclass
class ControllerMetrics:
    """Aggregate of one controller's episodes; zero-valued only before aggregation.

    Attributes
    ----------
    name : str
        Registry name of the controller.
    n_episodes : int
        Number of aggregated episodes.
    mean_reward, std_reward : float
        Mean and population standard deviation of the episode rewards.
    mean_r_error : float
        Mean of the episodes' mean absolute radial errors (m).
    p50_latency_us, p95_latency_us, p99_latency_us : float
        Linear-interpolation percentiles of the per-step wall times.
    error_limit_rate : float
        Fraction of episodes that exceeded :data:`MEAN_AXIS_ERROR_LIMIT_M`.
    mean_actuator_tracking : float
        Mean actuator tracking score.
    episodes : list of EpisodeResult
        The aggregated episodes, in run order.
    """

    name: str
    n_episodes: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_r_error: float = 0.0
    p50_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    error_limit_rate: float = 0.0
    mean_actuator_tracking: float = 0.0
    episodes: list[EpisodeResult] = field(default_factory=list)


def _episode_result(summary: dict[str, Any], elapsed_ns: int) -> EpisodeResult:
    """Turn one ``run_shot`` summary and its wall time into an episode result."""
    r_err = float(summary["mean_abs_r_error"])
    z_err = float(summary["mean_abs_z_error"])
    return EpisodeResult(
        mean_abs_r_error=r_err,
        mean_abs_z_error=z_err,
        reward=-(r_err + z_err),
        latency_us=elapsed_ns / 1e3 / max(int(summary["steps"]), 1),
        exceeded_error_limit=r_err > MEAN_AXIS_ERROR_LIMIT_M or z_err > MEAN_AXIS_ERROR_LIMIT_M,
        actuator_tracking=1.0 / (1.0 + float(summary["mean_abs_radial_actuator_lag"])),
    )


def _kernel_arguments(kernel_factory: KernelFactory | None) -> dict[str, Any]:
    """Return the simulator keyword arguments: quiet, with the kernel factory when one is given."""
    arguments: dict[str, Any] = {"verbose": False}
    if kernel_factory is not None:
        arguments["kernel_factory"] = kernel_factory
    return arguments


def _timed_shot(simulator: IsoFluxController, shot_duration: int) -> EpisodeResult:
    """Run one shot without a plot and time it."""
    started = time.perf_counter_ns()
    summary = simulator.run_shot(shot_duration=shot_duration, save_plot=False)
    return _episode_result(summary, time.perf_counter_ns() - started)


class _HInfinityFlightSimulator(IsoFluxController):
    """Flight simulator whose radial and vertical loops are two H-infinity controllers.

    The radial controller uses response gain 0.05 and the vertical one 0.02,
    both with the simulator's default actuator time constant of 0.06 s, and
    both are stepped at the simulator's control step.
    """

    def __init__(self, config_file: str, **arguments: Any) -> None:
        """Build the simulator, then synthesise one H-infinity controller per axis."""
        super().__init__(config_file, **arguments)
        self._radial = get_flight_sim_controller(response_gain=0.05, actuator_tau=0.06)
        self._vertical = get_flight_sim_controller(response_gain=0.02, actuator_tau=0.06)

    def pid_step(self, pid: PIDController, error: float) -> float:
        """Step the H-infinity controller of the loop that ``pid`` belongs to."""
        controller = self._radial if pid is self.pid_R else self._vertical
        return float(np.asarray(controller.step(error, self.control_dt_s)).item())


def _run_pid_episode(
    config_path: Any, shot_duration: int = 30, kernel_factory: KernelFactory | None = None
) -> EpisodeResult:
    """Run one shot with the simulator's own radial and vertical PID loops."""
    return _timed_shot(IsoFluxController(config_path, **_kernel_arguments(kernel_factory)), shot_duration)


def _run_hinf_episode(
    config_path: Any, shot_duration: int = 30, kernel_factory: KernelFactory | None = None
) -> EpisodeResult:
    """Run one shot with both position loops replaced by H-infinity controllers."""
    return _timed_shot(_HInfinityFlightSimulator(config_path, **_kernel_arguments(kernel_factory)), shot_duration)


CONTROLLERS: dict[str, EpisodeRunner] = {
    "PID": _run_pid_episode,
    "H-infinity": _run_hinf_episode,
}


def _aggregate(name: str, episodes: Sequence[EpisodeResult]) -> ControllerMetrics:
    """Aggregate a nonempty sequence of one controller's episodes."""
    rewards = [episode.reward for episode in episodes]
    latencies = [episode.latency_us for episode in episodes]
    return ControllerMetrics(
        name=name,
        n_episodes=len(episodes),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_r_error=float(np.mean([episode.mean_abs_r_error for episode in episodes])),
        p50_latency_us=float(np.percentile(latencies, 50)),
        p95_latency_us=float(np.percentile(latencies, 95)),
        p99_latency_us=float(np.percentile(latencies, 99)),
        error_limit_rate=float(np.mean([episode.exceeded_error_limit for episode in episodes])),
        mean_actuator_tracking=float(np.mean([episode.actuator_tracking for episode in episodes])),
        episodes=list(episodes),
    )


def run_campaign(
    n_episodes: int = 1000,
    shot_duration: int = 30,
    config_path: Any = None,
    *,
    kernel_factory: KernelFactory | None = None,
) -> dict[str, ControllerMetrics]:
    """Run every registered controller for ``n_episodes`` shots and aggregate them.

    Parameters
    ----------
    n_episodes : int
        Episodes per controller; at least 1.
    shot_duration : int
        Simulator steps per shot.
    config_path : Path or str or None
        Reactor configuration; defaults to the checkout's ``iter_config.json``.
    kernel_factory : callable or None
        Equilibrium kernel factory passed to the simulator; ``None`` keeps the
        simulator's default ``FusionKernel``.

    Returns
    -------
    dict of str to ControllerMetrics
        One aggregate per registered controller, in registry order.

    Raises
    ------
    ValueError
        ``n_episodes`` is below 1.
    EpisodeFailure
        An episode raised; no metrics are returned.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least 1")
    if config_path is None:
        config_path = repo_root / "iter_config.json"
    print(f"Episodes: {n_episodes} | Shot duration: {shot_duration} steps | Controllers: {', '.join(CONTROLLERS)}")
    results: dict[str, ControllerMetrics] = {}
    for name, run_episode in CONTROLLERS.items():
        episodes: list[EpisodeResult] = []
        for index in range(n_episodes):
            try:
                episodes.append(run_episode(config_path, shot_duration, kernel_factory))
            except Exception as exc:
                raise EpisodeFailure(f"{name} episode {index} failed: {type(exc).__name__}: {exc}") from exc
        results[name] = _aggregate(name, episodes)
    return results


def generate_summary_table(results: dict[str, ControllerMetrics]) -> str:
    """Return the campaign aggregates as a Markdown table, one row per controller."""
    lines = [
        "| Controller | Episodes | Mean Reward | Std Reward | Mean R Error (m) | "
        "P50 Lat (us) | P95 Lat (us) | P99 Lat (us) | Error-Limit Rate | Actuator Tracking |",
        "|------------|----------|-------------|------------|------------------|"
        "--------------|--------------|--------------|------------------|-------------------|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name:<10} | {m.n_episodes:>8} | {m.mean_reward:>11.4f} "
            f"| {m.std_reward:>10.4f} | {m.mean_r_error:>16.4f} "
            f"| {m.p50_latency_us:>12.0f} | {m.p95_latency_us:>12.0f} "
            f"| {m.p99_latency_us:>12.0f} | {m.error_limit_rate:>16.2%} "
            f"| {m.mean_actuator_tracking:>17.3f} |"
        )
    return "\n".join(lines)


def save_results_json(results: dict[str, ControllerMetrics], path: Path) -> None:
    """Write the campaign aggregates, without the per-episode records, to ``path`` as UTF-8 JSON."""
    data = {
        name: {
            "n_episodes": m.n_episodes,
            "mean_reward": m.mean_reward,
            "std_reward": m.std_reward,
            "mean_r_error": m.mean_r_error,
            "p50_latency_us": m.p50_latency_us,
            "p95_latency_us": m.p95_latency_us,
            "p99_latency_us": m.p99_latency_us,
            "error_limit_rate": m.error_limit_rate,
            "mean_actuator_tracking": m.mean_actuator_tracking,
        }
        for name, m in results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Results saved to {path}")


def main(argv: Sequence[str] | None = None, *, kernel_factory: KernelFactory | None = None) -> int:
    """Run the campaign from the command line.

    Parameters
    ----------
    argv : sequence of str or None
        Command-line arguments; ``None`` reads ``sys.argv``.
    kernel_factory : callable or None
        Equilibrium kernel factory for the simulator; ``None`` keeps ``FusionKernel``.

    Returns
    -------
    int
        0 after printing the table and writing any requested JSON; 1 when an
        episode failed, after printing the refusal to standard error.
    """
    parser = argparse.ArgumentParser(description="Repeated PID and H-infinity flight-simulator episodes")
    parser.add_argument("--episodes", type=int, default=1000, help="episodes per controller (default: 1000)")
    parser.add_argument("--quick", action="store_true", help="10 episodes of 10 steps")
    parser.add_argument("--output", type=str, default=None, help="path of the JSON aggregates")
    parser.add_argument("--config", type=str, default=None, help="reactor configuration JSON")
    parser.add_argument("--duration", type=int, default=30, help="simulator steps per shot (default: 30)")
    args = parser.parse_args(argv)
    if args.quick:
        args.episodes = 10
        args.duration = 10
    try:
        results = run_campaign(
            n_episodes=args.episodes,
            shot_duration=args.duration,
            config_path=args.config,
            kernel_factory=kernel_factory,
        )
    except EpisodeFailure as exc:
        print(f"Stress campaign refused: {exc}", file=sys.stderr)
        return 1
    print(generate_summary_table(results))
    if args.output:
        save_results_json(results, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
