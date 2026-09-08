# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Control Benchmark Suite.

"""Compare controllers on the synthetic scalar plant ``dx/dt = -0.5*x + u + d``.

State/reference use arbitrary state units; input and disturbance use state
units per second, not plasma units. Time is in seconds. Forward Euler and right-endpoint error quadrature define the
reported discrete metrics; results are not facility or real-time guarantees.
See ``docs/control_benchmark_suite.md`` for formulas and executable usage.

Examples
--------
A zero-output controller never reaches a constant unit reference:

>>> scenario = BenchmarkScenario("constant", {"dt": 1.0}, lambda t: 1.0, 2.0, n_episodes=1)
>>> result = BenchmarkRunner([PIDWrapper(0.0, 0.0)], [scenario]).run()[0]
>>> (result.iae, result.ise, result.itae)
(2.0, 2.0, 3.0)
>>> result.settling_time_s is None
True
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt

from scpn_control.benchmark_records import require_recorded_campaign

REPO_ROOT = Path(__file__).resolve().parents[1]


def _finite(value: object, label: str) -> float:
    """Reject non-real, boolean and nonfinite quantities before arithmetic."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _vector(value: npt.NDArray[np.float64], label: str) -> npt.NDArray[np.float64]:
    """Snapshot one real scalar channel without accepting broadcasting or NaNs."""
    array = np.asarray(value)
    if array.shape != (1,) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite real array with shape (1,)")
    return np.array(array, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class BenchmarkScenario:
    """Describe a synthetic scalar episode, validated by ``BenchmarkRunner.run``.

    Parameters
    ----------
    name
        Label. ``noise_robustness`` adds Gaussian observation noise (sigma 0.05);
        ``actuator_saturation`` clips applied input to [-20, 20].
    env_config
        Only ``dt`` is supported, in seconds, default 0.05. No Gym environment
        or facility model is configured by this mapping.
    reference_trajectory
        Pure scalar reference at the start of each interval, in state units.
        Sampled once per scenario per run and shared by all controllers.
    duration_s
        Positive duration, an integral number of positive ``dt`` intervals.
    disturbances
        Additive derivative inputs ``(onset_seconds, "step", magnitude)``.
        Onsets are nonnegative; steps persist from the first sample at/after onset.
    n_episodes
        Positive integer number of independently reset episodes.
    seed
        Nonnegative integer seed. Each controller sees the same noise sequence;
        repeated runs reproduce numerical metrics for deterministic controllers.
    """

    name: str
    env_config: dict[str, float]
    reference_trajectory: Callable[[float], float]
    duration_s: float
    disturbances: list[tuple[float, str, float]] = field(default_factory=list)
    n_episodes: int = 10
    seed: int = 0


class ControllerWrapper(Protocol):
    """Stateful scalar controller; runner calls are sequential, never concurrent."""

    def reset(self) -> None:
        """Reset all episode state before observing a plant starting at zero."""

    def step(self, obs: npt.NDArray[np.float64], ref: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
        """Consume private shape-(1,) snapshots and return a finite shape-(1,) input.

        ``obs`` and ``ref`` use state units; ``dt`` is seconds. Mutation of these
        snapshots cannot change plant/reference history. Exceptions abort the run.
        """

    def name(self) -> str:
        """Return the nonempty controller label captured once per run."""


@dataclass(frozen=True)
class BenchmarkResults:
    """Immutable episode aggregate under ``synthetic-control-benchmark/v2``.

    ``iae``, ``ise``, ``itae`` integrate post-step errors with right endpoints;
    units are state*s, state**2*s and state*s**2 respectively. ``control_effort``
    integrates squared applied input. These four values and measured controller
    call ``computation_time_us`` are episode means. ``violations`` sums samples
    with state > 3.5 across all episodes; it is a synthetic bound only.

    ``max_overshoot_pct`` is mean directed excursion beyond the final reference,
    normalised by the final step amplitude. It is None for zero amplitude.
    ``settling_time_s`` is mean first post-step sample remaining inside the 2%
    amplitude band through the end, relative to the final reference change;
    None if any episode does not settle within the observation window.
    ``settled_episodes`` retains the observed count. These are sampled-window
    metrics, not continuous-time stability proofs. Schema and authority fields
    cannot be supplied by callers through the constructor.
    """

    controller_name: str
    scenario_name: str
    iae: float
    ise: float
    itae: float
    max_overshoot_pct: float | None
    settling_time_s: float | None
    control_effort: float
    violations: int
    computation_time_us: float
    settled_episodes: int
    dt_s: float
    n_steps: int
    n_episodes: int
    seed: int
    schema: str = field(default="synthetic-control-benchmark/v2", init=False)
    synthetic: bool = field(default=True, init=False)
    facility_validation: bool = field(default=False, init=False)
    realtime_guarantee: bool = field(default=False, init=False)


@dataclass(frozen=True)
class _PreparedScenario:
    """Freeze sampled reference and disturbance inputs before controller mutation."""

    name: str
    dt: float
    reference: npt.NDArray[np.float64]
    disturbance: npt.NDArray[np.float64]
    episodes: int
    seed: int


def _prepare(scenario: BenchmarkScenario) -> _PreparedScenario:
    """Validate the complete time grid and input contract before any controller reset."""
    if not isinstance(scenario.name, str) or not scenario.name.strip():
        raise ValueError("scenario name must be nonempty")
    if set(scenario.env_config) - {"dt"}:
        raise ValueError("env_config supports only dt")
    dt = _finite(scenario.env_config.get("dt", 0.05), "dt")
    duration = _finite(scenario.duration_s, "duration_s")
    if dt <= 0 or duration <= 0:
        raise ValueError("dt and duration_s must be positive")
    ratio = duration / dt
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("duration_s must contain an integral positive number of dt intervals")
    for label, value, minimum in (("n_episodes", scenario.n_episodes, 1), ("seed", scenario.seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    times = np.arange(round(ratio), dtype=np.float64) * dt
    disturbance = np.zeros(len(times))
    for onset, kind, magnitude in scenario.disturbances:
        onset = _finite(onset, "disturbance onset")
        magnitude = _finite(magnitude, "disturbance magnitude")
        if onset < 0 or kind != "step":
            raise ValueError("disturbances require nonnegative onset and step kind")
        with np.errstate(over="ignore", invalid="ignore"):
            disturbance[times >= onset] += magnitude
    if not np.isfinite(disturbance).all():
        raise ValueError("summed disturbances must be finite")
    reference = np.array([_finite(scenario.reference_trajectory(float(t)), "reference") for t in times])
    return _PreparedScenario(scenario.name, dt, reference, disturbance, scenario.n_episodes, scenario.seed)


def _response_metrics(
    state: npt.NDArray[np.float64], reference: npt.NDArray[np.float64], dt: float
) -> tuple[float | None, float | None]:
    """Measure the last held-reference segment, with the initial state as initial reference."""
    changes = np.flatnonzero(np.diff(reference) != 0) + 1
    start = int(changes[-1]) if len(changes) else 0
    previous = float(reference[start - 1]) if start else 0.0
    amplitude = _finite(float(reference[-1]) - previous, "final reference amplitude")
    error = state[start:] - reference[-1]
    overshoot = float(max(0.0, np.max(np.sign(amplitude) * error)) / abs(amplitude) * 100) if amplitude else None
    outside = np.flatnonzero(np.abs(error) > 0.02 * abs(amplitude))
    first_settled = int(outside[-1]) + 1 if len(outside) else 0
    settling = (first_settled + 1) * dt if first_settled < len(error) else None
    if overshoot is not None:
        overshoot = _finite(overshoot, "overshoot")
    return overshoot, settling


class BenchmarkRunner:
    """Run a synthetic comparison sequentially with failure-atomic result publication.

    Parameters
    ----------
    controllers, scenarios
        Nonempty lists, copied at construction. Every successful run replaces
        ``results`` and returns a separate list of frozen records. On failure,
        prior results survive; controller side effects cannot be rolled back.
        Do not share a runner/controller across concurrent callers.
    """

    def __init__(self, controllers: list[ControllerWrapper], scenarios: list[BenchmarkScenario]):
        self.controllers = list(controllers)
        self.scenarios = list(scenarios)
        self.results: list[BenchmarkResults] = []

    def run(self) -> list[BenchmarkResults]:
        """Validate, reset and execute all pairs; publish only after every pair succeeds.

        Raises
        ------
        ValueError
            Empty inputs, invalid grid/scalars, invalid output shape, or nonfinite
            simulation/metric arithmetic. Controller/callback exceptions propagate.
        """
        if not self.controllers or not self.scenarios:
            raise ValueError("at least one controller and scenario are required")
        scenarios = [_prepare(s) for s in self.scenarios]
        controllers = [(c, c.name()) for c in self.controllers]
        if any(not isinstance(name, str) or not name.strip() for _, name in controllers):
            raise ValueError("controller name must be nonempty")
        results = [self._run_single(c, name, s) for s in scenarios for c, name in controllers]
        self.results = results
        return list(results)

    def _run_single(self, controller: ControllerWrapper, name: str, scenario: _PreparedScenario) -> BenchmarkResults:
        """Execute reset episodes against identical prepared inputs and seeded observation noise."""
        dt, reference = scenario.dt, scenario.reference
        n_steps = len(reference)
        rng = np.random.default_rng(scenario.seed)
        metrics = []
        overshoots: list[float | None] = []
        settling: list[float | None] = []
        violations = 0
        for _ in range(scenario.episodes):
            controller.reset()
            x: npt.NDArray[np.float64] = np.zeros(1)
            trajectory = np.zeros(n_steps)
            applied = np.zeros(n_steps)
            call_times = np.zeros(n_steps)
            for k, ref in enumerate(reference):
                obs = x.copy()
                if scenario.name == "noise_robustness":
                    obs += rng.normal(0, 0.05, size=1)
                start = time.perf_counter()
                output = controller.step(obs, np.array([ref]), dt)
                call_times[k] = (time.perf_counter() - start) * 1e6
                u = _vector(output, "controller output")
                if scenario.name == "actuator_saturation":
                    u = np.clip(u, -20.0, 20.0)
                with np.errstate(over="ignore", invalid="ignore"):
                    x = x + dt * (-0.5 * x + u + scenario.disturbance[k])
                x = _vector(x, "plant state")
                trajectory[k], applied[k] = x[0], u[0]
            error = trajectory - reference
            times = (np.arange(n_steps) + 1) * dt
            with np.errstate(over="ignore", invalid="ignore"):
                weighted_error = np.abs(error) * dt
                root_dt = math.sqrt(dt)
                row = [
                    np.sum(weighted_error),
                    np.sum((error * root_dt) ** 2),
                    np.sum(times * weighted_error),
                    np.sum((applied * root_dt) ** 2),
                    np.mean(call_times),
                ]
            if not np.isfinite(row).all():
                raise ValueError("benchmark metrics must be finite")
            metrics.append(row)
            violations += int(np.count_nonzero(trajectory > 3.5))
            overshoot, settled = _response_metrics(trajectory, reference, dt)
            overshoots.append(overshoot)
            settling.append(settled)
        mean = np.sum(np.array(metrics) / scenario.episodes, axis=0)
        return BenchmarkResults(
            controller_name=name,
            scenario_name=scenario.name,
            iae=float(mean[0]),
            ise=float(mean[1]),
            itae=float(mean[2]),
            max_overshoot_pct=float(sum(v / scenario.episodes for v in overshoots if v is not None))
            if all(v is not None for v in overshoots)
            else None,
            settling_time_s=float(sum(v / scenario.episodes for v in settling if v is not None))
            if all(v is not None for v in settling)
            else None,
            control_effort=float(mean[3]),
            violations=violations,
            computation_time_us=float(mean[4]),
            settled_episodes=sum(v is not None for v in settling),
            dt_s=dt,
            n_steps=n_steps,
            n_episodes=scenario.episodes,
            seed=scenario.seed,
        )

    def save_json(self, path: Path) -> None:
        """Write UTF-8 v2 records; refuse persistent evidence destinations without custody.

        ``None`` metrics encode as JSON null. Serialization completes before the
        destination opens; filesystem errors propagate. Existing files are replaced.
        """
        require_recorded_campaign(path, repository_root=REPO_ROOT)
        text = json.dumps([dataclasses.asdict(r) for r in self.results], indent=2, allow_nan=False)
        path.write_text(text + "\n", encoding="utf-8")

    def save_markdown(self, path: Path) -> None:
        """Render synthetic scope and unavailable metrics; enforce the same custody as JSON."""
        require_recorded_campaign(path, repository_root=REPO_ROOT)
        lines = [
            "Synthetic scalar benchmark (synthetic-control-benchmark/v2); no facility validation or real-time guarantee.",
            "",
            "| Controller | Scenario | IAE | Settling [s] | Overshoot [%] | Total violations | Time [µs] |",
            "|---|---|---|---|---|---|---|",
        ]
        for result in self.results:
            name = result.controller_name.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            scenario = result.scenario_name.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            settled = "not observed" if result.settling_time_s is None else f"{result.settling_time_s:.3f}"
            overshoot = "undefined" if result.max_overshoot_pct is None else f"{result.max_overshoot_pct:.3f}"
            lines.append(
                f"| {name} | {scenario} | {result.iae:.3f} | {settled} | {overshoot} | "
                f"{result.violations} | {result.computation_time_us:.3f} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def setpoint_tracking() -> BenchmarkScenario:
    """Return one 10 s episode: reference 2 to 2.5 at 5 s, sampled every 0.05 s."""

    def reference(t: float) -> float:
        """Hold each scalar target until the 5 s reference transition."""
        return 2.5 if t >= 5.0 else 2.0

    return BenchmarkScenario("setpoint_tracking", {"dt": 0.05}, reference, 10.0, n_episodes=1)


class PIDWrapper:
    """Scalar PI controller with unconstrained output and resettable integral state.

    ``Kp`` (1/s) multiplies state error; ``Ki`` (1/s**2) multiplies its time integral. Despite
    the historical PID name there is no derivative term or anti-windup. Both
    gains must be finite. Direct step calls validate inputs before mutating state.
    """

    def __init__(self, Kp: float = 1.0, Ki: float = 0.5):
        self.Kp = _finite(Kp, "Kp")
        self.Ki = _finite(Ki, "Ki")
        self.integral = 0.0

    def reset(self) -> None:
        """Clear accumulated error to zero for a new episode."""
        self.integral = 0.0

    def step(self, obs: npt.NDArray[np.float64], ref: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
        """Integrate current scalar error over positive dt seconds and return one input.

        Invalid shape/scalars or nonfinite arithmetic raise ValueError and preserve
        the previous integral. Inputs are read through copies and never mutated.
        """
        observation, reference = _vector(obs, "obs"), _vector(ref, "ref")
        dt = _finite(dt, "dt")
        if dt <= 0:
            raise ValueError("dt must be positive")
        error = float(reference[0]) - float(observation[0])
        integral = _finite(self.integral + error * dt, "integral")
        output = _finite(self.Kp * error + self.Ki * integral, "PI output")
        self.integral = integral
        return np.array([output])

    def name(self) -> str:
        """Return the stable historical label PID (the implementation is PI only)."""
        return "PID"
