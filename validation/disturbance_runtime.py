# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disturbance runtime.

"""Execute finite synthetic vertical-proxy scenarios without fabricating trace tails.

The Euler model is z'' = gamma^2*z - 10*z' + u + 0.5*d. Density and ELM
names select acceleration forcings, not density, beta_N or energy state.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid

from validation.disturbance_controllers import ControllerProtocol
from validation.disturbance_inputs import FloatArray, _finite, _positive, _state

DT = 1.0e-4
SCENARIO_DURATIONS = {"VDE": 2.0, "Density ramp": 4.0, "ELM pacing": 3.0}


class LinearPlant:
    """Integrate the reduced two-state vertical model with explicit Euler.

    Parameters
    ----------
    gamma_growth : float
        Positive coefficient in s^-1; the actual unstable pole also depends
        on the fixed 10 s^-1 damping. Position and velocity start at zero.

    Notes
    -----
    u and d are acceleration inputs in m/s^2, with disturbance gain 0.5.
    No device geometry, plasma density, current, beta or energy is simulated.
    Instances hold mutable state; stepping/reset is not thread safe.
    """

    def __init__(self, gamma_growth: float = 100.0) -> None:
        """Validate the growth coefficient and initialize zero position/velocity."""
        self.gamma_growth = _positive(gamma_growth, "gamma_growth")
        squared = _finite(self.gamma_growth * self.gamma_growth, "gamma_growth squared")
        self.A = np.array([[0.0, 1.0], [squared, -10.0]])
        self.B, self.B_d = np.array([0.0, 1.0]), np.array([0.0, 0.5])
        self.x: FloatArray = np.zeros(2)

    def step(self, u: float, d: float, dt: float) -> float:
        """Advance a finite step; invalid input/arithmetic leaves plant state unchanged."""
        control, disturbance, interval = _finite(u, "u"), _finite(d, "d"), _positive(dt, "dt")
        with np.errstate(over="raise", invalid="raise"):
            try:
                candidate = self.x + interval * (self.A @ self.x + self.B * control + self.B_d * disturbance)
            except FloatingPointError as exc:
                raise ValueError("plant arithmetic must remain finite") from exc
        self.x = _state(candidate)
        return self.z

    @property
    def z(self) -> float:
        """Return the current vertical position in metres."""
        return float(self.x[0])

    @property
    def dz(self) -> float:
        """Return the current vertical velocity in metres per second."""
        return float(self.x[1])

    def reset(self, x0: npt.ArrayLike | None = None) -> None:
        """Copy a finite position/velocity vector, or restore the zero state."""
        self.x = np.zeros(2) if x0 is None else _state(x0)


def _disturbance_vde(t: float) -> float:
    """Apply a 5000 m/s^2 acceleration proxy during the first millisecond."""
    return 5000.0 if t < 1.0e-3 else 0.0


def _disturbance_density_ramp(t: float) -> float:
    """Convert a synthetic 0.5-to-1.2 fraction ramp into acceleration forcing."""
    fraction = 1.2 if t >= 2.0 else 0.5 + 0.7 * (t / 2.0)
    return 200.0 * (fraction - 1.0)


def _disturbance_elm_pacing(t: float) -> float:
    """Apply 1000 m/s^2 proxy pulses, width 0.5 ms and period 100 ms."""
    return 1000.0 if t % 0.1 < 0.5e-3 else 0.0


SCENARIOS: dict[str, dict[str, object]] = {
    name: dict(
        disturbance=fn,
        duration_s=SCENARIO_DURATIONS[name],
        dt_s=DT,
        target_z=0.0,
        x0=np.array([initial, 0.0]),
        settling_threshold=0.05,
        description=description,
    )
    for name, fn, initial, description in [
        ("VDE", _disturbance_vde, 0.01, "Initial 1 cm displacement and first-millisecond acceleration kick."),
        (
            "Density ramp",
            _disturbance_density_ramp,
            0.0,
            "Synthetic fraction ramp mapped to vertical acceleration; no density state.",
        ),
        ("ELM pacing", _disturbance_elm_pacing, 0.0, "10 Hz acceleration pulses; no beta_N drop or energy evolution."),
    ]
}


@dataclass
class ScenarioMetrics:
    """Store observed finite-interval metrics and the actual completion contract.

    ISE is m^2*s; settling time is seconds; peak_overshoot is maximum absolute
    error in metres, not signed overshoot. Effort is integral |u| in m/s.
    wall_clock_s uses perf_counter around the conditional Euler/control loop,
    including forcing, finite checks and trace accumulation. It excludes
    configuration, plant/controller construction and reset, metric reduction,
    report writes and plotting; it is an uncontrolled local observation.
    stable means finite/bounded completion, not spectral or physical admission.
    Settling uses the recorded band and only the observed interval; settled
    distinguishes never-settled from reaching the band at the final sample.
    Defaults on added metadata preserve the old eight-field constructor.
    """

    controller: str
    scenario: str
    ise: float
    settling_time_s: float
    peak_overshoot: float
    control_effort: float
    wall_clock_s: float
    stable: bool
    requested_duration_s: float = 0.0
    completed_duration_s: float = 0.0
    requested_steps: int = 0
    completed_steps: int = 0
    settled: bool = False
    termination: str = "unspecified"
    target_z_m: float = 0.0
    dt_s: float = DT
    settling_band_m: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a fresh JSON-compatible declaration; no origin authentication is implied."""
        return asdict(self)


@dataclass
class TraceData:
    """Store aligned state/error/time samples including the actual terminal state.

    All vectors have completed_steps+1 samples. controls[k] is the issued
    control on interval k; its last value repeats the last held action solely
    to align the terminal timestamp. With zero completed steps it is zero.
    Effort integrates only real completed intervals, never that terminal slot.
    """

    times: FloatArray
    positions: FloatArray
    errors: FloatArray
    controls: FloatArray


def _compute_settling_time(
    times: FloatArray, errors: FloatArray, threshold_frac: float, reference_amplitude: float
) -> float:
    """Return the first recorded time after the last band violation, or the end."""
    band = threshold_frac * abs(reference_amplitude)
    exceeded = np.flatnonzero(np.abs(errors) > band)
    if len(exceeded) == 0:
        return float(times[0])
    return float(times[min(int(exceeded[-1]) + 1, len(times) - 1)])


def _scenario_inputs(
    cfg: Mapping[str, object],
) -> tuple[Callable[[float], object], float, float, float, FloatArray, float, int]:
    """Validate a complete scenario before resetting a controller or allocating its trace."""
    required = {"disturbance", "duration_s", "dt_s", "target_z", "x0", "settling_threshold"}
    if not required <= cfg.keys():
        raise ValueError("scenario configuration is missing required fields")
    forcing = cfg["disturbance"]
    if not callable(forcing):
        raise ValueError("disturbance must be callable")
    duration, dt = _positive(cfg["duration_s"], "duration_s"), _positive(cfg["dt_s"], "dt_s")
    ratio = _positive(duration / dt, "duration_s/dt_s")
    steps = round(ratio)
    if steps < 1 or not math.isclose(steps * dt, duration, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("duration_s must be a positive whole number of dt_s intervals")
    target, state = _finite(cfg["target_z"], "target_z"), _state(cfg["x0"])
    threshold = _positive(cfg["settling_threshold"], "settling_threshold")
    if threshold > 1.0:
        raise ValueError("settling_threshold must not exceed one")
    return forcing, duration, dt, target, state, threshold, steps


def run_scenario(
    controller_name: str, controller: ControllerProtocol, scenario_name: str, scenario_cfg: Mapping[str, object]
) -> tuple[ScenarioMetrics, TraceData]:
    """Reset and execute one scalar-error controller on actual finite Euler steps.

    Parameters
    ----------
    controller_name, scenario_name : str
        Caller labels recorded in the resulting diagnostic metrics.
    controller : ControllerProtocol
        Mutable scalar error controller; reset once before execution.
    scenario_cfg : Mapping
        Callable disturbance, positive duration_s/dt_s with integral interval
        count, finite target_z and shape-(2,) x0, and threshold in (0,1].

    Returns
    -------
    ScenarioMetrics, TraceData
        Actual initial/terminal states, truncated at the first |z|>10 m.
        No unexecuted tail is padded. Band is threshold times the maximum of
        initial absolute error, first 100 observed absolute errors and 1 mm.

    Raises
    ------
    ValueError
        Invalid configuration, scalar output/forcing, or nonfinite arithmetic.
        Such runs never return a stable metric. Controller reset/step failures
        otherwise propagate; no alternative controller is substituted.
    """
    forcing, duration, dt, target, x0, threshold, n_steps = _scenario_inputs(scenario_cfg)
    plant = LinearPlant()
    plant.reset(x0)
    controller.reset()
    positions = [plant.z]
    actions: list[float] = []
    reason = "initial_bound_exceeded" if abs(plant.z) > 10.0 else "completed"
    started = time.perf_counter()
    if reason == "completed":
        for k in range(n_steps):
            error = _finite(target - plant.z, "position error")
            u = _finite(controller.step(error, dt), "controller output")
            d = _finite(forcing(k * dt), "disturbance output")
            plant.step(u, d, dt)
            actions.append(u)
            positions.append(plant.z)
            if abs(plant.z) > 10.0:
                reason = "position_bound_exceeded"
                break
    elapsed = time.perf_counter() - started
    count = len(actions)
    times = np.arange(count + 1, dtype=np.float64) * dt
    position_array = np.asarray(positions, dtype=np.float64)
    errors = target - position_array
    held_actions = np.asarray([*actions, actions[-1] if actions else 0.0], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            ise = _finite(float(trapezoid(errors**2, times)), "ISE")
            effort = _finite(float(np.sum(np.abs(held_actions[:-1]))) * dt, "control effort")
        except FloatingPointError as exc:
            raise ValueError("metric arithmetic must remain finite") from exc
    reference = max(abs(float(errors[0])), float(np.max(np.abs(errors[:100]))), 1.0e-3)
    band = threshold * reference
    stable = reason == "completed" and count == n_steps
    metrics = ScenarioMetrics(
        controller_name,
        scenario_name,
        ise,
        _compute_settling_time(times, errors, threshold, reference),
        _finite(float(np.max(np.abs(errors))), "peak deviation"),
        effort,
        elapsed,
        stable,
        duration,
        float(times[-1]),
        n_steps,
        count,
        stable and bool(abs(errors[-1]) <= band),
        reason,
        target,
        dt,
        band,
    )
    return metrics, TraceData(times, position_array, errors, held_actions)
