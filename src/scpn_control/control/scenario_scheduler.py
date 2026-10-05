# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Scenario Scheduler.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Feedforward Scenario Scheduler
# ──────────────────────────────────────────────────────────────────────
"""Feedforward scenario waveforms and schedule generation for tokamak scenario control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray


@dataclass
class ScenarioWaveform:
    """A named time-series waveform with interpolated lookup.

    Attributes
    ----------
    name
        Waveform identifier (e.g. ``"Ip"``, ``"P_aux"``).
    times
        Knot times in seconds, monotonically increasing.
    values
        Waveform values at the knot times.
    interp_kind
        Interpolation kind (default linear with flat extrapolation).
    """

    name: str
    times: AnyFloatArray
    values: AnyFloatArray
    interp_kind: str = "linear"

    def _validation_error(self) -> str | None:
        """Return the first structural error in this interpolation contract."""
        try:
            times = np.asarray(self.times, dtype=np.float64)
            values = np.asarray(self.values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return "knots and values must be finite numeric arrays"
        if times.ndim != 1 or values.ndim != 1 or len(times) == 0 or len(times) != len(values):
            return "knots and values must have the same nonzero length in one dimension"
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            return "knots and values must be finite"
        if np.any(np.diff(times) <= 0.0):
            return "has non-monotonic knots; times must be strictly increasing"
        if self.interp_kind != "linear":
            return "supports only linear interpolation"
        return None

    def __call__(self, t: float) -> float:
        """Evaluate the waveform at time t."""
        error = self._validation_error()
        if error is not None:
            raise ValueError(f"Waveform {self.name} {error}")
        if not math.isfinite(t):
            raise ValueError("waveform evaluation time must be finite")
        return float(np.interp(t, self.times, self.values))


class ScenarioSchedule:
    """A collection of named waveforms defining a full discharge scenario.

    Parameters
    ----------
    waveforms
        Mapping of waveform name to :class:`ScenarioWaveform`.
    """

    def __init__(self, waveforms: dict[str, ScenarioWaveform]):
        self.waveforms = waveforms

    def evaluate(self, t: float) -> dict[str, float]:
        """Evaluate all waveforms at time t."""
        return {name: wf(t) for name, wf in self.waveforms.items()}

    def duration(self) -> float:
        """Total duration of the scenario."""
        if not self.waveforms:
            return 0.0
        for name, wf in self.waveforms.items():
            error = wf._validation_error()
            if error is not None:
                raise ValueError(f"Waveform {name} {error}")
        return float(max(wf.times[-1] for wf in self.waveforms.values()))

    def validate(self) -> list[str]:
        """Check for physical bounds and monotonicity."""
        errors: list[str] = []
        for name, wf in self.waveforms.items():
            error = wf._validation_error()
            if error is not None:
                errors.append(f"Waveform {name} {error}.")
                continue

            if "Ip" in name and np.any(wf.values < 0):
                errors.append(f"Waveform {name} has negative plasma current.")

            if "P_" in name and np.any(wf.values < 0):
                errors.append(f"Waveform {name} has negative heating power.")

            if "n_e" in name and np.any(wf.values <= 0):
                errors.append(f"Waveform {name} has non-positive density.")

        return errors


class FeedforwardController:
    """Combines pre-computed feedforward trajectories with a feedback trim."""

    def __init__(self, schedule: ScenarioSchedule, feedback: Callable[..., AnyFloatArray]):
        self.schedule = schedule
        self.feedback = feedback

    def step(self, x: AnyFloatArray, t: float, dt: float) -> FloatArray:
        """Return the control action ``u = u_ff(t) + u_fb(x_err)``."""
        state = np.asarray(x, dtype=np.float64)
        if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError("controller state must be a nonempty finite vector")
        if not math.isfinite(t) or not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("controller time must be finite and dt must be positive and finite")
        errors = self.schedule.validate()
        if errors:
            raise ValueError(f"invalid scenario schedule: {'; '.join(errors)}")
        ff_dict = self.schedule.evaluate(t)

        # Standardize mapping from dict to control vector [P_aux, Ip_ref, n_gas]
        # In a real system this would map properly to the plant inputs
        u_ff = np.zeros(3)
        u_ff[0] = ff_dict.get("P_aux", 0.0)
        u_ff[1] = ff_dict.get("Ip", 0.0)
        u_ff[2] = ff_dict.get("n_gas", 0.0)

        # Standardize reference vector
        x_ref = np.zeros(len(state))
        x_ref[0] = ff_dict.get("Ip", 0.0)

        # Calculate feedback
        u_fb = np.asarray(self.feedback(state, x_ref, t, dt), dtype=np.float64)
        if u_fb.shape != (3,) or not np.all(np.isfinite(u_fb)):
            raise ValueError("feedback must contain three finite control values")
        with np.errstate(over="raise", invalid="raise"):
            try:
                action = np.add(u_ff, u_fb)
            except FloatingPointError as exc:
                raise ValueError("combined control action must be finite") from exc
        return np.asarray(action, dtype=np.float64)


class ScenarioOptimizer:
    """Offline trajectory design."""

    def __init__(
        self, plant_model: Callable[..., AnyFloatArray], target_state: AnyFloatArray, T_total: float, dt: float = 0.5
    ):
        if not math.isfinite(T_total) or T_total <= 0.0:
            raise ValueError("T_total must be positive and finite")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        target = np.asarray(target_state, dtype=np.float64)
        if target.ndim != 1 or target.size == 0 or not np.all(np.isfinite(target)):
            raise ValueError("target_state must be a nonempty finite vector")
        self.plant_model = plant_model
        self.target_state = target.copy()
        self.T_total = T_total
        self.dt = dt

    def optimize(self, n_iter: int = 100) -> ScenarioSchedule:
        """Gradient-free optimization of breakpoint values."""
        if isinstance(n_iter, bool) or not isinstance(n_iter, Integral) or n_iter <= 0:
            raise ValueError("n_iter must be a positive integer")
        # Define 3 breakpoints for simplicity: 0, T/2, T
        times = np.array([0.0, self.T_total / 2.0, self.T_total])

        # Initial guess (flat at target)
        # Assuming u = [P_aux, Ip_ref]
        n_u = 2
        p0 = np.zeros(n_u * len(times))

        # Objective function
        def objective(p: AnyFloatArray) -> float:
            p = np.maximum(np.asarray(p, dtype=np.float64).reshape(n_u, len(times)), 0.0)
            wfs = {"P_aux": ScenarioWaveform("P_aux", times, p[0]), "Ip": ScenarioWaveform("Ip", times, p[1])}
            sched = ScenarioSchedule(wfs)

            x: AnyFloatArray = np.zeros(len(self.target_state))
            cost = 0.0

            t = 0.0
            while t < self.T_total:
                u_dict = sched.evaluate(t)
                u = np.array([u_dict["P_aux"], u_dict["Ip"]])

                x = self.plant_model(x, u, self.dt)

                # Tracking cost
                err = x - self.target_state
                cost += np.sum(err**2) * self.dt

                t += self.dt

            return cost

        try:
            import scipy.optimize

            res = scipy.optimize.minimize(objective, p0, method="Nelder-Mead", options={"maxiter": n_iter})
            p_opt = np.maximum(res.x.reshape(n_u, len(times)), 0.0)
        except ImportError:
            # Fallback if no scipy
            p_opt = p0.reshape(n_u, len(times))

        wfs = {"P_aux": ScenarioWaveform("P_aux", times, p_opt[0]), "Ip": ScenarioWaveform("Ip", times, p_opt[1])}
        schedule = ScenarioSchedule(wfs)
        errors = schedule.validate()
        if errors:
            raise ValueError(f"optimizer produced an invalid scenario schedule: {'; '.join(errors)}")
        return schedule


def iter_15ma_baseline() -> ScenarioSchedule:
    """Return the ITER 15 MA inductive baseline scenario schedule."""
    times = np.array([0, 10, 30, 60, 400, 430, 480], dtype=float)
    ip_vals = np.array([0.5, 5.0, 10.0, 15.0, 15.0, 10.0, 2.0])
    p_nbi = np.array([0.0, 0.0, 10.0, 33.0, 33.0, 10.0, 0.0])
    p_eccd = np.array([0.0, 0.0, 5.0, 17.0, 17.0, 5.0, 0.0])
    n_e = np.array([0.5, 1.0, 3.0, 5.0, 5.0, 3.0, 0.5])

    wfs = {
        "Ip": ScenarioWaveform("Ip", times, ip_vals),
        "P_NBI": ScenarioWaveform("P_NBI", times, p_nbi),
        "P_ECCD": ScenarioWaveform("P_ECCD", times, p_eccd),
        "n_e": ScenarioWaveform("n_e", times, n_e),
        "P_aux": ScenarioWaveform("P_aux", times, p_nbi + p_eccd),
    }
    return ScenarioSchedule(wfs)


def nstx_u_1ma_standard() -> ScenarioSchedule:
    """Return the NSTX-U 1 MA standard scenario schedule."""
    times = np.array([0.0, 0.2, 0.5, 1.5, 1.8, 2.0])
    ip_vals = np.array([0.1, 0.5, 1.0, 1.0, 0.5, 0.1])
    p_aux = np.array([0.0, 2.0, 8.0, 8.0, 2.0, 0.0])

    wfs = {"Ip": ScenarioWaveform("Ip", times, ip_vals), "P_aux": ScenarioWaveform("P_aux", times, p_aux)}
    return ScenarioSchedule(wfs)
