# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disturbance controllers.

"""Scalar error adapters for the Python-only reduced disturbance benchmark.

The local time-scaled PID and heuristic MPC differ from the canonical Rust
flight-simulator PID and neural MPC. No native benchmark parity is claimed.
"""

from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np

from validation.disturbance_inputs import FloatArray, _count, _finite, _positive


class ControllerProtocol(Protocol):
    """Consume scalar position error in metres and return acceleration control."""

    def step(self, error: float, dt: float) -> float:
        """Advance one finite error sample with a positive plant interval in seconds."""

    def reset(self) -> None:
        """Restore controller state before an independent scenario."""


class PIDController:
    """Apply a time-scaled PID with symmetric saturation and integral rollback.

    Parameters
    ----------
    kp, ki, kd : float
        Finite gains in s^-2, s^-3 and s^-1. Negative gains are permitted.
    u_max : float
        Positive absolute acceleration limit in m/s^2.

    Notes
    -----
    The first derivative is zero; later derivatives use actual dt. Every
    saturated step rolls back that step's integral, regardless of error sign.
    Instances hold mutable history and are not safe for concurrent stepping.
    """

    _integral: float
    _prev_error: float
    _first: bool

    def __init__(self, kp: float = 1.5e4, ki: float = 3.0e3, kd: float = 150.0, u_max: float = 1.0e6) -> None:
        """Validate gains and create zero integral/derivative history."""
        self.kp = _finite(kp, "kp")
        self.ki = _finite(ki, "ki")
        self.kd = _finite(kd, "kd")
        self.u_max = _positive(u_max, "u_max")
        self.reset()

    def step(self, error: float, dt: float) -> float:
        """Return bounded acceleration; refuse nonfinite arithmetic before state commit."""
        e, interval = _finite(error, "error"), _positive(dt, "dt")
        integral = _finite(self._integral + e * interval, "PID integral")
        derivative = 0.0 if self._first else (e - self._prev_error) / interval
        raw = _finite(self.kp * e + self.ki * integral + self.kd * derivative, "PID output")
        result = float(np.clip(raw, -self.u_max, self.u_max))
        self._integral = self._integral if abs(raw) > self.u_max else integral
        self._prev_error, self._first = e, False
        return result

    def reset(self) -> None:
        """Clear integral and previous error; suppress the next derivative sample."""
        self._integral = self._prev_error = 0.0
        self._first = True


class MPCController:
    """Apply the retained short-horizon approximate-gradient position controller.

    Parameters
    ----------
    gamma_growth : float
        Positive plant coefficient in s^-1; damping remains 10 s^-1.
    horizon, iterations : int
        Positive prediction-step and optimisation-iteration counts.
    q_weight, r_weight : float
        Nonnegative position weight and positive action regularisation weight.
    learning_rate, u_max : float
        Positive update scale and acceleration bound in m/s^2.

    Notes
    -----
    Reconstruct position/velocity from negative error and its finite difference.
    Each call starts a zero action sequence. The gradient uses one-step
    dt^2 sensitivity, not the full horizon adjoint; optimal MPC is not claimed.
    Instances hold mutable measurement history and are not thread safe.
    """

    _x_hat: FloatArray
    _prev_error: float
    _first: bool

    def __init__(
        self,
        gamma_growth: float = 100.0,
        horizon: int = 10,
        q_weight: float = 1.0e4,
        r_weight: float = 1.0e-2,
        iterations: int = 15,
        learning_rate: float = 0.1,
        u_max: float = 1.0e6,
    ) -> None:
        """Validate the prediction domain and initialize zero measurement history."""
        gamma = _positive(gamma_growth, "gamma_growth")
        squared = _finite(gamma * gamma, "gamma_growth squared")
        self.horizon = _count(horizon, "horizon")
        self.iterations = _count(iterations, "iterations")
        self.q_weight, self.r_weight = _finite(q_weight, "q_weight"), _positive(r_weight, "r_weight")
        if self.q_weight < 0.0:
            raise ValueError("q_weight must be nonnegative")
        self.lr, self.u_max = _positive(learning_rate, "learning_rate"), _positive(u_max, "u_max")
        self.A = np.array([[0.0, 1.0], [squared, -10.0]])
        self.B, self.C = np.array([0.0, 1.0]), np.array([1.0, 0.0])
        self.reset()

    def step(self, error: float, dt: float) -> float:
        """Return the first clipped action; reject nonfinite predictions before history commit."""
        e, interval = _finite(error, "error"), _positive(dt, "dt")
        de = 0.0 if self._first else _finite((e - self._prev_error) / interval, "MPC velocity")
        estimate = np.array([-e, -de])
        actions = np.zeros(self.horizon)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                for _ in range(self.iterations):
                    states = np.zeros((self.horizon + 1, 2))
                    states[0] = estimate
                    for k in range(self.horizon):
                        states[k + 1] = states[k] + interval * (self.A @ states[k] + self.B * actions[k])
                    gradient = 2.0 * self.r_weight * actions + 2.0 * self.q_weight * states[1:, 0] * (
                        self.B[1] * interval**2
                    )
                    actions = np.clip(actions - self.lr * gradient, -self.u_max, self.u_max)
            except (FloatingPointError, OverflowError) as exc:
                raise ValueError("MPC prediction arithmetic must remain finite") from exc
        result = _finite(float(actions[0]), "MPC output")
        self._x_hat, self._prev_error, self._first = estimate, e, False
        return result

    def reset(self) -> None:
        """Clear the estimated state and finite-difference measurement history."""
        self._x_hat = np.zeros(2)
        self._prev_error, self._first = 0.0, True


class HInfinityErrorController:
    """Adapt target-minus-position error to the real DGKF measurement convention.

    Parameters
    ----------
    gamma_growth : float
        Positive unstable plant coefficient in s^-1, with damping 10 s^-1.

    Notes
    -----
    The defining factory closes feedback on positive measured position. With
    the benchmark's zero target, feed negative error into that same controller.
    A nonzero target is an offset diagnostic, not an admitted tracking design.
    Synthesis/discretisation and reset remain the defining public implementation.
    """

    def __init__(self, gamma_growth: float = 100.0) -> None:
        """Construct the actual normalized controller without substituting a provider."""
        from scpn_control.control.h_infinity_controller import get_radial_robust_controller

        self.controller = get_radial_robust_controller(gamma_growth=_positive(gamma_growth, "gamma_growth"))

    def step(self, error: float, dt: float) -> float:
        """Convert error sign and advance the real single-input DGKF controller."""
        value = self.controller.step(-_finite(error, "error"), _positive(dt, "dt"))
        return _finite(float(np.asarray(value).item()), "H-infinity output")

    def reset(self) -> None:
        """Clear the actual dynamic DGKF controller state through its public reset."""
        self.controller.reset()


class SNNControllerWrapper:
    """Scale the real SC-NeuroCore rate-coded pool into acceleration control.

    Parameters
    ----------
    n_neurons, tau_window : int
        Positive neuron-per-population and rate-history sample counts.
    gain : float
        Finite acceleration gain multiplying the normalised rate difference.
    seed : int
        Nonnegative declared seed. The SC-NeuroCore provider uses its own
        fixed neuron seeds; this wrapper does not claim native seed control.

    Notes
    -----
    The plant dt is validated but does not set the provider neuron clock.
    Each reset reconstructs a fresh pool, including provider neurons and RNG
    state. Legacy NumPy fallback and quantum entropy are not enabled.
    """

    def __init__(self, n_neurons: int = 50, gain: float = 5.0e3, tau_window: int = 20, seed: int = 42) -> None:
        """Validate the declared dimensions and construct the real provider pool."""
        self._n, self._tau = _count(n_neurons, "n_neurons"), _count(tau_window, "tau_window")
        self._seed, self._gain = _count(seed, "seed", 0), _finite(gain, "gain")
        self.reset()

    def reset(self) -> None:
        """Reconstruct seeded provider cells and rate history without private state mutation."""
        from scpn_control.control.neuro_cybernetic_controller import SpikingControllerPool

        self._pool = SpikingControllerPool(
            n_neurons=self._n, gain=1.0, tau_window=self._tau, seed=self._seed, allow_numpy_fallback=False
        )

    @property
    def backend(self) -> str:
        """Return the actual provider label; no NumPy fallback is substituted."""
        return self._pool.backend

    def step(self, error: float, dt: float) -> float:
        """Advance one provider sample and return finite scaled acceleration."""
        e = _finite(error, "error")
        _positive(dt, "dt")
        return _finite(self._pool.step(e) * self._gain, "SNN output")


def build_controllers() -> dict[str, ControllerProtocol]:
    """Build actual available controllers and warn if optional construction fails.

    Returns
    -------
    dict of str to ControllerProtocol
        Fresh PID/MPC and successfully constructed real DGKF/SNN providers.
        Dictionary order is PID, H-infinity, MPC, SNN when all are available.

    Notes
    -----
    Optional failures omit their named controller with a fixed warning; they
    never install, train or relabel another algorithm. Report metadata must
    retain missing names. The native SNN clock is not synchronised to plant dt.
    """
    controllers: dict[str, ControllerProtocol] = {"PID": PIDController()}
    try:
        controllers["H-infinity"] = HInfinityErrorController()
    except (ImportError, ValueError, np.linalg.LinAlgError):
        warnings.warn("H-infinity provider construction failed; controller omitted.", stacklevel=2)
    controllers["MPC"] = MPCController()
    try:
        controllers["SNN"] = SNNControllerWrapper()
    except (ImportError, ValueError, RuntimeError):
        warnings.warn("SC-NeuroCore provider unavailable; SNN controller omitted.", stacklevel=2)
    return controllers
