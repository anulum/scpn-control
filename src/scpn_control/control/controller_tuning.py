# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller Tuning
"""Controller tuning through actual rollout and normalized-plant synthesis.

Optimises PID and H-infinity parameters against Gymnasium environments to
minimise tracking error. The PID objective evaluates the full parallel-form
control law ``u = Kp·e + Ki·∫e dt + Kd·de/dt`` with anti-windup, so the tuned
integral and derivative gains are the ones actually applied during the rollout
(not merely suggested and discarded).

Optuna is an optional dependency for PID rollout optimisation; install the
``tuning`` extra (``pip install scpn-control[tuning]``) to enable it. When it is
absent PID tuning refuses to fabricate gains.
H-infinity attenuation comes from the normalized DGKF plant, independent of
Optuna. No bandwidth value is inferred from the plant matrices.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.control.gym_tokamak_env import TokamakEnv
from scpn_control.control.h_infinity_controller import HInfinityController

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Fallback control period when the environment exposes no explicit ``dt`` [s].
_DEFAULT_DT: float = 1.0
# Minimum time-step guard for the derivative denominator [s].
_DT_EPS: float = 1e-6
# Symmetric anti-windup clamp on the accumulated integral term [error·s].
# Bounds integrator wind-up during sustained error so the integral contribution
# cannot dominate the command (Åström & Murray, *Feedback Systems*, 2008, Ch. 11
# — integrator anti-windup).
_INTEGRAL_CLAMP: float = 100.0
# Episodes averaged per candidate gain evaluation.
_TUNE_EPISODES: int = 5


def _resolve_control_period(env: Any, dt: float | None) -> float:
    """Resolve the discrete control period used by the PID rollout.

    Parameters
    ----------
    env : Any
        Environment being tuned. When ``dt`` is not supplied, the environment
        and its ``unwrapped`` view are probed for a ``dt`` attribute (the
        Gymnasium convention for a fixed integration step).
    dt : float or None
        Explicit control period in seconds. When ``None`` the environment is
        probed and, failing that, :data:`_DEFAULT_DT` is used.

    Returns
    -------
    float
        Strictly positive control period in seconds.

    Raises
    ------
    ValueError
        If the resolved period is nonfinite or not strictly positive.
    """
    candidate: float | None = dt
    if candidate is None:
        for holder in (env, getattr(env, "unwrapped", None)):
            attr = getattr(holder, "dt", None)
            if attr is not None:
                candidate = float(attr)
                break
    period = _DEFAULT_DT if candidate is None else float(candidate)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError(f"control period dt must be finite and strictly positive, got {period}")
    return period


def _pid_episode_iae(
    env: Any,
    *,
    kp: float,
    ki: float,
    kd: float,
    dt: float,
    integral_clamp: float = _INTEGRAL_CLAMP,
    seed: int | None = None,
) -> float:
    """Roll out one episode under a discrete PID law and return its IAE.

    The control law is the parallel-form PID
    ``u = Kp·e + Ki·∫e dt + Kd·de/dt`` with a symmetric anti-windup clamp on the
    integrator. The derivative is taken on the tracking error and is zero on the
    first step (``prev_error`` is initialised to the initial error) to avoid a
    derivative kick. For ``TokamakEnv``, the error is
    ``T_target - T_axis`` and the computed command changes heating while the
    current command stays zero. A scalar-error environment instead supplies
    error in observation element zero and accepts a scalar action.

    Parameters
    ----------
    env : Any
        ``TokamakEnv`` or a scalar-error environment exposing ``reset`` and
        ``step``; the latter reports tracking error in observation element 0.
    kp, ki, kd : float
        Proportional, integral and derivative gains.
    dt : float
        Control period in seconds.
    integral_clamp : float, optional
        Symmetric bound applied to the accumulated integral term.
    seed : int or None, optional
        For ``TokamakEnv``, reset with this seed so each gain candidate sees
        the same disturbance realization for the matching episode.

    Returns
    -------
    float
        Integral of the absolute error ``Σ |e|·dt`` accumulated over the
        episode.
    """
    tokamak_env = isinstance(env, TokamakEnv)
    obs, _info = env.reset(seed=seed) if tokamak_env and seed is not None else env.reset()
    integral = 0.0
    prev_error = float(env.T_target - obs[0]) if tokamak_env else float(obs[0])
    if not math.isfinite(prev_error):
        raise ValueError("initial tracking error must be finite")
    total_iae = 0.0
    done = False
    while not done:
        error = float(env.T_target - obs[0]) if tokamak_env else float(obs[0])
        if not math.isfinite(error):
            raise ValueError("tracking error must be finite")
        integral += error * dt
        integral = min(max(integral, -integral_clamp), integral_clamp)
        derivative = (error - prev_error) / max(dt, _DT_EPS)
        action = kp * error + ki * integral + kd * derivative
        if not math.isfinite(action):
            raise ValueError("PID action must be finite")
        prev_error = error
        if tokamak_env:
            heating_delta = float(np.clip(action, env.action_low[0], env.action_high[0]))
            commanded_action: float | FloatArray = np.array([heating_delta, 0.0], dtype=np.float64)
        else:
            commanded_action = action
        obs, _reward, terminated, truncated, _info = env.step(commanded_action)
        total_iae += abs(error) * dt
        if not math.isfinite(total_iae):
            raise ValueError("integrated tracking error must be finite")
        done = bool(terminated or truncated)
    return total_iae


def tune_pid(env: Any, n_trials: int = 50, dt: float | None = None) -> dict[str, float]:
    """Tune parallel-form PID gains (Kp, Ki, Kd) with Optuna.

    Every trial evaluates a full proportional-integral-derivative rollout — the
    suggested integral and derivative gains are applied, not just the
    proportional term — minimising the mean integral-of-absolute-error across
    :data:`_TUNE_EPISODES` episodes.

    Parameters
    ----------
    env : Any
        ``TokamakEnv`` (temperature target and two actuator channels) or a
        scalar-error environment with the documented reset/step contract.
    n_trials : int, optional
        Positive integer number of Optuna trials.
    dt : float or None, optional
        Control period in seconds. When ``None`` the environment's ``dt`` is
        used if present, otherwise :data:`_DEFAULT_DT`.

    Returns
    -------
    dict[str, float]
        Optimized ``{"Kp", "Ki", "Kd"}`` gains for the supplied environment.

    Raises
    ------
    ImportError
        If Optuna is unavailable; no defaults are represented as tuned gains.
    ValueError
        If the period or trial count is invalid.
    """
    control_period = _resolve_control_period(env, dt)
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    if not HAS_OPTUNA:
        raise ImportError("Optuna is required for PID tuning; install scpn-control[tuning]")

    def objective(trial: optuna.Trial) -> float:
        kp = trial.suggest_float("Kp", 0.1, 10.0, log=True)
        ki = trial.suggest_float("Ki", 0.01, 1.0, log=True)
        kd = trial.suggest_float("Kd", 0.01, 1.0, log=True)

        total_iae = 0.0
        for episode in range(_TUNE_EPISODES):
            total_iae += _pid_episode_iae(env, kp=kp, ki=ki, kd=kd, dt=control_period, seed=42 + episode)
        return total_iae / _TUNE_EPISODES

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return dict(study.best_params)


def tune_hinf(plant: dict[str, Any]) -> dict[str, float]:
    """Return a feasible near-infimum DGKF attenuation for a normalized plant.

    The plant must supply exactly ``A``, ``B1``, ``B2``, ``C1``, ``C2``,
    ``D12`` and ``D21``. The public synthesis constructor validates shapes,
    finite values, normalization, stabilizability and strict feasibility.
    There is no data-independent default or inferred bandwidth. The returned
    attenuation applies to the unsaturated linear continuous-time model.
    """
    required = {"A", "B1", "B2", "C1", "C2", "D12", "D21"}
    if set(plant) != required:
        raise ValueError("plant must contain exactly A, B1, B2, C1, C2, D12 and D21")
    controller = HInfinityController(
        A=plant["A"],
        B1=plant["B1"],
        B2=plant["B2"],
        C1=plant["C1"],
        C2=plant["C2"],
        D12=plant["D12"],
        D21=plant["D21"],
    )
    return {"gamma": float(controller.gamma)}
