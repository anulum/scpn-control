# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller Tuning
"""Automated controller gain tuning using Bayesian optimisation.

Optimises PID and H-infinity parameters against Gymnasium environments to
minimise tracking error. The PID objective evaluates the full parallel-form
control law ``u = Kp·e + Ki·∫e dt + Kd·de/dt`` with anti-windup, so the tuned
integral and derivative gains are the ones actually applied during the rollout
(not merely suggested and discarded).

Optuna is an optional dependency; install the ``tuning`` extra
(``pip install scpn-control[tuning]``) to enable optimisation. When it is
absent the tuners log a warning and return conservative default gains.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)

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
        If the resolved period is not strictly positive.
    """
    candidate: float | None = dt
    if candidate is None:
        for holder in (env, getattr(env, "unwrapped", None)):
            attr = getattr(holder, "dt", None)
            if attr is not None:
                candidate = float(attr)
                break
    period = _DEFAULT_DT if candidate is None else float(candidate)
    if not period > 0.0:
        raise ValueError(f"control period dt must be strictly positive, got {period}")
    return period


def _pid_episode_iae(
    env: Any,
    *,
    kp: float,
    ki: float,
    kd: float,
    dt: float,
    integral_clamp: float = _INTEGRAL_CLAMP,
) -> float:
    """Roll out one episode under a discrete PID law and return its IAE.

    The control law is the parallel-form PID
    ``u = Kp·e + Ki·∫e dt + Kd·de/dt`` with a symmetric anti-windup clamp on the
    integrator. The derivative is taken on the tracking error and is zero on the
    first step (``prev_error`` is initialised to the initial error) to avoid a
    derivative kick.

    Parameters
    ----------
    env : Any
        Environment exposing the Gymnasium ``reset``/``step`` contract; the
        observation vector reports the tracking error in element ``0``.
    kp, ki, kd : float
        Proportional, integral and derivative gains.
    dt : float
        Control period in seconds.
    integral_clamp : float, optional
        Symmetric bound applied to the accumulated integral term.

    Returns
    -------
    float
        Integral of the absolute error ``Σ |e|·dt`` accumulated over the
        episode.
    """
    obs, _info = env.reset()
    integral = 0.0
    prev_error = float(obs[0])
    total_iae = 0.0
    done = False
    while not done:
        error = float(obs[0])
        integral += error * dt
        integral = min(max(integral, -integral_clamp), integral_clamp)
        derivative = (error - prev_error) / max(dt, _DT_EPS)
        action = kp * error + ki * integral + kd * derivative
        prev_error = error
        obs, _reward, terminated, truncated, _info = env.step(action)
        total_iae += abs(error) * dt
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
        Gymnasium tokamak control environment; observation element ``0`` is the
        tracking error.
    n_trials : int, optional
        Number of Optuna trials.
    dt : float or None, optional
        Control period in seconds. When ``None`` the environment's ``dt`` is
        used if present, otherwise :data:`_DEFAULT_DT`.

    Returns
    -------
    dict[str, float]
        Optimal ``{"Kp", "Ki", "Kd"}`` gains. When Optuna is unavailable a
        warning is logged and conservative default gains are returned.
    """
    if not HAS_OPTUNA:
        logger.warning("Optuna not installed; returning default PID gains.")
        return {"Kp": 1.0, "Ki": 0.1, "Kd": 0.05}

    control_period = _resolve_control_period(env, dt)

    def objective(trial: optuna.Trial) -> float:
        kp = trial.suggest_float("Kp", 0.1, 10.0, log=True)
        ki = trial.suggest_float("Ki", 0.01, 1.0, log=True)
        kd = trial.suggest_float("Kd", 0.01, 1.0, log=True)

        total_iae = 0.0
        for _ in range(_TUNE_EPISODES):
            total_iae += _pid_episode_iae(env, kp=kp, ki=ki, kd=kd, dt=control_period)
        return total_iae / _TUNE_EPISODES

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return dict(study.best_params)


def tune_hinf(plant: dict[str, Any], n_trials: int = 50) -> dict[str, float]:
    """Tune H-infinity parameters (gamma, bandwidth) using Optuna."""
    if not HAS_OPTUNA:
        return {"gamma": 1.1, "bandwidth": 0.5}

    def objective(trial: optuna.Trial) -> float:
        gamma = trial.suggest_float("gamma", 1.01, 2.0)
        return float(abs(gamma - 1.1))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return dict(study.best_params)
