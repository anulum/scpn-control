# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gymnasium policy evaluation
"""Expose the actual Gymnasium wrapper and seeded reduced-order policy evaluation.

Observations/actions use float32 at the Gym boundary; the unchanged underlying
TokamakEnv evolves its float64 model. These episode statistics describe that
model, not experimental plasma performance or a controller safety certificate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from tools.rl_training_results import EvaluationStats, validate_stats

Observation = NDArray[np.float32]
Action = NDArray[np.float32]


class PredictivePolicy(Protocol):
    """A deterministic prediction interface supplied by actual SB3 policies."""

    def predict(self, observation: Observation, *, deterministic: bool = False) -> tuple[Action, object]:
        """Return a model action and opaque recurrent state for an observation."""
        ...


class GymTokamakEnv(gym.Env[Observation, Action]):
    """Wrap TokamakEnv with float32 Box spaces and the five-result Gym interface.

    Parameters
    ----------
    dt : float
        Physical model timestep in seconds, default 1e-3.
    max_steps : int
        Positive episode horizon, default 500.
    T_target : float
        Positive target axis temperature in keV, default 20.

    Notes
    -----
    The underlying model validates construction and actions. Gym reset options
    are accepted but unused; resetting with a seed also seeds the actual model.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dt: float = 1e-3, max_steps: int = 500, T_target: float = 20.0) -> None:
        """Construct the unchanged reduced-order model and its float32 spaces."""
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("episode horizon must be a positive integer")
        super().__init__()
        from scpn_control.control.gym_tokamak_env import TokamakEnv

        self._env = TokamakEnv(dt=dt, max_steps=max_steps, T_target=T_target)
        self.observation_space = spaces.Box(
            low=self._env.observation_low.astype(np.float32),
            high=self._env.observation_high.astype(np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=self._env.action_low.astype(np.float32), high=self._env.action_high.astype(np.float32), dtype=np.float32
        )

    def reset(
        self, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[Observation, dict[str, object]]:
        """Seed/reset both Gym and the actual model.

        Parameters
        ----------
        seed : int or None
            Seed supplied to Gym and the underlying observation RNG. None
            retains the underlying RNG stream.
        options : dict or None
            Accepted for Gym compatibility; no model option is applied.

        Returns
        -------
        tuple
            Float32 observation [T_axis, T_edge, beta_N, li, q95, Ip] and the
            underlying empty reset-information mapping.
        """
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed)
        return obs.astype(np.float32), info

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, object]]:
        """Advance the original model once, preserving reward and termination flags.

        Parameters
        ----------
        action : ndarray, shape (2,)
            Finite heating/current corrections. The underlying model clips
            them to [-5,5] MW and [-1,1] current-control units.

        Returns
        -------
        tuple
            Float32 observation, reward, disrupted flag, horizon flag and
            actual model diagnostics. Both flags may be true on the last step.

        Raises
        ------
        ValueError
            Invalid action shape/values or nonfinite model update. The
            underlying model preserves its episode state on refusal.
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self) -> None:
        """Delegate state logging to the underlying model without a graphical renderer."""
        self._env.render()


class PIDController:
    """Retain the legacy named baseline's two proportional control corrections.

    Parameters
    ----------
    T_target : float
        Axis-temperature target, default 20 keV.
    Kp_T : float
        Heating correction gain, default 0.5 MW/keV.
    Kp_Ip : float
        Current correction gain toward the fixed initial 15 MA target, default 0.1.

    Notes
    -----
    Despite the legacy name, this implementation has no integral/derivative term.
    Heating/current actions are clipped to [-5,5]/[-1,1]. A short observation uses
    the original 15 MA fallback; arithmetic and tuning are unchanged.
    """

    def __init__(self, T_target: float = 20.0, Kp_T: float = 0.5, Kp_Ip: float = 0.1) -> None:
        """Store the original gains and fixed current target without retuning."""
        self.T_target = T_target
        self.Kp_T = Kp_T
        self.Kp_Ip = Kp_Ip
        self.Ip_target = 15.0

    def act(self, obs: Observation) -> Action:
        """Return the original clipped heating/current proportional action.

        Parameters
        ----------
        obs : ndarray
            Temperature in element zero and current in element five. Short
            observations retain the legacy 15 MA current fallback.

        Returns
        -------
        ndarray, shape (2,), dtype float32
            Heating/current corrections clipped to [-5,5]/[-1,1].
        """
        T_ax = float(obs[0])
        Ip = float(obs[5]) if len(obs) > 5 else 15.0
        P_delta = np.clip(self.Kp_T * (self.T_target - T_ax), -5.0, 5.0)
        Ip_delta = np.clip(self.Kp_Ip * (self.Ip_target - Ip), -1.0, 1.0)
        return np.array([P_delta, Ip_delta], dtype=np.float32)


def evaluate_agent(
    env: GymTokamakEnv, predict_fn: Callable[[Observation], Action] | PredictivePolicy, n_episodes: int = 20
) -> EvaluationStats:
    """Evaluate actual policies on episode reset seeds 1000 through 1000+N-1.

    Call a supplied controller directly or request deterministic model.predict.
    Require positive integer N; count any terminated episode as disrupted and
    stop on either terminated or truncated. Return population reward deviation,
    mean episode length and termination fraction. Nonfinite accumulated rewards
    or summary values refuse. This advances the supplied environment in place;
    failed evaluation does not roll back already executed episode transitions.

    Parameters
    ----------
    env : GymTokamakEnv
        Actual environment advanced and reset for each episode.
    predict_fn : callable or PredictivePolicy
        Float32-observation controller or model with deterministic prediction.
    n_episodes : int
        Positive count, default 20, paired reset seeds start at 1000.

    Returns
    -------
    EvaluationStats
        Reward mean/population deviation, mean length, termination fraction
        and episode count for the reduced-order model.

    Raises
    ------
    ValueError
        Invalid episode count or nonfinite accumulated reward/summary.
    FloatingPointError
        Reward summary arithmetic overflows or is invalid.
    """
    if isinstance(n_episodes, bool) or not isinstance(n_episodes, int) or n_episodes < 1:
        raise ValueError("evaluation count must be a positive integer")
    rewards: list[float] = []
    lengths: list[int] = []
    disruptions = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 1000)
        total_reward = 0.0
        for step in range(env._env.max_steps):
            if callable(predict_fn):
                action = predict_fn(obs)
            else:
                action, _ = predict_fn.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if not math.isfinite(total_reward):
                raise ValueError("episode reward must remain finite")
            if terminated:
                disruptions += 1
                break
            if truncated:
                break
        rewards.append(total_reward)
        lengths.append(step + 1)
    with np.errstate(over="raise", invalid="raise"):
        stats = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
            "disruption_rate": disruptions / n_episodes,
            "n_episodes": n_episodes,
        }
    return validate_stats(stats)
