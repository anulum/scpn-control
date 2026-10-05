# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gymnasium-compatible reduced-order tokamak environment
"""
Gymnasium-compatible environment for tokamak plasma control.

Implements bounded reduced-order 0D physics dynamics for temperature and
plasma current using energy balance ($dW/dt = P_{heat} - P_{loss}$)
with IPB98(y,2) confinement scaling and Bremsstrahlung radiation.

Reference: Wesson, J. (2011). Tokamaks. 4th Edition.

Observation: [T_axis, T_edge, beta_N, li, q95, Ip]  (6-dim)
Action:      [P_aux_delta, Ip_delta]               (2-dim, continuous)
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

logger = logging.getLogger(__name__)


class TokamakEnv:
    """Minimal Gymnasium-compatible tokamak control environment.

    Implements a bounded reduced-order plasma response model for energy and current
    evolution based on 0D lumped-parameter equations.
    Ref: Wesson, J. (2011). Tokamaks. 4th Edition, Chapter 1.

    Follows the gymnasium.Env interface (reset/step/render) without
    requiring gymnasium as a hard dependency. If gymnasium is installed,
    this class can be registered via ``gymnasium.register()``.

    Parameters
    ----------
    dt : float
        Timestep per step call [s].
    max_steps : int
        Episode length.
    T_target : float
        Target axis temperature [keV].
    noise_std : float
        Observation noise standard deviation.
    n_e_20 : float
        Line-averaged electron density [10^20 m^-3]. Default 1.0.
    V_plasma : float
        Plasma volume [m^3]. Default 830.0 (ITER).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dt: float = 1e-3,
        max_steps: int = 500,
        T_target: float = 20.0,
        noise_std: float = 0.01,
        seed: int = 42,
        n_e_20: float = 1.0,
        V_plasma: float = 830.0,
    ):
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive for a physical timestep.")
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if not np.isfinite(T_target) or T_target <= 0.0:
            raise ValueError("T_target must be finite and positive.")
        if not np.isfinite(noise_std) or noise_std < 0.0:
            raise ValueError("noise_std must be finite and non-negative.")
        if not np.isfinite(n_e_20) or n_e_20 <= 0.0:
            raise ValueError("n_e_20 must be finite and positive.")
        if not np.isfinite(V_plasma) or V_plasma <= 0.0:
            raise ValueError("V_plasma must be finite and positive.")
        self.dt = dt
        self.max_steps = max_steps
        self.T_target = T_target
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self.n_e_20 = n_e_20
        self.V_plasma = V_plasma

        # State: [T_axis, T_edge, beta_N, li, q95, Ip_MA]
        self._state = np.zeros(6, dtype=np.float64)
        self._step_count = 0
        self._prev_temp_err = 0.0
        self.P_aux = 50.0  # [MW] base heating

        # Bounds
        self.observation_low = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        self.observation_high = np.array([50.0, 20.0, 5.0, 3.0, 10.0, 20.0])
        self.action_low = np.array([-5.0, -1.0])
        self.action_high = np.array([5.0, 1.0])

    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        """Shape of the observation vector (6 plasma-state features)."""
        return (6,)

    @property
    def action_space_shape(self) -> tuple[int, ...]:
        """Shape of the action vector (2 actuator commands)."""
        return (2,)

    def reset(self, seed: int | None = None) -> tuple[FloatArray, dict[str, Any]]:
        """Reset to an admitted initial plasma state and observation."""
        candidate_rng = np.random.default_rng(seed) if seed is not None else self._rng
        rng_state = copy.deepcopy(candidate_rng.bit_generator.state)

        # ITER-like initial condition with small perturbation
        try:
            initial_state = np.array(
                [
                    10.0 + candidate_rng.normal(0, 0.5),
                    2.0 + candidate_rng.normal(0, 0.1),
                    1.5 + candidate_rng.normal(0, 0.1),
                    0.85 + candidate_rng.normal(0, 0.05),
                    3.0 + candidate_rng.normal(0, 0.1),
                    15.0,
                ],
                dtype=np.float64,
            )
            observation = self._observe(initial_state, candidate_rng)
        except (FloatingPointError, ValueError):
            candidate_rng.bit_generator.state = rng_state
            raise

        self._rng = candidate_rng
        self._state[:] = initial_state
        self._step_count = 0
        self._prev_temp_err = abs(float(initial_state[0]) - self.T_target)
        self.P_aux = 50.0
        return observation, {}

    def step(self, action: AnyFloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Advance one timestep using physics-based energy balance.

        Return (obs, reward, terminated, truncated, info) after finite admission.

        A failed numerical update leaves the episode and observation RNG unchanged.
        """
        action = self._validate_action(action)
        action = np.clip(action, self.action_low, self.action_high)
        P_aux_delta, Ip_delta = float(action[0]), float(action[1])

        s = self._state
        T_ax: float = float(s[0])
        T_edge: float = float(s[1])
        Ip: float = float(s[5])

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                # Candidate actuators are local until the whole step is admitted.
                next_heating = float(np.clip(self.P_aux + P_aux_delta, 0.0, 150.0))
                Ip = float(np.clip(Ip + Ip_delta * self.dt * 10.0, 0.1, 20.0))

                # Energy balance (Wesson Ch. 3 & Ch. 14).
                T_avg = 0.5 * (T_ax + T_edge)
                W_th = 0.04806 * self.n_e_20 * T_avg * self.V_plasma
                tau_E = 2.0 * (Ip / 15.0) ** 0.93 * (max(next_heating, 1.0) / 50.0) ** -0.69
                P_loss = W_th / max(tau_E, 0.1)
                P_rad = 0.00535 * (self.n_e_20**2) * 1.5 * np.sqrt(max(T_avg, 0.1)) * self.V_plasma
                dW_dt = next_heating - P_loss - P_rad
                dT_avg_dt = dW_dt / (0.04806 * self.n_e_20 * self.V_plasma)
                T_avg += dT_avg_dt * self.dt
                T_ax = 1.8 * T_avg
                T_edge = 0.2 * T_avg

                # Reduced-order diagnostic estimates.
                beta_N: float = 0.27 * T_ax / max(abs(Ip), 0.1)
                q95: float = max(45.0 / max(Ip, 0.1), 1.5)
                li: float = 0.85 + 0.1 * (q95 - 3.0)
                disrupted = q95 < 2.0 or beta_N > 3.5

                temp_err = abs(T_ax - self.T_target)
                progress = max(0.0, self._prev_temp_err - temp_err)
                reward = -temp_err + 5.0 * progress + 0.5 - 50.0 * float(disrupted) - 0.01 * np.linalg.norm(action)
                next_state = np.array([T_ax, T_edge, beta_N, li, q95, Ip], dtype=np.float64)
        except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError("model step must remain finite") from exc

        rng_state = copy.deepcopy(self._rng.bit_generator.state)
        try:
            observation = self._observe(next_state)
        except (FloatingPointError, ValueError):
            self._rng.bit_generator.state = rng_state
            raise

        self.P_aux = next_heating
        self._state[:] = next_state
        self._step_count += 1
        self._prev_temp_err = temp_err

        terminated = bool(disrupted)
        truncated = bool(self._step_count >= self.max_steps)

        info = {
            "T_axis": float(T_ax),
            "beta_N": float(beta_N),
            "q95": float(q95),
            "disrupted": bool(disrupted),
            "step": self._step_count,
        }
        return observation, float(reward), terminated, truncated, info

    def _observe(
        self,
        state: FloatArray | None = None,
        rng: np.random.Generator | None = None,
    ) -> FloatArray:
        """Return a finite noisy observation of the supplied or current state."""
        noise = (self._rng if rng is None else rng).normal(0, self.noise_std, 6)
        with np.errstate(over="raise", invalid="raise"):
            noisy_state = (self._state if state is None else state) + noise
        if not np.all(np.isfinite(noisy_state)):
            raise ValueError("observation must remain finite")
        obs = np.clip(noisy_state, self.observation_low, self.observation_high)
        return obs.astype(np.float64)

    def _validate_action(self, action: AnyFloatArray) -> FloatArray:
        arr = np.asarray(action, dtype=np.float64)
        if arr.shape != (2,):
            raise ValueError("action must have shape (2,) for [P_aux_delta, Ip_delta].")
        if not np.all(np.isfinite(arr)):
            raise ValueError("action must contain only finite values.")
        return arr

    def render(self) -> None:
        """Print current state."""
        T_ax, T_edge, beta_N, li, q95, Ip = self._state
        logger.info("step=%4d  T_ax=%.1fkeV  beta_N=%.2f  q95=%.2f  Ip=%.1fMA", self._step_count, T_ax, beta_N, q95, Ip)
