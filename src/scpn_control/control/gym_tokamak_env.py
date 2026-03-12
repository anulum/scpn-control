# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Gymnasium Tokamak Environment
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""
Gymnasium-compatible environment wrapping the TokamakDigitalTwin.

Observation: [T_axis, T_edge, beta_N, li, q95, Ip_err]  (6-dim)
Action:      [P_aux_delta, Ip_delta]                      (2-dim, continuous)

Reward: -|T_axis - T_target| - 10*(disruption) - 0.01*|u|

Requires: ``pip install gymnasium`` (optional dependency).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class TokamakEnv:
    """Minimal Gymnasium-compatible tokamak control environment.

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
    def observation_space_shape(self) -> tuple:
        return (6,)

    @property
    def action_space_shape(self) -> tuple:
        return (2,)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset to initial plasma state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # ITER-like initial condition with small perturbation
        self._state = np.array(
            [
                10.0 + self._rng.normal(0, 0.5),  # T_axis [keV]
                2.0 + self._rng.normal(0, 0.1),  # T_edge [keV]
                1.5 + self._rng.normal(0, 0.1),  # beta_N
                0.85 + self._rng.normal(0, 0.05),  # li
                3.0 + self._rng.normal(0, 0.1),  # q95
                15.0,  # Ip [MA]
            ]
        )
        self._step_count = 0
        self._prev_temp_err = abs(self._state[0] - self.T_target)
        self.P_aux = 50.0
        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one timestep using physics-based energy balance.

        Returns (obs, reward, terminated, truncated, info).
        """
        action = np.clip(action, self.action_low, self.action_high)
        P_aux_delta, Ip_delta = float(action[0]), float(action[1])

        s = self._state
        T_ax: float = float(s[0])
        T_edge: float = float(s[1])
        Ip: float = float(s[5])

        # 1. Update heating and current
        self.P_aux = float(np.clip(self.P_aux + P_aux_delta, 0.0, 150.0))
        Ip = float(np.clip(Ip + Ip_delta * self.dt * 10.0, 0.1, 20.0))

        # 2. Physics: Energy Balance (Wesson Ch. 3 & Ch. 14)
        # Average temperature assuming parabolic profile: <T> = 0.5 * (T_ax + T_edge)
        T_avg = 0.5 * (T_ax + T_edge)
        
        # Stored Energy: W = 3/2 * (n_e + n_i) * <T> * V. Assume n_e = n_i.
        # W [MJ] = 3 * n_e_20 * T_avg_keV * V * 1.602e-19 * 1e20 * 1e-6
        # W [MJ] = 0.04806 * n_e_20 * T_avg * V
        W_th = 0.04806 * self.n_e_20 * T_avg * self.V_plasma
        
        # Confinement Time: Simplified IPB98(y,2) scaling
        # tau_E ~ Ip^0.93 * P^-0.69. Calibrated to ~2.0s for ITER at 15MA, 50MW.
        tau_E = 2.0 * (Ip / 15.0)**0.93 * (max(self.P_aux, 1.0) / 50.0)**-0.69
        
        # Losses: P_loss = W / tau_E [MW]
        P_loss = W_th / max(tau_E, 0.1)
        
        # Bremsstrahlung: P_br [MW] = 0.00535 * n_e_20^2 * Z_eff * sqrt(Te) * V
        # Reference: Wesson Ch. 14.5.1. Z_eff = 1.5.
        P_rad = 0.00535 * (self.n_e_20**2) * 1.5 * np.sqrt(max(T_avg, 0.1)) * self.V_plasma
        
        # Energy rate: dW/dt = P_aux - P_loss - P_rad
        dW_dt = self.P_aux - P_loss - P_rad
        
        # Temperature rate: d<T>/dt = dW_dt / (0.04806 * n_e_20 * V)
        dT_avg_dt = dW_dt / (0.04806 * self.n_e_20 * self.V_plasma)
        
        # Update T_ax and T_edge (simple profile relaxation)
        T_avg += dT_avg_dt * self.dt
        T_ax = 1.8 * T_avg # Maintain profile ratio
        T_edge = 0.2 * T_avg

        # beta_N = beta_t * a * B_T / Ip (Troyon, Phys. Rev. Lett. 53, 1984)
        _BETA_N_COEFF = 0.27  # [%-m-T/MA], ITER calibration
        beta_N: float = _BETA_N_COEFF * T_ax / max(abs(Ip), 0.1)
        
        # q95 ≈ 5 a² κ B_T / (R Ip); ITER: a=2m, κ=1.7, B_T=5.3T, R=6.2m
        # Wesson Ch.3 Eq.3.51
        _Q95_CONST = 45.0
        q95: float = max(_Q95_CONST / max(Ip, 0.1), 1.5)
        li: float = 0.85 + 0.1 * (q95 - 3.0)

        self._state = np.array([T_ax, T_edge, beta_N, li, q95, Ip])
        self._step_count += 1

        # Disruption check: q95 < 2 or beta_N > 3.5
        disrupted = q95 < 2.0 or beta_N > 3.5

        # Reward: tracking error + potential-based shaping (Ng et al. 1999)
        temp_err = abs(T_ax - self.T_target)
        progress = max(0.0, self._prev_temp_err - temp_err)
        self._prev_temp_err = temp_err
        reward = (
            -temp_err
            + 5.0 * progress
            + 0.5  # survival bonus
            - 50.0 * float(disrupted)
            - 0.01 * np.linalg.norm(action)
        )

        terminated = bool(disrupted)
        truncated = bool(self._step_count >= self.max_steps)

        info = {
            "T_axis": float(T_ax),
            "beta_N": float(beta_N),
            "q95": float(q95),
            "disrupted": bool(disrupted),
            "step": self._step_count,
        }
        return self._observe(), float(reward), terminated, truncated, info

    def _observe(self) -> np.ndarray:
        """Return noisy observation."""
        noise = self._rng.normal(0, self.noise_std, 6)
        obs = np.clip(
            self._state + noise,
            self.observation_low,
            self.observation_high,
        )
        return obs.astype(np.float64)

    def render(self) -> None:
        """Print current state."""
        T_ax, T_edge, beta_N, li, q95, Ip = self._state
        logger.info("step=%4d  T_ax=%.1fkeV  beta_N=%.2f  q95=%.2f  Ip=%.1fMA", self._step_count, T_ax, beta_N, q95, Ip)
