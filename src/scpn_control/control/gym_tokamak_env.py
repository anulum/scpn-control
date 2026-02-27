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

import numpy as np


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
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dt: float = 1e-3,
        max_steps: int = 500,
        T_target: float = 20.0,
        noise_std: float = 0.01,
        seed: int = 42,
    ):
        self.dt = dt
        self.max_steps = max_steps
        self.T_target = T_target
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

        # State: [T_axis, T_edge, beta_N, li, q95, Ip_MA]
        self._state = np.zeros(6, dtype=np.float64)
        self._step_count = 0

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
        self._state = np.array([
            10.0 + self._rng.normal(0, 0.5),   # T_axis [keV]
            2.0 + self._rng.normal(0, 0.1),    # T_edge [keV]
            1.5 + self._rng.normal(0, 0.1),    # beta_N
            0.85 + self._rng.normal(0, 0.05),  # li
            3.0 + self._rng.normal(0, 0.1),    # q95
            15.0,                               # Ip [MA]
        ])
        self._step_count = 0
        return self._observe(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one timestep.

        Returns (obs, reward, terminated, truncated, info).
        """
        action = np.clip(action, self.action_low, self.action_high)
        P_aux_delta, Ip_delta = float(action[0]), float(action[1])

        s = self._state
        T_ax: float = float(s[0])
        T_edge: float = float(s[1])
        Ip: float = float(s[5])

        # Simplified plasma response model
        T_ax += self.dt * (50.0 * P_aux_delta - 3.0 * (T_ax - T_edge))
        T_edge += self.dt * (0.5 * (T_ax - T_edge) - 1.0 * T_edge)
        Ip += Ip_delta * self.dt * 10.0
        beta_N: float = 0.02 * T_ax * np.sqrt(max(Ip, 0.1)) / max(abs(Ip), 0.1)
        q95: float = max(5.0 * 5.3 / max(Ip, 0.1), 1.5)
        li: float = 0.85 + 0.1 * (q95 - 3.0)

        self._state = np.array([T_ax, T_edge, beta_N, li, q95, Ip])
        self._step_count += 1

        # Disruption check: q95 < 2 or beta_N > 3.5
        disrupted = q95 < 2.0 or beta_N > 3.5

        # Reward
        temp_err = abs(T_ax - self.T_target)
        reward = -temp_err - 10.0 * float(disrupted) - 0.01 * np.linalg.norm(action)

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
        print(
            f"step={self._step_count:4d}  "
            f"T_ax={T_ax:.1f}keV  β_N={beta_N:.2f}  "
            f"q95={q95:.2f}  Ip={Ip:.1f}MA"
        )
