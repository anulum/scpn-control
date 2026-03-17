# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# Safety constraints for tokamak RL control.
#
# Disruption avoidance formulated as RL safety constraints:
#   β_N < β_limit  (Troyon limit: β_N < g I_p/(aB), g ≈ 3.5)
#   q_95 > 2       (kink stability: Wesson 2004, §3.4)
# Degrave et al. 2022, Nature 602, 414 (DeepMind magnetic control) —
#   shaped reward penalties for constraint proximity.
#
# Constrained Policy Optimization (CPO):
#   Achiam et al. 2017, ICML, "Constrained Policy Optimization", Eq. (5):
#   max_π  J_R(π)  s.t.  J_{C_i}(π) ≤ d_i  for all i
#
# Lagrangian dual relaxation (used here):
#   L(π,λ) = J_R(π) - Σ_i λ_i (J_{C_i}(π) - d_i)
#   λ_i update: λ_i ← max(0, λ_i + lr (C_i - d_i))
#   (Tessler et al. 2018, NeurIPS, "Reward-Constrained Policy Optimization",
#    Algorithm 1)
#
# Control barrier functions (CBF) for safety-critical systems:
#   h(x) > 0 in safe set S; safety filter ensures ḣ + γ h ≥ 0.
#   Ames et al. 2017, IEEE TAC 62, 3861, Definition 2.

# β_N Troyon limit [dimensionless]. Greenwald & Strait 1993,
# Phys. Plasmas 1, 1503: operational limit g ≈ 3.5 for standard H-mode.
BETA_N_LIMIT: float = 3.5

# q_95 kink-stability threshold [dimensionless].
# Wesson 2004, "Tokamaks", §3.4: disruptions occur for q_95 ≲ 2.
Q95_MIN: float = 2.0

# CBF decay rate γ [s⁻¹] for the exponential safety filter.
# Ames et al. 2017, Definition 3 (extended class-K function).
CBF_GAMMA: float = 1.0


@dataclass
class SafetyConstraint:
    """One inequality constraint: J_C(π) ≤ limit.

    Achiam et al. 2017, Eq. (5): constraint cost J_{C_i}(π) bounded by d_i.
    """

    name: str
    cost_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float]
    limit: float


def cbf_beta_n(x: np.ndarray, beta_n_limit: float = BETA_N_LIMIT) -> float:
    """Control barrier function for β_N safety.

    h(x) = β_limit - β_N  >  0 inside safe set.
    Ames et al. 2017, IEEE TAC 62, 3861, Definition 2.
    obs = [I_p, β_N, q_95, ...]
    """
    return beta_n_limit - x[1]


def cbf_q95(x: np.ndarray, q95_min: float = Q95_MIN) -> float:
    """Control barrier function for q_95 safety.

    h(x) = q_95 - q_min  >  0 inside safe set.
    Wesson 2004, §3.4; Ames et al. 2017, Definition 2.
    """
    return x[2] - q95_min


class ConstrainedGymTokamakEnv:
    """Gym wrapper augmenting step() with constraint cost signals.

    Constraint costs feed the Lagrangian multiplier update.
    Achiam et al. 2017, §4: cost signal C_i(s,a,s') evaluated per step.
    """

    def __init__(self, base_env: Any, constraints: list[SafetyConstraint]) -> None:
        self.base_env = base_env
        self.constraints = constraints
        self.n_constraints = len(constraints)

        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.base_env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        costs = [c.cost_fn(self._last_obs, action, obs) for c in self.constraints]
        info["constraint_costs"] = costs
        self._last_obs = obs

        return obs, reward, terminated, truncated, info


class LagrangianPPO:
    """PPO augmented with Lagrangian multipliers for constraint satisfaction.

    Objective (Tessler et al. 2018, Algorithm 1):
        r_aug = r - Σ_i λ_i c_i

    Multiplier update (dual gradient ascent):
        λ_i ← max(0,  λ_i + lr (C_i - d_i))

    Safety guarantee: at convergence J_{C_i}(π*) ≤ d_i for all i
    (Achiam et al. 2017, Theorem 1, assuming feasibility).

    Degrave et al. 2022, Nature 602, 414: penalty-based safety shaping
    used in DeepMind tokamak controller for disruption avoidance.
    """

    def __init__(
        self,
        env: ConstrainedGymTokamakEnv,
        lambda_lr: float = 0.01,
        gamma: float = 0.99,
    ) -> None:
        self.env = env
        self.n_constraints = env.n_constraints
        self.lambdas = np.zeros(self.n_constraints)
        self.lambda_lr = lambda_lr
        self.gamma = gamma
        self.trained = False

    def _augmented_reward(self, reward: float, costs: list[float]) -> float:
        """r_aug = r - Σ_i λ_i c_i  (Tessler et al. 2018, Eq. (3))."""
        penalty = sum(lam * c for lam, c in zip(self.lambdas, costs))
        return float(reward - penalty)

    def update_lambdas(self, episode_costs: list[float]) -> None:
        """Dual gradient ascent step.

        λ_i ← max(0,  λ_i + lr (C_i - d_i))
        Achiam et al. 2017, §5, Lagrangian update.
        """
        for i, c in enumerate(episode_costs):
            limit = self.env.constraints[i].limit
            self.lambdas[i] = max(0.0, self.lambdas[i] + self.lambda_lr * (c - limit))

    def train(self, total_timesteps: int) -> None:
        """Stub training loop demonstrating λ update mechanics.

        A production implementation replaces the random-action rollout
        with a PPO policy gradient update on the augmented reward.
        """
        current_step = 0
        while current_step < total_timesteps:
            obs, info = self.env.reset()
            done = False
            ep_costs = [0.0] * self.n_constraints
            steps = 0

            while not done and steps < 100:
                action = self.env.action_space.sample()
                obs, reward, term, trunc, info = self.env.step(action)
                done = term or trunc

                costs = info.get("constraint_costs", [0.0] * self.n_constraints)
                for i in range(self.n_constraints):
                    ep_costs[i] += costs[i]

                current_step += 1
                steps += 1

            self.update_lambdas(ep_costs)

        self.trained = True

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.env.action_space.sample())


# ── Default cost functions ────────────────────────────────────────────────────
# obs layout: [I_p [MA], β_N, q_95, ...]
# Constraint values: Wesson 2004, §3.4 (q_95); Greenwald & Strait 1993 (β_N).


def q95_cost_fn(obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray) -> float:
    """Cost for q_95 < Q95_MIN (kink instability region).

    C = max(0, Q95_MIN - q_95)
    Wesson 2004, §3.4: q_95 < 2 leads to disruptions.
    """
    return float(max(0.0, Q95_MIN - next_obs[2]))


def beta_n_cost_fn(obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray) -> float:
    """Cost for β_N > BETA_N_LIMIT (Troyon limit violation).

    C = max(0, β_N - β_limit)
    Greenwald & Strait 1993, Phys. Plasmas 1, 1503.
    """
    return float(max(0.0, next_obs[1] - BETA_N_LIMIT))


def ip_cost_fn(obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray) -> float:
    """Cost for I_p ≤ 0 (plasma lost).

    C = max(0, -I_p)
    """
    return float(max(0.0, -next_obs[0]))


def default_safety_constraints() -> list[SafetyConstraint]:
    return [
        SafetyConstraint("q95_lower_bound", q95_cost_fn, limit=0.0),
        SafetyConstraint("beta_n_upper_bound", beta_n_cost_fn, limit=0.0),
        SafetyConstraint("ip_positive", ip_cost_fn, limit=0.0),
    ]
