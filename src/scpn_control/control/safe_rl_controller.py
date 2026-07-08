# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
"""Safety-filtered reinforcement-learning controller wrappers and constraint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

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
    cost_fn: Callable[[AnyFloatArray, AnyFloatArray, AnyFloatArray], float]
    limit: float


@dataclass(frozen=True)
class _PolicyStep:
    observation: FloatArray
    action: FloatArray
    augmented_reward: float


def cbf_beta_n(x: AnyFloatArray, beta_n_limit: float = BETA_N_LIMIT) -> float:
    """Control barrier function for β_N safety.

    h(x) = β_limit - β_N  >  0 inside safe set.
    Ames et al. 2017, IEEE TAC 62, 3861, Definition 2.
    obs = [I_p, β_N, q_95, ...]
    """
    return float(beta_n_limit - x[1])


def cbf_q95(x: AnyFloatArray, q95_min: float = Q95_MIN) -> float:
    """Control barrier function for q_95 safety.

    h(x) = q_95 - q_min  >  0 inside safe set.
    Wesson 2004, §3.4; Ames et al. 2017, Definition 2.
    """
    return float(x[2] - q95_min)


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

    def reset(self, **kwargs: Any) -> tuple[FloatArray, dict[str, Any]]:
        """Reset the wrapped environment and cache the initial observation.

        Parameters
        ----------
        **kwargs
            Forwarded to the base environment's ``reset``.

        Returns
        -------
        tuple[FloatArray, dict[str, Any]]
            The initial observation and info mapping.
        """
        obs, info = self.base_env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action: AnyFloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Step the environment and append per-constraint costs to ``info``.

        Parameters
        ----------
        action
            The action applied to the base environment.

        Returns
        -------
        tuple[FloatArray, float, bool, bool, dict[str, Any]]
            The Gymnasium ``(obs, reward, terminated, truncated, info)`` tuple,
            with ``info["constraint_costs"]`` added.
        """
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        costs = [c.cost_fn(self._last_obs, action, obs) for c in self.constraints]
        info["constraint_costs"] = costs
        self._last_obs = obs

        return obs, reward, terminated, truncated, info


class LagrangianPPO:
    """Clipped policy-gradient controller with Lagrangian safety multipliers.

    Objective (Tessler et al. 2018, Algorithm 1):
        r_aug = r - Σ_i λ_i c_i

    Multiplier update (dual gradient ascent):
        λ_i ← max(0,  λ_i + lr (C_i - d_i))

    Safety target: under the feasibility and convergence assumptions of the
    constrained-policy literature, the Lagrangian optimum satisfies
    ``J_C_i(π*) ≤ d_i`` for each constraint. This lightweight NumPy controller
    exposes that optimisation structure; it does not replace a formal runtime
    safety certificate.

    The implementation is intentionally dependency-light: it uses a linear
    diagonal-Gaussian policy, clips the scalar advantage in the PPO spirit, and
    updates the policy from its own sampled rollouts instead of drawing actions
    directly from the environment's action-space sampler.
    """

    def __init__(
        self,
        env: ConstrainedGymTokamakEnv,
        lambda_lr: float = 0.01,
        gamma: float = 0.99,
        policy_lr: float = 0.05,
        exploration_std: float = 0.1,
        seed: int | None = 0,
    ) -> None:
        self.env = env
        self.n_constraints = env.n_constraints
        self.lambdas: FloatArray = np.zeros(self.n_constraints, dtype=np.float64)
        self.lambda_lr = lambda_lr
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.exploration_std = exploration_std
        self.rng = np.random.default_rng(seed)
        self.policy_weights: FloatArray | None = None
        self.policy_bias: FloatArray | None = None
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

    def _ensure_policy(self, obs: AnyFloatArray) -> FloatArray:
        """Initialise the linear policy for the observed state/action dimensions."""
        observation = np.asarray(obs, dtype=np.float64).reshape(-1)
        if self.policy_weights is None or self.policy_bias is None:
            sample = np.asarray(self.env.action_space.sample(), dtype=np.float64).reshape(-1)
            self.policy_weights = np.zeros((sample.size, observation.size), dtype=np.float64)
            self.policy_bias = np.zeros(sample.size, dtype=np.float64)
        return observation

    def _action_bounds(self, action_dim: int) -> tuple[FloatArray, FloatArray]:
        """Return finite action bounds, defaulting to ``[-1, 1]`` when absent."""
        low_raw = getattr(self.env.action_space, "low", None)
        high_raw = getattr(self.env.action_space, "high", None)
        if low_raw is None or high_raw is None:
            return (
                np.full(action_dim, -1.0, dtype=np.float64),
                np.full(action_dim, 1.0, dtype=np.float64),
            )
        low = np.asarray(low_raw, dtype=np.float64).reshape(-1)
        high = np.asarray(high_raw, dtype=np.float64).reshape(-1)
        return low, high

    def _policy_mean(self, obs: AnyFloatArray) -> tuple[FloatArray, FloatArray]:
        """Return flattened observation and deterministic linear-policy mean."""
        observation = self._ensure_policy(obs)
        assert self.policy_weights is not None
        assert self.policy_bias is not None
        mean = self.policy_weights @ observation + self.policy_bias
        low, high = self._action_bounds(mean.size)
        return observation, np.clip(mean, low, high)

    def _policy_action(self, obs: AnyFloatArray, *, explore: bool) -> tuple[FloatArray, FloatArray]:
        """Return a policy action and the flattened observation used to produce it."""
        observation, mean = self._policy_mean(obs)
        if explore:
            mean = mean + self.rng.normal(0.0, self.exploration_std, size=mean.shape)
        low, high = self._action_bounds(mean.size)
        return observation, np.clip(mean, low, high)

    def _discounted_returns(self, rewards: list[float]) -> FloatArray:
        """Compute discounted returns for one rollout."""
        returns = np.zeros(len(rewards), dtype=np.float64)
        running = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + self.gamma * running
            returns[index] = running
        return returns

    def _update_policy(self, steps: list[_PolicyStep]) -> None:
        """Apply a clipped REINFORCE-style update to the linear policy."""
        if not steps or self.policy_weights is None or self.policy_bias is None:
            return
        returns = self._discounted_returns([step.augmented_reward for step in steps])
        advantages = returns - float(np.mean(returns))
        for step, advantage in zip(steps, advantages):
            _, mean = self._policy_mean(step.observation)
            direction = step.action - mean
            clipped_advantage = float(np.clip(advantage, -1.0, 1.0))
            self.policy_weights += self.policy_lr * clipped_advantage * np.outer(direction, step.observation)
            self.policy_bias += self.policy_lr * clipped_advantage * direction

    def train(self, total_timesteps: int) -> None:
        """Train the linear policy on augmented rewards and update λ multipliers.

        The rollout action comes from the controller's current stochastic policy,
        not from ``action_space.sample()``. Episode returns update the linear
        policy, while accumulated constraint costs update the Lagrangian
        multipliers after each episode.
        """
        current_step = 0
        while current_step < total_timesteps:
            obs, info = self.env.reset()
            done = False
            ep_costs = [0.0] * self.n_constraints
            rollout: list[_PolicyStep] = []
            steps = 0

            while not done and steps < 100:
                observation, action = self._policy_action(obs, explore=True)
                obs, reward, term, trunc, info = self.env.step(action)
                done = term or trunc

                costs = info.get("constraint_costs", [0.0] * self.n_constraints)
                for i in range(self.n_constraints):
                    ep_costs[i] += costs[i]
                rollout.append(_PolicyStep(observation, action, self._augmented_reward(float(reward), costs)))

                current_step += 1
                steps += 1

            self._update_policy(rollout)
            self.update_lambdas(ep_costs)

        self.trained = True

    def predict(self, obs: AnyFloatArray) -> FloatArray:
        """Return an action for an observation.

        Parameters
        ----------
        obs
            The current observation.

        Returns
        -------
        FloatArray
            The deterministic mean action from the learned linear policy,
            clipped to the environment action bounds.
        """
        _, mean = self._policy_mean(obs)
        return mean


# ── Default cost functions ────────────────────────────────────────────────────
# obs layout: [I_p [MA], β_N, q_95, ...]
# Constraint values: Wesson 2004, §3.4 (q_95); Greenwald & Strait 1993 (β_N).


def q95_cost_fn(obs: AnyFloatArray, act: AnyFloatArray, next_obs: AnyFloatArray) -> float:
    """Cost for q_95 < Q95_MIN (kink instability region).

    C = max(0, Q95_MIN - q_95)
    Wesson 2004, §3.4: q_95 < 2 leads to disruptions.
    """
    return float(max(0.0, Q95_MIN - next_obs[2]))


def beta_n_cost_fn(obs: AnyFloatArray, act: AnyFloatArray, next_obs: AnyFloatArray) -> float:
    """Cost for β_N > BETA_N_LIMIT (Troyon limit violation).

    C = max(0, β_N - β_limit)
    Greenwald & Strait 1993, Phys. Plasmas 1, 1503.
    """
    return float(max(0.0, next_obs[1] - BETA_N_LIMIT))


def ip_cost_fn(obs: AnyFloatArray, act: AnyFloatArray, next_obs: AnyFloatArray) -> float:
    """Cost for I_p ≤ 0 (plasma lost).

    C = max(0, -I_p)
    """
    return float(max(0.0, -next_obs[0]))


def default_safety_constraints() -> list[SafetyConstraint]:
    """Return the default tokamak safety constraints (q95, beta_N, and limits)."""

    return [
        SafetyConstraint("q95_lower_bound", q95_cost_fn, limit=0.0),
        SafetyConstraint("beta_n_upper_bound", beta_n_cost_fn, limit=0.0),
        SafetyConstraint("ip_positive", ip_cost_fn, limit=0.0),
    ]
