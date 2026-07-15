# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: protoscience@anulum.li
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_control.control.safe_rl_controller import (
    BETA_N_LIMIT,
    Q95_MIN,
    ConstrainedGymTokamakEnv,
    LagrangianPPO,
    cbf_beta_n,
    cbf_q95,
    default_safety_constraints,
)

FloatArray = NDArray[np.float64]


class MockSpace:
    def __init__(
        self,
        sample_value: FloatArray,
        *,
        low: FloatArray | None = None,
        high: FloatArray | None = None,
    ) -> None:
        self.sample_value = sample_value
        self.low = low
        self.high = high
        self.sample_count = 0

    def sample(self) -> FloatArray:
        self.sample_count += 1
        return self.sample_value.copy()


class MockEnv:
    def __init__(self) -> None:
        self.action_space = MockSpace(np.array([0.0], dtype=np.float64))
        self.observation_space = MockSpace(np.array([10.0, 2.0, 3.0], dtype=np.float64))
        self.step_count = 0

    def reset(self) -> tuple[FloatArray, dict[str, Any]]:
        self.step_count = 0
        # obs: Ip, beta_N, q95
        return np.array([10.0, 2.0, 3.0], dtype=np.float64), {}

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        # Mock dynamics that violates q95 limit
        obs = np.array([10.0, 2.0, 1.5], dtype=np.float64)  # q95 = 1.5, which is < 2.0
        reward = 1.0 - abs(float(action[0]) - 0.25)
        term = self.step_count >= 10
        return obs, reward, term, False, {}


class BoundedMockEnv(MockEnv):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = MockSpace(
            np.array([0.0], dtype=np.float64),
            low=np.array([-0.2], dtype=np.float64),
            high=np.array([0.2], dtype=np.float64),
        )


def test_constrained_env_wrapper() -> None:
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    obs, info = env.reset()
    action = np.array([0.0])
    obs_next, reward, term, trunc, info = env.step(action)

    costs = info["constraint_costs"]
    assert len(costs) == 3
    # q95 went to 1.5, constraint is max(0, 2.0 - q95) -> 0.5
    assert costs[0] == 0.5
    # beta_N = 2.0, max(0, 2.0 - 3.5) = 0.0
    assert costs[1] == 0.0


def test_lagrangian_ppo_lambda_update() -> None:
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    ppo = LagrangianPPO(env, lambda_lr=0.1)

    # Simulate an episode with costs
    ep_costs = [5.0, 0.0, 0.0]  # Violated the first constraint heavily

    ppo.update_lambdas(ep_costs)

    assert ppo.lambdas[0] > 0.0  # Should have increased
    assert ppo.lambdas[1] == 0.0
    assert ppo.lambdas[2] == 0.0


def test_augmented_reward() -> None:
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    ppo = LagrangianPPO(env)
    ppo.lambdas = np.array([2.0, 1.0, 0.5], dtype=np.float64)

    # Base reward = 10, costs = [1.0, 0.0, 2.0]
    # Aug = 10 - (2*1 + 1*0 + 0.5*2) = 10 - 3 = 7
    aug = ppo._augmented_reward(10.0, [1.0, 0.0, 2.0])
    assert aug == 7.0

    # With lambdas=0, aug = base
    ppo.lambdas = np.zeros(3, dtype=np.float64)
    aug_zero = ppo._augmented_reward(10.0, [1.0, 0.0, 2.0])
    assert aug_zero == 10.0


def test_mock_training_loop() -> None:
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    ppo = LagrangianPPO(env)
    ppo.train(total_timesteps=50)

    assert ppo.trained
    # Since the mock env constantly violates q95, the first lambda should be > 0
    assert ppo.lambdas[0] > 0.0
    assert base_env.action_space.sample_count == 1


def test_safe_rl_constraint_satisfaction() -> None:
    """Lagrangian λ penalises repeated constraint violations.

    After training on a consistently-violating env, λ_0 must be positive,
    and the augmented reward penalises future q_95 violations.
    """
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    ppo = LagrangianPPO(env, lambda_lr=0.05)
    ppo.train(total_timesteps=200)

    # λ_0 > 0 means the solver is penalising q_95 violations
    assert ppo.lambdas[0] > 0.0, "λ_0 not updated despite persistent q_95 violation"

    # Augmented reward with active penalty must be less than base reward
    base_r = 1.0
    costs = [0.5, 0.0, 0.0]
    aug_r = ppo._augmented_reward(base_r, costs)
    assert aug_r < base_r, "Lagrangian penalty not reducing augmented reward"


def test_policy_training_updates_deterministic_prediction() -> None:
    """Training updates the controller policy instead of returning random samples."""
    base_env = MockEnv()
    env = ConstrainedGymTokamakEnv(base_env, default_safety_constraints())
    ppo = LagrangianPPO(env, policy_lr=0.2, exploration_std=0.2, seed=7)

    initial = ppo.predict(np.array([10.0, 2.0, 3.0], dtype=np.float64))
    ppo.train(total_timesteps=100)
    trained = ppo.predict(np.array([10.0, 2.0, 3.0], dtype=np.float64))

    np.testing.assert_array_equal(initial, np.array([0.0], dtype=np.float64))
    assert not np.allclose(trained, initial)
    # One sample infers action dimension; rollout and prediction use policy means.
    assert base_env.action_space.sample_count == 1


def test_policy_prediction_respects_action_bounds() -> None:
    """Deterministic policy output is clipped to declared action bounds."""
    env = ConstrainedGymTokamakEnv(BoundedMockEnv(), default_safety_constraints())
    ppo = LagrangianPPO(env)
    ppo.predict(np.array([10.0, 2.0, 3.0], dtype=np.float64))
    assert ppo.policy_bias is not None
    ppo.policy_bias[:] = 10.0
    action = ppo.predict(np.array([10.0, 2.0, 3.0], dtype=np.float64))
    np.testing.assert_array_equal(action, np.array([0.2], dtype=np.float64))


def test_empty_rollout_policy_update_is_noop() -> None:
    """Empty rollout updates are ignored by the policy updater."""
    env = ConstrainedGymTokamakEnv(MockEnv(), default_safety_constraints())
    ppo = LagrangianPPO(env)
    ppo._update_policy([])
    assert ppo.policy_weights is None


def test_cbf_positive() -> None:
    """Barrier function h(x) > 0 inside the safe set.

    Ames et al. 2017, IEEE TAC 62, 3861, Definition 2:
    h(x) > 0 for x in interior of safe set S.

    Safe set: β_N < BETA_N_LIMIT  and  q_95 > Q95_MIN.
    obs layout: [I_p, β_N, q_95, ...]
    """
    # Interior of safe set: β_N = 2.0 < 3.5, q_95 = 3.5 > 2.0
    obs_safe = np.array([10.0, 2.0, 3.5])
    assert cbf_beta_n(obs_safe) > 0.0, "cbf_beta_n should be positive inside safe set"
    assert cbf_q95(obs_safe) > 0.0, "cbf_q95 should be positive inside safe set"

    # Boundary: β_N == BETA_N_LIMIT
    obs_boundary = np.array([10.0, BETA_N_LIMIT, 3.0])
    assert cbf_beta_n(obs_boundary) == 0.0

    # Outside safe set: β_N > BETA_N_LIMIT
    obs_unsafe_beta = np.array([10.0, BETA_N_LIMIT + 0.5, 3.0])
    assert cbf_beta_n(obs_unsafe_beta) < 0.0, "cbf_beta_n must be negative outside safe set"

    # Outside safe set: q_95 < Q95_MIN
    obs_unsafe_q = np.array([10.0, 2.0, Q95_MIN - 0.5])
    assert cbf_q95(obs_unsafe_q) < 0.0, "cbf_q95 must be negative outside safe set"


def test_lagrangian_ppo_predict_returns_policy_mean() -> None:
    """An untrained Lagrangian policy returns its deterministic zero mean.

    The action-space sampler is used only to infer action dimensionality.
    """
    base_env = MockEnv()
    env = ConstrainedGymTokamakEnv(base_env, default_safety_constraints())
    ppo = LagrangianPPO(env)
    action = ppo.predict(np.array([10.0, 2.0, 3.0], dtype=np.float64))
    assert isinstance(action, np.ndarray)
    np.testing.assert_array_equal(action, np.array([0.0], dtype=np.float64))
    assert base_env.action_space.sample_count == 1


def test_policy_action_is_deterministic_without_exploration() -> None:
    """Skip the exploration noise term for a deterministic action (branch 236->238).

    With explore=False the Gaussian exploration perturbation is not added, so
    repeated evaluations of the same observation return the identical clipped
    action, and that action still respects the environment's action bounds.
    """
    env = ConstrainedGymTokamakEnv(MockEnv(), default_safety_constraints())
    ppo = LagrangianPPO(env)
    obs = np.array([10.0, 2.0, 3.0], dtype=np.float64)

    _, action_a = ppo._policy_action(obs, explore=False)
    _, action_b = ppo._policy_action(obs, explore=False)

    np.testing.assert_array_equal(action_a, action_b)
    assert np.all(action_a >= -0.2) and np.all(action_a <= 0.2)
