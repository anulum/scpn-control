# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Constrained Safe RL Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.control.safe_rl_controller import (
    ConstrainedGymTokamakEnv,
    LagrangianPPO,
    default_safety_constraints,
)


class MockEnv:
    def __init__(self):
        class Space:
            def sample(self):
                return np.array([0.0])

        self.action_space = Space()
        self.observation_space = Space()
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        # obs: Ip, beta_N, q95
        return np.array([10.0, 2.0, 3.0]), {}

    def step(self, action):
        self.step_count += 1
        # Mock dynamics that violates q95 limit
        obs = np.array([10.0, 2.0, 1.5])  # q95 = 1.5, which is < 2.0
        reward = 1.0
        term = self.step_count >= 10
        return obs, reward, term, False, {}


def test_constrained_env_wrapper():
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


def test_lagrangian_ppo_lambda_update():
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


def test_augmented_reward():
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    ppo = LagrangianPPO(env)
    ppo.lambdas = np.array([2.0, 1.0, 0.5])

    # Base reward = 10, costs = [1.0, 0.0, 2.0]
    # Aug = 10 - (2*1 + 1*0 + 0.5*2) = 10 - 3 = 7
    aug = ppo._augmented_reward(10.0, [1.0, 0.0, 2.0])
    assert aug == 7.0

    # With lambdas=0, aug = base
    ppo.lambdas = np.zeros(3)
    aug_zero = ppo._augmented_reward(10.0, [1.0, 0.0, 2.0])
    assert aug_zero == 10.0


def test_mock_training_loop():
    base_env = MockEnv()
    constraints = default_safety_constraints()
    env = ConstrainedGymTokamakEnv(base_env, constraints)

    ppo = LagrangianPPO(env)
    ppo.train(total_timesteps=50)

    assert ppo.trained
    # Since the mock env constantly violates q95, the first lambda should be > 0
    assert ppo.lambdas[0] > 0.0
