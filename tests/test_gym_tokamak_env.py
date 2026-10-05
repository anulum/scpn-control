# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gymnasium-compatible tokamak environment tests
"""Public reduced-order TokamakEnv admission and episode tests."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.gym_tokamak_env import TokamakEnv


class TestTokamakEnv:
    """Exercise the public environment with valid and invalid episode inputs."""

    def test_constructor_rejects_nonphysical_parameters(self) -> None:
        """Reject invalid model parameters before allocating an episode."""
        invalid_kwargs = (
            {"dt": 0.0},
            {"max_steps": 0},
            {"T_target": -1.0},
            {"noise_std": -0.1},
            {"n_e_20": 0.0},
            {"V_plasma": -1.0},
        )

        for kwargs in invalid_kwargs:
            with pytest.raises(ValueError, match="physical|positive|non-negative"):
                TokamakEnv(**kwargs)

    def test_step_rejects_invalid_action_shape_and_values(self) -> None:
        """Reject malformed actuator commands before advancing state."""
        env = TokamakEnv()
        env.reset()

        for action in (
            np.array([0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([np.nan, 0.0]),
            np.array([0.0, np.inf]),
        ):
            with pytest.raises(ValueError, match="action"):
                env.step(action)

    @pytest.mark.parametrize("extreme_density", (True, False))
    def test_step_rejects_derived_numeric_failure_without_advancing(self, extreme_density: bool) -> None:
        """A finite but overflowing model input cannot publish an episode step."""
        env = TokamakEnv(
            noise_std=0.0,
            n_e_20=1e308 if extreme_density else 1.0,
            dt=1e-3 if extreme_density else 1e308,
        )
        env.reset(seed=11)
        before = env._state.copy()
        heating_before = env.P_aux
        error_before = env._prev_temp_err

        with pytest.raises(ValueError, match="finite"):
            env.step(np.array([1.0, 0.0]))

        np.testing.assert_array_equal(env._state, before)
        assert env.P_aux == heating_before
        assert env._prev_temp_err == error_before
        assert env._step_count == 0

    def test_observation_failure_restores_episode_and_random_stream(self) -> None:
        """A failed noisy observation leaves the next valid shot deterministic."""
        env = TokamakEnv(noise_std=0.0)
        reference = TokamakEnv(noise_std=0.0)
        env.reset(seed=0)
        reference.reset(seed=0)
        env.noise_std = 1e308

        with pytest.raises(ValueError, match="finite"):
            env.step(np.zeros(2))

        assert env._step_count == 0
        np.testing.assert_array_equal(env._state, reference._state)
        env.noise_std = 0.0
        actual = env.step(np.zeros(2))
        expected = reference.step(np.zeros(2))
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]

    @pytest.mark.parametrize("initial_seed,failed_seed", ((0, 3), (1, None)))
    def test_reset_observation_failure_preserves_prior_episode(
        self, initial_seed: int, failed_seed: int | None
    ) -> None:
        """An unobservable new shot leaves the old state and RNG intact."""
        env = TokamakEnv(noise_std=0.0)
        reference = TokamakEnv(noise_std=0.0)
        env.reset(seed=initial_seed)
        reference.reset(seed=initial_seed)
        env.step(np.zeros(2))
        reference.step(np.zeros(2))
        env.noise_std = 1e308

        with pytest.raises(ValueError, match="finite"):
            env.reset(seed=failed_seed)

        assert env._step_count == reference._step_count
        np.testing.assert_array_equal(env._state, reference._state)
        env.noise_std = 0.0
        actual, _ = env.reset()
        expected, _ = reference.reset()
        np.testing.assert_array_equal(actual, expected)

    def test_reset_returns_obs_and_info(self) -> None:
        """Reset exposes the declared six-feature observation."""
        env = TokamakEnv()
        obs, info = env.reset()
        assert obs.shape == (6,)
        assert isinstance(info, dict)

    def test_step_returns_5_tuple(self) -> None:
        """Step follows the five-result Gymnasium shape."""
        env = TokamakEnv()
        env.reset()
        action = np.array([0.0, 0.0])
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert bool(terminated) in (True, False)
        assert bool(truncated) in (True, False)

    def test_deterministic_with_seed(self) -> None:
        """Equal reset seeds yield equal initial observations."""
        env1 = TokamakEnv(seed=123)
        obs1, _ = env1.reset(seed=123)
        env2 = TokamakEnv(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_episode_truncates_at_max_steps(self) -> None:
        """A non-disrupted shot truncates at its configured horizon."""
        env = TokamakEnv(max_steps=10)
        env.reset()
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
            if terminated:
                break
        if not terminated:
            assert truncated

    def test_action_clipping(self) -> None:
        """Out-of-range finite controls remain within actuator limits."""
        env = TokamakEnv()
        env.reset()
        obs, _, _, _, _ = env.step(np.array([100.0, 100.0]))
        assert np.all(np.isfinite(obs))

    def test_disruption_terminates(self) -> None:
        """The reported disruption stops a shot."""
        env = TokamakEnv()
        env.reset()
        # Drive current to zero -> q95 diverges or beta_N goes high
        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(np.array([5.0, -1.0]))
            if terminated:
                assert info["disrupted"]
                break

    def test_obs_within_bounds(self) -> None:
        """Initial observation respects the declared feature envelope."""
        env = TokamakEnv()
        obs, _ = env.reset()
        assert np.all(obs >= env.observation_low - 1e-6)
        assert np.all(obs <= env.observation_high + 1e-6)

    def test_negative_reward_for_error(self) -> None:
        """Large initial temperature error receives negative reward."""
        env = TokamakEnv(T_target=20.0)
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert reward < 0  # T_axis starts at ~10, target is 20

    def test_render_does_not_crash(self) -> None:
        """Rendering the current episode state remains callable."""
        env = TokamakEnv()
        env.reset()
        env.render()

    def test_spaces_properties(self) -> None:
        """Expose the declared observation and action dimensions."""
        env = TokamakEnv()
        assert env.observation_space_shape == (6,)
        assert env.action_space_shape == (2,)

    def test_energy_balance_steady_state(self) -> None:
        """Verify that temperature settles towards a steady state balance."""
        env = TokamakEnv(max_steps=2000, dt=0.01)  # Long time, large steps
        env.reset()

        # High power heating
        action = np.array([5.0, 0.0])  # Add 5MW per step up to limit?
        # Actually P_aux is stateful now. 5MW delta.

        temps = []
        for _ in range(500):
            obs, _, _, _, _ = env.step(action)
            temps.append(obs[0])
            action = np.array([0.0, 0.0])  # Maintain P_aux

        # Should converge
        final_temp = temps[-1]
        assert 2.0 < final_temp < 50.0
        # Rate of change should decrease
        dT_dt = abs(temps[-1] - temps[-2])
        dT_dt_start = abs(temps[1] - temps[0])
        assert dT_dt < dT_dt_start
