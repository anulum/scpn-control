# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller Tuning Tests
"""Tests for automated PID / H-infinity controller gain tuning.

Covers the discrete PID rollout primitive (integral accumulation, derivative,
anti-windup clamp), control-period resolution, the optional-Optuna fallback,
and — via an injected deterministic Optuna stand-in — the full optimisation
path, asserting that the suggested integral and derivative gains are actually
applied during the rollout (the regression this module was hardened for).
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace

import pytest

import scpn_control.control.controller_tuning as tuning_mod
from scpn_control.control.gym_tokamak_env import TokamakEnv


class _ScriptedEnv:
    """Environment replaying a fixed error sequence, then terminating.

    The tracking error is reported in observation element ``0``; the scripted
    sequence is exhausted one error per :meth:`step` call.
    """

    def __init__(self, errors: list[float], dt_attr: float | None = None) -> None:
        self._errors = list(errors)
        self._idx = 0
        self.actions: list[float] = []
        self.reset_calls = 0
        if dt_attr is not None:
            self.dt = dt_attr

    def reset(self) -> tuple[list[float], dict[str, object]]:
        """Reset to the first scripted error."""
        self.reset_calls += 1
        self._idx = 0
        return [self._errors[0]], {}

    def step(self, action: float) -> tuple[list[float], float, bool, bool, dict[str, object]]:
        """Record the action and advance through the scripted error sequence."""
        self.actions.append(float(action))
        self._idx += 1
        terminated = self._idx >= len(self._errors)
        observed = 0.0 if terminated else self._errors[self._idx]
        return [observed], 0.0, terminated, False, {}


def test_resolve_control_period_prefers_explicit_dt() -> None:
    """An explicit dt overrides any environment attribute."""
    assert tuning_mod._resolve_control_period(SimpleNamespace(dt=0.9), 0.5) == 0.5


def test_resolve_control_period_reads_env_dt() -> None:
    """When dt is None the environment's own dt is used."""
    assert tuning_mod._resolve_control_period(SimpleNamespace(dt=0.02), None) == 0.02


def test_resolve_control_period_reads_unwrapped_dt() -> None:
    """A Gymnasium wrapper exposes dt on its unwrapped view."""
    env = SimpleNamespace(unwrapped=SimpleNamespace(dt=0.04))
    assert tuning_mod._resolve_control_period(env, None) == 0.04


def test_resolve_control_period_defaults_when_absent() -> None:
    """With no explicit or environment dt the module default applies."""
    assert tuning_mod._resolve_control_period(object(), None) == tuning_mod._DEFAULT_DT


@pytest.mark.parametrize("bad_dt", [0.0, -0.5, float("inf"), float("nan")])
def test_resolve_control_period_rejects_non_positive(bad_dt: float) -> None:
    """A non-positive control period is rejected."""
    with pytest.raises(ValueError, match="strictly positive"):
        tuning_mod._resolve_control_period(object(), bad_dt)


def test_pid_episode_integrates_and_differentiates() -> None:
    """The rollout applies P, I and D terms and returns the true IAE.

    Hand-computed for errors [1.0, 0.5], dt=1, Kp=2, Ki=0.5, Kd=1:
      step 1: integral 1.0, derivative 0.0  -> action 2.5
      step 2: integral 1.5, derivative -0.5 -> action 1.25
      IAE = |1.0|*1 + |0.5|*1 = 1.5
    """
    env = _ScriptedEnv([1.0, 0.5])
    iae = tuning_mod._pid_episode_iae(env, kp=2.0, ki=0.5, kd=1.0, dt=1.0)
    assert env.actions == [pytest.approx(2.5), pytest.approx(1.25)]
    assert iae == pytest.approx(1.5)


def test_pid_episode_anti_windup_clamps_both_bounds() -> None:
    """The integrator saturates symmetrically at the anti-windup clamp."""
    pos = _ScriptedEnv([10.0, 10.0, 10.0])
    tuning_mod._pid_episode_iae(pos, kp=0.0, ki=1.0, kd=0.0, dt=1.0, integral_clamp=15.0)
    # integral: 10 -> clamp(20)=15 -> clamp(25)=15
    assert pos.actions == [pytest.approx(10.0), pytest.approx(15.0), pytest.approx(15.0)]

    neg = _ScriptedEnv([-10.0, -10.0, -10.0])
    tuning_mod._pid_episode_iae(neg, kp=0.0, ki=1.0, kd=0.0, dt=1.0, integral_clamp=15.0)
    assert neg.actions == [pytest.approx(-10.0), pytest.approx(-15.0), pytest.approx(-15.0)]


def test_tune_pid_without_optuna_refuses_to_claim_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent Optuna, no made-up gains can be returned as tuned."""
    monkeypatch.setattr(tuning_mod, "HAS_OPTUNA", False)
    with pytest.raises(ImportError, match=r"scpn-control\[tuning\]"):
        tuning_mod.tune_pid(_ScriptedEnv([0.25]), n_trials=3)


@pytest.mark.parametrize("n_trials", [0, -1, True])
def test_tune_pid_rejects_invalid_trial_count(n_trials: int) -> None:
    """A malformed search budget cannot be passed to Optuna."""
    with pytest.raises(ValueError, match="n_trials"):
        tuning_mod.tune_pid(_ScriptedEnv([0.25]), n_trials=n_trials)


def test_tune_hinf_without_optuna_rejects_missing_plant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent plant data cannot produce a made-up attenuation."""
    monkeypatch.setattr(tuning_mod, "HAS_OPTUNA", False)
    with pytest.raises(ValueError, match="plant"):
        tuning_mod.tune_hinf({})


def test_has_optuna_is_bool() -> None:
    """The Optuna availability flag is resolved to a bool on import."""
    assert isinstance(tuning_mod.HAS_OPTUNA, bool)


class _FakeTrial:
    """Deterministic Optuna trial stand-in returning fixed suggestions."""

    def __init__(self) -> None:
        self.params: dict[str, float] = {}

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        """Return a fixed value for known gains, else the interval midpoint."""
        del log
        values = {"Kp": 1.5, "Ki": 0.2, "Kd": 0.07, "gamma": 1.1}
        value = values.get(name, (float(low) + float(high)) / 2.0)
        self.params[name] = value
        return value


class _FakeStudy:
    """Deterministic Optuna study stand-in driving fixed trials."""

    def __init__(self) -> None:
        self.best_params: dict[str, float] = {}
        self.objective_values: list[float] = []

    def optimize(self, objective: Callable[[_FakeTrial], float], n_trials: int) -> None:
        """Evaluate the objective once per deterministic fake trial."""
        for _ in range(int(n_trials)):
            trial = _FakeTrial()
            self.objective_values.append(float(objective(trial)))
            self.best_params = dict(trial.params)


def _fake_optuna_module(studies: list[_FakeStudy]) -> ModuleType:
    """Build an Optuna-like module exposing the minimize study factory."""
    module = ModuleType("optuna")
    module.__dict__["Trial"] = _FakeTrial

    def create_study(*, direction: str) -> _FakeStudy:
        assert direction == "minimize"
        study = _FakeStudy()
        studies.append(study)
        return study

    module.__dict__["create_study"] = create_study
    return module


def test_tune_pid_applies_integral_and_derivative_gains(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full PID objective applies Ki/Kd, not just Kp (regression).

    For a single-step error of 0.25 with dt=1.0 and gains
    (Kp=1.5, Ki=0.2, Kd=0.07) the command is
      1.5*0.25 + 0.2*(0.25*1.0) + 0.07*0.0 = 0.425,
    proving the integral gain now contributes; the earlier P-only objective
    produced 0.375.
    """
    studies: list[_FakeStudy] = []
    try:
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "optuna", _fake_optuna_module(studies))
            importlib.reload(tuning_mod)

            env = _ScriptedEnv([0.25])
            pid_gains = tuning_mod.tune_pid(env, n_trials=1)

            assert tuning_mod.HAS_OPTUNA is True
            assert pid_gains == {"Kp": 1.5, "Ki": 0.2, "Kd": 0.07}
            assert env.reset_calls == tuning_mod._TUNE_EPISODES
            assert env.actions == [pytest.approx(0.425)] * tuning_mod._TUNE_EPISODES
            assert [len(study.objective_values) for study in studies] == [1]
    finally:
        importlib.reload(tuning_mod)


def test_tune_pid_runs_against_public_tokamak_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The actual two-action environment accepts the tuned heating command."""
    studies: list[_FakeStudy] = []
    try:
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "optuna", _fake_optuna_module(studies))
            importlib.reload(tuning_mod)
            env = TokamakEnv(max_steps=2, noise_std=0.0)
            gains = tuning_mod.tune_pid(env, n_trials=1)
            assert gains == {"Kp": 1.5, "Ki": 0.2, "Kd": 0.07}
            assert len(studies[0].objective_values) == 1
            assert studies[0].objective_values[0] > 0.0
    finally:
        importlib.reload(tuning_mod)


def test_tune_pid_pairs_tokamak_noise_across_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equal gain candidates face identical shot noise in the real environment."""
    studies: list[_FakeStudy] = []
    try:
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "optuna", _fake_optuna_module(studies))
            importlib.reload(tuning_mod)
            env = TokamakEnv(max_steps=2, noise_std=0.2)
            tuning_mod.tune_pid(env, n_trials=2)
            assert studies[0].objective_values[0] == studies[0].objective_values[1]
    finally:
        importlib.reload(tuning_mod)


@pytest.mark.parametrize(
    ("errors", "dt_s", "message"),
    [
        ([float("nan")], 1.0, "initial tracking error"),
        ([0.25, float("nan")], 1.0, "tracking error"),
        ([1.3e308], 1.0, "PID action"),
        ([1e200], 1e200, "integrated tracking error"),
    ],
)
def test_tune_pid_refuses_nonfinite_runtime_evidence(
    monkeypatch: pytest.MonkeyPatch,
    errors: list[float],
    dt_s: float,
    message: str,
) -> None:
    """Invalid environment evidence never becomes an Optuna objective value."""
    studies: list[_FakeStudy] = []
    try:
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "optuna", _fake_optuna_module(studies))
            importlib.reload(tuning_mod)
            with pytest.raises(ValueError, match=message):
                tuning_mod.tune_pid(_ScriptedEnv(errors), n_trials=1, dt=dt_s)
    finally:
        importlib.reload(tuning_mod)
