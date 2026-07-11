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
import logging
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace

import pytest

import scpn_control.control.controller_tuning as tuning_mod


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


@pytest.mark.parametrize("bad_dt", [0.0, -0.5])
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


def test_tune_pid_without_optuna_returns_defaults(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Absent Optuna, tune_pid logs a warning and returns default gains."""
    monkeypatch.setattr(tuning_mod, "HAS_OPTUNA", False)
    with caplog.at_level(logging.WARNING):
        gains = tuning_mod.tune_pid(None, n_trials=3)
    assert gains == {"Kp": 1.0, "Ki": 0.1, "Kd": 0.05}
    assert "Optuna not installed" in caplog.text


def test_tune_hinf_without_optuna_returns_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent Optuna, tune_hinf returns default gains."""
    monkeypatch.setattr(tuning_mod, "HAS_OPTUNA", False)
    assert tuning_mod.tune_hinf({}, n_trials=2) == {"gamma": 1.1, "bandwidth": 0.5}


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
            hinf_gains = tuning_mod.tune_hinf({}, n_trials=1)

            assert tuning_mod.HAS_OPTUNA is True
            assert pid_gains == {"Kp": 1.5, "Ki": 0.2, "Kd": 0.07}
            assert hinf_gains == {"gamma": 1.1}
            assert env.reset_calls == tuning_mod._TUNE_EPISODES
            assert env.actions == [pytest.approx(0.425)] * tuning_mod._TUNE_EPISODES
            assert [len(study.objective_values) for study in studies] == [1, 1]
    finally:
        importlib.reload(tuning_mod)
