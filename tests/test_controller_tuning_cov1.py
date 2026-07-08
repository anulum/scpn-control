# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Controller tuning coverage hardening tests
"""COV-1 regression tests for optional Optuna controller tuning paths."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType

import pytest

import scpn_control.control.controller_tuning as tuning_mod


class _FakeTrial:
    """Deterministic Optuna trial stand-in."""

    def __init__(self) -> None:
        self.params: dict[str, float] = {}

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        """Return a deterministic value within the requested range."""
        del log
        values = {
            "Kp": 1.5,
            "Ki": 0.2,
            "Kd": 0.07,
            "gamma": 1.1,
        }
        value = values.get(name, (float(low) + float(high)) / 2.0)
        self.params[name] = value
        return value


class _FakeStudy:
    """Deterministic Optuna study stand-in."""

    def __init__(self) -> None:
        self.best_params: dict[str, float] = {}
        self.objective_values: list[float] = []

    def optimize(self, objective: Callable[[_FakeTrial], float], n_trials: int) -> None:
        """Evaluate the objective using deterministic fake trials."""
        for _ in range(int(n_trials)):
            trial = _FakeTrial()
            self.objective_values.append(float(objective(trial)))
            self.best_params = dict(trial.params)


class _FakeEnv:
    """Minimal environment implementing the reset/step contract used by tune_pid."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.actions: list[float] = []

    def reset(self) -> tuple[list[float], dict[str, object]]:
        """Reset one deterministic one-step episode."""
        self.reset_calls += 1
        return [0.25], {}

    def step(self, action: float) -> tuple[list[float], float, bool, bool, dict[str, object]]:
        """Complete the episode after one control action."""
        self.actions.append(float(action))
        return [0.0], 0.0, True, False, {}


def _fake_optuna_module(studies: list[_FakeStudy]) -> ModuleType:
    """Create an Optuna-like module exposing the study factory used by tuning."""
    module = ModuleType("optuna")
    module.__dict__["Trial"] = _FakeTrial

    def create_study(*, direction: str) -> _FakeStudy:
        assert direction == "minimize"
        study = _FakeStudy()
        studies.append(study)
        return study

    module.__dict__["create_study"] = create_study
    return module


def test_optuna_tuning_paths_are_exercised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reload with fake Optuna and exercise both public tuning APIs."""
    studies: list[_FakeStudy] = []
    try:
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "optuna", _fake_optuna_module(studies))
            importlib.reload(tuning_mod)

            env = _FakeEnv()
            pid_gains = tuning_mod.tune_pid(env, n_trials=1)
            hinf_gains = tuning_mod.tune_hinf({}, n_trials=1)

            assert tuning_mod.HAS_OPTUNA is True
            assert pid_gains == {"Kp": 1.5, "Ki": 0.2, "Kd": 0.07}
            assert hinf_gains == {"gamma": 1.1}
            assert env.reset_calls == 5
            assert env.actions == [pytest.approx(0.375)] * 5
            assert [len(study.objective_values) for study in studies] == [1, 1]
    finally:
        importlib.reload(tuning_mod)
