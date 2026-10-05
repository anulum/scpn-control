# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Scenario scheduler admission regressions.
"""Exercise waveform, controller and optimizer input contracts publicly."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.control.scenario_scheduler import (
    FeedforwardController,
    ScenarioOptimizer,
    ScenarioSchedule,
    ScenarioWaveform,
)


def _zero_feedback(x: AnyFloatArray, x_ref: AnyFloatArray, t: float, dt: float) -> FloatArray:
    """Return a valid zero trim for scheduler admission tests."""
    del x, x_ref, t, dt
    return np.zeros(3, dtype=np.float64)


def _schedule(*, p_aux: float = 2.0, ip: float = 1.0) -> ScenarioSchedule:
    """Build a finite two-knot scenario using production waveforms."""
    times = np.array([0.0, 1.0], dtype=np.float64)
    return ScenarioSchedule(
        {
            "P_aux": ScenarioWaveform("P_aux", times, np.array([p_aux, p_aux], dtype=np.float64)),
            "Ip": ScenarioWaveform("Ip", times, np.array([ip, ip], dtype=np.float64)),
        }
    )


@pytest.mark.parametrize(
    ("times", "values", "match"),
    [
        ([0.0, 0.0], [1.0, 2.0], "strictly increasing"),
        ([0.0, 1.0], [1.0], "same nonzero length"),
        ([0.0, float("nan")], [1.0, 2.0], "finite"),
        ([0.0, 1.0], [1.0, float("inf")], "finite"),
    ],
)
def test_direct_waveform_refuses_invalid_knots(times: list[float], values: list[float], match: str) -> None:
    """Direct interpolation cannot convert malformed knots into a command."""
    waveform = ScenarioWaveform("P_aux", np.array(times), np.array(values))
    with pytest.raises(ValueError, match=match):
        waveform(0.5)


def test_direct_waveform_refuses_non_numeric_values() -> None:
    """Malformed source text cannot become an interpolated command."""
    waveform = ScenarioWaveform("P_aux", np.array([0.0, 1.0]), np.array(["bad", "2.0"]))
    with pytest.raises(ValueError, match="finite numeric arrays"):
        waveform(0.5)


def test_direct_waveform_refuses_unsupported_interpolation_or_time() -> None:
    """An unsupported interpolation claim or nonfinite time fails closed."""
    waveform = ScenarioWaveform("P_aux", np.array([0.0, 1.0]), np.array([1.0, 2.0]), interp_kind="cubic")
    with pytest.raises(ValueError, match="only linear"):
        waveform(0.5)
    waveform.interp_kind = "linear"
    with pytest.raises(ValueError, match="evaluation time must be finite"):
        waveform(float("nan"))


def test_schedule_duration_refuses_invalid_waveform() -> None:
    """Duration cannot be derived from a malformed final knot."""
    waveform = ScenarioWaveform("P_aux", np.array([0.0, float("inf")]), np.array([1.0, 2.0]))
    schedule = ScenarioSchedule({"P_aux": waveform})
    with pytest.raises(ValueError, match="must be finite"):
        schedule.duration()


def test_controller_step_rejects_invalid_schedule_domain() -> None:
    """An advisory validation error must block actual controller output."""
    controller = FeedforwardController(_schedule(ip=-1.0), _zero_feedback)
    with pytest.raises(ValueError, match="invalid scenario schedule"):
        controller.step(np.array([1.0]), 0.0, 0.1)


@pytest.mark.parametrize(
    ("feedback", "match"),
    [
        (np.array([1.0]), "three finite"),
        (np.array([float("nan"), 0.0, 0.0]), "three finite"),
    ],
)
def test_controller_step_refuses_malformed_feedback(feedback: FloatArray, match: str) -> None:
    """Scalar broadcasting and nonfinite trims cannot publish commands."""

    def produce_feedback(x: AnyFloatArray, x_ref: AnyFloatArray, t: float, dt: float) -> FloatArray:
        """Return the injected trim through the public callback contract."""
        del x, x_ref, t, dt
        return feedback

    controller = FeedforwardController(_schedule(), produce_feedback)
    with pytest.raises(ValueError, match=match):
        controller.step(np.array([1.0]), 0.0, 0.1)


def test_controller_step_refuses_finite_sum_overflow() -> None:
    """Finite schedule and trim values cannot combine into infinite action."""

    def large_feedback(x: AnyFloatArray, x_ref: AnyFloatArray, t: float, dt: float) -> FloatArray:
        """Return a large but finite auxiliary-power trim."""
        del x, x_ref, t, dt
        return np.array([1e308, 0.0, 0.0], dtype=np.float64)

    controller = FeedforwardController(_schedule(p_aux=1e308), large_feedback)
    with pytest.raises(ValueError, match="combined control action must be finite"):
        controller.step(np.array([1.0]), 0.0, 0.1)


@pytest.mark.parametrize(("state", "dt"), [(np.array([], dtype=float), 0.1), (np.array([1.0]), 0.0)])
def test_controller_step_refuses_invalid_state_or_timestep(state: FloatArray, dt: float) -> None:
    """The public controller requires a measurable state and positive cycle time."""
    controller = FeedforwardController(_schedule(), _zero_feedback)
    with pytest.raises(ValueError, match="state|dt"):
        controller.step(state, 0.0, dt)


def _linear_plant(x: AnyFloatArray, u: AnyFloatArray, dt: float) -> FloatArray:
    """Supply a finite plant callable for constructor-domain checks."""
    return np.asarray(x + u * dt, dtype=np.float64)


@pytest.mark.parametrize(("duration", "dt"), [(1.0, 0.0), (float("inf"), 0.1)])
def test_optimizer_refuses_invalid_time_domain(duration: float, dt: float) -> None:
    """Optimizer construction rejects time domains that cannot terminate."""
    plant: Callable[..., AnyFloatArray] = _linear_plant
    with pytest.raises(ValueError, match="T_total|dt"):
        ScenarioOptimizer(plant, np.array([1.0, 1.0]), duration, dt=dt)


def test_optimizer_refuses_invalid_target_state() -> None:
    """Offline optimization cannot target an undefined state vector."""
    with pytest.raises(ValueError, match="target_state must be a nonempty finite vector"):
        ScenarioOptimizer(_linear_plant, np.array([float("nan")]), 1.0, dt=0.1)


def test_optimizer_returns_physically_admissible_schedule() -> None:
    """Offline output cannot publish negative heating power or plasma current."""
    optimizer = ScenarioOptimizer(_linear_plant, np.array([10.0, 15.0]), 10.0, dt=1.0)
    schedule = optimizer.optimize(n_iter=50)
    assert schedule.validate() == []


@pytest.mark.parametrize("invalid", [True, 0, 1.5])
def test_optimizer_rejects_nonintegral_iteration_budget(invalid: object) -> None:
    """Offline optimization cannot silently coerce an invalid work budget."""
    optimizer = ScenarioOptimizer(_linear_plant, np.array([1.0, 1.0]), 1.0, dt=0.1)
    with pytest.raises(ValueError, match="n_iter must be a positive integer"):
        optimizer.optimize(n_iter=cast(int, invalid))


def test_optimizer_rejects_nonfinite_solver_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed numerical optimizer cannot publish an undefined schedule."""

    def invalid_candidate(*args: object, **kwargs: object) -> SimpleNamespace:
        """Inject an invalid candidate at the public optimizer boundary."""
        del args, kwargs
        return SimpleNamespace(x=np.full(6, float("nan")))

    monkeypatch.setattr("scipy.optimize.minimize", invalid_candidate)
    optimizer = ScenarioOptimizer(_linear_plant, np.array([1.0, 1.0]), 1.0, dt=0.1)
    with pytest.raises(ValueError, match="optimizer produced an invalid scenario schedule"):
        optimizer.optimize(n_iter=2)
