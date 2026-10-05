# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Gain-scheduled scenario tests.

"""Exercise the extracted scenario module through its public facade."""

import numpy as np
import pytest

from scpn_control._typing import FloatArray
from scpn_control.control import gain_scheduled_controller as facade
from scpn_control.control import gain_scheduled_scenario as scenario


def test_baseline_scenario_remains_available_through_historical_facade() -> None:
    """The split preserves public waveform types and an executable baseline."""
    assert facade.ScenarioWaveform is scenario.ScenarioWaveform
    assert facade.ScenarioSchedule is scenario.ScenarioSchedule
    schedule = facade.iter_baseline_schedule()
    assert schedule.validate() == []
    assert schedule.evaluate(60.0) == {"Ip": 15.0, "P_NBI": 33.0, "P_ECCD": 17.0}
    assert schedule.evaluate(500.0)["Ip"] == 2.0


@pytest.mark.parametrize(
    ("times", "values"),
    [
        (np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0, 2.0])),
        (np.array([0.0, 1.0]), np.array([0.0])),
        (np.array([0.0, float("nan")]), np.array([0.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([0.0, float("inf")])),
        (np.array([]), np.array([])),
    ],
)
def test_waveform_refuses_invalid_knots(times: FloatArray, values: FloatArray) -> None:
    """Malformed timing and values cannot masquerade as a valid schedule."""
    with pytest.raises(ValueError):
        facade.ScenarioWaveform("Ip", times, values)


def test_waveform_refuses_unsupported_interpolation_and_invalid_time() -> None:
    """The public mode and query time must match the implemented interpolation."""
    with pytest.raises(ValueError, match="interp_kind"):
        facade.ScenarioWaveform("Ip", np.array([0.0, 1.0]), np.array([0.0, 1.0]), "cubic")
    waveform = facade.ScenarioWaveform("Ip", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="time"):
        waveform(float("nan"))


def test_schedule_validator_detects_mutated_waveform_knots() -> None:
    """The public validator reports malformed knots after external mutation."""
    waveform = facade.ScenarioWaveform("Ip", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    waveform.times[1] = 0.0
    schedule = facade.ScenarioSchedule({"Ip": waveform})
    errors = schedule.validate()
    assert len(errors) == 1 and "strictly increasing" in errors[0]
    with pytest.raises(ValueError, match="strictly increasing"):
        schedule.evaluate(0.5)


def test_schedule_duration_refuses_corrupted_knots() -> None:
    """A mutated waveform cannot publish a misleading schedule duration."""
    waveform = facade.ScenarioWaveform("Ip", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    waveform.times = np.array([])
    schedule = facade.ScenarioSchedule({"Ip": waveform})
    assert schedule.validate()
    with pytest.raises(ValueError, match="non-empty"):
        schedule.duration()


def test_waveform_interpolates_large_finite_opposite_samples() -> None:
    """Representable interpolation must not overflow an intermediate delta."""
    waveform = facade.ScenarioWaveform("Ip", np.array([0.0, 1.0]), np.array([-1e308, 1e308]))
    assert waveform(0.5) == 0.0


def test_waveform_interpolates_large_finite_time_span() -> None:
    """A representable fraction survives a time span beyond float range."""
    waveform = facade.ScenarioWaveform("Ip", np.array([-1e308, 1e308]), np.array([0.0, 1.0]))
    assert waveform(0.0) == 0.5
