# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fueling admission contract.
"""Exercise public fueling admission before persistent controller state changes."""

import math
from typing import cast

import pytest

from scpn_control.control.fueling_mode import (
    IcePelletFuelingController,
    run_fueling_mode,
    simulate_iter_density_control,
)


@pytest.mark.parametrize(
    ("density", "step_index", "dt_s"),
    [
        (math.nan, 0, 0.001),
        (math.inf, 0, 0.001),
        (-0.1, 0, 0.001),
        (0.8, 0, math.nan),
        (0.8, 0, 0.0),
        (0.8, -1, 0.001),
        (0.8, True, 0.001),
        (0.8, cast(int, "0"), 0.001),
    ],
)
def test_invalid_public_step_preserves_controller_state(density: float, step_index: int, dt_s: float) -> None:
    """A refused sample cannot poison the next valid SNN and PI command."""
    rejected = IcePelletFuelingController()
    fresh = IcePelletFuelingController()
    with pytest.raises(ValueError):
        rejected.step(density, step_index, dt_s)
    assert rejected.integrator == 0.0
    assert rejected.step(0.8, 0, 0.001) == fresh.step(0.8, 0, 0.001)


@pytest.mark.parametrize("steps", [8.9, 9.1, math.inf, math.nan])
def test_simulation_refuses_fractional_or_nonfinite_step_count(steps: float) -> None:
    """The public run must execute exactly the declared number of steps."""
    with pytest.raises(ValueError, match="steps"):
        simulate_iter_density_control(steps=cast(int, steps))


def test_public_summary_refuses_fractional_step_count() -> None:
    """The public summary cannot relabel a truncated simulation as requested."""
    with pytest.raises(ValueError, match="steps"):
        run_fueling_mode(steps=cast(int, 8.9))


def test_duplicate_step_index_does_not_advance_controller() -> None:
    """Retrying an already admitted tick cannot change later controller output."""
    rejected = IcePelletFuelingController()
    fresh = IcePelletFuelingController()
    assert rejected.step(0.8, 0, 0.001) == fresh.step(0.8, 0, 0.001)
    with pytest.raises(ValueError, match="strictly increasing"):
        rejected.step(0.8, 0, 0.001)
    assert rejected.step(0.9, 1, 0.001) == fresh.step(0.9, 1, 0.001)


@pytest.mark.parametrize("dt_s", [0.001, 1e308])
def test_extreme_finite_sample_refuses_overflow_before_state_change(dt_s: float) -> None:
    """Finite inputs cannot overflow the PI path into a synthetic control command."""
    controller = IcePelletFuelingController(target_density=1e308)
    with pytest.raises(ValueError, match="integrator increment|PI command"):
        controller.step(0.0, 0, dt_s)
    assert controller.integrator == 0.0
