# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NMPC admission and output ownership tests.
"""Exercise the public NMPC configuration and control output boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control._typing import FloatArray
from scpn_control.control.nmpc_controller import NMPCConfig, NonlinearMPC


def _identity_plant(state: FloatArray, control: FloatArray) -> FloatArray:
    """Return a finite plant state for a deterministic control boundary test."""
    return state.copy()


@pytest.mark.parametrize("tolerance", [float("inf"), float("-inf"), float("nan"), 0.0, -1.0])
def test_rti_rejects_nonfinite_or_nonpositive_admission_tolerance(tolerance: float) -> None:
    """Reject thresholds that could accept any finite stationarity residual."""
    with pytest.raises(ValueError, match="rti_residual_tol must be positive finite"):
        NonlinearMPC(_identity_plant, NMPCConfig(horizon=1, rti_residual_tol=tolerance))


@pytest.mark.parametrize("method", ["step", "step_rti"])
def test_returned_control_cannot_mutate_controller_warm_start(method: str) -> None:
    """A caller modifying one output must leave the next tick's warm start intact."""
    controller = NonlinearMPC(_identity_plant, NMPCConfig(horizon=1, max_sqp_iter=1, qp_max_iter=2))
    state = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    previous = np.array([1.0, 1.0, 1.0])
    if method == "step":
        output = controller.step(state, state, previous)
    else:
        output = controller.step_rti(state, state, previous).u0
    warm_start = controller.u_traj.copy()
    output[0] = 999.0
    np.testing.assert_array_equal(controller.u_traj, warm_start)


class _FailingPlant:
    """Switch a public plant callback from finite output to a runtime failure."""

    fail = False

    def __call__(self, state: FloatArray, control: FloatArray) -> FloatArray:
        """Raise on demand while leaving successful plant steps deterministic."""
        if self.fail:
            raise RuntimeError("plant unavailable")
        return state.copy()


@pytest.mark.parametrize("method", ["step", "step_rti"])
def test_failed_tick_preserves_previous_control_and_warm_start(method: str) -> None:
    """An unavailable plant cannot publish a partially shifted control state."""
    plant = _FailingPlant()
    controller = NonlinearMPC(plant, NMPCConfig(horizon=2, max_sqp_iter=1, qp_max_iter=2))
    state = np.array([5.0, 1.0, 3.0, 1.0, 2.0, 3.0])
    previous = np.array([1.0, 1.0, 1.0])
    getattr(controller, method)(state, state, previous)
    controls_before = controller.u_traj.copy()
    states_before = controller.x_traj.copy()
    warm_before = controller._rti_warm_started

    plant.fail = True
    with pytest.raises(RuntimeError, match="plant unavailable"):
        getattr(controller, method)(state + 0.1, state, previous)

    np.testing.assert_array_equal(controller.u_traj, controls_before)
    np.testing.assert_array_equal(controller.x_traj, states_before)
    assert controller._rti_warm_started is warm_before
