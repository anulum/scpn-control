# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CODAC action mapping contract tests.
"""Tests for CODAC controller-action to EPICS process-variable mapping."""

from __future__ import annotations

from typing import Mapping

import pytest

from scpn_control.control.codac_interface import CODACConfig, CODACInterface
from scpn_control.scpn.contracts import ControlObservation


class _MappingController:
    """Controller fixture that returns a generic action mapping."""

    def step(self, _obs: ControlObservation, _k: int) -> Mapping[str, float]:
        """Return sparse controller actions for CODAC unpacking."""

        return {"dI_PF3": 125.0, "SPI_trigger": 1.0}


def _interface() -> CODACInterface:
    """Return a CODAC interface bound to the mapping controller fixture."""

    return CODACInterface(CODACConfig(), _MappingController())


def _nominal_pv_values() -> dict[str, float]:
    """Return a complete nominal hard-limit and external-interlock packet."""

    config = CODACConfig()
    values = {
        "Ip": 15.0,
        "beta_N": 2.5,
        "q95": 3.0,
        "n_e": 10.0,
        "Te_axis": 20.0,
        "locked_mode_amp": 5.0,
        "R_axis": 6.2,
        "Z_axis": 0.0,
    }
    values.update({pv: 0.0 for pv in config.interlock_pvs})
    return values


def test_unpack_action_accepts_generic_numeric_mapping() -> None:
    iface = _interface()
    action: Mapping[str, float] = {"dI_PF3": 125.0, "NBI_power": 12.5}

    pvs = iface.unpack_action(action)

    assert pvs["ITER-SCPN:dI_PF3"] == pytest.approx(125.0)
    assert pvs["ITER-SCPN:NBI_power"] == pytest.approx(12.5)
    assert pvs["ITER-SCPN:dI_PF1"] == pytest.approx(0.0)


def test_run_cycle_maps_sparse_controller_actions_to_all_output_pvs() -> None:
    iface = _interface()

    pvs = iface.run_cycle(_nominal_pv_values())

    assert pvs["ITER-SCPN:dI_PF3"] == pytest.approx(125.0)
    assert pvs["ITER-SCPN:SPI_trigger"] == pytest.approx(1.0)
    assert len(pvs) == len(iface.define_output_channels())
