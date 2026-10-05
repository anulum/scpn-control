# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CODAC axis admission regressions.
"""Exercise CODAC controller admission through its public cycle API."""

from __future__ import annotations

from typing import Mapping, cast

import pytest

from scpn_control.control.codac_interface import CODACConfig, CODACInterface
from scpn_control.scpn.contracts import ControlObservation


class _RecordingController:
    """Record public cycle calls without standing in for plant dynamics."""

    def __init__(self) -> None:
        self.observations: list[ControlObservation] = []

    def step(self, obs: ControlObservation, k: int) -> Mapping[str, float]:
        """Return a finite command only when the interface admits the input."""
        assert k == len(self.observations)
        self.observations.append(obs)
        return {"dI_PF1": 42.0}


def _nominal_packet(config: CODACConfig) -> dict[str, float]:
    """Provide the declared controller inputs and clear external interlocks."""
    packet = {
        "Ip": 15.0,
        "beta_N": 2.5,
        "q95": 3.0,
        "n_e": 10.0,
        "Te_axis": 20.0,
        "locked_mode_amp": 5.0,
        "R_axis": 6.2,
        "Z_axis": 0.0,
    }
    packet.update({pv: 0.0 for pv in config.interlock_pvs})
    return packet


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("R_axis", None),
        ("Z_axis", None),
        ("R_axis", float("nan")),
        ("Z_axis", float("inf")),
        ("R_axis", 8.1),
        ("Z_axis", -2.1),
    ],
)
def test_public_cycle_blocks_invalid_controller_axis(key: str, value: float | None) -> None:
    """Missing, nonfinite and out-of-channel-range axes cannot reach step."""
    config = CODACConfig()
    controller = _RecordingController()
    interface = CODACInterface(config, controller)
    packet = _nominal_packet(config)
    if value is None:
        packet.pop(key)
    else:
        packet[key] = value
    result = interface.run_cycle(packet)
    assert controller.observations == []
    assert result and set(result.values()) == {0.0}


def test_direct_pack_rejects_missing_axis() -> None:
    """Direct users of the public pack API cannot receive fabricated defaults."""
    interface = CODACInterface(CODACConfig(), _RecordingController())
    with pytest.raises(ValueError, match="R_axis"):
        interface.pack_observation({"Z_axis": 0.0})


@pytest.mark.parametrize("invalid", [True, "not-a-number"])
def test_direct_pack_rejects_non_numeric_axis(invalid: object) -> None:
    """Boolean and malformed text cannot enter a controller observation."""
    interface = CODACInterface(CODACConfig(), _RecordingController())
    with pytest.raises(ValueError, match="R_axis observation must be a finite number"):
        interface.pack_observation({"R_axis": cast(float, invalid), "Z_axis": 0.0})


def test_conflicting_axis_aliases_block_public_cycle() -> None:
    """Two incompatible names for the same observation cannot be admitted."""
    config = CODACConfig()
    controller = _RecordingController()
    interface = CODACInterface(config, controller)
    packet = _nominal_packet(config)
    packet["R_axis_m"] = 6.5
    result = interface.run_cycle(packet)
    assert controller.observations == []
    assert result and set(result.values()) == {0.0}


def test_axis_aliases_preserve_valid_public_cycle() -> None:
    """The documented meter-suffixed aliases remain usable on admission."""
    config = CODACConfig()
    controller = _RecordingController()
    interface = CODACInterface(config, controller)
    packet = _nominal_packet(config)
    packet["R_axis_m"] = packet.pop("R_axis")
    packet["Z_axis_m"] = packet.pop("Z_axis")
    result = interface.run_cycle(packet)
    assert result["ITER-SCPN:dI_PF1"] == 42.0
    assert controller.observations == [{"R_axis_m": 6.2, "Z_axis_m": 0.0}]


def test_matching_axis_aliases_preserve_valid_public_cycle() -> None:
    """Redundant equal axis values do not create a false interlock trip."""
    config = CODACConfig()
    controller = _RecordingController()
    interface = CODACInterface(config, controller)
    packet = _nominal_packet(config)
    packet["R_axis_m"] = packet["R_axis"]
    packet["Z_axis_m"] = packet["Z_axis"]
    result = interface.run_cycle(packet)
    assert result["ITER-SCPN:dI_PF1"] == 42.0
    assert controller.observations == [{"R_axis_m": 6.2, "Z_axis_m": 0.0}]
