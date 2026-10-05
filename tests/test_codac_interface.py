# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Codac Interface.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — CODAC Interface Tests
# ──────────────────────────────────────────────────────────────────────
"""CODAC interface and runtime-evidence behavior tests."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import pytest

from scpn_control.control.codac_interface import (
    CODACConfig,
    CODACInterface,
    CycleTimer,
)


class _StubController:
    """Minimal controller stub returning fixed actuator commands."""

    def __init__(self) -> None:
        self.calls = 0

    def step(self, obs: dict[str, float], k: int) -> dict[str, float]:
        self.calls += 1
        return {"dI_PF3": 100.0, "dI_PF_topbot_A": 50.0}


def _make_interface(config: CODACConfig | None = None) -> CODACInterface:
    ctrl = _StubController()
    cfg = config or CODACConfig()
    return CODACInterface(cfg, ctrl)


def _nominal_pv_values(config: CODACConfig | None = None) -> dict[str, float]:
    cfg = config or CODACConfig()
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
    values.update({pv: 0.0 for pv in cfg.interlock_pvs})
    return values


# ── Config ────────────────────────────────────────────────────────────


def test_config_defaults() -> None:
    """Check config defaults."""
    cfg = CODACConfig()
    assert cfg.pv_prefix == "ITER-SCPN"
    assert cfg.cycle_hz == 1000.0
    assert cfg.timeout_ms == 1.5
    assert len(cfg.interlock_pvs) == 3


# ── Channels ──────────────────────────────────────────────────────────


def test_input_channel_count_and_pv_names() -> None:
    """Check input channel count and pv names."""
    iface = _make_interface()
    inputs = iface.define_input_channels()
    assert len(inputs) == 12
    pv_names = {ch.pv_name for ch in inputs}
    assert "ITER-SCPN:Ip" in pv_names
    assert "ITER-SCPN:R_axis" in pv_names
    assert "ITER-SCPN:Z_axis" in pv_names
    assert all(ch.direction == "input" for ch in inputs)


def test_output_channel_count() -> None:
    """Check output channel count."""
    iface = _make_interface()
    outputs = iface.define_output_channels()
    assert len(outputs) == 11
    assert all(ch.direction == "output" for ch in outputs)
    spi = [ch for ch in outputs if "SPI" in ch.pv_name]
    assert len(spi) == 1
    assert spi[0].dtype == "bo"


# ── Pack / Unpack ─────────────────────────────────────────────────────


def test_pack_observation() -> None:
    """Check pack observation."""
    iface = _make_interface()
    pv = {"R_axis": 6.15, "Z_axis": 0.03, "Ip": 15.0}
    obs = iface.pack_observation(pv)
    assert obs["R_axis_m"] == 6.15
    assert obs["Z_axis_m"] == 0.03


def test_unpack_action() -> None:
    """Check unpack action."""
    iface = _make_interface()
    action = {"dI_PF3": 100.0, "NBI_power": 20.0, "SPI_trigger": 1.0}
    pvs = iface.unpack_action(action)
    assert "ITER-SCPN:dI_PF3" in pvs
    assert pvs["ITER-SCPN:dI_PF3"] == 100.0
    assert pvs["ITER-SCPN:NBI_power"] == 20.0
    assert pvs["ITER-SCPN:SPI_trigger"] == 1.0


def test_unpack_action_clamps_every_declared_output_limit() -> None:
    """Check unpack action clamps every declared output limit."""
    iface = _make_interface()
    pvs = iface.unpack_action(
        {
            "dI_PF1": -2e6,
            "dI_PF2": 2e6,
            "gas_valve_D2": -1.0,
            "gas_valve_Ne": 100.0,
            "ECRH_power": 21.0,
            "NBI_power": 41.0,
            "SPI_trigger": 2.0,
        }
    )

    assert pvs["ITER-SCPN:dI_PF1"] == -1e6
    assert pvs["ITER-SCPN:dI_PF2"] == 1e6
    assert pvs["ITER-SCPN:gas_valve_D2"] == 0.0
    assert pvs["ITER-SCPN:gas_valve_Ne"] == 50.0
    assert pvs["ITER-SCPN:ECRH_power"] == 20.0
    assert pvs["ITER-SCPN:NBI_power"] == 40.0
    assert pvs["ITER-SCPN:SPI_trigger"] == 1.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_unpack_action_rejects_non_finite_output(value: Any) -> None:
    """Check unpack action rejects non finite output."""
    with pytest.raises(ValueError, match="CODAC output NBI_power must be finite"):
        _make_interface().unpack_action({"NBI_power": value})


def test_unpack_action_rejects_non_numeric_output() -> None:
    """Check unpack action rejects non numeric output."""
    with pytest.raises(ValueError, match="CODAC output NBI_power must be numeric"):
        _make_interface().unpack_action({"NBI_power": cast(float, None)})


# ── Run cycle ─────────────────────────────────────────────────────────


def test_run_cycle() -> None:
    """Check run cycle."""
    iface = _make_interface()
    pv = _nominal_pv_values()
    result = iface.run_cycle(pv)
    assert isinstance(result, dict)
    assert "ITER-SCPN:dI_PF3" in result
    assert result["ITER-SCPN:dI_PF3"] == 100.0
    assert iface.controller.calls == 1


def test_run_cycle_blocks_controller_and_returns_stationary_packet_on_trip() -> None:
    """Check run cycle blocks controller and returns stationary packet on trip."""
    iface = _make_interface()
    pv = _nominal_pv_values()
    pv["Ip"] = 18.0

    result = iface.run_cycle(pv)

    assert iface.controller.calls == 0
    assert set(result) == {channel.pv_name for channel in iface.define_output_channels()}
    assert all(value == 0.0 for value in result.values())


# ── Safety interlock ──────────────────────────────────────────────────


def test_safety_interlock_triggers_on_out_of_range() -> None:
    """Check safety interlock triggers on out of range."""
    iface = _make_interface()
    pv_bad = _nominal_pv_values()
    pv_bad["Ip"] = 18.0  # Ip > 17.0 hard limit
    assert iface.safety_interlock(pv_bad) is True


def test_safety_interlock_passes_on_nominal() -> None:
    """Check safety interlock passes on nominal."""
    iface = _make_interface()
    pv_ok = _nominal_pv_values()
    assert iface.safety_interlock(pv_ok) is False


def test_safety_interlock_fails_closed_on_missing_or_invalid_required_input() -> None:
    """Check safety interlock fails closed on missing or invalid required input."""
    iface = _make_interface()
    assert iface.safety_interlock({}) is True

    missing = _nominal_pv_values()
    del missing["Ip"]
    assert iface.safety_interlock(missing) is True

    for value in (float("nan"), float("inf"), "invalid"):
        pv = _nominal_pv_values()
        pv["Ip"] = cast(float, value)
        assert iface.safety_interlock(pv) is True


def test_safety_interlock_fails_closed_on_missing_invalid_or_tripped_external_pv() -> None:
    """Check safety interlock fails closed on missing invalid or tripped external pv."""
    iface = _make_interface()
    interlock_pv = iface.config.interlock_pvs[0]

    missing = _nominal_pv_values()
    del missing[interlock_pv]
    assert iface.safety_interlock(missing) is True

    for value in (float("nan"), "invalid", 1.0, -1.0):
        pv = _nominal_pv_values()
        pv[interlock_pv] = cast(float, value)
        assert iface.safety_interlock(pv) is True


def test_safety_interlock_fails_closed_without_configured_external_pvs() -> None:
    """Check safety interlock fails closed without configured external pvs."""
    config = CODACConfig(interlock_pvs=())
    iface = _make_interface(config)

    assert iface.safety_interlock(_nominal_pv_values(config)) is True


# ── EPICS .db generation ─────────────────────────────────────────────


def test_generate_epics_db(tmp_path: Path) -> None:
    """Check generate epics db."""
    iface = _make_interface()
    db_path = tmp_path / "scpn.db"
    iface.generate_epics_db(db_path)
    text = db_path.read_text(encoding="utf-8")
    assert "record(ai" in text
    assert "record(ao" in text
    assert "record(bo" in text
    assert 'field(DESC, "Plasma current")' in text
    assert 'field(EGU, "MA")' in text
    # Verify all 23 channels present (12 input + 11 output)
    assert text.count("record(") == 23
    assert text.count("field(DRVH") == 10
    assert text.count("field(DRVL") == 10
    nbi_record = text.split('record(ao, "ITER-SCPN:NBI_power") {', maxsplit=1)[1].split("}", maxsplit=1)[0]
    assert 'field(DRVH, "40.0")' in nbi_record
    assert 'field(DRVL, "0.0")' in nbi_record


def test_render_epics_db_matches_written_export(tmp_path: Path) -> None:
    """Check render epics db matches written export."""
    iface = _make_interface()
    db_path = tmp_path / "scpn.db"
    iface.generate_epics_db(db_path)
    assert iface.render_epics_db() == db_path.read_text(encoding="utf-8")


# ── OPC-UA nodeset generation ────────────────────────────────────────


def test_generate_opcua_nodeset(tmp_path: Path) -> None:
    """Check generate opcua nodeset."""
    iface = _make_interface()
    xml_path = tmp_path / "nodeset.xml"
    iface.generate_opcua_nodeset(xml_path)
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    ns = {"ua": "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"}
    variables = root.findall(".//ua:UAVariable", ns)
    assert len(variables) == 23
    datatypes = {v.attrib["DataType"] for v in variables}
    assert "Double" in datatypes
    assert "Boolean" in datatypes


def test_render_opcua_nodeset_matches_written_export(tmp_path: Path) -> None:
    """Check render opcua nodeset matches written export."""
    iface = _make_interface()
    xml_path = tmp_path / "nodeset.xml"
    iface.generate_opcua_nodeset(xml_path)
    assert iface.render_opcua_nodeset() == xml_path.read_text(encoding="utf-8")


# ── CycleTimer ────────────────────────────────────────────────────────


def test_cycle_timer_detects_overrun() -> None:
    """Check cycle timer detects overrun."""
    timer = CycleTimer(1e6)  # 1 MHz → 1 us budget
    timer.start_cycle()
    # Burn at least a few microseconds
    _sum = 0.0
    for i in range(5000):
        _sum += float(i)
    timer.end_cycle()
    assert timer.check_overrun() is True


def test_cycle_timer_jitter_within_budget() -> None:
    """Check cycle timer jitter within budget."""
    timer = CycleTimer(1.0)  # 1 Hz → 1 s budget (huge)
    timer.start_cycle()
    jitter_ms = timer.end_cycle()
    assert jitter_ms < 0.0  # well under 1 s budget
    assert timer.check_overrun() is False
