# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Codac Interface
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — CODAC Interface Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from scpn_control.control.codac_interface import (
    CODACConfig,
    CODACInterface,
    CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY,
    CODAC_RUNTIME_EVIDENCE_QUALIFIED,
    CycleTimer,
    assert_codac_runtime_claim_admissible,
    codac_runtime_evidence,
    load_codac_runtime_evidence,
    save_codac_runtime_evidence,
)


class _StubController:
    """Minimal controller stub returning fixed actuator commands."""

    def step(self, obs, k):
        return {"dI_PF3": 100.0, "dI_PF_topbot_A": 50.0}


def _make_interface(config=None):
    ctrl = _StubController()
    cfg = config or CODACConfig()
    return CODACInterface(cfg, ctrl)


# ── Config ────────────────────────────────────────────────────────────


def test_config_defaults():
    cfg = CODACConfig()
    assert cfg.pv_prefix == "ITER-SCPN"
    assert cfg.cycle_hz == 1000.0
    assert cfg.timeout_ms == 1.5
    assert len(cfg.interlock_pvs) == 3


# ── Channels ──────────────────────────────────────────────────────────


def test_input_channel_count_and_pv_names():
    iface = _make_interface()
    inputs = iface.define_input_channels()
    assert len(inputs) == 12
    pv_names = {ch.pv_name for ch in inputs}
    assert "ITER-SCPN:Ip" in pv_names
    assert "ITER-SCPN:R_axis" in pv_names
    assert "ITER-SCPN:Z_axis" in pv_names
    assert all(ch.direction == "input" for ch in inputs)


def test_output_channel_count():
    iface = _make_interface()
    outputs = iface.define_output_channels()
    assert len(outputs) == 11
    assert all(ch.direction == "output" for ch in outputs)
    spi = [ch for ch in outputs if "SPI" in ch.pv_name]
    assert len(spi) == 1
    assert spi[0].dtype == "bo"


# ── Pack / Unpack ─────────────────────────────────────────────────────


def test_pack_observation():
    iface = _make_interface()
    pv = {"R_axis": 6.15, "Z_axis": 0.03, "Ip": 15.0}
    obs = iface.pack_observation(pv)
    assert obs["R_axis_m"] == 6.15
    assert obs["Z_axis_m"] == 0.03


def test_unpack_action():
    iface = _make_interface()
    action = {"dI_PF3": 100.0, "NBI_power": 20.0, "SPI_trigger": 1.0}
    pvs = iface.unpack_action(action)
    assert "ITER-SCPN:dI_PF3" in pvs
    assert pvs["ITER-SCPN:dI_PF3"] == 100.0
    assert pvs["ITER-SCPN:NBI_power"] == 20.0
    assert pvs["ITER-SCPN:SPI_trigger"] == 1.0


# ── Run cycle ─────────────────────────────────────────────────────────


def test_run_cycle():
    iface = _make_interface()
    pv = {"R_axis": 6.2, "Z_axis": 0.0}
    result = iface.run_cycle(pv)
    assert isinstance(result, dict)
    assert "ITER-SCPN:dI_PF3" in result
    assert result["ITER-SCPN:dI_PF3"] == 100.0


# ── Safety interlock ──────────────────────────────────────────────────


def test_safety_interlock_triggers_on_out_of_range():
    iface = _make_interface()
    pv_bad = {"Ip": 18.0, "q95": 3.0}  # Ip > 17.0 hard limit
    assert iface.safety_interlock(pv_bad) is True


def test_safety_interlock_passes_on_nominal():
    iface = _make_interface()
    pv_ok = {"Ip": 15.0, "q95": 3.0, "beta_N": 2.5, "n_e": 10.0, "Te_axis": 20.0}
    assert iface.safety_interlock(pv_ok) is False


# ── EPICS .db generation ─────────────────────────────────────────────


def test_generate_epics_db(tmp_path):
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


def test_render_epics_db_matches_written_export(tmp_path):
    iface = _make_interface()
    db_path = tmp_path / "scpn.db"
    iface.generate_epics_db(db_path)
    assert iface.render_epics_db() == db_path.read_text(encoding="utf-8")


# ── OPC-UA nodeset generation ────────────────────────────────────────


def test_generate_opcua_nodeset(tmp_path):
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


def test_render_opcua_nodeset_matches_written_export(tmp_path):
    iface = _make_interface()
    xml_path = tmp_path / "nodeset.xml"
    iface.generate_opcua_nodeset(xml_path)
    assert iface.render_opcua_nodeset() == xml_path.read_text(encoding="utf-8")


# ── Runtime evidence ─────────────────────────────────────────────────


def test_codac_runtime_evidence_binds_exports_and_payload(tmp_path):
    iface = _make_interface()
    evidence = codac_runtime_evidence(
        iface,
        controller_id="nsc-v0.19.2",
        observed_cycle_us=[410.0, 420.0, 430.0, 440.0],
        interlock_checks=2,
        interlock_blocks=1,
        backpressure_events=0,
        generated_utc="2026-05-24T00:00:00Z",
        facility_claim_allowed=True,
    )
    assert evidence.claim_status == CODAC_RUNTIME_EVIDENCE_QUALIFIED
    assert evidence.input_channel_count == 12
    assert evidence.output_channel_count == 11
    assert evidence.interlock_pv_count == len(iface.config.interlock_pvs)
    assert len(evidence.epics_db_sha256) == 64
    assert len(evidence.opcua_nodeset_sha256) == 64
    assert len(evidence.payload_sha256) == 64
    assert_codac_runtime_claim_admissible(evidence)

    path = tmp_path / "codac-runtime-evidence.json"
    save_codac_runtime_evidence(evidence, path)
    loaded = load_codac_runtime_evidence(path, require_facility_claim=True)
    assert loaded == evidence


def test_codac_runtime_evidence_keeps_local_runs_out_of_facility_claims():
    iface = _make_interface()
    evidence = codac_runtime_evidence(
        iface,
        controller_id="local-loopback",
        observed_cycle_us=[410.0, 420.0],
        interlock_checks=1,
        interlock_blocks=1,
        facility_claim_allowed=False,
    )
    assert evidence.claim_status == CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY
    with pytest.raises(ValueError, match="local-only"):
        assert_codac_runtime_claim_admissible(evidence)


def test_codac_runtime_evidence_requires_interlock_block_for_facility_claim():
    iface = _make_interface()
    with pytest.raises(ValueError, match="exercise and block"):
        codac_runtime_evidence(
            iface,
            controller_id="nsc-v0.19.2",
            observed_cycle_us=[410.0, 420.0],
            interlock_checks=2,
            interlock_blocks=0,
            facility_claim_allowed=True,
        )


def test_codac_runtime_evidence_rejects_deadline_overrun_for_facility_claim():
    iface = _make_interface()
    with pytest.raises(ValueError, match="cycle deadline"):
        codac_runtime_evidence(
            iface,
            controller_id="nsc-v0.19.2",
            observed_cycle_us=[410.0, 1500.0],
            interlock_checks=2,
            interlock_blocks=1,
            facility_claim_allowed=True,
        )


def test_codac_runtime_evidence_rejects_tampered_json(tmp_path):
    iface = _make_interface()
    evidence = codac_runtime_evidence(
        iface,
        controller_id="nsc-v0.19.2",
        observed_cycle_us=[410.0, 420.0],
        interlock_checks=2,
        interlock_blocks=1,
        facility_claim_allowed=True,
    )
    path = tmp_path / "codac-runtime-evidence.json"
    save_codac_runtime_evidence(evidence, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["observed_cycle_max_us"] = 421.0
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="payload_sha256"):
        load_codac_runtime_evidence(path)


def test_codac_runtime_evidence_rejects_duplicate_json_key(tmp_path):
    path = tmp_path / "codac-runtime-evidence.json"
    path.write_text('{"schema_version": "x", "schema_version": "y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_codac_runtime_evidence(path)


# ── CycleTimer ────────────────────────────────────────────────────────


def test_cycle_timer_detects_overrun():
    timer = CycleTimer(1e6)  # 1 MHz → 1 us budget
    timer.start_cycle()
    # Burn at least a few microseconds
    _sum = 0.0
    for i in range(5000):
        _sum += float(i)
    timer.end_cycle()
    assert timer.check_overrun() is True


def test_cycle_timer_jitter_within_budget():
    timer = CycleTimer(1.0)  # 1 Hz → 1 s budget (huge)
    timer.start_cycle()
    jitter_ms = timer.end_cycle()
    assert jitter_ms < 0.0  # well under 1 s budget
    assert timer.check_overrun() is False
