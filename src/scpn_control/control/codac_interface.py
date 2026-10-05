# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Codac Interface.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — ITER CODAC/EPICS Interface
# © 1996–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
ITER CODAC/EPICS integration prototype.

Generates EPICS .db and OPC-UA XML nodeset files for binding a
NeuroSymbolicController to the ITER Plant Instrumentation & Control
(I&C) infrastructure.  Does NOT require pyepics or any EPICS runtime.

References
----------
    ITER CODAC Handbook v7.0, §3.2 (PV naming conventions)
    IEC 62541 (OPC Unified Architecture)
"""

from __future__ import annotations

import io
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from scpn_control.control.codac_evidence import (
    CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY,
    CODAC_RUNTIME_EVIDENCE_QUALIFIED,
    CODAC_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    CODACRuntimeEvidence,
    _payload_sha256,
    _percentile,
    _reject_duplicate_keys,
    _require_finite_nonnegative,
    _require_nonnegative_int,
    _text_sha256,
    _utc_now,
)
from scpn_control.control.codac_evidence import (
    _validate_evidence_payload as _validate_evidence_payload_leaf,
)
from scpn_control.control.codac_observation import pack_codac_observation
from scpn_control.scpn.contracts import ControlObservation


@dataclass(frozen=True)
class CODACConfig:
    """CODAC plant-system configuration."""

    pv_prefix: str = "ITER-SCPN"
    cycle_hz: float = 1000.0  # Hz, ITER fast controller nominal rate
    timeout_ms: float = 1.5  # cycle budget [ms] at 1 kHz
    heartbeat_pv: str = "ITER-SCPN:HEARTBEAT"
    interlock_pvs: tuple[str, ...] = (
        "ITER-CIS:INTERLOCK:VDE",
        "ITER-CIS:INTERLOCK:HALO",
        "ITER-CIS:INTERLOCK:DISRUPTION",
    )


@dataclass(frozen=True)
class EPICSChannel:
    """Single EPICS process variable definition."""

    pv_name: str
    direction: str  # "input" | "output"
    dtype: str  # "ai" (analog in) | "ao" (analog out) | "bi" (binary in) | "bo" (binary out)
    units: str
    description: str
    low_limit: float = 0.0
    high_limit: float = 0.0


# ── Channel tables ────────────────────────────────────────────────────

_INPUT_CHANNELS: tuple[tuple[str, str, str, float, float, str], ...] = (
    # (suffix, units, desc, lo, hi, obs_key)
    ("Ip", "MA", "Plasma current", 0.0, 20.0, "Ip"),
    ("beta_N", "", "Normalised beta", 0.0, 5.0, "beta_N"),
    ("q95", "", "Edge safety factor", 1.0, 10.0, "q95"),
    ("n_e", "1e19 m-3", "Line-averaged electron density", 0.0, 20.0, "n_e"),
    ("Te_axis", "keV", "Electron temperature on axis", 0.0, 40.0, "Te_axis"),
    ("li", "", "Internal inductance", 0.0, 3.0, "li"),
    ("dBp_dt", "T/s", "Poloidal field rate of change", -10.0, 10.0, "dBp_dt"),
    ("locked_mode_amp", "G", "Locked mode amplitude", 0.0, 50.0, "locked_mode_amp"),
    ("n1_rms", "G", "n=1 RMS amplitude", 0.0, 50.0, "n1_rms"),
    ("Wmhd", "MJ", "Stored MHD energy", 0.0, 500.0, "Wmhd"),
    ("R_axis", "m", "Magnetic axis major radius", 4.0, 8.0, "R_axis_m"),
    ("Z_axis", "m", "Magnetic axis vertical position", -2.0, 2.0, "Z_axis_m"),
)

_OUTPUT_CHANNELS: tuple[tuple[str, str, str, float, float, str], ...] = (
    ("dI_PF1", "A", "PF1 current delta", -1e6, 1e6, "dI_PF1"),
    ("dI_PF2", "A", "PF2 current delta", -1e6, 1e6, "dI_PF2"),
    ("dI_PF3", "A", "PF3 current delta", -1e6, 1e6, "dI_PF3"),
    ("dI_PF4", "A", "PF4 current delta", -1e6, 1e6, "dI_PF4"),
    ("dI_PF5", "A", "PF5 current delta", -1e6, 1e6, "dI_PF5"),
    ("dI_PF6", "A", "PF6 current delta", -1e6, 1e6, "dI_PF6"),
    ("gas_valve_D2", "Pa m3/s", "D2 gas valve throughput", 0.0, 200.0, "gas_valve_D2"),
    ("gas_valve_Ne", "Pa m3/s", "Ne gas valve throughput", 0.0, 50.0, "gas_valve_Ne"),
    ("ECRH_power", "MW", "ECRH heating power", 0.0, 20.0, "ECRH_power"),
    ("NBI_power", "MW", "NBI heating power", 0.0, 40.0, "NBI_power"),
    ("SPI_trigger", "", "Shattered pellet injection trigger", 0.0, 1.0, "SPI_trigger"),
)

# Pre-built obs_key → column index for pack_observation
_INPUT_KEY_MAP: dict[str, str] = {row[0]: row[5] for row in _INPUT_CHANNELS}


def _validate_evidence_payload(payload: Mapping[str, Any], *, require_facility_claim: bool) -> CODACRuntimeEvidence:
    """Validate evidence against the current CODAC channel table widths."""
    return _validate_evidence_payload_leaf(
        payload,
        require_facility_claim=require_facility_claim,
        expected_input_count=len(_INPUT_CHANNELS),
        expected_output_count=len(_OUTPUT_CHANNELS),
    )


def _build_input_channels(prefix: str) -> list[EPICSChannel]:
    return [
        EPICSChannel(
            pv_name=f"{prefix}:{row[0]}",
            direction="input",
            dtype="ai",
            units=row[1],
            description=row[2],
            low_limit=row[3],
            high_limit=row[4],
        )
        for row in _INPUT_CHANNELS
    ]


def _build_output_channels(prefix: str) -> list[EPICSChannel]:
    return [
        EPICSChannel(
            pv_name=f"{prefix}:{row[0]}",
            direction="output",
            dtype="bo" if row[0] == "SPI_trigger" else "ao",
            units=row[1],
            description=row[2],
            low_limit=row[3],
            high_limit=row[4],
        )
        for row in _OUTPUT_CHANNELS
    ]


# ── Safety limits (hard interlock thresholds) ─────────────────────────
# ITER Physics Design Description Document, NF 39 (1999)

_HARD_LIMITS: dict[str, tuple[float, float]] = {
    "Ip": (0.0, 17.0),
    "beta_N": (0.0, 3.5),
    "q95": (2.0, float("inf")),
    "n_e": (0.0, 14.0),
    "Te_axis": (0.0, 30.0),
    "locked_mode_amp": (0.0, 20.0),
}


class CODACInterface:
    """Binds a NeuroSymbolicController to ITER CODAC I&C channels."""

    def __init__(self, config: CODACConfig, controller: Any) -> None:
        self.config = config
        self.controller = controller
        self._input_channels = _build_input_channels(config.pv_prefix)
        self._output_channels = _build_output_channels(config.pv_prefix)
        self._cycle_timer = CycleTimer(config.cycle_hz)
        self._step_k = 0

    def define_input_channels(self) -> list[EPICSChannel]:
        """Return the EPICS input (process-variable) channels read each cycle."""
        return list(self._input_channels)

    def define_output_channels(self) -> list[EPICSChannel]:
        """Return the EPICS output (actuator) channels written each cycle."""
        return list(self._output_channels)

    def pack_observation(self, pv_values: Mapping[str, float]) -> ControlObservation:
        """Convert validated EPICS axis PVs to a controller observation."""
        return pack_codac_observation(pv_values)

    def unpack_action(self, action: Mapping[str, float]) -> dict[str, float]:
        """Convert controller actions to EPICS process-variable values.

        Parameters
        ----------
        action : Mapping[str, float]
            Controller action values keyed by the names declared in
            ``_OUTPUT_CHANNELS``. Missing actions default to zero output, finite
            values are clamped to the channel envelope, and non-finite values
            are rejected before an actuator packet is returned.

        Returns
        -------
        dict[str, float]
            EPICS process-variable values keyed by fully qualified PV names.
        """
        out: dict[str, float] = {}
        for suffix, _units, _description, low_limit, high_limit, action_key in _OUTPUT_CHANNELS:
            raw_value = action.get(action_key, 0.0)
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CODAC output {action_key} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"CODAC output {action_key} must be finite")
            out[f"{self.config.pv_prefix}:{suffix}"] = min(max(value, low_limit), high_limit)
        return out

    def run_cycle(self, pv_values: Mapping[str, float]) -> dict[str, float]:
        """Run one fail-closed interlocked control cycle."""
        self._cycle_timer.start_cycle()
        try:
            packet = dict(pv_values)
            if self.safety_interlock(packet):
                return self.unpack_action({})
            obs = self.pack_observation(packet)
            action = self.controller.step(obs, self._step_k)
            result = self.unpack_action(action)
            self._step_k += 1
            return result
        finally:
            self._cycle_timer.end_cycle()

    def safety_interlock(self, pv_values: Mapping[str, float]) -> bool:
        """Return whether missing, invalid, tripped, or unsafe inputs block actuation.

        Configured external binary interlocks use ``0.0`` for clear. Missing,
        non-finite, non-numeric, or non-zero values are blocking.
        """
        if not self.config.interlock_pvs:
            return True
        try:
            pack_codac_observation(pv_values)
        except ValueError:
            return True
        for interlock_pv in self.config.interlock_pvs:
            raw_value = pv_values.get(interlock_pv)
            if raw_value is None:
                return True
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return True
            if not math.isfinite(value) or value != 0.0:
                return True
        for key, (lo, hi) in _HARD_LIMITS.items():
            raw_value = pv_values.get(key)
            if raw_value is None:
                return True
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return True
            if not math.isfinite(value) or value < lo or value > hi:
                return True
        return False

    def render_epics_db(self) -> str:
        """Return EPICS .db text with record() entries for all channels."""
        lines: list[str] = []
        lines.append("# Auto-generated EPICS database for SCPN-Control CODAC interface")
        lines.append(f"# Prefix: {self.config.pv_prefix}")
        lines.append("")
        for ch in self._input_channels + self._output_channels:
            lines.append(f'record({ch.dtype}, "{ch.pv_name}") {{')
            lines.append(f'    field(DESC, "{ch.description}")')
            if ch.units:
                lines.append(f'    field(EGU, "{ch.units}")')
            lines.append(f'    field(HOPR, "{ch.high_limit}")')
            lines.append(f'    field(LOPR, "{ch.low_limit}")')
            if ch.direction == "output" and ch.dtype == "ao":
                lines.append(f'    field(DRVH, "{ch.high_limit}")')
                lines.append(f'    field(DRVL, "{ch.low_limit}")')
            lines.append("}")
            lines.append("")
        return "\n".join(lines)

    def generate_epics_db(self, output_path: Path) -> None:
        """Write EPICS .db file with record() entries for all channels."""
        Path(output_path).write_text(self.render_epics_db(), encoding="utf-8")

    def _build_opcua_nodeset_tree(self) -> ET.ElementTree:
        ns_uri = "urn:iter:scpn-control:codac"
        root = ET.Element("UANodeSet", xmlns="http://opcfoundation.org/UA/2011/03/UANodeSet.xsd")
        ns_elem = ET.SubElement(root, "NamespaceUris")
        ET.SubElement(ns_elem, "Uri").text = ns_uri

        node_id = 1000
        for ch in self._input_channels + self._output_channels:
            var = ET.SubElement(
                root,
                "UAVariable",
                NodeId=f"ns=1;i={node_id}",
                BrowseName=f"1:{ch.pv_name}",
                DataType="Double" if ch.dtype in ("ai", "ao") else "Boolean",
                AccessLevel="3",
            )
            dn = ET.SubElement(var, "DisplayName")
            dn.text = ch.pv_name
            desc = ET.SubElement(var, "Description")
            desc.text = ch.description
            node_id += 1

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        return tree

    def render_opcua_nodeset(self) -> str:
        """Return OPC-UA XML nodeset text for ITER SDN integration."""
        buffer = io.BytesIO()
        self._build_opcua_nodeset_tree().write(buffer, encoding="utf-8", xml_declaration=True)
        return buffer.getvalue().decode("utf-8")

    def generate_opcua_nodeset(self, output_path: Path) -> None:
        """Write OPC-UA XML nodeset for ITER SDN integration."""
        Path(output_path).write_text(self.render_opcua_nodeset(), encoding="utf-8")


def codac_runtime_evidence(
    interface: CODACInterface,
    *,
    controller_id: str,
    observed_cycle_us: tuple[float, ...] | list[float],
    interlock_checks: int,
    interlock_blocks: int,
    backpressure_events: int = 0,
    plant_system: str = "ITER-CODAC-EPICS",
    generated_utc: str | None = None,
    facility_claim_allowed: bool = False,
) -> CODACRuntimeEvidence:
    """Build local CODAC evidence; caller counters cannot grant a facility claim."""
    if not isinstance(controller_id, str) or not controller_id.strip():
        raise ValueError("controller_id must be non-empty")
    if not isinstance(plant_system, str) or not plant_system.strip():
        raise ValueError("plant_system must be non-empty")
    raw_cycles = list(observed_cycle_us)
    if not raw_cycles:
        raise ValueError("observed_cycle_us must contain at least one sample")
    cycles = [_require_finite_nonnegative("observed_cycle_us", value) for value in raw_cycles]
    interlock_checks = _require_nonnegative_int("interlock_checks", interlock_checks)
    interlock_blocks = _require_nonnegative_int("interlock_blocks", interlock_blocks)
    backpressure_events = _require_nonnegative_int("backpressure_events", backpressure_events)
    if interlock_blocks > interlock_checks:
        raise ValueError("interlock_blocks cannot exceed interlock_checks")

    cycle_budget_us = 1_000_000.0 / float(interface.config.cycle_hz)
    timeout_budget_us = float(interface.config.timeout_ms) * 1000.0
    payload: dict[str, Any] = {
        "schema_version": CODAC_RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "generated_utc": generated_utc or _utc_now(),
        "controller_id": controller_id.strip(),
        "plant_system": plant_system.strip(),
        "pv_prefix": interface.config.pv_prefix,
        "cycle_hz": float(interface.config.cycle_hz),
        "cycle_budget_us": cycle_budget_us,
        "timeout_budget_us": timeout_budget_us,
        "deadline_us": min(cycle_budget_us, timeout_budget_us),
        "observed_cycle_p50_us": _percentile(cycles, 0.50),
        "observed_cycle_p95_us": _percentile(cycles, 0.95),
        "observed_cycle_p99_us": _percentile(cycles, 0.99),
        "observed_cycle_max_us": max(cycles),
        "input_channel_count": len(interface.define_input_channels()),
        "output_channel_count": len(interface.define_output_channels()),
        "interlock_pv_count": len(interface.config.interlock_pvs),
        "interlock_checks": interlock_checks,
        "interlock_blocks": interlock_blocks,
        "backpressure_events": backpressure_events,
        "output_limits_enforced": True,
        "epics_drive_limits_exported": True,
        "interlock_fail_closed": True,
        "epics_db_sha256": _text_sha256(interface.render_epics_db()),
        "opcua_nodeset_sha256": _text_sha256(interface.render_opcua_nodeset()),
        "facility_claim_allowed": bool(facility_claim_allowed),
        "claim_status": (
            CODAC_RUNTIME_EVIDENCE_QUALIFIED if facility_claim_allowed else CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY
        ),
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return _validate_evidence_payload(payload, require_facility_claim=bool(facility_claim_allowed))


def assert_codac_runtime_claim_admissible(evidence: CODACRuntimeEvidence) -> CODACRuntimeEvidence:
    """Fail closed unless CODAC runtime evidence can support a facility claim."""
    return _validate_evidence_payload(asdict(evidence), require_facility_claim=True)


def save_codac_runtime_evidence(evidence: CODACRuntimeEvidence, output_path: Path) -> None:
    """Validate and persist local CODAC runtime evidence as sorted JSON."""
    validated = _validate_evidence_payload(asdict(evidence), require_facility_claim=False)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(validated), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_codac_runtime_evidence(
    path: Path,
    *,
    require_facility_claim: bool = False,
) -> CODACRuntimeEvidence:
    """Load CODAC runtime evidence with duplicate-key and digest admission."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("CODAC runtime evidence must be a JSON object")
    return _validate_evidence_payload(payload, require_facility_claim=require_facility_claim)


class CycleTimer:
    """Enforces real-time cycle budget with deadline monitoring.

    Uses time.perf_counter_ns() for sub-microsecond resolution.
    """

    def __init__(self, cycle_hz: float) -> None:
        self.budget_ns = int(1e9 / cycle_hz)
        self._start_ns: int = 0
        self._elapsed_ns: int = 0

    def start_cycle(self) -> None:
        """Mark the start of a control cycle for jitter measurement."""
        self._start_ns = time.perf_counter_ns()

    def end_cycle(self) -> float:
        """Return jitter in milliseconds (elapsed - budget)."""
        self._elapsed_ns = time.perf_counter_ns() - self._start_ns
        return (self._elapsed_ns - self.budget_ns) / 1e6

    def check_overrun(self) -> bool:
        """Return True if the last cycle exceeded its budget."""
        return self._elapsed_ns > self.budget_ns
