# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Codac Interface
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — ITER CODAC/EPICS Interface
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
ITER CODAC/EPICS integration prototype.

Generates EPICS .db and OPC-UA XML nodeset files for binding a
NeuroSymbolicController to the ITER Plant Instrumentation & Control
(I&C) infrastructure.  Does NOT require pyepics or any EPICS runtime.

References:
    ITER CODAC Handbook v7.0, §3.2 (PV naming conventions)
    IEC 62541 (OPC Unified Architecture)
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scpn_control.scpn.contracts import ControlAction, ControlObservation


CODAC_RUNTIME_EVIDENCE_SCHEMA_VERSION = "scpn-control.codac-runtime-evidence.v1"
CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY = "bounded_codac_runtime_evidence_only"
CODAC_RUNTIME_EVIDENCE_QUALIFIED = "qualified_codac_runtime_evidence"


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


@dataclass(frozen=True)
class CODACRuntimeEvidence:
    """Tamper-evident CODAC/EPICS runtime admission evidence."""

    schema_version: str
    generated_utc: str
    controller_id: str
    plant_system: str
    pv_prefix: str
    cycle_hz: float
    cycle_budget_us: float
    timeout_budget_us: float
    deadline_us: float
    observed_cycle_p50_us: float
    observed_cycle_p95_us: float
    observed_cycle_p99_us: float
    observed_cycle_max_us: float
    input_channel_count: int
    output_channel_count: int
    interlock_pv_count: int
    interlock_checks: int
    interlock_blocks: int
    backpressure_events: int
    epics_db_sha256: str
    opcua_nodeset_sha256: str
    facility_claim_allowed: bool
    claim_status: str
    payload_sha256: str


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
# Pre-built action_key → column suffix for unpack_action
_OUTPUT_KEY_MAP: dict[str, str] = {row[5]: row[0] for row in _OUTPUT_CHANNELS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key in CODAC runtime evidence: {key}")
        seen.add(key)
        out[key] = value
    return out


def _require_finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _validate_evidence_payload(payload: Mapping[str, Any], *, require_facility_claim: bool) -> CODACRuntimeEvidence:
    if payload.get("schema_version") != CODAC_RUNTIME_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("CODAC runtime evidence schema_version is unsupported")
    declared_digest = payload.get("payload_sha256")
    if not _is_sha256(declared_digest):
        raise ValueError("CODAC runtime evidence payload_sha256 must be a SHA-256 hex digest")
    if declared_digest != _payload_sha256(payload):
        raise ValueError("CODAC runtime evidence payload_sha256 does not match payload")

    generated_utc = payload.get("generated_utc")
    if not isinstance(generated_utc, str) or not generated_utc.endswith("Z"):
        raise ValueError("CODAC runtime evidence generated_utc must be a UTC timestamp ending in Z")
    controller_id = payload.get("controller_id")
    plant_system = payload.get("plant_system")
    pv_prefix = payload.get("pv_prefix")
    if not isinstance(controller_id, str) or not controller_id.strip():
        raise ValueError("CODAC runtime evidence controller_id must be non-empty")
    if not isinstance(plant_system, str) or not plant_system.strip():
        raise ValueError("CODAC runtime evidence plant_system must be non-empty")
    if not isinstance(pv_prefix, str) or not pv_prefix.strip():
        raise ValueError("CODAC runtime evidence pv_prefix must be non-empty")

    cycle_hz = _require_finite_nonnegative("cycle_hz", payload.get("cycle_hz"))
    if cycle_hz <= 0.0:
        raise ValueError("cycle_hz must be positive")
    cycle_budget_us = _require_finite_nonnegative("cycle_budget_us", payload.get("cycle_budget_us"))
    timeout_budget_us = _require_finite_nonnegative("timeout_budget_us", payload.get("timeout_budget_us"))
    deadline_us = _require_finite_nonnegative("deadline_us", payload.get("deadline_us"))
    if not math.isclose(deadline_us, min(cycle_budget_us, timeout_budget_us), rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("deadline_us must equal min(cycle_budget_us, timeout_budget_us)")

    observed_p50 = _require_finite_nonnegative("observed_cycle_p50_us", payload.get("observed_cycle_p50_us"))
    observed_p95 = _require_finite_nonnegative("observed_cycle_p95_us", payload.get("observed_cycle_p95_us"))
    observed_p99 = _require_finite_nonnegative("observed_cycle_p99_us", payload.get("observed_cycle_p99_us"))
    observed_max = _require_finite_nonnegative("observed_cycle_max_us", payload.get("observed_cycle_max_us"))
    if not (observed_p50 <= observed_p95 <= observed_p99 <= observed_max):
        raise ValueError("CODAC runtime cycle percentiles must be ordered")

    input_count = _require_nonnegative_int("input_channel_count", payload.get("input_channel_count"))
    output_count = _require_nonnegative_int("output_channel_count", payload.get("output_channel_count"))
    interlock_pv_count = _require_nonnegative_int("interlock_pv_count", payload.get("interlock_pv_count"))
    if input_count != len(_INPUT_CHANNELS) or output_count != len(_OUTPUT_CHANNELS):
        raise ValueError("CODAC runtime evidence channel counts do not match current interface tables")
    if interlock_pv_count <= 0:
        raise ValueError("CODAC runtime evidence must bind at least one interlock PV")

    interlock_checks = _require_nonnegative_int("interlock_checks", payload.get("interlock_checks"))
    interlock_blocks = _require_nonnegative_int("interlock_blocks", payload.get("interlock_blocks"))
    backpressure_events = _require_nonnegative_int("backpressure_events", payload.get("backpressure_events"))
    if interlock_blocks > interlock_checks:
        raise ValueError("interlock_blocks cannot exceed interlock_checks")

    if not _is_sha256(payload.get("epics_db_sha256")):
        raise ValueError("epics_db_sha256 must be a SHA-256 hex digest")
    if not _is_sha256(payload.get("opcua_nodeset_sha256")):
        raise ValueError("opcua_nodeset_sha256 must be a SHA-256 hex digest")

    facility_claim_allowed = payload.get("facility_claim_allowed")
    claim_status = payload.get("claim_status")
    if not isinstance(facility_claim_allowed, bool):
        raise ValueError("facility_claim_allowed must be boolean")
    expected_status = CODAC_RUNTIME_EVIDENCE_QUALIFIED if facility_claim_allowed else CODAC_RUNTIME_EVIDENCE_LOCAL_ONLY
    if claim_status != expected_status:
        raise ValueError("CODAC runtime evidence claim_status does not match facility_claim_allowed")

    if require_facility_claim:
        if not facility_claim_allowed:
            raise ValueError("CODAC runtime evidence is local-only and cannot support a facility claim")
        if observed_p99 > deadline_us or observed_max > deadline_us:
            raise ValueError("CODAC runtime evidence exceeds the configured cycle deadline")
        if interlock_checks <= 0 or interlock_blocks <= 0:
            raise ValueError("CODAC runtime evidence must exercise and block at least one interlock path")
        if backpressure_events != 0:
            raise ValueError("CODAC runtime evidence with backpressure events cannot support a facility claim")

    return CODACRuntimeEvidence(
        schema_version=str(payload["schema_version"]),
        generated_utc=generated_utc,
        controller_id=controller_id,
        plant_system=plant_system,
        pv_prefix=pv_prefix,
        cycle_hz=cycle_hz,
        cycle_budget_us=cycle_budget_us,
        timeout_budget_us=timeout_budget_us,
        deadline_us=deadline_us,
        observed_cycle_p50_us=observed_p50,
        observed_cycle_p95_us=observed_p95,
        observed_cycle_p99_us=observed_p99,
        observed_cycle_max_us=observed_max,
        input_channel_count=input_count,
        output_channel_count=output_count,
        interlock_pv_count=interlock_pv_count,
        interlock_checks=interlock_checks,
        interlock_blocks=interlock_blocks,
        backpressure_events=backpressure_events,
        epics_db_sha256=str(payload["epics_db_sha256"]),
        opcua_nodeset_sha256=str(payload["opcua_nodeset_sha256"]),
        facility_claim_allowed=facility_claim_allowed,
        claim_status=str(claim_status),
        payload_sha256=str(declared_digest),
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
        return list(self._input_channels)

    def define_output_channels(self) -> list[EPICSChannel]:
        return list(self._output_channels)

    def pack_observation(self, pv_values: Mapping[str, float]) -> ControlObservation:
        """Convert raw EPICS PV dict to ControlObservation for the controller."""
        return ControlObservation(
            R_axis_m=float(pv_values.get("R_axis", pv_values.get("R_axis_m", 6.2))),
            Z_axis_m=float(pv_values.get("Z_axis", pv_values.get("Z_axis_m", 0.0))),
        )

    def unpack_action(self, action: ControlAction) -> dict[str, float]:
        """Convert controller ControlAction to EPICS PV dict."""
        prefix = self.config.pv_prefix
        out: dict[str, float] = {}
        for action_key, suffix in _OUTPUT_KEY_MAP.items():
            pv = f"{prefix}:{suffix}"
            out[pv] = float(action.get(action_key, 0.0))  # type: ignore[arg-type]
        return out

    def run_cycle(self, pv_values: Mapping[str, float]) -> dict[str, float]:
        """Single control cycle: pack -> step -> unpack."""
        self._cycle_timer.start_cycle()
        obs = self.pack_observation(pv_values)
        action = self.controller.step(obs, self._step_k)
        self._step_k += 1
        result = self.unpack_action(action)
        self._cycle_timer.end_cycle()
        return result

    def safety_interlock(self, pv_values: Mapping[str, float]) -> bool:
        """Return True if any hard limit is violated (actuation must be blocked)."""
        for key, (lo, hi) in _HARD_LIMITS.items():
            val = pv_values.get(key)
            if val is None:
                continue
            v = float(val)
            if v < lo or v > hi:
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
    """Build tamper-evident runtime evidence for a CODAC/EPICS boundary."""

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
    """Persist CODAC runtime evidence as canonical sorted JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        self._start_ns = time.perf_counter_ns()

    def end_cycle(self) -> float:
        """Return jitter in milliseconds (elapsed - budget)."""
        self._elapsed_ns = time.perf_counter_ns() - self._start_ns
        return (self._elapsed_ns - self.budget_ns) / 1e6

    def check_overrun(self) -> bool:
        """True if last cycle exceeded its budget."""
        return self._elapsed_ns > self.budget_ns
