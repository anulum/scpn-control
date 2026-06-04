# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Pulsed-scenario scheduler v2.
"""Reusable pulsed-fusion lifecycle scheduler for CONTROL-owned shot logic."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum


class PulsedScenarioState(StrEnum):
    """Canonical pulsed-fusion lifecycle states."""

    IDLE = "idle"
    RAMP_UP = "ramp_up"
    FLAT_TOP = "flat_top"
    BURN = "burn"
    EXPANSION = "expansion"
    DUMP = "dump"
    RECHARGE = "recharge"
    COOL_DOWN = "cool_down"


class PulsedScenarioAction(StrEnum):
    """Command emitted for the active lifecycle state."""

    ARM_PRECHARGE = "arm_precharge"
    RAMP_FIELD = "ramp_field"
    HOLD_FLAT_TOP = "hold_flat_top"
    FIRE_COMPRESSION = "fire_compression"
    RECOVER_ENERGY = "recover_energy"
    DUMP_RESIDUAL = "dump_residual"
    RECHARGE_BANK = "recharge_bank"
    COOL_DOWN = "cool_down"


@dataclass(frozen=True)
class PulsedScenarioSpec:
    """Guard thresholds for a reusable pulsed-fusion shot lifecycle."""

    min_precharge_energy_J: float
    ramp_current_A: float
    phase_tolerance_rad: float
    spatial_tolerance_m: float
    burn_temperature_eV: float
    min_fusion_power_W: float
    expansion_velocity_m_s: float
    dump_energy_floor_J: float
    recharge_voltage_fraction: float
    cooldown_temperature_eV: float
    cooldown_current_A: float
    min_burn_duration_s: float = 0.0

    def __post_init__(self) -> None:
        for field in (
            "min_precharge_energy_J",
            "ramp_current_A",
            "phase_tolerance_rad",
            "spatial_tolerance_m",
            "burn_temperature_eV",
            "min_fusion_power_W",
            "expansion_velocity_m_s",
            "dump_energy_floor_J",
            "cooldown_temperature_eV",
            "cooldown_current_A",
            "min_burn_duration_s",
        ):
            value = _finite(field, getattr(self, field))
            if value < 0.0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, value)
        if self.phase_tolerance_rad <= 0.0:
            raise ValueError("phase_tolerance_rad must be strictly positive")
        if self.spatial_tolerance_m <= 0.0:
            raise ValueError("spatial_tolerance_m must be strictly positive")
        recharge_fraction = _finite("recharge_voltage_fraction", self.recharge_voltage_fraction)
        if not 0.0 < recharge_fraction <= 1.0:
            raise ValueError("recharge_voltage_fraction must lie in (0, 1]")
        object.__setattr__(self, "recharge_voltage_fraction", recharge_fraction)


@dataclass(frozen=True)
class PulsedPlasmaTelemetry:
    """Plasma telemetry consumed by lifecycle transition guards."""

    coil_current_A: float
    temperature_eV: float
    phase_lock_error_rad: float
    reference_error_m: float
    fusion_power_W: float
    radial_velocity_m_s: float

    def __post_init__(self) -> None:
        for field in (
            "coil_current_A",
            "temperature_eV",
            "phase_lock_error_rad",
            "reference_error_m",
            "fusion_power_W",
            "radial_velocity_m_s",
        ):
            object.__setattr__(self, field, _finite(field, getattr(self, field)))
        if self.temperature_eV < 0.0:
            raise ValueError("temperature_eV must be non-negative")
        if self.phase_lock_error_rad < 0.0:
            raise ValueError("phase_lock_error_rad must be non-negative")
        if self.reference_error_m < 0.0:
            raise ValueError("reference_error_m must be non-negative")
        if self.fusion_power_W < 0.0:
            raise ValueError("fusion_power_W must be non-negative")


@dataclass(frozen=True)
class CapacitorBankTelemetry:
    """Capacitor-bank telemetry consumed by lifecycle transition guards."""

    voltage_V: float
    voltage_max_V: float
    energy_J: float

    def __post_init__(self) -> None:
        voltage = _finite("voltage_V", self.voltage_V)
        voltage_max = _finite("voltage_max_V", self.voltage_max_V)
        energy = _finite("energy_J", self.energy_J)
        if voltage < 0.0:
            raise ValueError("voltage_V must be non-negative")
        if voltage_max <= 0.0:
            raise ValueError("voltage_max_V must be strictly positive")
        if voltage > voltage_max:
            raise ValueError("voltage_V must not exceed voltage_max_V")
        if energy < 0.0:
            raise ValueError("energy_J must be non-negative")
        object.__setattr__(self, "voltage_V", voltage)
        object.__setattr__(self, "voltage_max_V", voltage_max)
        object.__setattr__(self, "energy_J", energy)

    @property
    def voltage_fraction(self) -> float:
        """Return bank voltage as a fraction of the declared maximum."""
        return self.voltage_V / self.voltage_max_V


@dataclass(frozen=True)
class PulsedScenarioTransition:
    """Single lifecycle transition audit entry."""

    t_s: float
    from_state: PulsedScenarioState
    to_state: PulsedScenarioState
    reason: str

    def to_json(self) -> dict[str, float | str]:
        """Return the stable JSON-serialisable audit representation."""
        return {
            "t_s": self.t_s,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PulsedScenarioCommand:
    """Command emitted by one pulsed-scenario scheduler step."""

    t_s: float
    state: PulsedScenarioState
    action: PulsedScenarioAction
    reason: str
    transition: bool
    dwell_s: float


_NEXT_STATE: dict[PulsedScenarioState, PulsedScenarioState] = {
    PulsedScenarioState.IDLE: PulsedScenarioState.RAMP_UP,
    PulsedScenarioState.RAMP_UP: PulsedScenarioState.FLAT_TOP,
    PulsedScenarioState.FLAT_TOP: PulsedScenarioState.BURN,
    PulsedScenarioState.BURN: PulsedScenarioState.EXPANSION,
    PulsedScenarioState.EXPANSION: PulsedScenarioState.DUMP,
    PulsedScenarioState.DUMP: PulsedScenarioState.RECHARGE,
    PulsedScenarioState.RECHARGE: PulsedScenarioState.COOL_DOWN,
    PulsedScenarioState.COOL_DOWN: PulsedScenarioState.IDLE,
}

_ACTION_BY_STATE: dict[PulsedScenarioState, PulsedScenarioAction] = {
    PulsedScenarioState.IDLE: PulsedScenarioAction.ARM_PRECHARGE,
    PulsedScenarioState.RAMP_UP: PulsedScenarioAction.RAMP_FIELD,
    PulsedScenarioState.FLAT_TOP: PulsedScenarioAction.HOLD_FLAT_TOP,
    PulsedScenarioState.BURN: PulsedScenarioAction.FIRE_COMPRESSION,
    PulsedScenarioState.EXPANSION: PulsedScenarioAction.RECOVER_ENERGY,
    PulsedScenarioState.DUMP: PulsedScenarioAction.DUMP_RESIDUAL,
    PulsedScenarioState.RECHARGE: PulsedScenarioAction.RECHARGE_BANK,
    PulsedScenarioState.COOL_DOWN: PulsedScenarioAction.COOL_DOWN,
}


class PulsedScenarioScheduler:
    """CONTROL-owned reusable eight-state scheduler for pulsed-fusion shots."""

    def __init__(self, spec: PulsedScenarioSpec) -> None:
        self.spec = spec
        self.state = PulsedScenarioState.IDLE
        self._last_sample_t_s: float | None = None
        self._last_transition_t_s = 0.0
        self._audit_log: list[PulsedScenarioTransition] = []

    @property
    def audit_log(self) -> tuple[PulsedScenarioTransition, ...]:
        """Return immutable transition audit entries."""
        return tuple(self._audit_log)

    def reset(self) -> None:
        """Return to ``idle`` and clear timestamp and audit state."""
        self.state = PulsedScenarioState.IDLE
        self._last_sample_t_s = None
        self._last_transition_t_s = 0.0
        self._audit_log.clear()

    def transition_to(
        self,
        next_state: PulsedScenarioState | str,
        t_s: float,
        reason: str,
    ) -> PulsedScenarioTransition:
        """Perform a validated manual adjacent transition."""
        time = self._validate_timestamp(t_s)
        target = PulsedScenarioState(next_state)
        if target is not _NEXT_STATE[self.state]:
            raise ValueError(f"invalid transition {self.state.value} -> {target.value}")
        if not reason.strip():
            raise ValueError("reason must not be empty")
        record = self._record_transition(time, target, reason)
        self._last_sample_t_s = time
        return record

    def step(
        self,
        t_s: float,
        plasma: PulsedPlasmaTelemetry,
        bank: CapacitorBankTelemetry,
    ) -> PulsedScenarioCommand:
        """Evaluate lifecycle guards at ``t_s`` and emit the active-state command."""
        time = self._validate_timestamp(t_s)
        dwell = time - self._last_transition_t_s
        next_state, reason = self._guard(plasma, bank, dwell)
        transition = next_state is not None
        if next_state is not None:
            self._record_transition(time, next_state, reason)
        self._last_sample_t_s = time
        return PulsedScenarioCommand(
            t_s=time,
            state=self.state,
            action=_ACTION_BY_STATE[self.state],
            reason=reason,
            transition=transition,
            dwell_s=dwell,
        )

    def audit_log_jsonl(self) -> str:
        """Return the transition audit log as newline-delimited JSON."""
        return "\n".join(json.dumps(record.to_json(), sort_keys=True) for record in self._audit_log)

    def _validate_timestamp(self, t_s: float) -> float:
        time = _finite("t_s", t_s)
        if time < 0.0:
            raise ValueError("t_s must be non-negative")
        if self._last_sample_t_s is not None and time < self._last_sample_t_s:
            raise ValueError("t_s must be monotone")
        return time

    def _record_transition(
        self,
        t_s: float,
        next_state: PulsedScenarioState,
        reason: str,
    ) -> PulsedScenarioTransition:
        previous = self.state
        self.state = next_state
        self._last_transition_t_s = t_s
        record = PulsedScenarioTransition(
            t_s=t_s,
            from_state=previous,
            to_state=next_state,
            reason=reason,
        )
        self._audit_log.append(record)
        return record

    def _guard(
        self,
        plasma: PulsedPlasmaTelemetry,
        bank: CapacitorBankTelemetry,
        dwell_s: float,
    ) -> tuple[PulsedScenarioState | None, str]:
        spec = self.spec
        match self.state:
            case PulsedScenarioState.IDLE:
                if bank.energy_J >= spec.min_precharge_energy_J:
                    return PulsedScenarioState.RAMP_UP, "precharge energy available"
                return None, "waiting for precharge energy"
            case PulsedScenarioState.RAMP_UP:
                if abs(plasma.coil_current_A) >= spec.ramp_current_A:
                    return PulsedScenarioState.FLAT_TOP, "ramp current reached"
                return None, "waiting for ramp current"
            case PulsedScenarioState.FLAT_TOP:
                phase_ok = plasma.phase_lock_error_rad <= spec.phase_tolerance_rad
                spatial_ok = plasma.reference_error_m <= spec.spatial_tolerance_m
                temperature_ok = plasma.temperature_eV >= spec.burn_temperature_eV
                energy_ok = bank.energy_J >= spec.min_precharge_energy_J
                if phase_ok and spatial_ok and temperature_ok and energy_ok:
                    return PulsedScenarioState.BURN, "phase, spatial, temperature, and energy guards passed"
                if not phase_ok or not spatial_ok:
                    return None, "waiting for phase and spatial lock"
                if not temperature_ok:
                    return None, "waiting for burn temperature"
                return None, "waiting for burn energy"
            case PulsedScenarioState.BURN:
                if dwell_s < spec.min_burn_duration_s:
                    return None, "waiting for minimum burn dwell"
                if plasma.fusion_power_W >= spec.min_fusion_power_W:
                    return PulsedScenarioState.EXPANSION, "fusion power threshold reached"
                return None, "waiting for fusion power"
            case PulsedScenarioState.EXPANSION:
                if plasma.radial_velocity_m_s >= spec.expansion_velocity_m_s:
                    return PulsedScenarioState.DUMP, "expansion velocity reached"
                return None, "waiting for expansion velocity"
            case PulsedScenarioState.DUMP:
                if bank.energy_J <= spec.dump_energy_floor_J:
                    return PulsedScenarioState.RECHARGE, "residual energy dumped"
                return None, "waiting for dump energy floor"
            case PulsedScenarioState.RECHARGE:
                if bank.voltage_fraction >= spec.recharge_voltage_fraction:
                    return PulsedScenarioState.COOL_DOWN, "bank recharged"
                return None, "waiting for recharge voltage"
            case PulsedScenarioState.COOL_DOWN:
                temperature_ok = plasma.temperature_eV <= spec.cooldown_temperature_eV
                current_ok = abs(plasma.coil_current_A) <= spec.cooldown_current_A
                if temperature_ok and current_ok:
                    return PulsedScenarioState.IDLE, "plasma cooled and coil current cleared"
                return None, "waiting for plasma cooldown"


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


__all__ = [
    "CapacitorBankTelemetry",
    "PulsedPlasmaTelemetry",
    "PulsedScenarioAction",
    "PulsedScenarioCommand",
    "PulsedScenarioScheduler",
    "PulsedScenarioSpec",
    "PulsedScenarioState",
    "PulsedScenarioTransition",
]
