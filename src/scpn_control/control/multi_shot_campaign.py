# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-shot pulsed-campaign orchestration.
"""Multi-shot campaign admission over pulsed scheduler and bank telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from typing import Any

from scpn_control.control.capacitor_bank_state import CapacitorBank, CapacitorBankSpec
from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    CapacitorBankTelemetry,
    PulsedPlasmaTelemetry,
    PulsedScenarioCommand,
    PulsedScenarioScheduler,
    PulsedScenarioSpec,
)

MULTI_SHOT_CAMPAIGN_SCHEMA_VERSION = "scpn-control.multi-shot-campaign.v1"
_EXPECTED_TRANSITION_STATES = (
    "ramp_up",
    "flat_top",
    "burn",
    "expansion",
    "dump",
    "recharge",
    "cool_down",
    "idle",
)


@dataclass(frozen=True)
class CampaignShotSample:
    """One timestamped plasma and bank telemetry sample for a pulsed shot."""

    t_s: float
    plasma: PulsedPlasmaTelemetry
    bank: CapacitorBankTelemetry | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_s", _non_negative("t_s", self.t_s))
        if not isinstance(self.plasma, PulsedPlasmaTelemetry):
            raise TypeError("plasma must be PulsedPlasmaTelemetry")
        if self.bank is not None and not isinstance(self.bank, CapacitorBankTelemetry):
            raise TypeError("bank must be CapacitorBankTelemetry or None")


@dataclass(frozen=True)
class CampaignShotPlan:
    """Input contract for one shot in a multi-shot campaign."""

    shot_id: str
    samples: tuple[CampaignShotSample, ...]
    initial_bank_voltage_V: float
    initial_bank_current_A: float = 0.0

    def __post_init__(self) -> None:
        shot_id = self.shot_id.strip()
        if not shot_id:
            raise ValueError("shot_id must not be empty")
        if not self.samples:
            raise ValueError("samples must not be empty")
        object.__setattr__(self, "shot_id", shot_id)
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(
            self, "initial_bank_voltage_V", _non_negative("initial_bank_voltage_V", self.initial_bank_voltage_V)
        )
        object.__setattr__(
            self, "initial_bank_current_A", _finite("initial_bank_current_A", self.initial_bank_current_A)
        )


@dataclass(frozen=True)
class CampaignCommandLog:
    """Stable command log row emitted by the scheduler."""

    t_s: float
    state: str
    action: str
    reason: str
    transition: bool
    dwell_s: float

    @classmethod
    def from_command(cls, command: PulsedScenarioCommand) -> CampaignCommandLog:
        """Convert a scheduler command to a JSON-stable row."""
        return cls(
            t_s=command.t_s,
            state=command.state.value,
            action=command.action.value,
            reason=command.reason,
            transition=command.transition,
            dwell_s=command.dwell_s,
        )

    def to_json(self) -> dict[str, float | str | bool]:
        """Return the JSON representation."""
        return {
            "t_s": self.t_s,
            "state": self.state,
            "action": self.action,
            "reason": self.reason,
            "transition": self.transition,
            "dwell_s": self.dwell_s,
        }


@dataclass(frozen=True)
class CampaignShotResult:
    """Admission result for one pulsed shot."""

    shot_id: str
    pulse_id: str
    passed: bool
    failure_reason: str | None
    terminal_state: str
    transition_states: tuple[str, ...]
    command_log: tuple[CampaignCommandLog, ...]
    shot_phase_log: tuple[dict[str, float | str], ...]
    capacitor_state_initial_J: float
    capacitor_state_final_J: float
    energy_recovered_J: float
    trigger_timestamp_ns: int | None

    def to_json(self) -> dict[str, Any]:
        """Return the JSON representation."""
        return {
            "shot_id": self.shot_id,
            "pulse_id": self.pulse_id,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "terminal_state": self.terminal_state,
            "transition_states": list(self.transition_states),
            "command_log": [row.to_json() for row in self.command_log],
            "shot_phase_log": list(self.shot_phase_log),
            "capacitor_state_initial_J": self.capacitor_state_initial_J,
            "capacitor_state_final_J": self.capacitor_state_final_J,
            "energy_recovered_J": self.energy_recovered_J,
            "trigger_timestamp_ns": self.trigger_timestamp_ns,
        }

    def replay_v1_1_extension(self) -> dict[str, Any]:
        """Return fields compatible with geometry-neutral replay v1.1 extensions."""
        return {
            "pulse_id": self.pulse_id,
            "capacitor_state_initial_J": self.capacitor_state_initial_J,
            "trigger_timestamp_ns": self.trigger_timestamp_ns,
            "energy_recovered_J": self.energy_recovered_J,
            "shot_phase_log": list(self.shot_phase_log),
        }


class MultiShotCampaignOrchestrator:
    """Run deterministic multi-shot admission over scheduler and bank contracts."""

    def __init__(
        self,
        campaign_id: str,
        scheduler_spec: PulsedScenarioSpec,
        bank_spec: CapacitorBankSpec,
        *,
        require_complete_lifecycle: bool = True,
    ) -> None:
        campaign = campaign_id.strip()
        if not campaign:
            raise ValueError("campaign_id must not be empty")
        if not isinstance(scheduler_spec, PulsedScenarioSpec):
            raise TypeError("scheduler_spec must be PulsedScenarioSpec")
        if not isinstance(bank_spec, CapacitorBankSpec):
            raise TypeError("bank_spec must be CapacitorBankSpec")
        self.campaign_id = campaign
        self.scheduler_spec = scheduler_spec
        self.bank_spec = bank_spec
        self.require_complete_lifecycle = bool(require_complete_lifecycle)

    def run(self, shots: tuple[CampaignShotPlan, ...] | list[CampaignShotPlan]) -> dict[str, Any]:
        """Run all shots and return a digest-bound campaign report."""
        shot_plans = tuple(shots)
        if not shot_plans:
            raise ValueError("shots must not be empty")
        seen: set[str] = set()
        results: list[CampaignShotResult] = []
        for shot in shot_plans:
            if not isinstance(shot, CampaignShotPlan):
                raise TypeError("shots must contain CampaignShotPlan entries")
            if shot.shot_id in seen:
                raise ValueError(f"duplicate shot_id: {shot.shot_id}")
            seen.add(shot.shot_id)
            results.append(self._run_shot(shot))
        passed_count = sum(1 for result in results if result.passed)
        report: dict[str, Any] = {
            "schema_version": MULTI_SHOT_CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "shot_count": len(results),
            "passed_count": passed_count,
            "failed_count": len(results) - passed_count,
            "require_complete_lifecycle": self.require_complete_lifecycle,
            "expected_transition_states": list(_EXPECTED_TRANSITION_STATES),
            "shots": [result.to_json() for result in results],
            "payload_sha256": "",
        }
        report["payload_sha256"] = _payload_sha256(report)
        return report

    def _run_shot(self, shot: CampaignShotPlan) -> CampaignShotResult:
        scheduler = PulsedScenarioScheduler(self.scheduler_spec)
        bank = CapacitorBank(
            self.bank_spec,
            initial_voltage_V=shot.initial_bank_voltage_V,
            initial_current_A=shot.initial_bank_current_A,
        )
        initial_energy = bank.state.energy_J
        last_energy = initial_energy
        min_energy = initial_energy
        command_log: list[CampaignCommandLog] = []
        failure_reason: str | None = None
        for sample in shot.samples:
            try:
                bank_telemetry = sample.bank if sample.bank is not None else bank.telemetry()
                last_energy = bank_telemetry.energy_J
                min_energy = min(min_energy, last_energy)
                command = scheduler.step(sample.t_s, sample.plasma, bank_telemetry)
            except ValueError as exc:
                failure_reason = str(exc)
                break
            command_log.append(CampaignCommandLog.from_command(command))

        phase_log = tuple(
            {
                "t": record.t_s,
                "state": record.to_state.value,
                "reason": record.reason,
            }
            for record in scheduler.audit_log
        )
        transition_states = tuple(str(row["state"]) for row in phase_log)
        terminal_state = scheduler.state.value
        if failure_reason is None:
            failure_reason = self._admission_failure_reason(terminal_state, transition_states)
        trigger_timestamp_ns = next(
            (int(round(float(row["t"]) * 1_000_000_000)) for row in phase_log if row["state"] == "burn"),
            None,
        )
        return CampaignShotResult(
            shot_id=shot.shot_id,
            pulse_id=_pulse_id(self.campaign_id, shot.shot_id),
            passed=failure_reason is None,
            failure_reason=failure_reason,
            terminal_state=terminal_state,
            transition_states=transition_states,
            command_log=tuple(command_log),
            shot_phase_log=phase_log,
            capacitor_state_initial_J=initial_energy,
            capacitor_state_final_J=last_energy,
            energy_recovered_J=max(last_energy - min_energy, 0.0),
            trigger_timestamp_ns=trigger_timestamp_ns,
        )

    def _admission_failure_reason(self, terminal_state: str, transition_states: tuple[str, ...]) -> str | None:
        if self.require_complete_lifecycle and transition_states != _EXPECTED_TRANSITION_STATES:
            return "shot did not traverse the complete pulsed lifecycle"
        if terminal_state != "idle":
            return "shot did not return to idle"
        return None


def _pulse_id(campaign_id: str, shot_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"scpn-control:{campaign_id}:{shot_id}"))


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(_stable_json(unsigned).encode("utf-8")).hexdigest()


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _non_negative(name: str, value: float) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


__all__ = [
    "CampaignCommandLog",
    "CampaignShotPlan",
    "CampaignShotResult",
    "CampaignShotSample",
    "MULTI_SHOT_CAMPAIGN_SCHEMA_VERSION",
    "MultiShotCampaignOrchestrator",
]
