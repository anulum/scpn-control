# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Geometry-Neutral Contracts
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────
"""Geometry-neutral control contracts for non-tokamak replay adapters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


def _require_text(name: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _require_finite(name: str, value: float) -> float:
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f"{name} must be finite.")
    return value_f


def _require_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


@dataclass(frozen=True)
class MagneticConfiguration:
    """Magnetic-device metadata independent of tokamak axis observations."""

    name: str
    device_class: str
    field_periods: int
    coordinate_system: str
    reference: str

    def __post_init__(self) -> None:
        _require_text("name", self.name)
        _require_text("device_class", self.device_class)
        _require_text("coordinate_system", self.coordinate_system)
        _require_text("reference", self.reference)
        if _require_int("field_periods", self.field_periods) < 1:
            raise ValueError("field_periods must be >= 1.")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "device_class": self.device_class,
            "field_periods": int(self.field_periods),
            "coordinate_system": self.coordinate_system,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class ActuatorChannel:
    """Bounded actuator channel with slew-rate and latency semantics."""

    name: str
    unit: str
    min_value: float
    max_value: float
    slew_rate_per_s: float
    latency_steps: int = 0
    failure_mode: str = "none"

    def __post_init__(self) -> None:
        _require_text("name", self.name)
        _require_text("unit", self.unit)
        min_value = _require_finite("min_value", self.min_value)
        max_value = _require_finite("max_value", self.max_value)
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value.")
        slew_rate = _require_finite("slew_rate_per_s", self.slew_rate_per_s)
        if slew_rate <= 0.0:
            raise ValueError("slew_rate_per_s must be > 0.")
        _require_text("failure_mode", self.failure_mode)
        if _require_int("latency_steps", self.latency_steps) < 0:
            raise ValueError("latency_steps must be >= 0.")

    def clamp(self, value: float) -> float:
        value_f = _require_finite("value", value)
        return min(max(value_f, float(self.min_value)), float(self.max_value))

    def apply_slew(self, *, previous: float, requested: float, dt_s: float) -> float:
        previous_f = _require_finite("previous", previous)
        requested_f = self.clamp(requested)
        dt = _require_finite("dt_s", dt_s)
        if dt <= 0.0:
            raise ValueError("dt_s must be > 0.")
        max_delta = float(self.slew_rate_per_s) * dt
        delta = min(max(requested_f - previous_f, -max_delta), max_delta)
        return self.clamp(previous_f + delta)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "min_value": float(self.min_value),
            "max_value": float(self.max_value),
            "slew_rate_per_s": float(self.slew_rate_per_s),
            "latency_steps": int(self.latency_steps),
            "failure_mode": self.failure_mode,
        }


@dataclass(frozen=True)
class ActuatorSet:
    """Unique named actuator channels."""

    channels: tuple[ActuatorChannel, ...]

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("ActuatorSet requires at least one channel.")
        names = [channel.name for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("Actuator channel names must be unique.")

    def by_name(self, name: str) -> ActuatorChannel:
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(f"unknown actuator channel: {name}")

    def to_dict(self) -> dict[str, object]:
        return {"channels": [channel.to_dict() for channel in self.channels]}


@dataclass(frozen=True)
class DiagnosticChannel:
    """One measured or replayed diagnostic value."""

    name: str
    value: float
    unit: str
    sigma: float
    provenance: str

    def __post_init__(self) -> None:
        _require_text("name", self.name)
        _require_text("unit", self.unit)
        _require_text("provenance", self.provenance)
        _require_finite("value", self.value)
        sigma = _require_finite("sigma", self.sigma)
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0.")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": float(self.value),
            "unit": self.unit,
            "sigma": float(self.sigma),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class DiagnosticFrame:
    """Replay diagnostic frame at one control tick."""

    step: int
    time_s: float
    channels: tuple[DiagnosticChannel, ...]

    def __post_init__(self) -> None:
        if _require_int("step", self.step) < 0:
            raise ValueError("step must be >= 0.")
        time_s = _require_finite("time_s", self.time_s)
        if time_s < 0.0:
            raise ValueError("time_s must be >= 0.")
        if not self.channels:
            raise ValueError("DiagnosticFrame requires at least one channel.")
        names = [channel.name for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("Diagnostic channel names must be unique.")

    def as_mapping(self) -> dict[str, float]:
        return {channel.name: float(channel.value) for channel in self.channels}

    def to_dict(self) -> dict[str, object]:
        return {
            "step": int(self.step),
            "time_s": float(self.time_s),
            "channels": [channel.to_dict() for channel in self.channels],
        }


@dataclass(frozen=True)
class ControlObjective:
    """Named targets, weights, and hard constraints for a replay scenario."""

    target_metrics: Mapping[str, float]
    weights: Mapping[str, float]
    constraints: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.target_metrics:
            raise ValueError("target_metrics must not be empty.")
        for group_name, group in (
            ("target_metrics", self.target_metrics),
            ("weights", self.weights),
            ("constraints", self.constraints),
        ):
            for key, value in group.items():
                _require_text(f"{group_name} key", key)
                _require_finite(f"{group_name}.{key}", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_metrics": {key: float(value) for key, value in self.target_metrics.items()},
            "weights": {key: float(value) for key, value in self.weights.items()},
            "constraints": {key: float(value) for key, value in self.constraints.items()},
        }


@dataclass(frozen=True)
class ReplayScenario:
    """Deterministic replay scenario contract for compact control adapters."""

    name: str
    seed: int
    steps: int
    dt_s: float
    magnetic_configuration: MagneticConfiguration
    actuator_set: ActuatorSet
    objective: ControlObjective
    initial_frame: DiagnosticFrame
    fault_schedule: Mapping[int, Mapping[str, str]]

    def __post_init__(self) -> None:
        _require_text("name", self.name)
        _require_int("seed", self.seed)
        steps = _require_int("steps", self.steps)
        if steps < 2:
            raise ValueError("steps must be >= 2.")
        dt = _require_finite("dt_s", self.dt_s)
        if dt <= 0.0:
            raise ValueError("dt_s must be > 0.")
        for step, faults in self.fault_schedule.items():
            step_i = _require_int("fault schedule step", step)
            if step_i < 0 or step_i >= steps:
                raise ValueError("fault schedule step out of replay bounds.")
            if not faults:
                raise ValueError("fault schedule entries must not be empty.")
            for channel_name, fault_mode in faults.items():
                _require_text("fault channel", channel_name)
                _require_text("fault mode", fault_mode)
                self.actuator_set.by_name(channel_name)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": int(self.seed),
            "steps": int(self.steps),
            "dt_s": float(self.dt_s),
            "magnetic_configuration": self.magnetic_configuration.to_dict(),
            "actuator_set": self.actuator_set.to_dict(),
            "objective": self.objective.to_dict(),
            "initial_frame": self.initial_frame.to_dict(),
            "fault_schedule": {str(int(step)): dict(faults) for step, faults in self.fault_schedule.items()},
        }


__all__ = [
    "ActuatorChannel",
    "ActuatorSet",
    "ControlObjective",
    "DiagnosticChannel",
    "DiagnosticFrame",
    "MagneticConfiguration",
    "ReplayScenario",
]
