# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Exact-current LIF runtime.

"""Digest-bound, stateful consumption of the SC-NeuroCore LIF contract.

This module deliberately keeps the historical :meth:`CompiledNet.lif_fire`
gate separate.  The runtime below owns persistent membrane state, consumes
timestamped piecewise-constant currents, and resets only at an explicit shot
boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module, metadata, resources
from pathlib import Path
from typing import Protocol, cast

PROFILE_SCHEMA = "sc-neurocore.exact-current-lif-profile.v1"
STATE_SCHEMA = "sc-neurocore.exact-current-lif-state.v1"
PACKET_SCHEMA = "sc-neurocore.exact-current-lif-packet.v1"
PROFILE_NAME = "sc_exact_current_hard_reset_lif_v1"
RUNTIME_CHECKPOINT_SCHEMA = "scpn-control.exact-current-lif-checkpoint.v1"
RUNTIME_EXECUTION_SCHEMA = "scpn-control.exact-current-lif-execution.v1"

SC_IMPLEMENTATION_COMMIT = "bc76e5b3c217fec191534bb650685316e645ad34"
SC_CONTRACT_COMMIT = "248e88a827acfe9be0d654855ae9d3b7d2dcd527"
SC_PROFILE_SHA256 = "8051be0ff173b0ff6434d3f5b54ab8a1c9f5078f62fddd3e359e6a77deb5c716"
SC_PROFILE_DIGEST = "c667f3885f564dcf968febaf62125a86abaaee4758df792d5f06b0e82d1f121a"
SC_MODEL_SOURCE_SHA256 = "064be334316184e50a85fb82b1a804cdf1342bb927c39588b4d4105c7a087762"
SC_CONTRACT_MODULE_SHA256 = "9e9bd13784c857829b569e3bd4bc32b214452f6d1d0205b45bd9d3117b7ec487"
SC_REFERENCE_PACKET_SHA256 = "d8752ce402c91ad6cae7170add60172454497811e83ae3e8b68930f03f36bcde"

_MAX_PROFILE_BYTES = 64 * 1024
_MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024


class ExactCurrentLIFError(RuntimeError):
    """Base class for stable, typed CONTROL LIF runtime failures."""

    code = "exact_current_lif_error"


class ExactCurrentLIFUnavailableError(ExactCurrentLIFError):
    """Raised when the required SC-NeuroCore contract is unavailable."""

    code = "exact_current_lif_unavailable"


class ExactCurrentLIFBindingError(ExactCurrentLIFError):
    """Raised when profile, source, schema, unit, or commit binding drifts."""

    code = "exact_current_lif_binding_mismatch"


class ExactCurrentLIFInputError(ExactCurrentLIFError):
    """Raised when a timestamped current input violates the contract."""

    code = "exact_current_lif_input_invalid"


class ExactCurrentLIFStateError(ExactCurrentLIFError):
    """Raised when checkpoint state cannot be restored atomically."""

    code = "exact_current_lif_state_invalid"


class ExactCurrentLIFExecutionError(ExactCurrentLIFError):
    """Raised when SC-NeuroCore rejects an otherwise bound execution."""

    code = "exact_current_lif_execution_failed"


class _Profile(Protocol):  # pragma: no cover - static typing contract
    digest: str
    v_rest: float

    def to_payload(self) -> dict[str, object]: ...

    def verify_source_binding(self) -> None: ...


class _State(Protocol):  # pragma: no cover - static typing contract
    shot_id: str


class _Packet(Protocol):  # pragma: no cover - static typing contract
    def to_json(self) -> str: ...


class _Session(Protocol):  # pragma: no cover - static typing contract
    @property
    def state(self) -> _State: ...

    def execute(self, ticks: Sequence[object]) -> _Packet: ...

    def serialize_state(self) -> str: ...

    def restore_state(self, serialized: str | bytes) -> _State: ...

    def reset_shot(self, shot_id: str) -> _State: ...


@dataclass(frozen=True)
class _SCContract:
    profile_from_json: Callable[[str | bytes], _Profile]
    tick_factory: Callable[[float, tuple[float, ...]], object]
    session_factory: Callable[..., _Session]
    packet_from_json: Callable[..., _Packet]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_json(name: str, serialized: str | bytes, *, maximum_bytes: int) -> object:
    size = len(serialized.encode("utf-8")) if isinstance(serialized, str) else len(serialized)
    if size > maximum_bytes:
        raise ValueError(f"{name} exceeds the {maximum_bytes}-byte limit")
    try:
        return json.loads(serialized, object_pairs_hook=_object_without_duplicates)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{name} must be valid JSON") from exc


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _exact_keys(name: str, payload: Mapping[str, object], expected: set[str]) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{name} fields mismatch; missing={sorted(expected - observed)}, unknown={sorted(observed - expected)}"
        )


def _load_sc_contract() -> _SCContract:
    try:
        solvers = import_module("sc_neurocore.solvers")
        contract_module = import_module("sc_neurocore.solvers.exact_lif_profile")
        distribution_version = metadata.version("sc-neurocore")
    except (ImportError, metadata.PackageNotFoundError) as exc:  # pragma: no cover - environment guard
        raise ExactCurrentLIFUnavailableError(
            "sc-neurocore 3.16.0 with the exact-current LIF contract is required"
        ) from exc

    if distribution_version != "3.16.0":  # pragma: no cover - environment guard
        raise ExactCurrentLIFBindingError(f"unsupported sc-neurocore distribution version: {distribution_version}")
    source_path = getattr(contract_module, "__file__", None)
    if not isinstance(source_path, str):  # pragma: no cover - environment guard
        raise ExactCurrentLIFBindingError("SC-NeuroCore contract module has no source path")
    try:
        observed_module_sha256 = _sha256(Path(source_path).read_bytes())
    except OSError as exc:  # pragma: no cover - environment guard
        raise ExactCurrentLIFBindingError("SC-NeuroCore contract source cannot be read") from exc
    if observed_module_sha256 != SC_CONTRACT_MODULE_SHA256:  # pragma: no cover - environment guard
        raise ExactCurrentLIFBindingError("SC-NeuroCore exact-current LIF contract source digest mismatch")

    try:
        profile_class = solvers.ExactCurrentLIFProfile
        tick_class = solvers.CurrentDriveTick
        session_class = solvers.ExactCurrentLIFSession
        packet_class = solvers.ExactLIFExecutionPacket
    except AttributeError as exc:  # pragma: no cover - environment guard
        raise ExactCurrentLIFUnavailableError(
            "SC-NeuroCore does not export the required exact-current LIF API"
        ) from exc

    return _SCContract(
        profile_from_json=cast(Callable[[str | bytes], _Profile], profile_class.from_json),
        tick_factory=cast(Callable[[float, tuple[float, ...]], object], tick_class),
        session_factory=cast(Callable[..., _Session], session_class),
        packet_from_json=cast(Callable[..., _Packet], packet_class.from_json),
    )


@dataclass(frozen=True)
class ExactCurrentLIFProfileBinding:
    """Complete immutable binding to one SC-NeuroCore execution contract."""

    profile_json: str
    profile_artifact_sha256: str
    profile_digest: str
    profile_schema: str
    profile_name: str
    state_schema: str
    packet_schema: str
    model_source_sha256: str
    implementation_commit: str
    contract_commit: str

    @classmethod
    def from_json(
        cls,
        serialized: str | bytes,
        *,
        implementation_commit: str,
        contract_commit: str,
    ) -> ExactCurrentLIFProfileBinding:
        """Validate exact profile bytes and bind them to immutable SC commits."""
        raw_bytes = serialized.encode("utf-8") if isinstance(serialized, str) else serialized
        try:
            payload = _mapping("profile", _load_json("profile", raw_bytes, maximum_bytes=_MAX_PROFILE_BYTES))
            if implementation_commit != SC_IMPLEMENTATION_COMMIT:
                raise ValueError("profile implementation commit mismatch")
            if contract_commit != SC_CONTRACT_COMMIT:
                raise ValueError("profile contract commit mismatch")
            profile_schema = payload.get("schema")
            profile_name = payload.get("profile")
            state = _mapping("profile state", payload.get("state"))
            model = _mapping("profile model", payload.get("model"))
            units = _mapping("profile units", payload.get("units"))
            if profile_schema != PROFILE_SCHEMA or profile_name != PROFILE_NAME:
                raise ValueError("unsupported profile schema or name")
            if state.get("serialization_schema") != STATE_SCHEMA:
                raise ValueError("unsupported state schema")
            if model.get("source_sha256") != SC_MODEL_SOURCE_SHA256:
                raise ValueError("model source digest mismatch")
            if units != {
                "time": "ms",
                "voltage": "normalized_voltage",
                "current": "normalized_current",
                "resistance": "normalized_resistance",
            }:
                raise ValueError("profile unit contract mismatch")
            contract = _load_sc_contract()
            profile = contract.profile_from_json(raw_bytes)
            profile.verify_source_binding()
            if _sha256(raw_bytes) != SC_PROFILE_SHA256:
                raise ValueError("profile artifact digest mismatch")
        except (OSError, TypeError, ValueError) as exc:
            raise ExactCurrentLIFBindingError(str(exc)) from exc

        return cls(
            profile_json=raw_bytes.decode("utf-8"),
            profile_artifact_sha256=SC_PROFILE_SHA256,
            profile_digest=SC_PROFILE_DIGEST,
            profile_schema=PROFILE_SCHEMA,
            profile_name=PROFILE_NAME,
            state_schema=STATE_SCHEMA,
            packet_schema=PACKET_SCHEMA,
            model_source_sha256=SC_MODEL_SOURCE_SHA256,
            implementation_commit=implementation_commit,
            contract_commit=contract_commit,
        )

    @classmethod
    def from_installed_reference(cls) -> ExactCurrentLIFProfileBinding:
        """Load and validate the profile shipped by the installed SC package."""
        try:
            profile_bytes = (
                resources.files("sc_neurocore.neurons")
                .joinpath("reference_trace_data/exact_current_lif_profile_v1.json")
                .read_bytes()
            )
        except (ImportError, ModuleNotFoundError, OSError) as exc:  # pragma: no cover - environment guard
            raise ExactCurrentLIFUnavailableError("installed SC-NeuroCore profile artifact is unavailable") from exc
        return cls.from_json(
            profile_bytes,
            implementation_commit=SC_IMPLEMENTATION_COMMIT,
            contract_commit=SC_CONTRACT_COMMIT,
        )

    def to_payload(self) -> dict[str, object]:
        """Return all compatibility and provenance fields except profile bytes."""
        return {
            "profile": {
                "name": self.profile_name,
                "schema": self.profile_schema,
                "artifact_sha256": self.profile_artifact_sha256,
                "canonical_sha256": self.profile_digest,
            },
            "state_schema": self.state_schema,
            "packet_schema": self.packet_schema,
            "model_source_sha256": self.model_source_sha256,
            "implementation_commit": self.implementation_commit,
            "contract_commit": self.contract_commit,
        }


@dataclass(frozen=True)
class ExactCurrentLIFTransitionTick:
    """One shared duration and simultaneous currents for every transition."""

    duration_ms: float
    transition_currents: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, (int, float)):
            raise ExactCurrentLIFInputError("duration_ms must be a finite positive real value")
        duration_ms = float(self.duration_ms)
        if not math.isfinite(duration_ms) or duration_ms <= 0.0:
            raise ExactCurrentLIFInputError("duration_ms must be a finite positive real value")
        if not isinstance(self.transition_currents, tuple) or any(
            not isinstance(currents, tuple) for currents in self.transition_currents
        ):
            raise ExactCurrentLIFInputError("transition_currents must be a tuple of current tuples")
        normalized: list[tuple[float, ...]] = []
        for currents in self.transition_currents:
            values: list[float] = []
            for current in currents:
                if isinstance(current, bool) or not isinstance(current, (int, float)):
                    raise ExactCurrentLIFInputError("current contributions must be finite real values")
                value = float(current)
                if not math.isfinite(value):
                    raise ExactCurrentLIFInputError("current contributions must be finite real values")
                values.append(value)
            try:
                math.fsum(values)
            except OverflowError as exc:
                raise ExactCurrentLIFInputError("summed current must remain finite") from exc
            normalized.append(tuple(values))
        object.__setattr__(self, "duration_ms", duration_ms)
        object.__setattr__(self, "transition_currents", tuple(normalized))


@dataclass(frozen=True)
class ExactCurrentLIFTransitionPacket:
    """One transition's complete canonical SC-NeuroCore execution packet."""

    transition_name: str
    packet_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.transition_name, str) or not self.transition_name:
            raise ExactCurrentLIFExecutionError("transition_name must be a non-empty string")
        try:
            packet = _mapping(
                "execution packet",
                _load_json(
                    "execution packet",
                    self.packet_json,
                    maximum_bytes=_MAX_CHECKPOINT_BYTES,
                ),
            )
            if packet.get("schema") != PACKET_SCHEMA:
                raise ValueError("unsupported execution packet schema")
            if _canonical_json(packet) != self.packet_json:
                raise ValueError("execution packet must use canonical JSON")
        except (TypeError, ValueError) as exc:
            raise ExactCurrentLIFExecutionError(str(exc)) from exc

    @property
    def sha256(self) -> str:
        """Return the digest of the complete canonical packet."""
        return _sha256(self.packet_json.encode("utf-8"))

    def to_payload(self) -> dict[str, object]:
        """Return the packet as structured data without dropping trace fields."""
        packet = _mapping(
            "execution packet",
            _load_json("execution packet", self.packet_json, maximum_bytes=_MAX_CHECKPOINT_BYTES),
        )
        return {
            "transition": self.transition_name,
            "packet_sha256": self.sha256,
            "packet": dict(packet),
        }


@dataclass(frozen=True)
class ExactCurrentLIFExecution:
    """Complete ordered multi-transition result for one runtime call."""

    packets: tuple[ExactCurrentLIFTransitionPacket, ...]

    def __post_init__(self) -> None:
        if not self.packets or any(not isinstance(packet, ExactCurrentLIFTransitionPacket) for packet in self.packets):
            raise ExactCurrentLIFExecutionError("packets must contain complete transition packets")
        names = [packet.transition_name for packet in self.packets]
        if len(set(names)) != len(names):
            raise ExactCurrentLIFExecutionError("execution transition names must be unique")

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic aggregate without reducing SC packet fidelity."""
        return {
            "schema": RUNTIME_EXECUTION_SCHEMA,
            "transitions": [packet.to_payload() for packet in self.packets],
        }

    def to_json(self) -> str:
        """Serialize the complete aggregate canonically."""
        return _canonical_json(self.to_payload())

    @property
    def sha256(self) -> str:
        """Return the digest of the complete aggregate."""
        return _sha256(self.to_json().encode("utf-8"))


class ExactCurrentLIFRuntime:
    """Failure-atomic persistent SC-NeuroCore sessions for compiled transitions."""

    def __init__(
        self,
        transition_names: Sequence[str],
        binding: ExactCurrentLIFProfileBinding,
        *,
        shot_id: str = "shot-0",
    ) -> None:
        names = tuple(transition_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ExactCurrentLIFBindingError("transition_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ExactCurrentLIFBindingError("transition_names must be unique")
        if not isinstance(binding, ExactCurrentLIFProfileBinding):
            raise ExactCurrentLIFBindingError("binding must be an ExactCurrentLIFProfileBinding")
        self.transition_names = names
        validated_binding = ExactCurrentLIFProfileBinding.from_json(
            binding.profile_json,
            implementation_commit=binding.implementation_commit,
            contract_commit=binding.contract_commit,
        )
        if binding != validated_binding:
            raise ExactCurrentLIFBindingError("binding fields differ from the validated SC-NeuroCore profile")
        self.binding = validated_binding
        self._contract = _load_sc_contract()
        try:
            self._profile = self._contract.profile_from_json(binding.profile_json)
            self._sessions = tuple(self._new_session(shot_id=shot_id) for _ in names)
        except (OSError, TypeError, ValueError) as exc:
            raise ExactCurrentLIFBindingError(str(exc)) from exc

    def _new_session(
        self,
        *,
        shot_id: str,
        serialized_state: str | None = None,
    ) -> _Session:
        session = self._contract.session_factory(
            self._profile,
            producer_commit=self.binding.implementation_commit,
            shot_id=shot_id,
        )
        if serialized_state is not None:
            session.restore_state(serialized_state)
        return session

    @property
    def serialized_states(self) -> tuple[str, ...]:
        """Return complete state envelopes in transition order."""
        return tuple(session.serialize_state() for session in self._sessions)

    def execute(self, ticks: Sequence[ExactCurrentLIFTransitionTick]) -> ExactCurrentLIFExecution:
        """Execute ticks transactionally while preserving state across calls."""
        frozen_ticks = tuple(ticks)
        if any(not isinstance(tick, ExactCurrentLIFTransitionTick) for tick in frozen_ticks):
            raise ExactCurrentLIFInputError("ticks must contain only ExactCurrentLIFTransitionTick values")
        for tick in frozen_ticks:
            if len(tick.transition_currents) != len(self.transition_names):
                raise ExactCurrentLIFInputError("each tick must provide currents for every compiled transition")

        snapshots = self.serialized_states
        try:
            candidates = tuple(
                self._new_session(
                    shot_id=self._sessions[index].state.shot_id,
                    serialized_state=snapshots[index],
                )
                for index in range(len(self.transition_names))
            )
            packets: list[ExactCurrentLIFTransitionPacket] = []
            for transition_index, candidate in enumerate(candidates):
                sc_ticks = tuple(
                    self._contract.tick_factory(tick.duration_ms, tick.transition_currents[transition_index])
                    for tick in frozen_ticks
                )
                packet = candidate.execute(sc_ticks)
                packet_json = packet.to_json()
                self._contract.packet_from_json(
                    packet_json,
                    profile=self._profile,
                    expected_producer_commit=self.binding.implementation_commit,
                )
                packets.append(
                    ExactCurrentLIFTransitionPacket(
                        transition_name=self.transition_names[transition_index],
                        packet_json=packet_json,
                    )
                )
        except (ArithmeticError, OSError, TypeError, ValueError) as exc:
            raise ExactCurrentLIFExecutionError(str(exc)) from exc

        self._sessions = candidates
        return ExactCurrentLIFExecution(tuple(packets))

    def serialize_checkpoint(self) -> str:
        """Serialize all transition states and their exact compatibility binding."""
        states = [
            _mapping(
                "state",
                _load_json("state", state, maximum_bytes=_MAX_CHECKPOINT_BYTES),
            )
            for state in self.serialized_states
        ]
        return _canonical_json(
            {
                "schema": RUNTIME_CHECKPOINT_SCHEMA,
                "binding": self.binding.to_payload(),
                "transition_names": list(self.transition_names),
                "states": [dict(state) for state in states],
            }
        )

    def restore_checkpoint(self, serialized: str | bytes) -> None:
        """Restore every transition atomically after strict compatibility checks."""
        try:
            payload = _mapping(
                "checkpoint",
                _load_json("checkpoint", serialized, maximum_bytes=_MAX_CHECKPOINT_BYTES),
            )
            _exact_keys(
                "checkpoint",
                payload,
                {"schema", "binding", "transition_names", "states"},
            )
            if payload["schema"] != RUNTIME_CHECKPOINT_SCHEMA:
                raise ValueError("unsupported checkpoint schema")
            if payload["binding"] != self.binding.to_payload():
                raise ValueError("checkpoint compatibility binding mismatch")
            if payload["transition_names"] != list(self.transition_names):
                raise ValueError("checkpoint transition ordering mismatch")
            states = payload["states"]
            if not isinstance(states, list) or len(states) != len(self.transition_names):
                raise ValueError("checkpoint state count mismatch")
            candidates: list[_Session] = []
            timeline: tuple[object, object, object] | None = None
            for state in states:
                state_payload = _mapping("checkpoint state", state)
                state_json = _canonical_json(state_payload)
                state_body = _mapping("checkpoint state body", state_payload.get("state"))
                shot_id = state_body.get("shot_id")
                if not isinstance(shot_id, str):
                    raise ValueError("checkpoint shot_id must be a string")
                candidate_timeline = (
                    shot_id,
                    state_body.get("time_ms"),
                    state_body.get("reset_epoch"),
                )
                if timeline is None:
                    timeline = candidate_timeline
                elif candidate_timeline != timeline:
                    raise ValueError("checkpoint states must share shot_id, time_ms, and reset_epoch")
                candidates.append(self._new_session(shot_id=shot_id, serialized_state=state_json))
        except (ArithmeticError, OSError, TypeError, ValueError) as exc:
            raise ExactCurrentLIFStateError(str(exc)) from exc
        self._sessions = tuple(candidates)

    def reset_shot(self, shot_id: str) -> None:
        """Reset all transitions only at an explicit new-shot boundary."""
        snapshots = self.serialized_states
        try:
            candidates = tuple(
                self._new_session(
                    shot_id=session.state.shot_id,
                    serialized_state=snapshots[index],
                )
                for index, session in enumerate(self._sessions)
            )
            for candidate in candidates:
                candidate.reset_shot(shot_id)
        except (OSError, TypeError, ValueError) as exc:
            raise ExactCurrentLIFStateError(str(exc)) from exc
        self._sessions = candidates


__all__ = [
    "ExactCurrentLIFBindingError",
    "ExactCurrentLIFError",
    "ExactCurrentLIFExecution",
    "ExactCurrentLIFExecutionError",
    "ExactCurrentLIFInputError",
    "ExactCurrentLIFProfileBinding",
    "ExactCurrentLIFRuntime",
    "ExactCurrentLIFStateError",
    "ExactCurrentLIFTransitionPacket",
    "ExactCurrentLIFTransitionTick",
    "ExactCurrentLIFUnavailableError",
    "PACKET_SCHEMA",
    "PROFILE_NAME",
    "PROFILE_SCHEMA",
    "RUNTIME_CHECKPOINT_SCHEMA",
    "RUNTIME_EXECUTION_SCHEMA",
    "SC_CONTRACT_COMMIT",
    "SC_IMPLEMENTATION_COMMIT",
    "SC_MODEL_SOURCE_SHA256",
    "SC_PROFILE_DIGEST",
    "SC_PROFILE_SHA256",
    "SC_REFERENCE_PACKET_SHA256",
    "STATE_SCHEMA",
]
