# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Rust Control-Plane Engine

"""Rust-accelerated control-plane wrapper.

This module exposes a single orchestrator class for launch-time control of the
compiled Rust transport primitives.

Responsibility is split by design:
- Python owns policy, campaign configuration, and lifecycle boundaries.
- Rust owns the per-tick heavy control/transport work and back-pressured UDP path.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsInt, cast

from collections.abc import Callable, Mapping, Sequence

from scpn_control.core._rust_compat import (
    RustPIDController,
    RustSnnController,
    RustUdpTransportBridge,
)
from scpn_control.core.runtime_admission import (
    RuntimeAdmissionRequest,
    collect_runtime_admission,
    normalise_runtime_admission_policy,
    skipped_runtime_admission,
)

try:  # pragma: no cover - optional runtime dependency
    from scpn_control_rs import PySpikingControllerPool as _NativeSpikingControllerPool

    _NATIVE_CONTROLLER_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    _NativeSpikingControllerPool = None
    _NATIVE_CONTROLLER_AVAILABLE = False

try:  # pragma: no cover - optional runtime dependency
    from scpn_control_rs import PyUdpTransportBridge as _RustTransportBridge

    _RUST_TRANSPORT_CONTROLLER_AVAILABLE = True
except Exception:  # pragma: no cover - optional runtime dependency
    _RustTransportBridge = None
    _RUST_TRANSPORT_CONTROLLER_AVAILABLE = False

_NATIVE_BACKEND_AVAILABLE = bool(_RUST_TRANSPORT_CONTROLLER_AVAILABLE and _NativeSpikingControllerPool is not None)

_LOGGER = logging.getLogger("SCPN.Control.RustEngine")

_ITPA_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "validation" / "reference_data" / "itpa" / "gyro_bohm_coefficients.json"
)

_PYO3_ARG_ORDER: dict[str, tuple[str, ...]] = {
    "set_transport_settings": (
        "endpoint",
        "port",
        "ttl",
        "max_queue",
        "backend",
        "heartbeat_port",
        "heartbeat_timeout_ms",
    ),
    "configure_transport": (
        "endpoint",
        "port",
        "ttl",
        "max_queue",
        "backend",
        "heartbeat_port",
        "heartbeat_timeout_ms",
    ),
    "set_execution_affinity": (
        "core_snn",
        "core_z3",
        "core_net",
        "core_hb",
    ),
    "configure_native_formal_verification": (
        "enabled",
        "max_marking",
        "max_depth",
        "dispatch_interval_steps",
        "channel_capacity",
        "mode",
    ),
    "configure_runtime_budget": (
        "max_iterations",
        "tick_interval",
        "initial_state",
        "plant_gain",
        "pacing_mode",
    ),
    "start": (
        "core_snn",
        "core_z3",
        "max_steps",
    ),
}


def _normalise_execution_backend(value: str) -> str:
    """Validate and canonicalise the campaign execution backend selector."""

    backend = str(value).strip().lower().replace("_", "-")
    aliases = {
        "fallback": "python",
        "hybrid": "python",
        "rs": "native",
        "rust": "native",
        "rust-native": "native",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"auto", "native", "python"}:
        raise ValueError("execution_backend must be one of: auto, native, python")
    return backend


def _normalise_pacing_mode(value: str) -> str:
    """Validate and canonicalise the native pacing selector."""

    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "yield": "sleep",
        "scheduler": "sleep",
        "spin_loop": "spin",
        "busy_wait": "spin",
        "busywait": "spin",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"sleep", "spin"}:
        raise ValueError("pacing_mode must be one of: sleep, spin")
    return mode


def _coerce_telemetry_int(value: object, default: int = 0) -> int:
    """Parse integer telemetry values emitted by Python or PyO3 bindings."""

    if value is None:
        return default
    try:
        if isinstance(value, (str, bytes, bytearray)):
            return int(value)
        if isinstance(value, SupportsInt):
            return int(value)
    except (TypeError, ValueError):
        return default
    return default


@dataclass(slots=True)
class _TransportSettings:
    endpoint: str = "239.0.0.1"
    port: int = 5555
    ttl: int = 1
    max_queue: int = 4
    backend: str = "std"
    heartbeat_port: int = 0
    heartbeat_timeout_ms: int = 3


@dataclass(slots=True)
class _ExecutionSettings:
    core_snn: int = 1
    core_z3: int = 2
    core_net: int = 3
    core_hb: int = 4


@dataclass(slots=True)
class _FormalVerificationSettings:
    enabled: bool = True
    mode: str = "async_drop"
    max_marking: int = 100
    max_depth: int = 4
    dispatch_interval_steps: int = 30
    channel_capacity: int = 2


@dataclass(slots=True)
class _CampaignSnapshot:
    step: int
    measured_r: float
    measured_z: float
    r_error: float
    z_error: float
    r_command: float
    z_command: float
    publish_ok: bool
    heartbeat_expired: bool
    cycle_us: float
    acados_time_ns: int
    snn_time_ns: int


class NeuroCyberneticEngine:
    """Coordinate Rust-native control execution from Python.

    Python owns campaign configuration, safety policy, transport policy and telemetry.
    Execution primitives (SNN, PID, transport publisher) are executed through
    compiled Rust bindings.
    """

    def __init__(
        self,
        n_neurons: int = 64,
        seed: int = 7,
        state_init_r: float = 6.2,
        state_init_z: float = 0.0,
        plant_gain: float = 0.0,
    ) -> None:
        self._n_neurons = int(n_neurons)
        self._seed = int(seed)
        self._plant_gain = float(plant_gain)

        self._acados_targets: dict[str, float] = {
            "target_r": float(state_init_r),
            "target_z": float(state_init_z),
            "rho_tor_target": float(state_init_r),
            "z_tor_target": float(state_init_z),
            "beta_n_limit": 2.5,
            "c_gB": 0.0424,
        }

        self._transport = _TransportSettings()
        self._execution = _ExecutionSettings()
        self._formal_verification = _FormalVerificationSettings()
        self._snn: RustSnnController | None = None
        self._pid_r: RustPIDController | None = None
        self._pid_z: RustPIDController | None = None
        self._bridge: Any | None = None
        self._native_pool: object | None = None
        self._running = False
        self._state_sampler: Callable[[], tuple[float, float]] | None = None
        self._itpa_constraints: dict[str, float | str] = {}
        self._kuramoto_weights: dict[str, float] = {}
        self._max_publish_failures = 128
        self._last_runtime_admission: dict[str, Any] | None = None

    @property
    def transport_backend(self) -> str:
        return self._transport.backend

    @property
    def is_running(self) -> bool:
        return bool(self._running)

    @property
    def is_native_backend(self) -> bool:
        return self._native_pool is not None

    @property
    def native_backend_available(self) -> bool:
        """Whether transport-native Rust bindings are importable."""
        return bool(_NATIVE_BACKEND_AVAILABLE)

    def configure_acados_targets(self, targets: Mapping[str, Any]) -> dict[str, float]:
        """Configure control targets and execution-safe scalar bounds.

        Supported keys:
        - target_r / rho_tor_target / R_target
        - target_z / z_tor_target / Z_target
        - beta_n_limit
        - c_gB / c_gB_nominal
        - optional u_min/u_max safety clamps
        """

        target_r = _coalesce_key(
            targets, ("target_r", "rho_tor_target", "R_target"), default=self._acados_targets["target_r"]
        )
        target_z = _coalesce_key(
            targets, ("target_z", "z_tor_target", "Z_target"), default=self._acados_targets["target_z"]
        )
        beta_n_limit = _coalesce_key(targets, ("beta_n_limit",), default=self._acados_targets["beta_n_limit"])
        c_gb = _coalesce_key(
            targets,
            ("c_gB", "c_gB_nominal"),
            default=self._acados_targets["c_gB"],
        )

        payload: dict[str, float] = {
            "target_r": _finite_float("target_r", target_r),
            "target_z": _finite_float("target_z", target_z),
            "rho_tor_target": _finite_float("rho_tor_target", target_r),
            "z_tor_target": _finite_float("z_tor_target", target_z),
            "beta_n_limit": _finite_float("beta_n_limit", beta_n_limit),
            "c_gB": _finite_float("c_gB", c_gb, min_value=1e-12),
        }

        if "u_min" in targets:
            payload["u_min"] = _finite_float("u_min", targets["u_min"], min_value=-1.0e9)
        if "u_max" in targets:
            payload["u_max"] = _finite_float("u_max", targets["u_max"], min_value=-1.0e9)

        self._acados_targets.update(payload)
        _LOGGER.info(
            "ACADOS target payload updated (target_r=%s, target_z=%s, c_gB=%s, beta_n=%s)",
            payload["target_r"],
            payload["target_z"],
            payload["c_gB"],
            payload["beta_n_limit"],
        )
        return dict(payload)

    def configure_transport(
        self,
        *,
        endpoint: str = "239.0.0.1",
        port: int = 5555,
        ttl: int = 1,
        max_queue: int = 4,
        backend: str = "std",
        heartbeat_port: int = 0,
        heartbeat_timeout_ms: int = 3,
    ) -> None:
        self._transport = _TransportSettings(
            endpoint=str(endpoint),
            port=_coerce_int("port", port, minimum=1, maximum=65535),
            ttl=_coerce_int("ttl", ttl, minimum=1, maximum=255),
            max_queue=_coerce_int("max_queue", max_queue, minimum=1, maximum=4096),
            backend=str(backend),
            heartbeat_port=_coerce_int("heartbeat_port", heartbeat_port, minimum=0, maximum=65535),
            heartbeat_timeout_ms=_coerce_int(
                "heartbeat_timeout_ms",
                heartbeat_timeout_ms,
                minimum=1,
                maximum=120_000,
            ),
        )

    def configure_kuramoto_weights(self, weights: Mapping[str, Any] | str | Path) -> dict[str, float]:
        """Load Kuramoto weights that remain under Python policy control.

        Accepts either:

        - a mapping of weight names to float coefficients,
        - a filesystem path to a JSON object containing weight names.
        """

        payload: Mapping[str, Any]
        if isinstance(weights, (str, Path)):
            payload = _read_json_dict(weights)
        else:
            payload = weights

        if not isinstance(payload, Mapping):
            raise TypeError("weights must be a mapping or JSON file path")

        parsed: dict[str, float] = {}
        for name, raw_value in payload.items():
            key = str(name)
            parsed[key] = _finite_float(f"kuramoto_weight[{key}]", raw_value)

        self._kuramoto_weights = parsed
        _LOGGER.info("Configured %s Kuramoto weights", len(parsed))
        return dict(parsed)

    def configure_execution_affinity(
        self, *, core_snn: int = 1, core_z3: int = 2, core_net: int = 3, core_hb: int = 4
    ) -> None:
        """Record preferred execution affinity for campaign telemetry.

        The compiled transport bridge does not currently expose explicit core pinning.
        These fields are retained for reproducibility, campaign replay, and future
        native controller migration.
        """

        self._execution = _ExecutionSettings(
            core_snn=_coerce_int("core_snn", core_snn, minimum=0, maximum=4095),
            core_z3=_coerce_int("core_z3", core_z3, minimum=0, maximum=4095),
            core_net=_coerce_int("core_net", core_net, minimum=0, maximum=4095),
            core_hb=_coerce_int("core_hb", core_hb, minimum=0, maximum=4095),
        )

    def configure_native_formal_verification(
        self,
        *,
        enabled: bool = True,
        mode: str = "async_drop",
        max_marking: int = 100,
        max_depth: int = 4,
        dispatch_interval_steps: int = 30,
        channel_capacity: int = 2,
    ) -> dict[str, int | bool | str]:
        """Configure native formal checking for the fused hardware loop."""

        mode_value = str(mode).strip().lower().replace("-", "_")
        if mode_value == "disabled":
            enabled = False
            mode_value = "async_drop"
        aliases = {
            "async": "async_drop",
            "drop": "async_drop",
            "blocking": "sync_stride",
            "strict": "sync_stride",
            "sync": "sync_stride",
            "aot": "aot_certificate",
            "certificate": "aot_certificate",
            "runtime_certificate": "aot_certificate",
        }
        mode_value = aliases.get(mode_value, mode_value)
        if mode_value not in {"async_drop", "sync_stride", "aot_certificate"}:
            raise ValueError(
                "native formal verification mode must be async_drop, sync_stride, aot_certificate, or disabled"
            )

        max_marking_value = _coerce_int("max_marking", max_marking, minimum=1, maximum=2_147_483_647)
        max_depth_value = _coerce_int("max_depth", max_depth, minimum=1, maximum=64)
        dispatch_interval_value = _coerce_int(
            "dispatch_interval_steps",
            dispatch_interval_steps,
            minimum=1,
            maximum=1_000_000,
        )
        channel_capacity_value = _coerce_int(
            "channel_capacity",
            channel_capacity,
            minimum=1,
            maximum=1024,
        )

        self._formal_verification = _FormalVerificationSettings(
            enabled=bool(enabled),
            mode=mode_value,
            max_marking=max_marking_value,
            max_depth=max_depth_value,
            dispatch_interval_steps=dispatch_interval_value,
            channel_capacity=channel_capacity_value,
        )
        return {
            "enabled": self._formal_verification.enabled,
            "mode": self._formal_verification.mode,
            "max_marking": self._formal_verification.max_marking,
            "max_depth": self._formal_verification.max_depth,
            "dispatch_interval_steps": self._formal_verification.dispatch_interval_steps,
            "channel_capacity": self._formal_verification.channel_capacity,
        }

    def configure_itpa_gyro_bohm(self, path: str | Path | None = None) -> dict[str, float | str]:
        """Load and apply ITPA gyro-Bohm profile constraints.

        Both legacy (`{"c_gB": ...}`) and upgraded schema
        (`scaling_parameters.{c_gB_nominal,...}`) are supported.
        """

        source = Path(path) if path is not None else _ITPA_DEFAULT_PATH
        payload = _read_json_dict(source)

        if "scaling_parameters" in payload and isinstance(payload["scaling_parameters"], dict):
            scaling = payload["scaling_parameters"]
            c_gb = scaling.get("c_gB_nominal", payload.get("c_gB", self._acados_targets["c_gB"]))
            alpha_t = scaling.get("alpha_Te")
            alpha_b = scaling.get("alpha_B")
            rho_bounds = scaling.get("normalized_radius_bounds", {})
            loaded: dict[str, float | str] = {
                "c_gB": float(_finite_scalar("itpa c_gB_nominal", c_gb)),
            }
            if alpha_t is not None:
                loaded["alpha_Te"] = _finite_scalar("itpa alpha_Te", alpha_t)
            if alpha_b is not None:
                loaded["alpha_B"] = _finite_scalar("itpa alpha_B", alpha_b)
            if isinstance(rho_bounds, dict):
                for key in ("rho_tor_min", "rho_tor_max"):
                    if key in rho_bounds:
                        loaded[key] = _finite_scalar(f"itpa {key}", rho_bounds[key])
            self._itpa_constraints = loaded
            self._acados_targets["c_gB"] = float(loaded["c_gB"])
            return loaded

        if "c_gB" in payload:
            loaded = {"c_gB": _finite_scalar("itpa c_gB", payload["c_gB"])}
            self._itpa_constraints = loaded
            self._acados_targets["c_gB"] = float(loaded["c_gB"])
            return loaded

        raise ValueError(f"gyro-Bohm constraint payload is missing required coefficient in {source}")

    def set_max_publish_failures(self, max_failures: int) -> None:
        self._max_publish_failures = _coerce_int("max_publish_failures", max_failures, minimum=0, maximum=1_000_000)

    def set_state_sampler(self, sampler: Callable[[], tuple[float, float]]) -> None:
        if not callable(sampler):
            raise TypeError("sampler must be callable")
        self._state_sampler = sampler

    def execute_campaign(
        self,
        *,
        steps: int | None = None,
        runtime_s: float | None = None,
        tick_interval_s: float = 0.001,
        max_publish_failures: int | None = None,
        initial_state: tuple[float, float] | None = None,
        plant_gain: float | None = None,
        core_snn: int = 1,
        core_z3: int = 2,
        core_net: int = 3,
        core_hb: int = 4,
        execution_backend: str = "auto",
        pacing_mode: str = "sleep",
        runtime_admission_policy: str = "warn",
    ) -> dict[str, Any]:
        """Campaign-facing wrapper used by command-line and external orchestrators."""
        return self.execute_hardware_loop(
            steps=steps,
            runtime_s=runtime_s,
            tick_interval_s=tick_interval_s,
            initial_state=initial_state,
            plant_gain=plant_gain,
            max_publish_failures=max_publish_failures,
            core_snn=core_snn,
            core_z3=core_z3,
            core_net=core_net,
            core_hb=core_hb,
            execution_backend=execution_backend,
            pacing_mode=pacing_mode,
            runtime_admission_policy=runtime_admission_policy,
        )

    def execute_hardware_loop(
        self,
        *,
        steps: int | None = None,
        runtime_s: float | None = None,
        tick_interval_s: float = 0.001,
        initial_state: tuple[float, float] | None = None,
        plant_gain: float | None = None,
        max_publish_failures: int | None = None,
        core_snn: int = 1,
        core_z3: int = 2,
        core_net: int = 3,
        core_hb: int = 4,
        execution_backend: str = "auto",
        pacing_mode: str = "sleep",
        runtime_admission_policy: str = "warn",
    ) -> dict[str, Any]:
        """Run the Rust-accelerated control loop.

        Configuration is owned by Python; transport execution is delegated to
        compiled Rust primitives (`RustSnnController`, `RustPIDController`,
        `RustUdpTransportBridge`).
        """

        execution_backend = _normalise_execution_backend(execution_backend)
        pacing_mode_value = _normalise_pacing_mode(pacing_mode)
        runtime_admission_policy = normalise_runtime_admission_policy(runtime_admission_policy)
        if execution_backend == "python" and pacing_mode_value == "spin":
            raise ValueError("spin pacing is only available with the native execution backend")

        if max_publish_failures is not None:
            self.set_max_publish_failures(max_publish_failures)

        if self._running:
            raise RuntimeError("campaign already running")

        target_r = float(self._acados_targets["target_r"])
        target_z = float(self._acados_targets["target_z"])
        plant_gain = float(self._plant_gain if plant_gain is None else plant_gain)

        if initial_state is None:
            measured_r, measured_z = target_r, target_z
        else:
            measured_r = _finite_float("initial_state[0]", initial_state[0])
            measured_z = _finite_float("initial_state[1]", initial_state[1])

        tick_interval = _finite_float("tick_interval_s", tick_interval_s, min_value=0.0)
        if runtime_s is not None and tick_interval <= 0.0:
            raise ValueError("tick_interval_s must be > 0 when runtime_s is provided")

        runtime_steps = (
            None if runtime_s is None else int(_finite_float("runtime_s", runtime_s, min_value=0.0) / tick_interval)
        )
        if steps is None and runtime_steps is None:
            max_iterations = None
        else:
            if steps is not None and steps < 0:
                raise ValueError("steps must be non-negative")
            if runtime_steps is not None and runtime_steps < 0:
                runtime_steps = 0
            if steps is None:
                max_iterations = runtime_steps
            elif runtime_steps is None:
                max_iterations = steps
            else:
                max_iterations = min(steps, runtime_steps)

        self.configure_execution_affinity(core_snn=core_snn, core_z3=core_z3, core_net=core_net, core_hb=core_hb)

        native_available = _NATIVE_CONTROLLER_AVAILABLE and _NativeSpikingControllerPool is not None
        if execution_backend == "native" and not native_available:
            raise RuntimeError("SC-NEUROCORE native controller bridge is not available")
        if execution_backend == "auto" and pacing_mode_value == "spin" and not native_available:
            raise RuntimeError("spin pacing requires an available native controller bridge")

        self.admit_runtime(
            execution_backend="native" if execution_backend == "auto" and native_available else execution_backend,
            pacing_mode=pacing_mode_value,
            tick_interval_s=tick_interval,
            native_backend_available=native_available,
            policy=runtime_admission_policy,
        )

        if execution_backend in {"auto", "native"} and native_available:
            try:
                return self._execute_native_handoff(
                    target_r=target_r,
                    target_z=target_z,
                    max_iterations=max_iterations,
                    tick_interval=tick_interval,
                    initial_state=(measured_r, measured_z),
                    plant_gain=plant_gain,
                    pacing_mode=pacing_mode_value,
                )
            except RuntimeError:
                raise
            except Exception:
                if execution_backend == "native" or pacing_mode_value == "spin":
                    raise
                _LOGGER.exception("Native controller handoff failed; falling back to Python orchestrated mode")

        return self._execute_hybrid_loop(
            target_r=target_r,
            target_z=target_z,
            initial_state=(measured_r, measured_z),
            plant_gain=plant_gain,
            tick_interval=tick_interval,
            max_iterations=max_iterations,
        )

    def admit_runtime(
        self,
        *,
        execution_backend: str,
        pacing_mode: str,
        tick_interval_s: float,
        native_backend_available: bool,
        policy: str = "warn",
    ) -> dict[str, Any]:
        """Collect runtime admission evidence and optionally fail closed."""

        policy_value = normalise_runtime_admission_policy(policy)
        if policy_value == "off":
            self._last_runtime_admission = skipped_runtime_admission(policy_value)
            return dict(self._last_runtime_admission)

        require = policy_value == "require"
        request = RuntimeAdmissionRequest(
            execution_backend=execution_backend,
            pacing_mode=pacing_mode,
            transport_backend=self._transport.backend,
            formal_mode=self._formal_verification.mode if self._formal_verification.enabled else "disabled",
            tick_interval_s=tick_interval_s,
            core_snn=self._execution.core_snn,
            core_z3=self._execution.core_z3,
            core_net=self._execution.core_net,
            core_hb=self._execution.core_hb,
            heartbeat_port=self._transport.heartbeat_port,
            native_backend_available=native_backend_available,
            require_preempt_rt=require,
            require_realtime_scheduler=require,
            require_performance_governor=require,
            require_heartbeat=require,
        )
        report = collect_runtime_admission(request)
        report["policy"] = policy_value
        self._last_runtime_admission = report

        raw_errors = report.get("errors", ())
        raw_warnings = report.get("warnings", ())
        problems: list[object] = []
        if isinstance(raw_errors, (list, tuple)):
            problems.extend(raw_errors)
        if isinstance(raw_warnings, (list, tuple)):
            problems.extend(raw_warnings)
        if require and report.get("status") != "pass":
            detail = "; ".join(str(problem) for problem in problems[:6])
            raise RuntimeError(f"runtime admission failed: {detail}")
        if policy_value == "warn" and problems:
            _LOGGER.warning("Runtime admission is not production-qualified: %s", "; ".join(map(str, problems[:6])))
        return dict(report)

    def _execute_native_handoff(
        self,
        *,
        target_r: float,
        target_z: float,
        max_iterations: int | None,
        tick_interval: float,
        initial_state: tuple[float, float],
        plant_gain: float,
        pacing_mode: str,
    ) -> dict[str, Any]:
        """Execute via a fused native loop when available."""
        if _NativeSpikingControllerPool is None:
            raise RuntimeError("native controller pool not available")

        if self._native_pool is None:
            self._native_pool = _NativeSpikingControllerPool(
                n_neurons=self._n_neurons,
                seed=self._seed,
            )

        # Keep all execution policy in Python, including campaign budget and safety thresholds.
        self._call_optional(self._native_pool, "set_nmpc_targets", self._acados_targets)
        self._call_optional(self._native_pool, "set_acados_targets", self._acados_targets)
        self._call_optional(self._native_pool, "configure_acados_targets", self._acados_targets)
        self._call_optional(
            self._native_pool,
            "set_transport_settings",
            endpoint=self._transport.endpoint,
            port=self._transport.port,
            ttl=self._transport.ttl,
            max_queue=self._transport.max_queue,
            backend=self._transport.backend,
            heartbeat_port=self._transport.heartbeat_port,
            heartbeat_timeout_ms=self._transport.heartbeat_timeout_ms,
        )
        self._call_optional(
            self._native_pool,
            "set_execution_affinity",
            core_snn=self._execution.core_snn,
            core_z3=self._execution.core_z3,
            core_net=self._execution.core_net,
            core_hb=self._execution.core_hb,
        )
        self._call_optional(
            self._native_pool,
            "set_kuramoto_weights",
            self._kuramoto_weights,
        )
        self._call_optional(
            self._native_pool,
            "configure_itpa_gyro_bohm",
            self._itpa_constraints,
        )
        self._call_optional(
            self._native_pool,
            "configure_transport",
            endpoint=self._transport.endpoint,
            port=self._transport.port,
            ttl=self._transport.ttl,
            max_queue=self._transport.max_queue,
            backend=self._transport.backend,
            heartbeat_port=self._transport.heartbeat_port,
            heartbeat_timeout_ms=self._transport.heartbeat_timeout_ms,
        )

        max_published = 0
        dropped = 0

        self._running = True
        start_ns = time.perf_counter()
        status = "normal"

        try:
            self._call_optional(
                self._native_pool,
                "configure_native_formal_verification",
                enabled=self._formal_verification.enabled,
                max_marking=self._formal_verification.max_marking,
                max_depth=self._formal_verification.max_depth,
                dispatch_interval_steps=self._formal_verification.dispatch_interval_steps,
                channel_capacity=self._formal_verification.channel_capacity,
                mode=self._formal_verification.mode,
            )
            self._call_optional(
                self._native_pool,
                "set_max_publish_failures",
                self._max_publish_failures,
            )
            start_obj = getattr(self._native_pool, "start", None)
            if not callable(start_obj):
                raise RuntimeError("native controller pool does not expose start()")
            start = cast(Callable[..., Any], start_obj)
            self._call_optional(
                self._native_pool,
                "configure_runtime_budget",
                max_iterations=max_iterations,
                tick_interval=tick_interval,
                initial_state=list(initial_state),
                plant_gain=plant_gain,
                pacing_mode=pacing_mode,
            )
            max_steps = 0 if max_iterations is None else max_iterations
            last_start_error: Exception | None = None
            # Native `start()` signatures vary across releases and feature sets;
            # probe permissive call patterns for compatibility.
            for args in (
                (self._execution.core_snn, self._execution.core_z3, max_steps),
                (self._execution.core_snn, self._execution.core_z3),
                (max_steps,),
            ):
                try:
                    start(*args)
                    last_start_error = None
                    break
                except TypeError as exc:
                    last_start_error = exc
            if last_start_error is not None:
                for kwargs in (
                    {"core_snn": self._execution.core_snn, "core_z3": self._execution.core_z3, "max_steps": max_steps},
                    {"core_snn": self._execution.core_snn, "core_z3": self._execution.core_z3},
                    {"max_steps": max_steps},
                    {},
                ):
                    try:
                        if kwargs:
                            start(**kwargs)
                        else:
                            start()
                        last_start_error = None
                        break
                    except TypeError as exc:
                        last_start_error = exc
                        continue
            if last_start_error is not None:
                raise last_start_error

            elapsed_s = time.perf_counter() - start_ns
            native_summary = self._call_optional(self._native_pool, "extract_slab_telemetry")
            if not isinstance(native_summary, dict):
                native_summary = {
                    "status": status,
                    "steps": 0 if max_iterations is None else max_iterations,
                    "elapsed_s": elapsed_s,
                    "published": int(max_published),
                    "dropped": int(dropped),
                    "heartbeat_enabled": bool(self._transport.heartbeat_port > 0),
                    "execution": {
                        "core_snn": self._execution.core_snn,
                        "core_z3": self._execution.core_z3,
                        "core_net": self._execution.core_net,
                        "core_hb": self._execution.core_hb,
                        "mode": "native",
                    },
                }
            else:
                native_summary["execution"] = dict(native_summary.get("execution", {}))
                native_summary.setdefault("mode", "native")
                native_summary.setdefault("heartbeat_enabled", bool(self._transport.heartbeat_port > 0))
            native_steps = _coerce_telemetry_int(
                native_summary.get("steps"),
                0 if max_iterations is None else max_iterations,
            )
            native_total_cycle_ns = _coerce_telemetry_int(
                native_summary.get("total_cycle_ns"),
                _coerce_telemetry_int(native_summary.get("last_cycle_ns"), 0) * max(1, native_steps),
            )
            return self._campaign_summary(
                steps=native_steps,
                elapsed_s=elapsed_s,
                total_cycle_ns=native_total_cycle_ns,
                publish_failures=_coerce_telemetry_int(native_summary.get("publish_failures"), 0),
                dropped=_coerce_telemetry_int(native_summary.get("dropped"), 0),
                status=status,
                snapshot=None,
                heartbeat_expired=False,
                native_summary=native_summary,
                execution_mode="native",
            )
        except KeyboardInterrupt:
            status = "keyboard_interrupt"
            self._call_optional(self._native_pool, "force_shutdown")
            elapsed_s = time.perf_counter() - start_ns
            native_summary = self._call_optional(self._native_pool, "extract_slab_telemetry")
            if not isinstance(native_summary, dict):
                native_summary = {"status": status}
            return self._campaign_summary(
                steps=0 if max_iterations is None else max_iterations,
                elapsed_s=elapsed_s,
                total_cycle_ns=0,
                publish_failures=0,
                dropped=0,
                status=status,
                snapshot=None,
                heartbeat_expired=False,
                execution_mode="native",
                native_summary=native_summary,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            status = "runtime_failure"
            self._call_optional(self._native_pool, "force_shutdown")
            native_summary = self._call_optional(self._native_pool, "extract_slab_telemetry")
            if not isinstance(native_summary, dict):
                native_summary = {"status": status, "mode": "native"}
            raise RuntimeError("native controller execution failed") from exc
        finally:
            self._running = False

    def _execute_hybrid_loop(
        self,
        *,
        target_r: float,
        target_z: float,
        initial_state: tuple[float, float],
        plant_gain: float,
        tick_interval: float,
        max_iterations: int | None,
    ) -> dict[str, Any]:
        measured_r, measured_z = initial_state
        self._snn = RustSnnController(target_r, target_z)
        self._pid_r = RustPIDController.radial()
        self._pid_z = RustPIDController.vertical()
        self._bridge = RustUdpTransportBridge(
            endpoint=self._transport.endpoint,
            port=self._transport.port,
            ttl=self._transport.ttl,
            max_queue=self._transport.max_queue,
            backend=self._transport.backend,
            heartbeat_port=self._transport.heartbeat_port,
            heartbeat_timeout_ms=self._transport.heartbeat_timeout_ms,
        )

        heartbeat_enabled = self._transport.heartbeat_port > 0
        self._running = True
        step = 0
        publish_failures = 0
        dropped = 0
        total_cycle_ns = 0
        latest_snapshot: _CampaignSnapshot | None = None
        status_reason = "normal"
        start_ns = time.perf_counter()

        self._bridge.start()
        try:
            while True:
                if max_iterations is not None and step >= max_iterations:
                    status_reason = "completed"
                    break

                iteration_start_ns = time.perf_counter_ns()

                if heartbeat_enabled and self._bridge.heartbeat_expired():
                    status_reason = "heartbeat_timeout"
                    raise RuntimeError("heartbeat timeout: transport monitor not healthy")

                if self._state_sampler is not None:
                    sample = self._state_sampler()
                    if not isinstance(sample, tuple) or len(sample) != 2:
                        raise TypeError("state_sampler must return tuple[float, float]")
                    measured_r = _finite_float("state_sampler[0]", sample[0])
                    measured_z = _finite_float("state_sampler[1]", sample[1])

                r_error = measured_r - target_r
                z_error = measured_z - target_z

                snn_start_ns = time.perf_counter_ns()
                r_cmd_raw, z_cmd_raw = self._snn.step(measured_r, measured_z)
                snn_end_ns = time.perf_counter_ns()

                pid_start_ns = time.perf_counter_ns()
                r_pid = self._pid_r.step(r_error)
                z_pid = self._pid_z.step(z_error)
                pid_end_ns = time.perf_counter_ns()

                r_cmd = r_cmd_raw + r_pid
                z_cmd = z_cmd_raw + z_pid

                u_min = self._acados_targets.get("u_min")
                u_max = self._acados_targets.get("u_max")
                if u_min is not None:
                    r_cmd = max(float(u_min), r_cmd)
                    z_cmd = max(float(u_min), z_cmd)
                if u_max is not None:
                    r_cmd = min(float(u_max), r_cmd)
                    z_cmd = min(float(u_max), z_cmd)

                publish_ok = self._bridge.publish(
                    float(r_error),
                    float(z_error),
                    float(r_cmd),
                    float(z_cmd),
                    int(pid_end_ns - pid_start_ns),
                    int(snn_end_ns - snn_start_ns),
                    0,
                )

                if not publish_ok:
                    dropped += 1
                    publish_failures += 1
                    _LOGGER.warning("transport backpressure at step=%s", step)
                    if publish_failures > self._max_publish_failures:
                        status_reason = "publisher_backpressure"
                        raise RuntimeError("transport backpressure exceeded configured tolerance")
                else:
                    publish_failures = 0

                cycle_ns = time.perf_counter_ns() - iteration_start_ns
                total_cycle_ns += cycle_ns

                if tick_interval > 0.0:
                    delta_ns = tick_interval * 1_000_000_000.0
                    sleep_ns = int(delta_ns - cycle_ns)
                    if sleep_ns > 0:
                        time.sleep(sleep_ns / 1_000_000_000.0)

                measured_r = measured_r + plant_gain * r_cmd * tick_interval
                measured_z = measured_z + plant_gain * z_cmd * tick_interval

                latest_snapshot = _CampaignSnapshot(
                    step=step,
                    measured_r=measured_r,
                    measured_z=measured_z,
                    r_error=r_error,
                    z_error=z_error,
                    r_command=r_cmd,
                    z_command=z_cmd,
                    publish_ok=bool(publish_ok),
                    heartbeat_expired=False,
                    cycle_us=cycle_ns / 1000.0,
                    acados_time_ns=int(pid_end_ns - pid_start_ns),
                    snn_time_ns=int(snn_end_ns - snn_start_ns),
                )
                step += 1

            elapsed_s = time.perf_counter() - start_ns
            return self._campaign_summary(
                steps=step,
                elapsed_s=elapsed_s,
                total_cycle_ns=total_cycle_ns,
                publish_failures=publish_failures,
                dropped=dropped,
                status=status_reason,
                snapshot=latest_snapshot,
                heartbeat_expired=False,
                execution_mode="python",
            )
        except KeyboardInterrupt:
            status_reason = "keyboard_interrupt"
            raise
        except Exception:
            status_reason = "runtime_failure"
            raise
        finally:
            if self._bridge is not None:
                try:
                    self._bridge.stop()
                except Exception as exc:
                    _LOGGER.warning("bridge stop failed: %s", exc)
            self._running = False

    def stop(self) -> None:
        """Request immediate campaign stop."""
        if self._native_pool is not None:
            self._call_optional(self._native_pool, "force_shutdown")
            self._call_optional(self._native_pool, "stop")
            self._native_pool = None
        if self._bridge is not None:
            try:
                self._bridge.stop()
            except Exception as exc:
                _LOGGER.warning("bridge stop failed: %s", exc)
        self._running = False

    def run_hardware_campaign(self, **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias used by external orchestration scripts."""
        return self.execute_campaign(**kwargs)

    def execute_emergency_shutdown(self) -> dict[str, Any]:
        """Best-effort emergency shutdown hook exposed to callers."""
        self.stop()
        return self.extract_slab_telemetry()

    def extract_slab_telemetry(self) -> dict[str, Any]:
        bridge = self._bridge
        payload: dict[str, Any] = {
            "running": bool(self._running),
            "backend": self._transport.backend,
            "target": {
                "r": float(self._acados_targets.get("target_r", 0.0)),
                "z": float(self._acados_targets.get("target_z", 0.0)),
            },
            "bridge_payload_bytes": 0,
            "bridge_running": bool(bridge.is_running()) if bridge is not None else False,
            "bridge_stopped": bool(bridge.stopped()) if bridge is not None else False,
            "heartbeat_expired": bool(bridge.heartbeat_expired()) if bridge is not None else False,
            "heartbeat_age_ns": int(bridge.heartbeat_age_ns()) if bridge is not None else 0,
            "itpa_profile": self._itpa_constraints,
        }
        if bridge is not None:
            payload["bridge_payload_bytes"] = bridge.payload_bytes()
        if self._last_runtime_admission is not None:
            payload["runtime_admission"] = dict(self._last_runtime_admission)
        if self._native_pool is not None:
            telemetry = self._call_optional(self._native_pool, "extract_slab_telemetry")
            if telemetry is None:
                telemetry = self._call_optional(self._native_pool, "campaign_summary")
            if isinstance(telemetry, dict):
                payload["native"] = telemetry
        return payload

    def _campaign_summary(
        self,
        *,
        steps: int,
        elapsed_s: float,
        total_cycle_ns: int,
        publish_failures: int,
        dropped: int,
        status: str,
        snapshot: _CampaignSnapshot | None,
        heartbeat_expired: bool,
        execution_mode: str = "python",
        native_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        avg_cycle_us = 0.0 if steps == 0 else total_cycle_ns / max(1, steps) / 1000.0
        result: dict[str, Any] = {
            "status": status,
            "steps": int(steps),
            "elapsed_s": float(elapsed_s),
            "target_r": float(self._acados_targets["target_r"]),
            "target_z": float(self._acados_targets["target_z"]),
            "c_gB": float(self._acados_targets["c_gB"]),
            "avg_cycle_us": avg_cycle_us,
            "publish_failures": int(publish_failures),
            "dropped": int(dropped),
            "heartbeat_expired": heartbeat_expired,
            "transport": {
                "endpoint": self._transport.endpoint,
                "port": self._transport.port,
                "backend": self._transport.backend,
                "heartbeat_port": self._transport.heartbeat_port,
            },
            "execution": {
                "core_snn": self._execution.core_snn,
                "core_z3": self._execution.core_z3,
                "core_net": self._execution.core_net,
                "core_hb": self._execution.core_hb,
                "mode": execution_mode,
            },
            "itpa_profile": self._itpa_constraints,
        }
        if self._last_runtime_admission is not None:
            result["runtime_admission"] = dict(self._last_runtime_admission)

        if snapshot is not None:
            result["snapshot"] = {
                "step": snapshot.step,
                "measured_r": snapshot.measured_r,
                "measured_z": snapshot.measured_z,
                "r_error": snapshot.r_error,
                "z_error": snapshot.z_error,
                "r_command": snapshot.r_command,
                "z_command": snapshot.z_command,
                "publish_ok": snapshot.publish_ok,
                "heartbeat_expired": snapshot.heartbeat_expired,
                "cycle_us": snapshot.cycle_us,
                "acados_time_ns": snapshot.acados_time_ns,
                "snn_time_ns": snapshot.snn_time_ns,
            }

        if native_summary is not None:
            result["native"] = native_summary

        return result

    def _call_optional(
        self,
        target: object,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if target is None or not hasattr(target, method_name):
            return None
        method = getattr(target, method_name)
        if not callable(method):
            return None

        try:
            return method(*args, **kwargs)
        except TypeError as exc:
            if kwargs:
                preferred_order = _PYO3_ARG_ORDER.get(method_name)
                if preferred_order is not None:
                    payload = [kwargs[key] for key in preferred_order if key in kwargs]
                    if len(payload) == len(kwargs):
                        try:
                            return method(*args, *payload)
                        except Exception as fallback_exc:  # pragma: no cover - backend-specific
                            _LOGGER.debug(
                                "native optional method fallback for %s failed: %s",
                                method_name,
                                fallback_exc,
                            )
                            return None
            _LOGGER.debug(
                "native optional method %s() not compatible with provided signature: %s",
                method_name,
                exc,
            )
            return None
        except Exception as exc:  # pragma: no cover - optional path
            _LOGGER.debug(
                "native optional method %s() failed without hard-fail: %s",
                method_name,
                exc,
            )
            return None


def _finite_float(
    name: str,
    value: Any,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc

    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} must be <= {max_value}")

    return float(parsed)


def _finite_scalar(name: str, value: Any) -> float:
    return _finite_float(name, value)


def _coerce_int(name: str, value: Any, *, minimum: int | None = None, maximum: int | None = None) -> int:
    parsed = int(_finite_float(name, value))
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return int(parsed)


def _coalesce_key(data: Mapping[str, Any], keys: Sequence[str], *, default: Any) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def _read_json_dict(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"file does not exist: {file_path}")
    with file_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {file_path}")
    return payload
