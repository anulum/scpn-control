# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Rust Engine Tests

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

import scpn_control.core.rust_engine
from scpn_control.core import rust_engine

assert scpn_control.core.rust_engine is rust_engine


def test_execute_hardware_loop_rejects_unknown_execution_backend() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    with pytest.raises(ValueError, match="execution_backend"):
        engine.execute_hardware_loop(steps=1, execution_backend="unknown")


def test_execute_hardware_loop_rejects_unknown_pacing_mode() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    with pytest.raises(ValueError, match="pacing_mode"):
        engine.execute_hardware_loop(steps=1, pacing_mode="timer_magic")


def test_execute_hardware_loop_rejects_spin_pacing_on_python_backend() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    with pytest.raises(ValueError, match="native execution backend"):
        engine.execute_hardware_loop(steps=1, execution_backend="python", pacing_mode="spin")


def test_execute_hardware_loop_force_native_requires_native_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = rust_engine.NeuroCyberneticEngine()
    monkeypatch.setattr(rust_engine, "_NATIVE_CONTROLLER_AVAILABLE", False)
    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", None)

    with pytest.raises(RuntimeError, match="native controller bridge"):
        engine.execute_hardware_loop(steps=1, execution_backend="native")


def test_execute_hardware_loop_force_python_skips_native(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = rust_engine.NeuroCyberneticEngine()
    monkeypatch.setattr(rust_engine, "_NATIVE_CONTROLLER_AVAILABLE", True)
    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", object)

    def fail_native(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("native path should not be called")

    def fake_python(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"status": "completed", "execution": {"mode": "python"}}

    monkeypatch.setattr(rust_engine.NeuroCyberneticEngine, "_execute_native_handoff", fail_native)
    monkeypatch.setattr(rust_engine.NeuroCyberneticEngine, "_execute_hybrid_loop", fake_python)

    summary = engine.execute_hardware_loop(steps=1, execution_backend="python")

    assert summary["execution"]["mode"] == "python"


def test_execute_hardware_loop_auto_falls_back_after_non_runtime_native_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = rust_engine.NeuroCyberneticEngine()
    monkeypatch.setattr(rust_engine, "_NATIVE_CONTROLLER_AVAILABLE", True)
    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", object)

    def fail_native(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise OSError("native extension mismatch")

    def fake_python(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"status": "completed", "execution": {"mode": "python"}}

    monkeypatch.setattr(rust_engine.NeuroCyberneticEngine, "_execute_native_handoff", fail_native)
    monkeypatch.setattr(rust_engine.NeuroCyberneticEngine, "_execute_hybrid_loop", fake_python)

    summary = engine.execute_hardware_loop(steps=1, execution_backend="auto")

    assert summary["execution"]["mode"] == "python"


def test_configure_native_formal_verification_accepts_sync_stride_mode() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    settings = engine.configure_native_formal_verification(
        mode="sync_stride",
        dispatch_interval_steps=5,
        channel_capacity=3,
    )

    assert settings["enabled"] is True
    assert settings["mode"] == "sync_stride"
    assert settings["dispatch_interval_steps"] == 5
    assert settings["channel_capacity"] == 3


def test_configure_native_formal_verification_accepts_aot_certificate_mode() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    settings = engine.configure_native_formal_verification(mode="certificate")

    assert settings["enabled"] is True
    assert settings["mode"] == "aot_certificate"


def test_configure_native_formal_verification_disabled_alias_turns_off_checks() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    settings = engine.configure_native_formal_verification(mode="disabled")

    assert settings["enabled"] is False
    assert settings["mode"] == "async_drop"


def test_configure_native_formal_verification_rejects_unknown_mode() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    with pytest.raises(ValueError, match="native formal verification mode"):
        engine.configure_native_formal_verification(mode="inline_magic")


def test_configure_acados_targets_normalises_aliases_and_safety_clamps() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    payload = engine.configure_acados_targets(
        {
            "R_target": 6.37,
            "Z_target": -0.12,
            "c_gB_nominal": 0.041,
            "beta_n_limit": 3.4,
            "u_min": -9.0,
            "u_max": 11.0,
        }
    )

    assert payload == {
        "target_r": 6.37,
        "target_z": -0.12,
        "rho_tor_target": 6.37,
        "z_tor_target": -0.12,
        "beta_n_limit": 3.4,
        "c_gB": 0.041,
        "u_min": -9.0,
        "u_max": 11.0,
    }


def test_configure_acados_targets_rejects_invalid_gyro_bohm_coefficient() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    with pytest.raises(ValueError, match="c_gB"):
        engine.configure_acados_targets({"c_gB": 0.0})


def test_configure_transport_validates_network_bounds() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    engine.configure_transport(
        endpoint="239.7.7.7",
        port=6000,
        ttl=16,
        max_queue=128,
        backend="io-uring",
        heartbeat_port=6001,
        heartbeat_timeout_ms=5,
    )

    assert engine.transport_backend == "io-uring"
    with pytest.raises(ValueError, match="port"):
        engine.configure_transport(port=0)
    with pytest.raises(ValueError, match="ttl"):
        engine.configure_transport(ttl=0)
    with pytest.raises(ValueError, match="heartbeat_timeout_ms"):
        engine.configure_transport(heartbeat_timeout_ms=0)


def test_configure_kuramoto_weights_loads_json_and_rejects_nonfinite(tmp_path: Path) -> None:
    engine = rust_engine.NeuroCyberneticEngine()
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"k_phase": 0.25, "k_error": -0.1}), encoding="utf-8")

    loaded = engine.configure_kuramoto_weights(weights)

    assert loaded == {"k_phase": 0.25, "k_error": -0.1}
    with pytest.raises(ValueError, match="kuramoto_weight"):
        engine.configure_kuramoto_weights({"bad": math.inf})


def test_configure_itpa_gyro_bohm_accepts_upgraded_and_legacy_schema(tmp_path: Path) -> None:
    engine = rust_engine.NeuroCyberneticEngine()
    upgraded = tmp_path / "gyro_bohm_coefficients.json"
    upgraded.write_text(
        json.dumps(
            {
                "scaling_parameters": {
                    "c_gB_nominal": 0.0424,
                    "alpha_Te": 1.5,
                    "alpha_B": -2.0,
                    "normalized_radius_bounds": {"rho_tor_min": 0.1, "rho_tor_max": 0.95},
                }
            }
        ),
        encoding="utf-8",
    )
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"c_gB": 0.03}), encoding="utf-8")

    upgraded_loaded = engine.configure_itpa_gyro_bohm(upgraded)
    legacy_loaded = engine.configure_itpa_gyro_bohm(legacy)

    assert upgraded_loaded == {
        "c_gB": 0.0424,
        "alpha_Te": 1.5,
        "alpha_B": -2.0,
        "rho_tor_min": 0.1,
        "rho_tor_max": 0.95,
    }
    assert legacy_loaded == {"c_gB": 0.03}


def test_configure_itpa_gyro_bohm_rejects_missing_coefficient(tmp_path: Path) -> None:
    engine = rust_engine.NeuroCyberneticEngine()
    bad_payload = tmp_path / "bad.json"
    bad_payload.write_text(json.dumps({"metadata": {"source": "empty"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required coefficient"):
        engine.configure_itpa_gyro_bohm(bad_payload)


def test_rust_engine_scalar_helpers_fail_closed_on_bad_inputs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    not_object = tmp_path / "array.json"
    not_object.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    assert rust_engine._normalise_execution_backend("rust-native") == "native"
    assert rust_engine._normalise_execution_backend("fallback") == "python"
    assert rust_engine._normalise_pacing_mode("busy-wait") == "spin"
    assert rust_engine._coerce_telemetry_int("not-int", default=7) == 7
    with pytest.raises(TypeError, match="must be numeric"):
        rust_engine._finite_float("gain", object())
    with pytest.raises(ValueError, match="must be <="):
        rust_engine._finite_float("gain", 2.0, max_value=1.0)
    with pytest.raises(ValueError, match="must be <="):
        rust_engine._coerce_int("core", 4096, maximum=4095)
    with pytest.raises(FileNotFoundError, match="file does not exist"):
        rust_engine._read_json_dict(missing)
    with pytest.raises(TypeError, match="expected JSON object"):
        rust_engine._read_json_dict(not_object)


class _FakeNativePool:
    instances: list["_FakeNativePool"] = []

    def __init__(self, *, n_neurons: int, seed: int) -> None:
        self.n_neurons = n_neurons
        self.seed = seed
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.started_with: tuple[Any, ...] | None = None
        _FakeNativePool.instances.append(self)

    def set_nmpc_targets(self, payload: dict[str, float]) -> None:
        self.calls.append(("set_nmpc_targets", (payload,), {}))

    def configure_transport(self, *args: Any) -> None:
        self.calls.append(("configure_transport", args, {}))

    def configure_native_formal_verification(self, *args: Any) -> None:
        self.calls.append(("configure_native_formal_verification", args, {}))

    def configure_runtime_budget(self, *args: Any) -> None:
        self.calls.append(("configure_runtime_budget", args, {}))

    def start(self, core_snn: int, core_z3: int, max_steps: int) -> None:
        self.started_with = (core_snn, core_z3, max_steps)

    def extract_slab_telemetry(self) -> dict[str, object]:
        return {
            "steps": 4,
            "total_cycle_ns": 800,
            "publish_failures": 1,
            "dropped": 2,
            "execution": {"native_binding": "fake"},
        }


def test_execute_native_handoff_forwards_policy_and_summarises_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeNativePool.instances.clear()
    monkeypatch.setattr(rust_engine, "_NATIVE_CONTROLLER_AVAILABLE", True)
    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", _FakeNativePool)
    monkeypatch.setattr(
        rust_engine,
        "collect_runtime_admission",
        lambda _request: {
            "status": "pass",
            "production_claim_allowed": True,
            "errors": [],
            "warnings": [],
            "request": {},
            "probe": {},
        },
    )
    engine = rust_engine.NeuroCyberneticEngine(n_neurons=8, seed=9)
    engine.configure_acados_targets({"target_r": 6.3, "target_z": 0.2, "c_gB": 0.05})
    engine.configure_transport(endpoint="239.8.8.8", port=6000, max_queue=32, backend="std")
    engine.configure_native_formal_verification(mode="aot", dispatch_interval_steps=1, channel_capacity=8)

    summary = engine.execute_hardware_loop(
        steps=4,
        tick_interval_s=0.0001,
        execution_backend="native",
        runtime_admission_policy="warn",
    )

    pool = _FakeNativePool.instances[0]
    assert pool.n_neurons == 8
    assert pool.seed == 9
    assert pool.started_with == (1, 2, 4)
    assert ("set_nmpc_targets", (engine._acados_targets,), {}) in pool.calls
    assert any(name == "configure_transport" and args[0] == "239.8.8.8" for name, args, _kwargs in pool.calls)
    assert any(
        name == "configure_native_formal_verification" and args[0] is True and args[-1] == "aot_certificate"
        for name, args, _kwargs in pool.calls
    )
    assert summary["execution"]["mode"] == "native"
    assert summary["steps"] == 4
    assert summary["publish_failures"] == 1
    assert summary["dropped"] == 2


class _FailingNativePool(_FakeNativePool):
    def start(self, core_snn: int, core_z3: int, max_steps: int) -> None:
        raise RuntimeError(f"contract failure on cores {core_snn}/{core_z3}/{max_steps}")

    def force_shutdown(self) -> None:
        self.calls.append(("force_shutdown", (), {}))


def test_execute_native_handoff_fails_closed_and_forces_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    _FailingNativePool.instances.clear()
    monkeypatch.setattr(rust_engine, "_NATIVE_CONTROLLER_AVAILABLE", True)
    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", _FailingNativePool)
    monkeypatch.setattr(
        rust_engine,
        "collect_runtime_admission",
        lambda _request: {
            "status": "pass",
            "production_claim_allowed": True,
            "errors": [],
            "warnings": [],
            "request": {},
            "probe": {},
        },
    )
    engine = rust_engine.NeuroCyberneticEngine()

    with pytest.raises(RuntimeError, match="native controller execution failed"):
        engine.execute_hardware_loop(steps=1, execution_backend="native")

    assert any(name == "force_shutdown" for name, _args, _kwargs in _FailingNativePool.instances[0].calls)


class _FakeSnnController:
    def __init__(self, target_r: float, target_z: float) -> None:
        self.target = (target_r, target_z)

    def step(self, measured_r: float, measured_z: float) -> tuple[float, float]:
        return (2.0 + measured_r * 0.0, -2.0 + measured_z * 0.0)


class _FakePidController:
    def __init__(self, gain: float) -> None:
        self._gain = gain

    @classmethod
    def radial(cls) -> "_FakePidController":
        return cls(0.5)

    @classmethod
    def vertical(cls) -> "_FakePidController":
        return cls(-0.5)

    def step(self, error: float) -> float:
        return self._gain * error


class _FakeTransportBridge:
    instances: list["_FakeTransportBridge"] = []

    def __init__(
        self,
        *,
        endpoint: str,
        port: int,
        ttl: int,
        max_queue: int,
        backend: str,
        heartbeat_port: int,
        heartbeat_timeout_ms: int,
    ) -> None:
        self.endpoint = endpoint
        self.port = port
        self.ttl = ttl
        self.max_queue = max_queue
        self.backend = backend
        self.heartbeat_port = heartbeat_port
        self.heartbeat_timeout_ms = heartbeat_timeout_ms
        self.publish_results = [False, True]
        self.published: list[tuple[float, float, float, float, int, int, int]] = []
        self.started = False
        self.stop_count = 0
        _FakeTransportBridge.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stop_count += 1
        self.started = False

    def publish(
        self,
        r_error: float,
        z_error: float,
        r_command: float,
        z_command: float,
        acados_time_ns: int,
        snn_time_ns: int,
        reserved_ns: int,
    ) -> bool:
        self.published.append((r_error, z_error, r_command, z_command, acados_time_ns, snn_time_ns, reserved_ns))
        return self.publish_results.pop(0) if self.publish_results else True

    def heartbeat_expired(self) -> bool:
        return False

    def heartbeat_age_ns(self) -> int:
        return 17

    def payload_bytes(self) -> int:
        return 64

    def is_running(self) -> bool:
        return self.started

    def stopped(self) -> bool:
        return not self.started


class _ExpiredHeartbeatBridge(_FakeTransportBridge):
    def heartbeat_expired(self) -> bool:
        return True


def _patch_hybrid_primitives(monkeypatch: pytest.MonkeyPatch, bridge_type: type[_FakeTransportBridge]) -> None:
    _FakeTransportBridge.instances.clear()
    monkeypatch.setattr(rust_engine, "RustSnnController", _FakeSnnController)
    monkeypatch.setattr(rust_engine, "RustPIDController", _FakePidController)
    monkeypatch.setattr(rust_engine, "RustUdpTransportBridge", bridge_type)


def test_execute_hybrid_loop_tracks_backpressure_snapshot_and_bridge_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hybrid_primitives(monkeypatch, _FakeTransportBridge)
    engine = rust_engine.NeuroCyberneticEngine()
    engine.configure_acados_targets({"target_r": 6.0, "target_z": 0.0, "u_min": -1.0, "u_max": 1.0})
    engine.configure_transport(endpoint="239.9.9.9", port=6010, backend="std")

    summary = engine.execute_hardware_loop(
        steps=2,
        tick_interval_s=0.0,
        initial_state=(6.2, -0.2),
        plant_gain=0.1,
        max_publish_failures=2,
        execution_backend="python",
        runtime_admission_policy="off",
    )
    bridge = _FakeTransportBridge.instances[0]
    telemetry = engine.extract_slab_telemetry()

    assert summary["status"] == "completed"
    assert summary["steps"] == 2
    assert summary["dropped"] == 1
    assert summary["publish_failures"] == 0
    assert summary["snapshot"]["publish_ok"] is True
    assert bridge.published[0][2] == 1.0
    assert bridge.published[0][3] == -1.0
    assert bridge.stop_count == 1
    assert telemetry["bridge_payload_bytes"] == 64
    assert telemetry["bridge_stopped"] is True


def test_execute_hybrid_loop_rejects_bad_state_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hybrid_primitives(monkeypatch, _FakeTransportBridge)
    engine = rust_engine.NeuroCyberneticEngine()

    def bad_sampler() -> Any:
        return (1.0, 2.0, 3.0)

    engine.set_state_sampler(bad_sampler)

    with pytest.raises(TypeError, match="state_sampler"):
        engine.execute_hardware_loop(
            steps=1,
            tick_interval_s=0.0,
            execution_backend="python",
            runtime_admission_policy="off",
        )

    assert _FakeTransportBridge.instances[0].stop_count == 1


def test_execute_hybrid_loop_fails_closed_on_heartbeat_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hybrid_primitives(monkeypatch, _ExpiredHeartbeatBridge)
    engine = rust_engine.NeuroCyberneticEngine()
    engine.configure_transport(heartbeat_port=6020)

    with pytest.raises(RuntimeError, match="heartbeat timeout"):
        engine.execute_hardware_loop(
            steps=1,
            tick_interval_s=0.0,
            execution_backend="python",
            runtime_admission_policy="off",
        )

    assert _FakeTransportBridge.instances[0].stop_count == 1


def test_emergency_shutdown_stops_native_and_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hybrid_primitives(monkeypatch, _FakeTransportBridge)
    engine = rust_engine.NeuroCyberneticEngine()
    native = _FakeNativePool(n_neurons=4, seed=5)
    bridge = _FakeTransportBridge(
        endpoint="239.0.0.1",
        port=5555,
        ttl=1,
        max_queue=4,
        backend="std",
        heartbeat_port=0,
        heartbeat_timeout_ms=3,
    )
    bridge.start()
    engine._native_pool = native
    engine._bridge = bridge
    engine._running = True

    telemetry = engine.execute_emergency_shutdown()

    assert telemetry["running"] is False
    assert telemetry["bridge_stopped"] is True
    assert engine.is_running is False
