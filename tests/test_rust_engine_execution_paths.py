# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neuro-cybernetic engine native handoff and hybrid loop branches

"""Native-handoff, hybrid-loop and configuration branches of the neuro-cybernetic engine.

Drives the configuration validators and execution-loop guards, the native
controller handoff (via a stubbed controller pool: start-signature probing,
telemetry fallbacks, keyboard interrupt and failure escalation), and the Python
hybrid loop edges (state sampler, transport backpressure, bridge-stop failure)
via a stubbed transport bridge.
"""

from __future__ import annotations

from typing import Any

import pytest

import scpn_control.core.rust_engine as rust_engine
from scpn_control.core.rust_engine import NeuroCyberneticEngine, _coerce_telemetry_int


def _engine() -> NeuroCyberneticEngine:
    engine = NeuroCyberneticEngine(n_neurons=8, seed=3)
    engine.configure_acados_targets({"target_r": 6.2, "target_z": 0.0})
    return engine


# ── small helpers and property accessors ──────────────────────────────


def test_coerce_telemetry_int_falls_back_on_unparsable_value() -> None:
    assert _coerce_telemetry_int(None, default=4) == 4
    assert _coerce_telemetry_int([1, 2], default=9) == 9  # non-SupportsInt → default
    assert _coerce_telemetry_int("12") == 12


def test_engine_exposes_native_backend_properties() -> None:
    engine = _engine()
    assert engine.is_native_backend is False
    assert isinstance(engine.native_backend_available, bool)


def test_configure_kuramoto_weights_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="weights must be a mapping or JSON file path"):
        _engine().configure_kuramoto_weights([1.0, 2.0])  # type: ignore[arg-type]


def test_set_state_sampler_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="sampler must be callable"):
        _engine().set_state_sampler(object())  # type: ignore[arg-type]


# ── execute_hardware_loop guards ──────────────────────────────────────


def test_hardware_loop_rejects_spin_pacing_on_python_backend() -> None:
    with pytest.raises(ValueError, match="spin pacing is only available with the native"):
        _engine().execute_hardware_loop(steps=1, execution_backend="python", pacing_mode="spin")


def test_hardware_loop_rejects_reentrant_run() -> None:
    engine = _engine()
    engine._running = True
    with pytest.raises(RuntimeError, match="campaign already running"):
        engine.execute_hardware_loop(steps=1)


def test_hardware_loop_requires_positive_tick_with_runtime() -> None:
    with pytest.raises(ValueError, match="tick_interval_s must be > 0 when runtime_s"):
        _engine().execute_hardware_loop(runtime_s=1.0, tick_interval_s=0.0)


def test_hardware_loop_rejects_negative_steps() -> None:
    with pytest.raises(ValueError, match="steps must be non-negative"):
        _engine().execute_hardware_loop(steps=-1)


def test_hardware_loop_clamps_negative_runtime_steps_to_zero(bridge: type["_FakeBridge"]) -> None:
    # runtime_s/tick yields zero iterations; the loop completes immediately
    result = _engine().execute_hardware_loop(runtime_s=0.0, tick_interval_s=0.001, runtime_admission_policy="off")
    assert result["steps"] == 0


def test_hardware_loop_takes_min_of_steps_and_runtime_budget(bridge: type["_FakeBridge"]) -> None:
    result = _engine().execute_hardware_loop(
        steps=2, runtime_s=0.05, tick_interval_s=0.001, runtime_admission_policy="off"
    )
    assert result["steps"] <= 2


def test_native_only_backend_without_bridge_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fail-closed guard fires only when the native controller pool is
    # unavailable; force that precondition so the test is independent of whether
    # ``scpn_control_rs`` is installed in the running environment.
    monkeypatch.setattr(rust_engine, "_NATIVE_CONTROLLER_AVAILABLE", False)
    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", None)
    with pytest.raises(RuntimeError, match="native controller bridge is not available"):
        _engine().execute_hardware_loop(steps=1, execution_backend="native", runtime_admission_policy="off")


def test_execute_campaign_and_run_hardware_campaign_aliases_delegate(bridge: type["_FakeBridge"]) -> None:
    engine = _engine()
    campaign = engine.execute_campaign(steps=1, tick_interval_s=0.0, runtime_admission_policy="off")
    assert campaign["execution"]["mode"] == "python"
    alias = engine.run_hardware_campaign(steps=1, tick_interval_s=0.0, runtime_admission_policy="off")
    assert alias["status"] in {"completed", "normal"}


# ── native controller handoff (stubbed pool) ──────────────────────────


class _FakePool:
    mode = "positional"
    telemetry: Any = {"steps": 3, "total_cycle_ns": 30, "publish_failures": 0, "dropped": 0}
    fail_init = False

    def __init__(self, *, n_neurons: int, seed: int) -> None:
        if type(self).fail_init:
            raise ValueError("native pool construction failed")
        self.n_neurons = n_neurons
        self.seed = seed
        self.shutdown = False

    def start(self, *args: Any, **kwargs: Any) -> None:
        mode = type(self).mode
        if mode == "kbi":
            raise KeyboardInterrupt
        if mode == "fail":
            raise TypeError("incompatible signature")
        if mode == "kwargs" and args:
            raise TypeError("positional signature not supported")
        return None

    def extract_slab_telemetry(self) -> Any:
        return type(self).telemetry

    def force_shutdown(self) -> None:
        self.shutdown = True


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> type[_FakePool]:
    _FakePool.mode = "positional"
    _FakePool.telemetry = {"steps": 3, "total_cycle_ns": 30, "publish_failures": 0, "dropped": 0}
    _FakePool.fail_init = False
    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", _FakePool)
    monkeypatch.setattr(rust_engine, "_NATIVE_CONTROLLER_AVAILABLE", True)
    return _FakePool


def test_native_handoff_success_with_positional_start(native: type[_FakePool]) -> None:
    result = _engine().execute_hardware_loop(steps=3, execution_backend="native", runtime_admission_policy="off")
    assert result["execution"]["mode"] == "native"
    assert result["steps"] == 3


def test_native_handoff_supports_unbounded_iteration_budget(native: type[_FakePool]) -> None:
    # neither steps nor runtime_s set: the native budget is unbounded (max_iterations=None)
    result = _engine().execute_hardware_loop(execution_backend="native", runtime_admission_policy="off")
    assert result["execution"]["mode"] == "native"


def test_native_handoff_probes_keyword_start_signatures(native: type[_FakePool]) -> None:
    native.mode = "kwargs"
    result = _engine().execute_hardware_loop(steps=2, execution_backend="native", runtime_admission_policy="off")
    assert result["execution"]["mode"] == "native"


def test_native_handoff_synthesises_summary_when_telemetry_absent(native: type[_FakePool]) -> None:
    native.telemetry = None  # non-dict telemetry forces the synthesised summary
    result = _engine().execute_hardware_loop(steps=2, execution_backend="native", runtime_admission_policy="off")
    assert result["native"]["execution"]["mode"] == "native"
    assert result["native"]["steps"] == 2


def test_native_handoff_reports_keyboard_interrupt(native: type[_FakePool]) -> None:
    native.mode = "kbi"
    native.telemetry = None  # non-dict telemetry on interrupt forces the minimal summary
    result = _engine().execute_hardware_loop(steps=2, execution_backend="native", runtime_admission_policy="off")
    assert result["status"] == "keyboard_interrupt"


def test_native_handoff_escalates_incompatible_start(native: type[_FakePool]) -> None:
    native.mode = "fail"
    with pytest.raises(RuntimeError, match="native controller execution failed"):
        _engine().execute_hardware_loop(steps=2, execution_backend="native", runtime_admission_policy="off")


def test_native_handoff_rejects_pool_without_start(native: type[_FakePool], monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoStartPool:
        def __init__(self, *, n_neurons: int, seed: int) -> None:
            self.start = 5  # not callable

        def extract_slab_telemetry(self) -> dict[str, Any]:
            return {}

        def force_shutdown(self) -> None:
            return None

    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", _NoStartPool)
    # the missing start() raises inside the handoff try-block and is re-wrapped
    with pytest.raises(RuntimeError, match="native controller execution failed"):
        _engine().execute_hardware_loop(steps=1, execution_backend="native", runtime_admission_policy="off")


def test_native_handoff_direct_call_requires_pool_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", None)
    engine = _engine()
    with pytest.raises(RuntimeError, match="native controller pool not available"):
        engine._execute_native_handoff(
            target_r=6.2,
            target_z=0.0,
            max_iterations=1,
            tick_interval=0.0,
            initial_state=(6.2, 0.0),
            plant_gain=0.0,
            pacing_mode="sleep",
        )


def test_native_construction_failure_escalates_on_native_backend(native: type[_FakePool]) -> None:
    native.fail_init = True
    with pytest.raises(ValueError, match="native pool construction failed"):
        _engine().execute_hardware_loop(steps=1, execution_backend="native", runtime_admission_policy="off")


def test_native_construction_failure_falls_back_to_python_on_auto(
    native: type[_FakePool], bridge: type["_FakeBridge"]
) -> None:
    native.fail_init = True
    result = _engine().execute_hardware_loop(
        steps=1, execution_backend="auto", tick_interval_s=0.0, runtime_admission_policy="off"
    )
    assert result["execution"]["mode"] == "python"


# ── hybrid loop edges (stubbed transport bridge) ──────────────────────


class _FakeBridge:
    publish_ok = True
    stop_raises = False
    heartbeat = False

    def __init__(self, **kwargs: Any) -> None:
        self._running = False

    def start(self) -> None:
        self._running = True

    def heartbeat_expired(self) -> bool:
        return type(self).heartbeat

    def publish(self, *args: Any) -> bool:
        return type(self).publish_ok

    def stop(self) -> None:
        if type(self).stop_raises:
            raise RuntimeError("bridge stop failed")
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def stopped(self) -> bool:
        return not self._running

    def heartbeat_age_ns(self) -> int:
        return 0

    def payload_bytes(self) -> int:
        return 0


class _FakeSnn:
    def __init__(self, target_r: float, target_z: float) -> None:
        self.target_r = target_r
        self.target_z = target_z

    def step(self, measured_r: float, measured_z: float) -> tuple[float, float]:
        return (0.01, 0.01)


class _FakePID:
    @classmethod
    def radial(cls) -> "_FakePID":
        return cls()

    @classmethod
    def vertical(cls) -> "_FakePID":
        return cls()

    def step(self, error: float) -> float:
        return -0.5 * float(error)


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> type[_FakeBridge]:
    _FakeBridge.publish_ok = True
    _FakeBridge.stop_raises = False
    _FakeBridge.heartbeat = False
    monkeypatch.setattr(rust_engine, "RustUdpTransportBridge", _FakeBridge)
    monkeypatch.setattr(rust_engine, "RustSnnController", _FakeSnn)
    monkeypatch.setattr(rust_engine, "RustPIDController", _FakePID)
    # Pin the orchestrated Python hybrid path. The optional ``scpn_control_rs``
    # native controller pool is importable in some environments (a local
    # maturin build) but absent in others (CI), which would otherwise route
    # these loop-edge tests through the native handoff and bypass the stubbed
    # bridge. The native handoff is covered separately via a stubbed pool.
    monkeypatch.setattr(rust_engine, "_NATIVE_CONTROLLER_AVAILABLE", False)
    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", None)
    return _FakeBridge


def test_hybrid_loop_samples_live_plant_state(bridge: type[_FakeBridge]) -> None:
    engine = _engine()
    engine.set_state_sampler(lambda: (6.2, 0.0))
    result = engine.execute_hardware_loop(steps=2, tick_interval_s=0.0, runtime_admission_policy="off")
    assert result["steps"] == 2


def test_hybrid_loop_halts_on_transport_backpressure(bridge: type[_FakeBridge]) -> None:
    bridge.publish_ok = False
    engine = _engine()
    with pytest.raises(RuntimeError, match="transport backpressure exceeded"):
        engine.execute_hardware_loop(
            steps=4, max_publish_failures=0, tick_interval_s=0.0, runtime_admission_policy="off"
        )


def test_hybrid_loop_paces_with_positive_tick_interval(bridge: type[_FakeBridge]) -> None:
    result = _engine().execute_hardware_loop(steps=1, tick_interval_s=0.001, runtime_admission_policy="off")
    assert result["steps"] == 1


def test_hybrid_loop_propagates_keyboard_interrupt(bridge: type[_FakeBridge]) -> None:
    engine = _engine()

    def _interrupt() -> tuple[float, float]:
        raise KeyboardInterrupt

    engine.set_state_sampler(_interrupt)
    with pytest.raises(KeyboardInterrupt):
        engine.execute_hardware_loop(steps=2, tick_interval_s=0.0, runtime_admission_policy="off")


def test_hybrid_loop_tolerates_bridge_stop_failure(bridge: type[_FakeBridge]) -> None:
    bridge.stop_raises = True
    result = _engine().execute_hardware_loop(steps=1, tick_interval_s=0.0, runtime_admission_policy="off")
    assert result["steps"] == 1


# ── stop / telemetry / optional dispatch ──────────────────────────────


def test_stop_tolerates_bridge_failure(bridge: type[_FakeBridge]) -> None:
    bridge.stop_raises = True
    engine = _engine()
    engine.execute_hardware_loop(steps=1, tick_interval_s=0.0, runtime_admission_policy="off")
    engine._bridge = _FakeBridge()
    engine._bridge.start()
    engine.stop()
    assert engine.is_running is False


def test_extract_slab_telemetry_includes_native_pool_summary() -> None:
    engine = _engine()

    class _Pool:
        def extract_slab_telemetry(self) -> dict[str, Any]:
            return {"steps": 5}

    engine._native_pool = _Pool()
    telemetry = engine.extract_slab_telemetry()
    assert telemetry["native"] == {"steps": 5}


def test_extract_slab_telemetry_falls_back_to_native_campaign_summary() -> None:
    engine = _engine()

    class _Pool:
        def extract_slab_telemetry(self) -> None:
            return None

        def campaign_summary(self) -> dict[str, Any]:
            return {"summary": "native"}

    engine._native_pool = _Pool()
    assert engine.extract_slab_telemetry()["native"] == {"summary": "native"}


def test_call_optional_returns_none_for_non_callable_attribute() -> None:
    class _Holder:
        marker = 7

    assert _engine()._call_optional(_Holder(), "marker") is None


def test_call_optional_swallows_incompatible_signature() -> None:
    class _Holder:
        def configure(self, value: int) -> int:
            return value

    # method_name absent from the PyO3 arg-order table → the TypeError is swallowed
    assert _engine()._call_optional(_Holder(), "configure", value=1, extra=2) is None


def test_native_handoff_reuses_existing_pool(native: type[_FakePool]) -> None:
    """Exercise rust_engine.py:739->746 — a second native run reuses the built pool."""
    engine = _engine()
    first = engine.execute_hardware_loop(steps=1, execution_backend="native", runtime_admission_policy="off")
    assert first["execution"]["mode"] == "native"
    pool_after_first = engine._native_pool
    assert pool_after_first is not None
    second = engine.execute_hardware_loop(steps=1, execution_backend="native", runtime_admission_policy="off")
    assert second["execution"]["mode"] == "native"
    assert engine._native_pool is pool_after_first


def test_native_handoff_keyboard_interrupt_preserves_dict_summary(native: type[_FakePool]) -> None:
    """Exercise rust_engine.py:908->910 — dict telemetry on interrupt is kept as-is."""
    native.mode = "kbi"
    native.telemetry = {"steps": 1, "status": "interrupted"}
    result = _engine().execute_hardware_loop(steps=2, execution_backend="native", runtime_admission_policy="off")
    assert result["status"] == "keyboard_interrupt"
    assert result["native"]["steps"] == 1


def test_hybrid_loop_skips_sleep_when_cycle_exceeds_tick(bridge: type["_FakeBridge"]) -> None:
    """Exercise rust_engine.py:1036->1039 — a sub-cycle tick budget skips the pacing sleep."""
    result = _engine().execute_hardware_loop(steps=1, tick_interval_s=1e-9, runtime_admission_policy="off")
    assert result["steps"] == 1


def test_stop_on_fresh_engine_is_a_noop() -> None:
    """Exercise rust_engine.py:1090->1095 — stop() with no bridge (nor pool) is a no-op."""
    engine = _engine()
    engine.stop()
    assert engine._bridge is None
    assert engine._native_pool is None


def test_extract_slab_telemetry_skips_non_dict_native_summary() -> None:
    """Exercise rust_engine.py:1139->1141 — non-dict native telemetry is omitted."""
    engine = _engine()

    class _Pool:
        def extract_slab_telemetry(self) -> str:
            return "not-a-mapping"

    engine._native_pool = _Pool()
    telemetry = engine.extract_slab_telemetry()
    assert "native" not in telemetry


def test_campaign_summary_omits_runtime_admission_when_unset() -> None:
    """Exercise rust_engine.py:1184->1187 — no recorded admission omits the block."""
    engine = _engine()
    assert engine._last_runtime_admission is None
    summary = engine._campaign_summary(
        steps=0,
        elapsed_s=0.0,
        total_cycle_ns=0,
        publish_failures=0,
        dropped=0,
        status="idle",
        snapshot=None,
        heartbeat_expired=False,
        execution_mode="python",
    )
    assert "runtime_admission" not in summary


def test_call_optional_reraises_typeerror_without_kwargs() -> None:
    """Exercise rust_engine.py:1224->1238 — a no-kwargs TypeError skips the reorder fallback."""

    class _Holder:
        def blow_up(self) -> None:
            raise TypeError("positional boom")

    assert _engine()._call_optional(_Holder(), "blow_up") is None


def test_call_optional_arg_reorder_skips_on_key_mismatch() -> None:
    """Exercise rust_engine.py:1228->1238 — an unknown kwarg foils the arg-order reorder."""

    class _Holder:
        def set_transport_settings(self, **kwargs: Any) -> None:
            raise TypeError("positional signature required")

    result = _engine()._call_optional(_Holder(), "set_transport_settings", endpoint="udp://x", bogus="y")
    assert result is None
