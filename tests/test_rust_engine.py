# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Rust Engine Tests

from __future__ import annotations

import pytest

import scpn_control.core.rust_engine
from scpn_control.core import rust_engine

assert scpn_control.core.rust_engine is rust_engine


def test_execute_hardware_loop_rejects_unknown_execution_backend() -> None:
    engine = rust_engine.NeuroCyberneticEngine()

    with pytest.raises(ValueError, match="execution_backend"):
        engine.execute_hardware_loop(steps=1, execution_backend="unknown")


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
