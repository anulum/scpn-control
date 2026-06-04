# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime Admission Engine Tests

from __future__ import annotations

import pytest

from scpn_control.core import rust_engine
from scpn_control.core.rust_engine import NeuroCyberneticEngine


def _admission_report(status: str = "fail") -> dict[str, object]:
    return {
        "schema_version": "scpn-control.runtime-admission.v1",
        "status": status,
        "production_claim_allowed": False,
        "errors": [] if status == "pass" else ["PREEMPT_RT kernel evidence is required but was not detected"],
        "warnings": [] if status == "pass" else ["spin pacing is local regression evidence only"],
        "request": {},
        "probe": {},
    }


def test_engine_runtime_admission_require_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_engine, "collect_runtime_admission", lambda request: _admission_report("fail"))
    engine = NeuroCyberneticEngine()
    engine.configure_execution_affinity(core_snn=1, core_z3=2, core_net=3, core_hb=4)

    with pytest.raises(RuntimeError, match="runtime admission failed"):
        engine.execute_hardware_loop(
            steps=0,
            execution_backend="python",
            runtime_admission_policy="require",
        )


def test_engine_runtime_admission_warn_records_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_engine, "collect_runtime_admission", lambda request: _admission_report("fail"))
    engine = NeuroCyberneticEngine()

    report = engine.admit_runtime(
        execution_backend="native",
        pacing_mode="spin",
        tick_interval_s=0.0001,
        native_backend_available=True,
        policy="warn",
    )
    summary = engine.extract_slab_telemetry()

    assert report["status"] == "fail"
    assert summary["runtime_admission"]["status"] == "fail"


def test_engine_runtime_admission_off_records_skipped_report() -> None:
    engine = NeuroCyberneticEngine()

    report = engine.admit_runtime(
        execution_backend="python",
        pacing_mode="sleep",
        tick_interval_s=0.001,
        native_backend_available=False,
        policy="off",
    )

    assert report["status"] == "skipped"
    assert report["production_claim_allowed"] is False


def test_engine_auto_spin_requires_native_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_engine, "_NATIVE_CONTROLLER_AVAILABLE", False)
    monkeypatch.setattr(rust_engine, "_NativeSpikingControllerPool", None)
    engine = NeuroCyberneticEngine()

    with pytest.raises(RuntimeError, match="spin pacing requires an available native controller bridge"):
        engine.execute_hardware_loop(
            steps=0,
            execution_backend="auto",
            pacing_mode="spin",
            runtime_admission_policy="off",
        )
