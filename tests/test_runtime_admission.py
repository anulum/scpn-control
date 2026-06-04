# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime Admission Tests

from __future__ import annotations

import sys
import types

import pytest

from scpn_control.core.runtime_admission import (
    RuntimeAdmissionProbe,
    RuntimeAdmissionRequest,
    collect_runtime_probe,
    evaluate_runtime_admission,
    normalise_runtime_admission_policy,
    skipped_runtime_admission,
)


def _probe(**overrides: object) -> RuntimeAdmissionProbe:
    values = {
        "generated_ns": 1,
        "platform_system": "Linux",
        "kernel_release": "6.17.0-generic",
        "is_linux": True,
        "preempt_rt": False,
        "realtime_sysfs": None,
        "affinity": (1, 2, 3, 4),
        "available_parallelism": 8,
        "scheduler_policy": "SCHED_OTHER",
        "scheduler_priority": 0,
        "governors": {1: "powersave", 2: "powersave", 3: "powersave", 4: "powersave"},
        "memlock_soft_bytes": 8192,
        "memlock_hard_bytes": 8192,
        "native_snapshot": None,
    }
    values.update(overrides)
    return RuntimeAdmissionProbe(**values)  # type: ignore[arg-type]


def test_runtime_admission_require_fails_without_rt_evidence() -> None:
    request = RuntimeAdmissionRequest(
        execution_backend="native",
        pacing_mode="spin",
        native_backend_available=True,
        require_preempt_rt=True,
        require_realtime_scheduler=True,
        require_performance_governor=True,
        require_heartbeat=True,
        heartbeat_port=0,
    )

    report = evaluate_runtime_admission(request, _probe())

    assert report["status"] == "fail"
    assert any("PREEMPT_RT" in error for error in report["errors"])
    assert any("real-time scheduler" in error for error in report["errors"])
    assert any("heartbeat" in error for error in report["errors"])
    assert report["production_claim_allowed"] is False


def test_runtime_admission_passes_with_rt_scheduler_and_governor() -> None:
    request = RuntimeAdmissionRequest(
        execution_backend="native",
        pacing_mode="spin",
        native_backend_available=True,
        require_preempt_rt=True,
        require_realtime_scheduler=True,
        require_performance_governor=True,
        require_heartbeat=True,
        heartbeat_port=5556,
    )
    probe = _probe(
        kernel_release="6.12.0-rt",
        preempt_rt=True,
        scheduler_policy="SCHED_FIFO",
        scheduler_priority=99,
        governors={1: "performance", 2: "performance", 3: "performance", 4: "performance"},
        memlock_soft_bytes="unlimited",
        memlock_hard_bytes="unlimited",
    )

    report = evaluate_runtime_admission(request, probe)

    assert report["status"] == "pass"
    assert report["errors"] == []
    assert report["production_claim_allowed"] is True


def test_runtime_admission_rejects_duplicate_or_unavailable_cores() -> None:
    request = RuntimeAdmissionRequest(core_snn=1, core_z3=1, core_net=7, core_hb=4)

    report = evaluate_runtime_admission(request, _probe())

    assert report["status"] == "fail"
    assert any("distinct" in error for error in report["errors"])
    assert any("outside current process affinity" in error for error in report["errors"])


def test_runtime_admission_policy_normalisation_and_skip_record() -> None:
    assert normalise_runtime_admission_policy("fail-closed") == "require"
    assert normalise_runtime_admission_policy("disabled") == "off"
    assert skipped_runtime_admission()["status"] == "skipped"

    with pytest.raises(ValueError, match="runtime admission policy"):
        normalise_runtime_admission_policy("maybe")


def test_runtime_probe_binds_native_snapshot_when_extension_exposes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    native = types.ModuleType("scpn_control_rs")

    def runtime_admission_snapshot(
        core_snn: int,
        core_z3: int,
        core_net: int,
        core_hb: int,
        tick_interval_s: float,
        pacing_mode: str,
    ) -> dict[str, object]:
        return {
            "schema_version": "scpn-control.runtime-admission-native.v1",
            "requested_cores": [core_snn, core_z3, core_net, core_hb],
            "tick_interval_s": tick_interval_s,
            "pacing_mode": pacing_mode,
        }

    native.__dict__["runtime_admission_snapshot"] = runtime_admission_snapshot
    monkeypatch.setitem(sys.modules, "scpn_control_rs", native)

    probe = collect_runtime_probe(
        RuntimeAdmissionRequest(
            core_snn=4,
            core_z3=5,
            core_net=6,
            core_hb=7,
            tick_interval_s=0.0001,
            pacing_mode="spin",
        )
    )

    assert probe.native_snapshot is not None
    assert probe.native_snapshot["schema_version"] == "scpn-control.runtime-admission-native.v1"
    assert probe.native_snapshot["requested_cores"] == [4, 5, 6, 7]
