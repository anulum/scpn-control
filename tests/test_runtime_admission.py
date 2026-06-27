# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime Admission Tests

from __future__ import annotations

import os
import sys
import types
import importlib.util
from pathlib import Path
from typing import cast

import pytest

import scpn_control.core.runtime_admission as ra
from scpn_control.core.runtime_admission import (
    RuntimeAdmissionProbe,
    RuntimeAdmissionRequest,
    _current_affinity,
    _format_limit,
    _limit_to_int,
    _memlock_limits,
    _native_runtime_snapshot,
    _read_governors,
    _scheduler_policy,
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


def _errors(report: dict[str, object]) -> list[str]:
    return cast(list[str], report["errors"])


def _warnings(report: dict[str, object]) -> list[str]:
    return cast(list[str], report["warnings"])


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
    assert any("PREEMPT_RT" in error for error in _errors(report))
    assert any("real-time scheduler" in error for error in _errors(report))
    assert any("heartbeat" in error for error in _errors(report))
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
    assert any("distinct" in error for error in _errors(report))
    assert any("outside current process affinity" in error for error in _errors(report))


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


def test_collect_runtime_admission_evaluates_current_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public collection wrapper binds the live probe to admission evaluation."""
    request = RuntimeAdmissionRequest()
    probe = _probe()

    def collect_probe(observed_request: RuntimeAdmissionRequest) -> RuntimeAdmissionProbe:
        assert observed_request is request
        return probe

    monkeypatch.setattr(ra, "collect_runtime_probe", collect_probe)
    report = ra.collect_runtime_admission(request)

    assert report["status"] == "pass"
    assert report["production_claim_allowed"] is False
    assert report["probe"] == probe.as_dict()


class TestEvaluateRuntimeAdmissionBranches:
    def test_non_linux_probe_is_rejected(self) -> None:
        report = evaluate_runtime_admission(RuntimeAdmissionRequest(), _probe(is_linux=False))
        assert any("requires Linux" in error for error in _errors(report))

    def test_negative_core_is_rejected(self) -> None:
        request = RuntimeAdmissionRequest(core_snn=-1)
        report = evaluate_runtime_admission(request, _probe(affinity=(-1, 2, 3, 4)))
        assert any("non-negative" in error for error in _errors(report))

    def test_spin_pacing_requires_small_tick_interval(self) -> None:
        request = RuntimeAdmissionRequest(
            pacing_mode="spin", execution_backend="native", native_backend_available=True, tick_interval_s=0.02
        )
        report = evaluate_runtime_admission(request, _probe())
        assert any("tick_interval_s <= 0.01" in error for error in _errors(report))

    def test_spin_pacing_rejects_python_backend(self) -> None:
        request = RuntimeAdmissionRequest(
            pacing_mode="spin", execution_backend="python", native_backend_available=True, tick_interval_s=0.005
        )
        report = evaluate_runtime_admission(request, _probe())
        assert any("native execution backend" in error for error in _errors(report))

    def test_spin_pacing_requires_native_bridge(self) -> None:
        request = RuntimeAdmissionRequest(
            pacing_mode="spin", execution_backend="native", native_backend_available=False, tick_interval_s=0.005
        )
        report = evaluate_runtime_admission(request, _probe())
        assert any("native PyO3 controller bridge" in error for error in _errors(report))

    def test_io_uring_transport_is_linux_only(self) -> None:
        request = RuntimeAdmissionRequest(transport_backend="io_uring")
        report = evaluate_runtime_admission(request, _probe(is_linux=False))
        assert any("io-uring transport is Linux-only" in error for error in _errors(report))

    def test_spin_pacing_warns_without_rt_kernel_and_scheduler(self) -> None:
        request = RuntimeAdmissionRequest(
            pacing_mode="spin",
            execution_backend="native",
            native_backend_available=True,
            tick_interval_s=0.005,
            require_preempt_rt=False,
            require_realtime_scheduler=False,
        )
        report = evaluate_runtime_admission(request, _probe(preempt_rt=False, scheduler_policy="SCHED_OTHER"))
        warnings = _warnings(report)
        assert any("PREEMPT_RT" in warning for warning in warnings)
        assert any("SCHED_FIFO/SCHED_RR" in warning for warning in warnings)

    def test_unknown_governor_emits_warning(self) -> None:
        probe = _probe(governors={1: None, 2: None, 3: None, 4: None})
        report = evaluate_runtime_admission(RuntimeAdmissionRequest(), probe)
        assert any("CPU governor unavailable" in warning for warning in _warnings(report))


class TestRuntimeProbeHelpers:
    def test_resource_module_falls_back_to_none_on_win32_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Import-time resource detection has a deterministic Windows fallback."""
        module_name = "_runtime_admission_win32_probe"
        source_path = Path(ra.__file__).resolve()
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, module_name, module)

        spec.loader.exec_module(module)

        assert module._resource is None

    def test_current_affinity_falls_back_when_getaffinity_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_pid: int) -> set[int]:
            raise OSError("affinity unavailable")

        monkeypatch.setattr(os, "sched_getaffinity", _raise, raising=False)
        affinity = _current_affinity()
        assert affinity == tuple(range(os.cpu_count() or 1))

    def test_read_governors_returns_none_for_unreadable_core(self) -> None:
        governors = _read_governors([10_000_000])
        assert governors == {10_000_000: None}

    def test_scheduler_policy_unknown_without_getscheduler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(os, "sched_getscheduler", raising=False)
        assert _scheduler_policy() == ("unknown", None)

    def test_scheduler_policy_unknown_on_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_pid: int) -> int:
            raise OSError("no scheduler")

        monkeypatch.setattr(os, "sched_getscheduler", _raise, raising=False)
        assert _scheduler_policy() == ("unknown", None)

    def test_scheduler_policy_priority_none_on_param_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_pid: int) -> object:
            raise OSError("no param")

        monkeypatch.setattr(os, "sched_getparam", _raise, raising=False)
        _label, priority = _scheduler_policy()
        assert priority is None

    def test_memlock_limits_unknown_without_resource_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ra, "_resource", None)
        assert _memlock_limits() == ("unknown", "unknown")

    @pytest.mark.skipif(sys.platform == "win32", reason="resource module is POSIX-only")
    def test_memlock_limits_unknown_on_getrlimit_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import resource

        def _raise(_which: int) -> tuple[int, int]:
            raise OSError("rlimit unavailable")

        monkeypatch.setattr(resource, "getrlimit", _raise)
        assert _memlock_limits() == ("unknown", "unknown")

    @pytest.mark.skipif(sys.platform == "win32", reason="resource module is POSIX-only")
    def test_format_limit_reports_unlimited_for_infinity(self) -> None:
        import resource

        assert _format_limit(resource.RLIM_INFINITY) == "unlimited"

    def test_limit_to_int_returns_none_for_unknown_string(self) -> None:
        assert _limit_to_int("unknown") is None

    def test_native_runtime_snapshot_none_when_extension_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = types.ModuleType("scpn_control_rs")
        monkeypatch.setitem(sys.modules, "scpn_control_rs", fake)
        assert _native_runtime_snapshot(RuntimeAdmissionRequest()) is None

    def test_native_runtime_snapshot_none_when_snapshot_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = types.ModuleType("scpn_control_rs")

        def _raise(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("native snapshot failed")

        fake.runtime_admission_snapshot = _raise  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "scpn_control_rs", fake)
        assert _native_runtime_snapshot(RuntimeAdmissionRequest()) is None
