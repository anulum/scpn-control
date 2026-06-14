# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — PREEMPT_RT Runtime Admission

"""Runtime admission checks for deterministic native control campaigns.

The control loop can execute a fast native hot path, but that does not prove the
host is qualified for production real-time execution. This module separates
local developer runs from production-admissible runs by binding the observed
kernel, affinity, scheduler, governor, heartbeat, and memory-lock assumptions to
a schema-versioned report.
"""

from __future__ import annotations

import os
import platform
import time
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

if sys.platform != "win32":
    import resource as _resource
else:
    _resource = None  # type: ignore[assignment]

RUNTIME_ADMISSION_SCHEMA_VERSION = "scpn-control.runtime-admission.v1"
DEFAULT_MIN_MEMLOCK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RuntimeAdmissionRequest:
    """Operator-requested runtime contract for a native control campaign."""

    execution_backend: str = "auto"
    pacing_mode: str = "sleep"
    transport_backend: str = "std"
    formal_mode: str = "async_drop"
    tick_interval_s: float = 0.001
    core_snn: int = 1
    core_z3: int = 2
    core_net: int = 3
    core_hb: int = 4
    heartbeat_port: int = 0
    native_backend_available: bool = False
    require_preempt_rt: bool = False
    require_realtime_scheduler: bool = False
    require_performance_governor: bool = False
    require_heartbeat: bool = False
    min_memlock_bytes: int = DEFAULT_MIN_MEMLOCK_BYTES

    @property
    def requested_cores(self) -> tuple[int, int, int, int]:
        return (self.core_snn, self.core_z3, self.core_net, self.core_hb)

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_backend": self.execution_backend,
            "pacing_mode": self.pacing_mode,
            "transport_backend": self.transport_backend,
            "formal_mode": self.formal_mode,
            "tick_interval_s": self.tick_interval_s,
            "core_snn": self.core_snn,
            "core_z3": self.core_z3,
            "core_net": self.core_net,
            "core_hb": self.core_hb,
            "heartbeat_port": self.heartbeat_port,
            "native_backend_available": self.native_backend_available,
            "require_preempt_rt": self.require_preempt_rt,
            "require_realtime_scheduler": self.require_realtime_scheduler,
            "require_performance_governor": self.require_performance_governor,
            "require_heartbeat": self.require_heartbeat,
            "min_memlock_bytes": self.min_memlock_bytes,
        }


@dataclass(frozen=True, slots=True)
class RuntimeAdmissionProbe:
    """Observed host runtime properties used for admission."""

    generated_ns: int
    platform_system: str
    kernel_release: str
    is_linux: bool
    preempt_rt: bool
    realtime_sysfs: bool | None
    affinity: tuple[int, ...]
    available_parallelism: int
    scheduler_policy: str
    scheduler_priority: int | None
    governors: Mapping[int, str | None]
    memlock_soft_bytes: int | str
    memlock_hard_bytes: int | str
    native_snapshot: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_ns": self.generated_ns,
            "platform_system": self.platform_system,
            "kernel_release": self.kernel_release,
            "is_linux": self.is_linux,
            "preempt_rt": self.preempt_rt,
            "realtime_sysfs": self.realtime_sysfs,
            "affinity": list(self.affinity),
            "available_parallelism": self.available_parallelism,
            "scheduler_policy": self.scheduler_policy,
            "scheduler_priority": self.scheduler_priority,
            "governors": {str(core): value for core, value in self.governors.items()},
            "memlock_soft_bytes": self.memlock_soft_bytes,
            "memlock_hard_bytes": self.memlock_hard_bytes,
            "native_snapshot": dict(self.native_snapshot) if self.native_snapshot is not None else None,
        }


def normalise_runtime_admission_policy(value: str) -> str:
    """Return canonical admission policy: off, warn, or require."""

    policy = str(value).strip().lower().replace("-", "_")
    aliases = {
        "disabled": "off",
        "skip": "off",
        "none": "off",
        "permissive": "warn",
        "warning": "warn",
        "strict": "require",
        "required": "require",
        "fail_closed": "require",
    }
    policy = aliases.get(policy, policy)
    if policy not in {"off", "warn", "require"}:
        raise ValueError("runtime admission policy must be one of: off, warn, require")
    return policy


def collect_runtime_probe(request: RuntimeAdmissionRequest | None = None) -> RuntimeAdmissionProbe:
    """Collect current host runtime properties without mutating the host."""

    system = platform.system()
    kernel_release = platform.release()
    is_linux = system.lower() == "linux"
    realtime_sysfs = _read_realtime_sysfs() if is_linux else None
    preempt_rt = _detect_preempt_rt(kernel_release, realtime_sysfs)
    requested_cores = request.requested_cores if request is not None else ()
    affinity = _current_affinity()
    governors = _read_governors(requested_cores)
    scheduler_policy, scheduler_priority = _scheduler_policy()
    memlock_soft, memlock_hard = _memlock_limits()
    native_snapshot = _native_runtime_snapshot(request) if request is not None else None

    return RuntimeAdmissionProbe(
        generated_ns=time.time_ns(),
        platform_system=system,
        kernel_release=kernel_release,
        is_linux=is_linux,
        preempt_rt=preempt_rt,
        realtime_sysfs=realtime_sysfs,
        affinity=affinity,
        available_parallelism=os.cpu_count() or 1,
        scheduler_policy=scheduler_policy,
        scheduler_priority=scheduler_priority,
        governors=governors,
        memlock_soft_bytes=memlock_soft,
        memlock_hard_bytes=memlock_hard,
        native_snapshot=native_snapshot,
    )


def evaluate_runtime_admission(
    request: RuntimeAdmissionRequest,
    probe: RuntimeAdmissionProbe,
) -> dict[str, object]:
    """Evaluate a runtime admission request against an observed probe."""

    errors: list[str] = []
    warnings: list[str] = []
    requested = request.requested_cores
    requested_set = set(requested)

    if not probe.is_linux:
        errors.append("runtime admission requires Linux for native control campaigns")

    if len(requested_set) != len(requested):
        errors.append("execution cores must be distinct for SNN, Z3, network, and heartbeat threads")

    if any(core < 0 for core in requested):
        errors.append("execution cores must be non-negative")

    missing_affinity = sorted(core for core in requested_set if core not in set(probe.affinity))
    if missing_affinity:
        errors.append(f"requested cores outside current process affinity: {missing_affinity}")

    if request.pacing_mode == "spin" and request.tick_interval_s > 0.01:
        errors.append("spin pacing requires tick_interval_s <= 0.01")

    if request.pacing_mode == "spin" and request.execution_backend == "python":
        errors.append("spin pacing is only admissible with the native execution backend")

    if request.pacing_mode == "spin" and not request.native_backend_available:
        errors.append("spin pacing requires the native PyO3 controller bridge")

    if request.transport_backend.replace("_", "-") == "io-uring" and not probe.is_linux:
        errors.append("io-uring transport is Linux-only")

    if request.require_preempt_rt and not probe.preempt_rt:
        errors.append("PREEMPT_RT kernel evidence is required but was not detected")
    elif request.pacing_mode == "spin" and not probe.preempt_rt:
        warnings.append("spin pacing is local regression evidence only without PREEMPT_RT or realtime sysfs evidence")

    rt_scheduler = probe.scheduler_policy in {"SCHED_FIFO", "SCHED_RR"}
    if request.require_realtime_scheduler and not rt_scheduler:
        errors.append(f"real-time scheduler policy required, observed {probe.scheduler_policy}")
    elif request.pacing_mode == "spin" and not rt_scheduler:
        warnings.append(f"spin pacing is not protected by SCHED_FIFO/SCHED_RR, observed {probe.scheduler_policy}")

    non_performance = sorted(
        core for core, governor in probe.governors.items() if governor not in {"performance", None}
    )
    unknown_governor = sorted(core for core, governor in probe.governors.items() if governor is None)
    if request.require_performance_governor and non_performance:
        errors.append(f"performance CPU governor required on cores {non_performance}")
    elif non_performance:
        warnings.append(f"non-performance CPU governor observed on cores {non_performance}")
    if unknown_governor:
        warnings.append(f"CPU governor unavailable for cores {unknown_governor}")

    soft_memlock = _limit_to_int(probe.memlock_soft_bytes)
    if soft_memlock is not None and soft_memlock < request.min_memlock_bytes:
        message = (
            f"RLIMIT_MEMLOCK soft limit below runtime admission floor: {soft_memlock} < {request.min_memlock_bytes}"
        )
        if request.require_realtime_scheduler:
            errors.append(message)
        else:
            warnings.append(message)

    if request.require_heartbeat and request.heartbeat_port <= 0:
        errors.append("heartbeat dead-man switch port is required for production admission")
    elif request.heartbeat_port <= 0:
        warnings.append("heartbeat dead-man switch is disabled")

    status = "fail" if errors else "pass"
    production_claim_allowed = bool(
        status == "pass"
        and request.require_preempt_rt
        and request.require_realtime_scheduler
        and request.require_performance_governor
        and (not request.require_heartbeat or request.heartbeat_port > 0)
    )

    return {
        "schema_version": RUNTIME_ADMISSION_SCHEMA_VERSION,
        "status": status,
        "production_claim_allowed": production_claim_allowed,
        "errors": errors,
        "warnings": warnings,
        "request": request.as_dict(),
        "probe": probe.as_dict(),
    }


def collect_runtime_admission(request: RuntimeAdmissionRequest) -> dict[str, object]:
    """Collect and evaluate current host runtime admission evidence."""

    return evaluate_runtime_admission(request, collect_runtime_probe(request))


def skipped_runtime_admission(policy: str = "off") -> dict[str, object]:
    """Return a schema-valid skipped admission record."""

    return {
        "schema_version": RUNTIME_ADMISSION_SCHEMA_VERSION,
        "status": "skipped",
        "production_claim_allowed": False,
        "policy": normalise_runtime_admission_policy(policy),
        "errors": [],
        "warnings": ["runtime admission was explicitly disabled"],
    }


def _detect_preempt_rt(kernel_release: str, realtime_sysfs: bool | None) -> bool:
    release = kernel_release.lower()
    return bool(realtime_sysfs) or "preempt_rt" in release or "-rt" in release or "realtime" in release


def _read_realtime_sysfs() -> bool | None:
    path = Path("/sys/kernel/realtime")
    try:
        return path.read_text(encoding="utf-8").strip() == "1"
    except OSError:
        return None


def _current_affinity() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        try:
            return tuple(sorted(os.sched_getaffinity(0)))
        except OSError:
            pass
    count = os.cpu_count() or 1
    return tuple(range(count))


def _read_governors(cores: Sequence[int]) -> dict[int, str | None]:
    governors: dict[int, str | None] = {}
    for core in cores:
        path = Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_governor")
        try:
            governors[int(core)] = path.read_text(encoding="utf-8").strip()
        except OSError:
            governors[int(core)] = None
    return governors


def _scheduler_policy() -> tuple[str, int | None]:
    if not hasattr(os, "sched_getscheduler"):
        return "unknown", None
    try:
        policy_id = os.sched_getscheduler(0)
    except OSError:
        return "unknown", None

    labels = {
        getattr(os, "SCHED_OTHER", -1): "SCHED_OTHER",
        getattr(os, "SCHED_FIFO", -2): "SCHED_FIFO",
        getattr(os, "SCHED_RR", -3): "SCHED_RR",
        getattr(os, "SCHED_BATCH", -4): "SCHED_BATCH",
        getattr(os, "SCHED_IDLE", -5): "SCHED_IDLE",
    }
    priority: int | None = None
    try:
        priority = int(os.sched_getparam(0).sched_priority)
    except OSError:
        priority = None
    return labels.get(policy_id, f"policy:{policy_id}"), priority


def _memlock_limits() -> tuple[int | str, int | str]:
    if _resource is None:
        return "unknown", "unknown"
    try:
        soft, hard = _resource.getrlimit(_resource.RLIMIT_MEMLOCK)
    except (OSError, ValueError):
        return "unknown", "unknown"
    return _format_limit(soft), _format_limit(hard)


def _format_limit(value: int) -> int | str:
    if _resource is not None and value == _resource.RLIM_INFINITY:
        return "unlimited"
    return int(value)


def _limit_to_int(value: int | str) -> int | None:
    if isinstance(value, int):
        return value
    if value == "unlimited":
        return None
    return None


def _native_runtime_snapshot(request: RuntimeAdmissionRequest) -> Mapping[str, object] | None:
    try:
        from scpn_control_rs import runtime_admission_snapshot
    except Exception:
        return None

    try:
        snapshot = runtime_admission_snapshot(
            request.core_snn,
            request.core_z3,
            request.core_net,
            request.core_hb,
            request.tick_interval_s,
            request.pacing_mode,
        )
    except Exception:
        return None
    return snapshot if isinstance(snapshot, Mapping) else None
