#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Pulsed-scenario scheduler side-by-side benchmark.
"""Benchmark the eight-step pulsed scheduler in Python and native Rust."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    CapacitorBankTelemetry,
    PulsedPlasmaTelemetry,
    PulsedScenarioScheduler,
    PulsedScenarioSpec,
)

ROOT = Path(__file__).resolve().parents[1]
RUST_MANIFEST = ROOT / "scpn-control-rs" / "Cargo.toml"
EXPECTED_STATES = [
    "ramp_up",
    "flat_top",
    "burn",
    "expansion",
    "dump",
    "recharge",
    "cool_down",
    "idle",
]


def _spec() -> PulsedScenarioSpec:
    return PulsedScenarioSpec(
        min_precharge_energy_J=100.0,
        ramp_current_A=2.0e6,
        phase_tolerance_rad=0.01,
        spatial_tolerance_m=0.002,
        burn_temperature_eV=1.0e3,
        min_fusion_power_W=2.0e6,
        expansion_velocity_m_s=1.0e3,
        dump_energy_floor_J=40.0,
        recharge_voltage_fraction=0.95,
        cooldown_temperature_eV=20.0,
        cooldown_current_A=1.0e3,
        min_burn_duration_s=0.0,
    )


def _plasma(values: tuple[float, float, float, float, float, float]) -> PulsedPlasmaTelemetry:
    return PulsedPlasmaTelemetry(
        coil_current_A=values[0],
        temperature_eV=values[1],
        phase_lock_error_rad=values[2],
        reference_error_m=values[3],
        fusion_power_W=values[4],
        radial_velocity_m_s=values[5],
    )


def _bank(voltage: float, energy: float) -> CapacitorBankTelemetry:
    return CapacitorBankTelemetry(voltage_V=voltage, voltage_max_V=10_000.0, energy_J=energy)


def _campaign() -> tuple[tuple[float, PulsedPlasmaTelemetry, CapacitorBankTelemetry], ...]:
    rows = (
        (0.0, (0.0, 10.0, 0.02, 0.01, 0.0, 0.0), 9800.0, 200.0),
        (1.0e-3, (2.5e6, 10.0, 0.02, 0.01, 0.0, 0.0), 9800.0, 200.0),
        (2.0e-3, (2.5e6, 1200.0, 0.004, 0.001, 0.0, 0.0), 9800.0, 200.0),
        (3.0e-3, (2.5e6, 1500.0, 0.004, 0.001, 3.0e6, 0.0), 9800.0, 200.0),
        (4.0e-3, (0.0, 200.0, 0.02, 0.01, 0.0, 1500.0), 9800.0, 200.0),
        (5.0e-3, (0.0, 120.0, 0.02, 0.01, 0.0, 0.0), 2000.0, 20.0),
        (6.0e-3, (0.0, 40.0, 0.02, 0.01, 0.0, 0.0), 9700.0, 180.0),
        (7.0e-3, (100.0, 15.0, 0.02, 0.01, 0.0, 0.0), 9800.0, 200.0),
    )
    return tuple((t_s, _plasma(plasma), _bank(voltage, energy)) for t_s, plasma, voltage, energy in rows)


def _run_python_campaign() -> list[str]:
    scheduler = PulsedScenarioScheduler(_spec())
    states = [scheduler.step(t_s, plasma, bank).state.value for t_s, plasma, bank in _campaign()]
    if states != EXPECTED_STATES or scheduler.state.value != "idle" or len(scheduler.audit_log) != 8:
        raise RuntimeError("Python scheduler did not complete the canonical eight-state campaign")
    return states


def _stats(samples_ns: list[int]) -> dict[str, float | int]:
    ordered = sorted(samples_ns)
    count = len(ordered)
    return {
        "samples": count,
        "mean_us": statistics.fmean(ordered) / 1_000.0,
        "median_us": statistics.median(ordered) / 1_000.0,
        "p95_us": ordered[min(count - 1, int(count * 0.95))] / 1_000.0,
        "p99_us": ordered[min(count - 1, int(count * 0.99))] / 1_000.0,
        "min_us": ordered[0] / 1_000.0,
        "max_us": ordered[-1] / 1_000.0,
    }


def _measure(fn: Callable[[], list[str]], *, iterations: int, warmup: int) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    samples: list[int] = []
    states: list[str] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        states = fn()
        samples.append(time.perf_counter_ns() - started)
    return {"stats": _stats(samples), "last_states": states}


def _rust_command(iterations: int, warmup: int) -> list[str]:
    return [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--manifest-path",
        str(RUST_MANIFEST),
        "-p",
        "control-control",
        "--example",
        "bench_pulsed_scenario_scheduler",
        "--",
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
    ]


def _run_rust(iterations: int, warmup: int) -> dict[str, object]:
    command = _rust_command(iterations, warmup)
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True, timeout=300)
    payload: object = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Rust scheduler benchmark must emit a JSON object")
    if payload.get("last_states") != EXPECTED_STATES:
        raise ValueError("Rust scheduler benchmark did not preserve the eight-state campaign")
    return payload


def _loadavg() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except OSError:
        return None


def _affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(*, iterations: int, warmup: int, include_rust: bool = True) -> dict[str, Any]:
    """Run the bounded side-by-side campaign and return sealed evidence."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    load_start = _loadavg()
    python_result = _measure(_run_python_campaign, iterations=iterations, warmup=warmup)
    rust_result = _run_rust(iterations, warmup) if include_rust else None
    parity_passed = rust_result is not None and rust_result["last_states"] == python_result["last_states"]
    payload: dict[str, Any] = {
        "schema_version": "scpn-control.pulsed-scenario-scheduler-benchmark.v1",
        "generated_utc": _utc_now(),
        "command": " ".join(sys.argv),
        "evidence_class": "local_proxy",
        "orientation_only": True,
        "scientific_admission": False,
        "production_admission": False,
        "public_claim_allowed": False,
        "claim_boundary": (
            "Loaded-workstation orientation only; not PREEMPT_RT, HIL, target-hardware, "
            "facility, plant-cycle, or production timing evidence."
        ),
        "iterations": iterations,
        "warmup": warmup,
        "campaign_steps": 8,
        "python_result": python_result,
        "rust_result": rust_result,
        "parity_passed": parity_passed,
        "rust_command": _rust_command(iterations, warmup) if include_rust else None,
        "context": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": _affinity(),
            "loadavg_start": load_start,
            "loadavg_end": _loadavg(),
            "isolation": "loaded_workstation_process_affinity_only",
        },
        "payload_sha256": "",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    python_stats = payload["python_result"]["stats"]
    rust_result = payload["rust_result"]
    rows = [
        (
            "Python",
            python_stats["samples"],
            python_stats["mean_us"],
            python_stats["median_us"],
            python_stats["p95_us"],
            python_stats["p99_us"],
        )
    ]
    if isinstance(rust_result, dict):
        rust_stats = rust_result["stats"]
        rows.append(
            (
                "Rust native",
                rust_stats["samples"],
                rust_stats["mean_us"],
                rust_stats["median_us"],
                rust_stats["p95_us"],
                rust_stats["p99_us"],
            )
        )
    table = "\n".join(
        f"| {name} | {samples} | {mean:.6f} | {median:.6f} | {p95:.6f} | {p99:.6f} |"
        for name, samples, mean, median, p95, p99 in rows
    )
    body = "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- SCPN Control — Pulsed-scenario scheduler benchmark report. -->",
            "",
            "# Pulsed-Scenario Scheduler Benchmark",
            "",
            f"- Generated UTC: `{payload['generated_utc']}`",
            f"- Evidence class: `{payload['evidence_class']}`",
            f"- Orientation only: `{payload['orientation_only']}`",
            f"- Public claim allowed: `{payload['public_claim_allowed']}`",
            f"- Parity passed: `{payload['parity_passed']}`",
            f"- Claim boundary: {payload['claim_boundary']}",
            "",
            "| Surface | Samples | Mean us | Median us | p95 us | p99 us |",
            "|---|---:|---:|---:|---:|---:|",
            table,
            "",
            f"Payload SHA-256: `{payload['payload_sha256']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main() -> None:
    """CLI entry point for the scheduler benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()
    payload = run(iterations=args.iterations, warmup=args.warmup, include_rust=not args.skip_rust)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        _write_markdown(args.md_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
