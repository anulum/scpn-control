# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-shot campaign benchmark harness.
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.control.capacitor_bank_state import CapacitorBankSpec
from scpn_control.control.multi_shot_campaign import CampaignShotPlan, CampaignShotSample, MultiShotCampaignOrchestrator
from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    CapacitorBankTelemetry,
    PulsedPlasmaTelemetry,
    PulsedScenarioSpec,
)


def _scheduler_spec() -> PulsedScenarioSpec:
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


def _bank_spec() -> CapacitorBankSpec:
    return CapacitorBankSpec(
        capacitance_F=100e-6,
        inductance_H=100e-6,
        series_resistance_ohm=0.5,
        voltage_max_V=10_000.0,
        recharge_power_kW=20.0,
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


def _bank(voltage_V: float, energy_J: float) -> CapacitorBankTelemetry:
    return CapacitorBankTelemetry(voltage_V=voltage_V, voltage_max_V=10_000.0, energy_J=energy_J)


def _admission_digest(label: str) -> str:
    return hashlib.sha256(f"scpn-control.multi-shot-campaign.benchmark:{label}".encode()).hexdigest()


def _complete_shot(shot_id: str) -> CampaignShotPlan:
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
    samples = tuple(
        CampaignShotSample(t_s, _plasma(plasma), _bank(voltage, energy)) for t_s, plasma, voltage, energy in rows
    )
    return CampaignShotPlan(
        shot_id=shot_id,
        samples=samples,
        initial_bank_voltage_V=5000.0,
        pulsed_mpc_admission_digest=_admission_digest(shot_id),
    )


def _orchestrator() -> MultiShotCampaignOrchestrator:
    return MultiShotCampaignOrchestrator("campaign-a", _scheduler_spec(), _bank_spec())


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


def _stats(samples_ns: list[int]) -> dict[str, float | int]:
    ordered = sorted(samples_ns)
    n = len(ordered)
    return {
        "samples": n,
        "mean_us": statistics.fmean(ordered) / 1_000.0,
        "median_us": statistics.median(ordered) / 1_000.0,
        "p95_us": ordered[min(n - 1, int(n * 0.95))] / 1_000.0,
        "p99_us": ordered[min(n - 1, int(n * 0.99))] / 1_000.0,
        "min_us": ordered[0] / 1_000.0,
        "max_us": ordered[-1] / 1_000.0,
    }


def _measure(fn: Callable[[], Any], *, steps: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    samples: list[int] = []
    last: Any = None
    for _ in range(steps):
        start = time.perf_counter_ns()
        last = fn()
        samples.append(time.perf_counter_ns() - start)
    return {
        "stats": _stats(samples),
        "last_passed_count": last["passed_count"],
        "last_pulsed_mpc_admission_digest_count": last["pulsed_mpc_admission_digest_count"],
    }


def _pyo3_measurement(shots: list[CampaignShotPlan], *, steps: int, warmup: int) -> tuple[dict[str, Any] | None, str]:
    try:
        scpn_control_rs = importlib.import_module("scpn_control_rs")
    except ImportError:
        return None, "optional PyO3 extension is not installed"
    if not hasattr(scpn_control_rs, "PyMultiShotCampaignOrchestrator"):
        return None, "optional PyO3 extension is installed but was not rebuilt with multi-shot bindings"

    shot_ids = [shot.shot_id for shot in shots]
    sample_shot_index: list[int] = []
    sample_t_s: list[float] = []
    plasma_rows: list[list[float]] = []
    bank_rows: list[list[float]] = []
    for shot_index, shot in enumerate(shots):
        for sample in shot.samples:
            if sample.bank is None:
                return None, "PyO3 benchmark requires explicit bank telemetry for every sample"
            sample_shot_index.append(shot_index)
            sample_t_s.append(sample.t_s)
            plasma_rows.append(
                [
                    sample.plasma.coil_current_A,
                    sample.plasma.temperature_eV,
                    sample.plasma.phase_lock_error_rad,
                    sample.plasma.reference_error_m,
                    sample.plasma.fusion_power_W,
                    sample.plasma.radial_velocity_m_s,
                ]
            )
            bank_rows.append([sample.bank.voltage_V, sample.bank.voltage_max_V, sample.bank.energy_J])

    bridge = scpn_control_rs.PyMultiShotCampaignOrchestrator(
        "campaign-a",
        scpn_control_rs.PyPulsedScenarioSpec(
            100.0,
            2.0e6,
            0.01,
            0.002,
            1.0e3,
            2.0e6,
            1.0e3,
            40.0,
            0.95,
            20.0,
            1.0e3,
            0.0,
        ),
        scpn_control_rs.PyCapacitorBankSpec(100e-6, 100e-6, 0.5, 10_000.0, 20.0),
        True,
    )
    sample_shot_index_array = np.asarray(sample_shot_index, dtype=np.uintp)
    sample_t_s_array = np.asarray(sample_t_s, dtype=np.float64)
    plasma_array = np.asarray(plasma_rows, dtype=np.float64)
    bank_array = np.asarray(bank_rows, dtype=np.float64)
    initial_voltage_array = np.asarray([shot.initial_bank_voltage_V for shot in shots], dtype=np.float64)
    admission_digests = [shot.pulsed_mpc_admission_digest for shot in shots]
    return (
        _measure(
            lambda: bridge.run_table(
                shot_ids,
                sample_shot_index_array,
                sample_t_s_array,
                plasma_array,
                bank_array,
                initial_voltage_array,
                admission_digests,
            ),
            steps=steps,
            warmup=warmup,
        ),
        "ok",
    )


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = payload["result"]["stats"]
    rows = [
        (
            "Python",
            stats["samples"],
            stats["mean_us"],
            stats["median_us"],
            stats["p95_us"],
            stats["p99_us"],
        )
    ]
    if payload["pyo3_result"] is not None:
        pyo3_stats = payload["pyo3_result"]["stats"]
        rows.append(
            (
                "PyO3",
                pyo3_stats["samples"],
                pyo3_stats["mean_us"],
                pyo3_stats["median_us"],
                pyo3_stats["p95_us"],
                pyo3_stats["p99_us"],
            )
        )
    table_rows = "\n".join(
        f"| {surface} | {samples} | {mean_us:.6f} | {median_us:.6f} | {p95_us:.6f} | {p99_us:.6f} |"
        for surface, samples, mean_us, median_us, p95_us, p99_us in rows
    )
    body = "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- SCPN Control — Multi-shot campaign benchmark report. -->",
            "",
            "# Multi-Shot Campaign Benchmark",
            "",
            f"- Generated UTC: `{payload['generated_utc']}`",
            f"- Evidence class: `{payload['evidence_class']}`",
            f"- Production claim allowed: `{payload['production_claim_allowed']}`",
            f"- CPU affinity: `{payload['context']['cpu_affinity']}`",
            f"- Load average start: `{payload['context']['loadavg_start']}`",
            f"- Load average end: `{payload['context']['loadavg_end']}`",
            "",
            f"- PyO3 status: `{payload['pyo3_status']}`",
            "",
            "| Surface | Samples | Mean us | Median us | p95 us | p99 us |",
            "|---|---:|---:|---:|---:|---:|",
            table_rows,
            "",
            "This is local regression evidence for the CONTROL multi-shot campaign adapter.",
            "Each measured campaign carries two digest-bound pulsed-MPC admission references.",
            "It is not target-hardware timing evidence and does not admit facility PCS claims.",
            "",
            f"Payload SHA-256: `{payload['payload_sha256']}`",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    orchestrator = _orchestrator()
    shots = [_complete_shot("shot-001"), _complete_shot("shot-002")]
    load_start = _loadavg()
    pyo3_result, pyo3_status = _pyo3_measurement(shots, steps=args.steps, warmup=args.warmup)
    payload: dict[str, Any] = {
        "schema_version": "scpn-control.multi-shot-campaign-benchmark.v1.1",
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": " ".join(sys.argv),
        "evidence_class": args.evidence_class,
        "production_claim_allowed": False,
        "steps": args.steps,
        "warmup": args.warmup,
        "context": {
            "cwd": str(Path.cwd()),
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_affinity": _affinity(),
            "loadavg_start": load_start,
            "loadavg_end": None,
        },
        "result": _measure(lambda: orchestrator.run(shots), steps=args.steps, warmup=args.warmup),
        "pyo3_result": pyo3_result,
        "pyo3_status": pyo3_status,
        "payload_sha256": "",
    }
    payload["context"]["loadavg_end"] = _loadavg()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the multi-shot campaign orchestrator.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--evidence-class", default="local_regression")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    payload = run(args)
    if args.json_out is not None:
        _write_json(args.json_out, payload)
    if args.md_out is not None:
        _write_markdown(args.md_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
