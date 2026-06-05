# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Pulsed-shot MPC adapter benchmark harness.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.control.capacitor_bank_state import CapacitorBank, CapacitorBankSpec, PulseSpec
from scpn_control.control.fusion_sota_mpc import ModelPredictiveController, NeuralSurrogate, PulsedShotMPCAdapter
from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    PulsedScenarioScheduler,
    PulsedScenarioSpec,
    PulsedScenarioState,
)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _sha256_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    return hashlib.sha256(encoded).hexdigest()


def _affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _loadavg() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except OSError:
        return None


def _stats(samples_ns: list[int]) -> dict[str, float | int]:
    ordered = sorted(samples_ns)
    n = len(ordered)
    if n == 0:
        raise ValueError("at least one sample is required")
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
    timings: list[int] = []
    last_value: Any = None
    for _ in range(steps):
        start = time.perf_counter_ns()
        last_value = fn()
        timings.append(time.perf_counter_ns() - start)
    return {"stats": _stats(timings), "last_value": _normalise_result(last_value)}


def _normalise_result(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalise_result(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _normalise_result(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scheduler(state: PulsedScenarioState) -> PulsedScenarioScheduler:
    scheduler = PulsedScenarioScheduler(
        PulsedScenarioSpec(
            min_precharge_energy_J=5.0,
            ramp_current_A=10.0,
            phase_tolerance_rad=0.1,
            spatial_tolerance_m=0.01,
            burn_temperature_eV=100.0,
            min_fusion_power_W=50.0,
            expansion_velocity_m_s=5.0,
            dump_energy_floor_J=0.5,
            recharge_voltage_fraction=0.8,
            cooldown_temperature_eV=10.0,
            cooldown_current_A=1.0,
            min_burn_duration_s=0.0,
        )
    )
    scheduler.state = state
    return scheduler


def _bank(*, initial_voltage_V: float, resistance_ohm: float) -> CapacitorBank:
    return CapacitorBank(
        CapacitorBankSpec(
            capacitance_F=1.0,
            inductance_H=1.0,
            series_resistance_ohm=resistance_ohm,
            voltage_max_V=20.0,
            recharge_power_kW=1.0,
        ),
        initial_voltage_V=initial_voltage_V,
    )


def _mpc() -> ModelPredictiveController:
    surrogate = NeuralSurrogate(n_coils=2, n_state=2, verbose=False)
    surrogate.B[:, :] = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64)
    return ModelPredictiveController(surrogate, np.array([6.0, 0.0], dtype=np.float64))


def _python_cases() -> dict[str, Callable[[], Any]]:
    state = np.array([5.0, 1.0], dtype=np.float64)
    ref = np.array([6.0, 0.0], dtype=np.float64)
    non_burn = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.FLAT_TOP),
        _bank(initial_voltage_V=10.0, resistance_ohm=0.05),
        burn_action_mask=np.array([True, False]),
    )
    feasible = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.BURN),
        _bank(initial_voltage_V=10.0, resistance_ohm=0.01),
    )
    infeasible = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.BURN),
        _bank(initial_voltage_V=0.0, resistance_ohm=1.0),
        pulse_duration_s=1.0,
    )

    return {
        "python_non_burn_mask": lambda: (
            non_burn.step(state, ref),
            non_burn.explain_last_decision(),
        ),
        "python_burn_feasible": lambda: (
            feasible.step(
                state,
                ref,
                pulse=PulseSpec(peak_current_A=0.5, duration_s=0.001),
            ),
            feasible.explain_last_decision(),
        ),
        "python_burn_infeasible_safe": lambda: (
            infeasible.step(state, ref),
            infeasible.explain_last_decision(),
        ),
    }


def _rust_cases() -> tuple[dict[str, Callable[[], Any]], str | None]:
    try:
        import scpn_control_rs  # type: ignore[import-not-found]
    except ImportError as exc:
        return {}, f"optional PyO3 extension unavailable: {exc}"

    rust_mpc = scpn_control_rs.PyMpcController(
        np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64),
        np.array([6.0, 0.0], dtype=np.float64),
    )
    if not hasattr(rust_mpc, "plan_pulsed"):
        return {}, "optional PyO3 extension is installed but was not rebuilt with plan_pulsed"

    state = np.array([5.0, 1.0], dtype=np.float64)
    safe = np.zeros(2, dtype=np.float64)
    _, probe_decision = rust_mpc.plan_pulsed(
        state,
        "flat_top",
        True,
        np.array([True, False], dtype=bool),
        safe,
        12.0,
    )
    if "evidence_schema_version" not in probe_decision:
        return {}, "optional PyO3 extension is installed but was not rebuilt with pulsed decision evidence"

    return {
        "pyo3_non_burn_mask": lambda: rust_mpc.plan_pulsed(
            state,
            "flat_top",
            True,
            np.array([True, False], dtype=bool),
            safe,
            12.0,
        ),
        "pyo3_burn_infeasible_safe": lambda: rust_mpc.plan_pulsed(
            state,
            "burn",
            False,
            np.array([True, True], dtype=bool),
            safe,
            -0.5,
        ),
    }, None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "| Case | Samples | Mean us | Median us | p95 us | p99 us |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["results"].items():
        stats = result["stats"]
        rows.append(
            f"| `{name}` | {stats['samples']} | {stats['mean_us']:.6f} | "
            f"{stats['median_us']:.6f} | {stats['p95_us']:.6f} | {stats['p99_us']:.6f} |"
        )
    body = "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- SCPN Control — Pulsed-shot MPC adapter benchmark report. -->",
            "",
            "# Pulsed MPC Adapter Benchmark",
            "",
            f"- Generated UTC: `{payload['generated_utc']}`",
            f"- Evidence class: `{payload['evidence_class']}`",
            f"- Production claim allowed: `{payload['production_claim_allowed']}`",
            f"- Steps: `{payload['steps']}`",
            f"- Warmup: `{payload['warmup']}`",
            f"- CPU affinity: `{payload['context']['cpu_affinity']}`",
            f"- Load average start: `{payload['context']['loadavg_start']}`",
            f"- Load average end: `{payload['context']['loadavg_end']}`",
            f"- PyO3 status: `{payload['pyo3_status']}`",
            "",
            "## Results",
            "",
            *rows,
            "",
            "## Claim boundary",
            "",
            "This is local regression evidence for the pulsed MPC admission adapter.",
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

    load_start = _loadavg()
    results: dict[str, Any] = {}
    for name, fn in _python_cases().items():
        results[name] = _measure(fn, steps=args.steps, warmup=args.warmup)
    rust_cases, pyo3_status = _rust_cases()
    if rust_cases:
        pyo3_status = "available"
        for name, fn in rust_cases.items():
            results[name] = _measure(fn, steps=args.steps, warmup=args.warmup)

    payload: dict[str, Any] = {
        "schema_version": "scpn-control.pulsed-mpc-adapter-benchmark.v1.1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": " ".join(sys.argv),
        "evidence_class": args.evidence_class,
        "production_claim_allowed": False,
        "steps": args.steps,
        "warmup": args.warmup,
        "pyo3_status": pyo3_status or "available",
        "context": {
            "cwd": str(Path.cwd()),
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_affinity": _affinity(),
            "loadavg_start": load_start,
            "loadavg_end": _loadavg(),
        },
        "results": results,
    }
    payload["payload_sha256"] = _sha256_payload(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the pulsed-shot MPC adapter.")
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
