# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capacitor-bank energy ledger benchmark harness.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scpn_control.control.capacitor_bank_state import CapacitorBank, CapacitorBankSpec, PulseSpec


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


def _sha256_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _spec() -> CapacitorBankSpec:
    return CapacitorBankSpec(
        capacitance_F=100e-6,
        inductance_H=100e-6,
        series_resistance_ohm=0.5,
        voltage_max_V=10_000.0,
        recharge_power_kW=20.0,
    )


_SPEC_ARGS = (100e-6, 100e-6, 0.5, 10_000.0, 20.0)
_INITIAL_VOLTAGE_V = 5_000.0
_INITIAL_CURRENT_A = 12.0
_PEAK_CURRENT_A = 500.0


def _run_python(*, discharge_steps: int, dt_s: float) -> dict[str, float | bool | str]:
    bank = CapacitorBank(_spec(), initial_voltage_V=_INITIAL_VOLTAGE_V, initial_current_A=_INITIAL_CURRENT_A)
    report = bank.discharge(
        PulseSpec(peak_current_A=_PEAK_CURRENT_A, duration_s=discharge_steps * dt_s, waveform="half_sine"),
        dt=dt_s,
        n_steps=discharge_steps,
    )
    return {
        "energy_initial_J": report.energy_initial_J,
        "energy_remaining_J": report.energy_remaining_J,
        "energy_delivered_J": report.energy_delivered_J,
        "resistive_loss_J": report.resistive_loss_J,
        "load_energy_J": report.load_energy_J,
        "energy_balance_residual_J": report.energy_balance_residual_J,
        "energy_balance_relative_error": report.energy_balance_relative_error,
        "energy_balance_passed": report.energy_balance_passed,
        "rlc_regime": report.rlc_regime.value,
    }


def _load_rust_backend() -> Any:
    try:
        import scpn_control_rs as rust
    except ImportError:
        return None
    return rust if hasattr(rust, "PyCapacitorBankModel") else None


def _run_rust(rust: Any, *, discharge_steps: int, dt_s: float) -> dict[str, float | bool | str]:
    bank = rust.PyCapacitorBankModel(*_SPEC_ARGS, _INITIAL_VOLTAGE_V, _INITIAL_CURRENT_A)
    report = bank.discharge(_PEAK_CURRENT_A, discharge_steps * dt_s, "half_sine", dt_s, discharge_steps)
    return {
        "energy_initial_J": report["energy_initial_J"],
        "energy_remaining_J": report["energy_remaining_J"],
        "energy_delivered_J": report["energy_delivered_J"],
        "resistive_loss_J": report["resistive_loss_J"],
        "load_energy_J": report["load_energy_J"],
        "energy_balance_residual_J": report["energy_balance_residual_J"],
        "energy_balance_relative_error": report["energy_balance_relative_error"],
        "energy_balance_passed": report["energy_balance_passed"],
        "rlc_regime": report["rlc_regime"],
    }


def _measure_backend(run_once: Any, *, steps: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        run_once()
    samples: list[int] = []
    last: dict[str, float | bool | str] | None = None
    for _ in range(steps):
        start = time.perf_counter_ns()
        last = run_once()
        samples.append(time.perf_counter_ns() - start)
    return {"stats": _stats(samples), "last_report": last}


def _cross_language_parity(python_report: dict[str, Any], rust_report: dict[str, Any]) -> dict[str, float]:
    keys = ("energy_delivered_J", "resistive_loss_J", "load_energy_J")
    worst = 0.0
    for key in keys:
        py_value = float(python_report[key])
        rs_value = float(rust_report[key])
        scale = max(abs(py_value), abs(rs_value), 1.0)
        worst = max(worst, abs(py_value - rs_value) / scale)
    return {"max_relative_difference": worst}


def _measure(*, steps: int, warmup: int, discharge_steps: int, dt_s: float) -> dict[str, Any]:
    python_result = _measure_backend(
        lambda: _run_python(discharge_steps=discharge_steps, dt_s=dt_s), steps=steps, warmup=warmup
    )
    languages: dict[str, Any] = {"python": python_result}

    rust = _load_rust_backend()
    if rust is not None:
        rust_result = _measure_backend(
            lambda: _run_rust(rust, discharge_steps=discharge_steps, dt_s=dt_s), steps=steps, warmup=warmup
        )
        languages["rust"] = rust_result
        py_mean = python_result["stats"]["mean_us"]
        rs_mean = rust_result["stats"]["mean_us"]
        languages["rust_speedup_vs_python"] = py_mean / rs_mean if rs_mean > 0.0 else None
        languages["cross_language_parity"] = _cross_language_parity(
            python_result["last_report"], rust_result["last_report"]
        )
    else:
        languages["rust"] = None
        languages["rust_speedup_vs_python"] = None
        languages["cross_language_parity"] = None

    # Backwards-compatible top-level view of the Python reference path.
    return {
        "stats": python_result["stats"],
        "last_report": python_result["last_report"],
        "languages": languages,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _language_rows(languages: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for name in ("python", "rust"):
        backend = languages.get(name)
        if backend is None:
            rows.append(f"| {name} | not available | — | — | — | — |")
            continue
        stats = backend["stats"]
        rows.append(
            f"| {name} | {stats['samples']} | {stats['mean_us']:.6f} | {stats['median_us']:.6f} | "
            f"{stats['p95_us']:.6f} | {stats['p99_us']:.6f} |"
        )
    return rows


def _comparison_lines(languages: dict[str, Any]) -> list[str]:
    speedup = languages.get("rust_speedup_vs_python")
    parity = languages.get("cross_language_parity")
    if speedup is None or parity is None:
        return ["- Rust path: not available in this run.", ""]
    return [
        f"- Rust speedup vs Python (mean): `{speedup:.3f}x`",
        f"- Cross-language ledger parity (max relative difference): `{parity['max_relative_difference']:.3e}`",
        "",
    ]


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = payload["result"]["stats"]
    report = payload["result"]["last_report"]
    body = "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- SCPN Control — Capacitor-bank energy ledger benchmark report. -->",
            "",
            "# Capacitor-Bank Energy Ledger Benchmark",
            "",
            f"- Generated UTC: `{payload['generated_utc']}`",
            f"- Evidence class: `{payload['evidence_class']}`",
            f"- Production claim allowed: `{payload['production_claim_allowed']}`",
            f"- CPU affinity: `{payload['context']['cpu_affinity']}`",
            f"- Load average start: `{payload['context']['loadavg_start']}`",
            f"- Load average end: `{payload['context']['loadavg_end']}`",
            f"- Discharge steps per sample: `{payload['settings']['discharge_steps']}`",
            f"- Step size s: `{payload['settings']['dt_s']}`",
            "",
            "## Per-language discharge timing",
            "",
            "| Language | Samples | Mean us | Median us | p95 us | p99 us |",
            "|---|---:|---:|---:|---:|---:|",
            *_language_rows(payload["result"]["languages"]),
            "",
            *_comparison_lines(payload["result"]["languages"]),
            "## Last energy ledger",
            "",
            f"- Energy balance passed: `{report['energy_balance_passed']}`",
            f"- Relative residual: `{report['energy_balance_relative_error']}`",
            f"- Residual J: `{report['energy_balance_residual_J']}`",
            f"- RLC regime: `{report['rlc_regime']}`",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark capacitor-bank RLC energy ledger admission.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--discharge-steps", type=int, default=200)
    parser.add_argument("--dt-s", type=float, default=1.0e-7)
    parser.add_argument("--evidence-class", default="local_regression")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    load_start = _loadavg()
    result = _measure(steps=args.steps, warmup=args.warmup, discharge_steps=args.discharge_steps, dt_s=args.dt_s)
    payload: dict[str, Any] = {
        "schema_version": "scpn-control.capacitor-bank-energy-benchmark.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": args.evidence_class,
        "production_claim_allowed": False,
        "settings": {
            "steps": args.steps,
            "warmup": args.warmup,
            "discharge_steps": args.discharge_steps,
            "dt_s": args.dt_s,
        },
        "context": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_affinity": _affinity(),
            "loadavg_start": load_start,
            "loadavg_end": _loadavg(),
            "isolation_method": "caller-provided affinity only; not a production benchmark",
        },
        "result": result,
    }
    payload["payload_sha256"] = _sha256_payload(payload)

    if args.json_out is not None:
        _write_json(args.json_out, payload)
    if args.markdown_out is not None:
        _write_markdown(args.markdown_out, payload)
    if args.json_out is None and args.markdown_out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
