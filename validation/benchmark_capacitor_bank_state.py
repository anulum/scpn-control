# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Capacitor-bank state model benchmark runner.
"""Benchmark the Python and Rust capacitor-bank state surfaces."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from scpn_control.control.capacitor_bank_state import CapacitorBank, CapacitorBankSpec, PulseSpec, free_response

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUST_CRATE = PROJECT_ROOT / "scpn-control-rs" / "crates" / "control-control"
DEFAULT_JSON = PROJECT_ROOT / "validation" / "reports" / "capacitor_bank_state_soft_isolated.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--step-iterations", type=int, default=50_000)
    parser.add_argument("--free-response-iterations", type=int, default=50_000)
    parser.add_argument("--discharge-iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=2_000)
    args = parser.parse_args()

    if args.step_iterations <= 0 or args.free_response_iterations <= 0 or args.discharge_iterations <= 0:
        raise SystemExit("benchmark iterations must be positive")
    if args.warmup < 0:
        raise SystemExit("warmup must be non-negative")

    load_before = _loadavg()
    started_ns = time.time_ns()
    python_results = _python_benchmarks(
        step_iterations=args.step_iterations,
        free_response_iterations=args.free_response_iterations,
        discharge_iterations=args.discharge_iterations,
        warmup=args.warmup,
    )
    rust_results = _rust_benchmarks(
        step_iterations=args.step_iterations,
        free_response_iterations=args.free_response_iterations,
        discharge_iterations=args.discharge_iterations,
        warmup=args.warmup,
    )
    finished_ns = time.time_ns()
    report = {
        "schema_version": "scpn-control.capacitor-bank-benchmark.v1",
        "claim_boundary": (
            "soft-isolated local benchmark evidence only; not target-hardware, HIL, "
            "facility interlock, or plant PCS timing evidence"
        ),
        "project_root": str(PROJECT_ROOT),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "started_unix_ns": started_ns,
        "finished_unix_ns": finished_ns,
        "duration_s": (finished_ns - started_ns) / 1.0e9,
        "host": _host_context(load_before=load_before, load_after=_loadavg()),
        "parameters": {
            "step_iterations": args.step_iterations,
            "free_response_iterations": args.free_response_iterations,
            "discharge_iterations": args.discharge_iterations,
            "warmup": args.warmup,
        },
        "benchmarks": {
            "python": python_results,
            "rust": rust_results,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_out = args.md_out or args.json_out.with_suffix(".md")
    md_out.write_text(_markdown_report(report, args.json_out), encoding="utf-8")


def _python_benchmarks(
    *,
    step_iterations: int,
    free_response_iterations: int,
    discharge_iterations: int,
    warmup: int,
) -> dict[str, dict[str, float]]:
    spec = CapacitorBankSpec(
        capacitance_F=100e-6,
        inductance_H=100e-6,
        series_resistance_ohm=0.5,
        voltage_max_V=10_000.0,
        recharge_power_kW=20.0,
    )
    for _ in range(warmup):
        free_response(spec, 5_000.0, 0.0, 2.0e-5)
    free_response_times = _measure_ns(
        free_response_iterations,
        lambda: free_response(spec, 5_000.0, 0.0, 2.0e-5),
    )

    bank = CapacitorBank(spec, initial_voltage_V=5_000.0)
    for _ in range(warmup):
        bank.step(1.0e-7)
    step_times = _measure_ns(step_iterations, lambda: bank.step(1.0e-7))

    pulse = PulseSpec(peak_current_A=100.0, duration_s=1.0e-4, waveform="half_sine")
    for _ in range(max(1, warmup // 100)):
        CapacitorBank(spec, initial_voltage_V=5_000.0).discharge(pulse, dt=1.0e-6, n_steps=10)
    discharge_times = _measure_ns(
        discharge_iterations,
        lambda: CapacitorBank(spec, initial_voltage_V=5_000.0).discharge(pulse, dt=1.0e-6, n_steps=10),
    )
    return {
        "free_response_ns": _stats(free_response_times),
        "step_ns": _stats(step_times),
        "discharge_10_step_ns": _stats(discharge_times),
    }


def _rust_benchmarks(
    *,
    step_iterations: int,
    free_response_iterations: int,
    discharge_iterations: int,
    warmup: int,
) -> dict[str, object]:
    if shutil.which("cargo") is None:
        return {"skipped": True, "reason": "cargo not found"}
    with tempfile.TemporaryDirectory(prefix="scpn_capacitor_bench_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "src").mkdir()
        (tmp_path / "Cargo.toml").write_text(
            "\n".join(
                [
                    "[package]",
                    'name = "scpn_capacitor_bench"',
                    'version = "0.0.0"',
                    'edition = "2021"',
                    "",
                    "[dependencies]",
                    f'control-control = {{ path = "{RUST_CRATE}" }}',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (tmp_path / "src" / "main.rs").write_text(_RUST_BENCH_SOURCE, encoding="utf-8")
        cmd = [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--",
            str(step_iterations),
            str(free_response_iterations),
            str(discharge_iterations),
            str(warmup),
        ]
        completed = subprocess.run(
            cmd,
            cwd=tmp_path,
            check=True,
            text=True,
            capture_output=True,
            timeout=180,
        )
    parsed: dict[str, object] = json.loads(completed.stdout)
    return parsed


def _measure_ns(n: int, fn: Callable[[], object]) -> list[int]:
    samples: list[int] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return samples


def _stats(samples_ns: list[int]) -> dict[str, float]:
    sorted_ns = sorted(samples_ns)
    return {
        "n": float(len(samples_ns)),
        "mean": statistics.fmean(samples_ns),
        "median": statistics.median(samples_ns),
        "p95": float(sorted_ns[min(len(sorted_ns) - 1, int(len(sorted_ns) * 0.95))]),
        "p99": float(sorted_ns[min(len(sorted_ns) - 1, int(len(sorted_ns) * 0.99))]),
        "min": float(sorted_ns[0]),
        "max": float(sorted_ns[-1]),
    }


def _host_context(
    *, load_before: tuple[float, float, float] | None, load_after: tuple[float, float, float] | None
) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "affinity": _affinity(),
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "governor": _governor(),
        "rustc": _command_version(["rustc", "--version"]),
        "cargo": _command_version(["cargo", "--version"]),
    }


def _affinity() -> list[int] | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    return sorted(int(cpu) for cpu in getter(0))


def _loadavg() -> tuple[float, float, float] | None:
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        return None
    return (one, five, fifteen)


def _governor() -> str | None:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _command_version(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=10)
    return completed.stdout.strip() or completed.stderr.strip() or None


def _markdown_report(report: dict[str, object], json_path: Path) -> str:
    benchmarks = report["benchmarks"]
    assert isinstance(benchmarks, dict)
    python_results = benchmarks["python"]
    rust_results = benchmarks["rust"]
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- SCPN Control — Capacitor-bank state model benchmark report. -->",
            "",
            "# Capacitor-Bank State Model Benchmark",
            "",
            f"- JSON evidence: `{json_path.as_posix()}`",
            f"- Claim boundary: {report['claim_boundary']}",
            f"- Command: `{report['command']}`",
            f"- Duration: {report['duration_s']:.3f} s",
            "",
            "## Host Context",
            "",
            "```json",
            json.dumps(report["host"], indent=2, sort_keys=True),
            "```",
            "",
            "## Results",
            "",
            "```json",
            json.dumps({"python": python_results, "rust": rust_results}, indent=2, sort_keys=True),
            "```",
            "",
            "The Rust measurements use a temporary release-mode Cargo harness that imports the checked-in",
            "`control-control::capacitor_bank` crate by path. The Python measurements use",
            "`scpn_control.control.capacitor_bank_state` in the same process.",
            "",
        ]
    )


_RUST_BENCH_SOURCE = r"""
use control_control::capacitor_bank::{
    free_response, CapacitorBank, CapacitorBankSpec, PulseSpec, PulseWaveform,
};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let step_iterations: usize = args[1].parse().expect("step iterations");
    let free_response_iterations: usize = args[2].parse().expect("free response iterations");
    let discharge_iterations: usize = args[3].parse().expect("discharge iterations");
    let warmup: usize = args[4].parse().expect("warmup");
    let spec = CapacitorBankSpec::new(100e-6, 100e-6, 0.5, 10_000.0, 20.0).expect("valid spec");

    for _ in 0..warmup {
        let _ = free_response(spec, 5_000.0, 0.0, 2.0e-5).expect("free response");
    }
    let free_response_ns = measure(free_response_iterations, || {
        let _ = free_response(spec, 5_000.0, 0.0, 2.0e-5).expect("free response");
    });

    let mut bank = CapacitorBank::new(spec, 5_000.0, 0.0).expect("valid bank");
    for _ in 0..warmup {
        let _ = bank.step(1.0e-7, 0.0).expect("step");
    }
    let step_ns = measure(step_iterations, || {
        let _ = bank.step(1.0e-7, 0.0).expect("step");
    });

    let pulse = PulseSpec::new(100.0, 1.0e-4, PulseWaveform::HalfSine).expect("valid pulse");
    for _ in 0..(warmup.max(100) / 100) {
        let mut warm = CapacitorBank::new(spec, 5_000.0, 0.0).expect("valid bank");
        let _ = warm.discharge(pulse, 1.0e-6, 10).expect("discharge");
    }
    let discharge_10_step_ns = measure(discharge_iterations, || {
        let mut local = CapacitorBank::new(spec, 5_000.0, 0.0).expect("valid bank");
        let _ = local.discharge(pulse, 1.0e-6, 10).expect("discharge");
    });

    println!(
        "{{\"free_response_ns\":{},\"step_ns\":{},\"discharge_10_step_ns\":{}}}",
        stats_json(&free_response_ns),
        stats_json(&step_ns),
        stats_json(&discharge_10_step_ns)
    );
}

fn measure<F: FnMut()>(n: usize, mut f: F) -> Vec<u128> {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_nanos());
    }
    samples
}

fn stats_json(samples: &[u128]) -> String {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let sum: u128 = sorted.iter().sum();
    format!(
        "{{\"n\":{},\"mean\":{},\"median\":{},\"p95\":{},\"p99\":{},\"min\":{},\"max\":{}}}",
        n,
        sum as f64 / n as f64,
        sorted[n / 2],
        sorted[((n as f64 * 0.95) as usize).min(n - 1)],
        sorted[((n as f64 * 0.99) as usize).min(n - 1)],
        sorted[0],
        sorted[n - 1]
    )
}
"""


if __name__ == "__main__":
    main()
