# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — AER observation benchmark runner.
"""Benchmark Python and Rust AER decoder paths."""

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

from scpn_control.scpn.observation import (
    SpikeBuffer,
    SpikeEvent,
    decode_isi,
    decode_rate,
    decode_temporal,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_CORE = PROJECT_ROOT / "scpn-control-rs" / "crates" / "control-core"
DEFAULT_JSON = PROJECT_ROOT / "validation" / "reports" / "aer_observation_soft_isolated.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--warmup", type=int, default=2_000)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise SystemExit("iterations must be positive")
    if args.warmup < 0:
        raise SystemExit("warmup must be non-negative")

    load_before = _loadavg()
    started_ns = time.time_ns()
    python_results = _python_bench(args.iterations, args.warmup)
    rust_results = _rust_bench(args.iterations, args.warmup)
    finished_ns = time.time_ns()
    report = {
        "schema_version": "scpn-control.aer-observation-benchmark.v1",
        "evidence_class": "local_regression",
        "production_claim_allowed": False,
        "claim_boundary": (
            "soft-isolated local decoder benchmark evidence only; not target-hardware, "
            "neuromorphic-device, FPGA, HIL, or plant PCS timing evidence"
        ),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "outer_command": os.environ.get("SCPN_BENCH_OUTER_COMMAND"),
        "duration_s": (finished_ns - started_ns) / 1.0e9,
        "host": _host(load_before, _loadavg()),
        "parameters": {"iterations": args.iterations, "warmup": args.warmup},
        "benchmarks": {"python": python_results, "rust": rust_results},
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_out = args.md_out or args.json_out.with_suffix(".md")
    md_out.write_text(_markdown(report, args.json_out), encoding="utf-8")


def _events() -> list[SpikeEvent]:
    return [SpikeEvent(neuron_id=idx % 64, timestamp_ns=idx * 100) for idx in range(256)]


def _python_bench(iterations: int, warmup: int) -> dict[str, dict[str, float]]:
    events = _events()
    admission_buffer = SpikeBuffer(capacity=len(events))
    admission_buffer.extend(events)
    for _ in range(warmup):
        decode_rate(events, 30_000, 64)
        decode_temporal(events, 30_000, 64, now_ns=30_000)
        decode_isi(events, 30_000, 64)
        admission_buffer.admission_report()
        _admission_push_report(events)
    return {
        "rate_ns": _stats(_measure(iterations, lambda: decode_rate(events, 30_000, 64))),
        "temporal_ns": _stats(_measure(iterations, lambda: decode_temporal(events, 30_000, 64, now_ns=30_000))),
        "isi_ns": _stats(_measure(iterations, lambda: decode_isi(events, 30_000, 64))),
        "admission_report_ns": _stats(_measure(iterations, admission_buffer.admission_report)),
        "admission_push_report_ns": _stats(_measure(iterations, lambda: _admission_push_report(events))),
    }


def _admission_push_report(events: list[SpikeEvent]) -> dict[str, int | bool]:
    buffer = SpikeBuffer(capacity=len(events))
    buffer.extend(events)
    return buffer.admission_report()


def _rust_bench(iterations: int, warmup: int) -> dict[str, object]:
    if shutil.which("cargo") is None:
        return {"skipped": True, "reason": "cargo not found"}
    with tempfile.TemporaryDirectory(prefix="scpn_aer_bench_") as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "Cargo.toml").write_text(
            "\n".join(
                [
                    "[package]",
                    'name = "scpn_aer_bench"',
                    'version = "0.0.0"',
                    'edition = "2021"',
                    "",
                    "[dependencies]",
                    f'control-core = {{ path = "{CONTROL_CORE}" }}',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (root / "src" / "main.rs").write_text(_RUST_SOURCE, encoding="utf-8")
        completed = subprocess.run(
            ["cargo", "run", "--release", "--quiet", "--", str(iterations), str(warmup)],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
            timeout=180,
        )
    return json.loads(completed.stdout)


def _measure(n: int, fn: Callable[[], object]) -> list[int]:
    samples: list[int] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return samples


def _stats(samples: list[int]) -> dict[str, float]:
    sorted_samples = sorted(samples)
    return {
        "n": float(len(samples)),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p95": float(sorted_samples[min(len(sorted_samples) - 1, int(len(sorted_samples) * 0.95))]),
        "p99": float(sorted_samples[min(len(sorted_samples) - 1, int(len(sorted_samples) * 0.99))]),
        "min": float(sorted_samples[0]),
        "max": float(sorted_samples[-1]),
    }


def _host(
    load_before: tuple[float, float, float] | None, load_after: tuple[float, float, float] | None
) -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "governor": _governor(),
        "rustc": _version(["rustc", "--version"]),
        "cargo": _version(["cargo", "--version"]),
    }


def _loadavg() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except OSError:
        return None


def _governor() -> str | None:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def _version(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=10)
    return completed.stdout.strip() or completed.stderr.strip() or None


def _markdown(report: dict[str, object], json_path: Path) -> str:
    return "\n".join(
        [
            "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
            "<!-- Commercial license available -->",
            "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
            "<!-- ORCID: 0009-0009-3560-0851 -->",
            "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
            "<!-- SCPN Control — AER observation benchmark report. -->",
            "",
            "# AER Observation Benchmark",
            "",
            f"- JSON evidence: `{json_path.as_posix()}`",
            f"- Evidence class: `{report['evidence_class']}`",
            f"- Production claim allowed: `{report['production_claim_allowed']}`",
            f"- Claim boundary: {report['claim_boundary']}",
            f"- Command: `{report['command']}`",
            f"- Outer command: `{report.get('outer_command')}`",
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
            json.dumps(report["benchmarks"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


_RUST_SOURCE = r"""
use control_core::spike_buffer::{
    decode_isi, decode_rate, decode_temporal, SpikeBuffer, SpikeEvent,
};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let iterations: usize = args[1].parse().expect("iterations");
    let warmup: usize = args[2].parse().expect("warmup");
    let events: Vec<SpikeEvent> = (0..256)
        .map(|idx| SpikeEvent::new(idx % 64, idx as u64 * 100))
        .collect();
    let admission_buffer = build_admission_buffer(&events);
    for _ in 0..warmup {
        let _ = decode_rate(&events, 30_000, 64).expect("rate");
        let _ = decode_temporal(&events, 30_000, 64, 30_000).expect("temporal");
        let _ = decode_isi(&events, 30_000, 64).expect("isi");
        let _ = admission_buffer.admission_report();
        admission_push_report(&events);
    }
    let rate = measure(iterations, || {
        let _ = decode_rate(&events, 30_000, 64).expect("rate");
    });
    let temporal = measure(iterations, || {
        let _ = decode_temporal(&events, 30_000, 64, 30_000).expect("temporal");
    });
    let isi = measure(iterations, || {
        let _ = decode_isi(&events, 30_000, 64).expect("isi");
    });
    let admission = measure(iterations, || {
        std::hint::black_box(admission_buffer.admission_report());
    });
    let admission_push = measure(iterations, || {
        admission_push_report(&events);
    });
    println!(
        "{{\"rate_ns\":{},\"temporal_ns\":{},\"isi_ns\":{},\"admission_report_ns\":{},\"admission_push_report_ns\":{}}}",
        stats_json(&rate),
        stats_json(&temporal),
        stats_json(&isi),
        stats_json(&admission),
        stats_json(&admission_push)
    );
}

fn build_admission_buffer(events: &[SpikeEvent]) -> SpikeBuffer {
    let mut buffer = SpikeBuffer::new(events.len()).expect("buffer");
    for event in events {
        buffer.push(*event);
    }
    buffer
}

fn admission_push_report(events: &[SpikeEvent]) {
    let buffer = build_admission_buffer(events);
    std::hint::black_box(buffer.admission_report());
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
