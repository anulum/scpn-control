# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime Admission Benchmark

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

from scpn_control.core.runtime_admission import RuntimeAdmissionRequest, collect_runtime_admission


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx] / 1000.0


def _measure(fn: Callable[[], object], iterations: int, warmup: int) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    samples: list[int] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - start)
    return {
        "samples": iterations,
        "min_us": min(samples) / 1000.0,
        "median_us": statistics.median(samples) / 1000.0,
        "mean_us": statistics.mean(samples) / 1000.0,
        "p95_us": _percentile(samples, 0.95),
        "p99_us": _percentile(samples, 0.99),
        "max_us": max(samples) / 1000.0,
    }


def _affinity() -> list[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def _loadavg() -> list[float | str]:
    if hasattr(os, "getloadavg"):
        return list(os.getloadavg())
    return ["unavailable"]


def _payload_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    stats = payload["stats"]
    assert isinstance(stats, dict)
    context = payload["context"]
    assert isinstance(context, dict)
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Control — Runtime admission benchmark report. -->",
        "",
        "# Runtime Admission Benchmark",
        "",
        f"- Generated UTC: `{payload['generated_utc']}`",
        f"- Command: `{payload['command']}`",
        f"- Evidence class: `{payload['evidence_class']}`",
        f"- Production claim allowed: `{payload['production_claim_allowed']}`",
        f"- CPU affinity: `{context['cpu_affinity']}`",
        f"- Isolation method: `{context['isolation_method']}`",
        f"- Load average: `{context['loadavg_start']}` -> `{context['loadavg_end']}`",
        f"- Samples: `{stats['samples']}`",
        f"- Median: `{stats['median_us']:.3f} us`",
        f"- p95: `{stats['p95_us']:.3f} us`",
        f"- p99: `{stats['p99_us']:.3f} us`",
        f"- Max: `{stats['max_us']:.3f} us`",
        "",
        "This is an admission-probe benchmark, not a control-loop hot-path benchmark.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runtime admission probe overhead")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--core-snn", type=int, default=1)
    parser.add_argument("--core-z3", type=int, default=2)
    parser.add_argument("--core-net", type=int, default=3)
    parser.add_argument("--core-hb", type=int, default=4)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    loadavg_start = _loadavg()
    request = RuntimeAdmissionRequest(
        execution_backend="native",
        pacing_mode="spin",
        native_backend_available=True,
        require_preempt_rt=True,
        require_realtime_scheduler=True,
        require_performance_governor=True,
        require_heartbeat=True,
        heartbeat_port=5556,
        core_snn=args.core_snn,
        core_z3=args.core_z3,
        core_net=args.core_net,
        core_hb=args.core_hb,
    )
    last_report: dict[str, object] = {}

    def run_probe() -> dict[str, object]:
        nonlocal last_report
        last_report = collect_runtime_admission(request)
        return last_report

    stats = _measure(run_probe, args.iterations, args.warmup)
    payload: dict[str, object] = {
        "schema_version": "scpn-control.runtime-admission-benchmark.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "evidence_class": "local_regression",
        "production_claim_allowed": False,
        "context": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_affinity": _affinity(),
            "isolation_method": "process-affinity-inherited-or-taskset",
            "loadavg_start": loadavg_start,
            "loadavg_end": _loadavg(),
        },
        "stats": stats,
        "last_admission_status": last_report.get("status"),
        "last_admission_errors": last_report.get("errors", []),
        "last_admission_warnings": last_report.get("warnings", []),
        "payload_sha256": "",
    }
    payload["payload_sha256"] = _payload_digest(payload)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(args.md_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
