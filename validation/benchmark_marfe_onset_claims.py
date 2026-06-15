#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MARFE onset validation local-regression benchmark
"""Record local-regression timing for MARFE onset validation.

The report is not production benchmark evidence. It records local functional
timing for the sealed MARFE validation payload and leaves production claims
blocked because no isolated-core benchmark context or measured MARFE reference
artefact is supplied.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from validation.validate_marfe_onset import build_evidence, validate_marfe_onset

REPORT_DIR = Path(__file__).resolve().parent / "reports"
JSON_REPORT = REPORT_DIR / "marfe_onset_claims.json"
MARKDOWN_REPORT = REPORT_DIR / "marfe_onset_claims.md"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    timings_us: list[float] = []
    for _ in range(40):
        started = time.perf_counter_ns()
        result = validate_marfe_onset()
        timings_us.append((time.perf_counter_ns() - started) / 1000.0)

    evidence = build_evidence(result, target_id="local-marfe-onset-benchmark")
    payload = {
        "schema_version": "scpn-control.marfe-onset-claims-benchmark.v1",
        "claim_status": "bounded_model",
        "production_claim_allowed": False,
        "benchmark_context": "local_regression_non_isolated",
        "iterations": len(timings_us),
        "timing_us": {
            "mean": statistics.fmean(timings_us),
            "p50": _percentile(timings_us, 0.50),
            "p95": _percentile(timings_us, 0.95),
            "p99": _percentile(timings_us, 0.99),
            "min": min(timings_us),
            "max": max(timings_us),
        },
        "validation_schema_version": evidence["schema_version"],
        "validation_payload_sha256": evidence["payload_sha256"],
        "validation_passed": evidence["passed"],
        "public_claim_allowed": evidence["public_claim_allowed"],
        "critical_density_20": evidence["critical_density"]["critical_density_20"],
        "onset_temperature_eV": evidence["onset_temperature"]["onset_temperature_eV"],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    timing = payload["timing_us"]
    MARKDOWN_REPORT.write_text(
        "\n".join(
            [
                "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
                "",
                "# MARFE Onset Claim-Admission Benchmark",
                "",
                "This report records local-regression timing for the MARFE onset",
                "closed-form validator. It is not production benchmark evidence.",
                "",
                f"- Claim status: `{payload['claim_status']}`",
                f"- Production claim allowed: `{payload['production_claim_allowed']}`",
                f"- Benchmark context: `{payload['benchmark_context']}`",
                f"- Iterations: `{payload['iterations']}`",
                f"- Validation payload SHA-256: `{payload['validation_payload_sha256']}`",
                f"- Critical density: `{payload['critical_density_20']:.12g}` x 10^20 m^-3",
                f"- Onset temperature: `{payload['onset_temperature_eV']:.12g}` eV",
                f"- Mean latency: `{timing['mean']:.12g}` us",
                f"- P50/P95/P99 latency: `{timing['p50']:.12g}` / `{timing['p95']:.12g}` / `{timing['p99']:.12g}` us",
                "",
                "Measured MARFE campaign or documented public-reference artefacts remain required for facility density-limit claims.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
