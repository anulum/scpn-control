# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime Admission Benchmark Tests

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_runtime_admission_benchmark_emits_schema_bound_report(tmp_path: Path) -> None:
    report = tmp_path / "runtime_admission.json"
    markdown = tmp_path / "runtime_admission.md"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_runtime_admission.py",
            "--iterations",
            "2",
            "--warmup",
            "0",
            "--core-snn",
            "0",
            "--core-z3",
            "1",
            "--core-net",
            "2",
            "--core-hb",
            "3",
            "--json-out",
            str(report),
            "--md-out",
            str(markdown),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    stdout_payload = json.loads(result.stdout)
    assert payload["schema_version"] == "scpn-control.runtime-admission-benchmark.v1"
    assert payload["evidence_class"] == "local_regression"
    assert payload["production_claim_allowed"] is False
    assert payload["stats"]["samples"] == 2
    assert stdout_payload["schema_version"] == payload["schema_version"]
    assert markdown.read_text(encoding="utf-8").startswith("# Runtime Admission Benchmark")
