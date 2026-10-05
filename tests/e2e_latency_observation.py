# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — E2E Latency Evidence Validation

"""Actual tiny E2E producer observation and explicitly authored report derivatives."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scpn_control.benchmark_records import load_verified_latest
from validation.validate_e2e_latency_evidence import build_e2e_latency_evidence_payload

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def measured_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the installed Python E2E producer once and verify software-only custody.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Session-owned temporary root factory; canonical report paths are unused.

    Returns
    -------
    pathlib.Path
        Actual three-iteration/one-warmup 16x16 report, with unqualified hardware
        labels. Wrapper manifest/immutable bytes are verified and argv/stdout/
        stderr retained beside the report.

    Notes
    -----
    Timings come from the real sensor/SOR/transport/H-infinity/clamp path.
    These tiny samples are diagnostic, not tail statistics, isolated performance
    or operator-qualified hardware evidence. No mock/native substitution occurs.
    """
    root = tmp_path_factory.mktemp("e2e-observation")
    report = root / "actual.json"
    argv = [
        sys.executable,
        str(ROOT / "tools/run_recorded_benchmark.py"),
        "--repository-root",
        str(root),
        "--records-root",
        "records",
        "--family",
        "e2e-contract-software",
        "--campaign-id",
        "actual-observation",
        "--evidence-class",
        "software_boundary_test",
        "--artifact",
        f"report={report}",
        "--",
        sys.executable,
        str(ROOT / "benchmarks/e2e_control_latency.py"),
        "--iterations",
        "3",
        "--warmup",
        "1",
        "--output-json",
        str(report),
        "--json",
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.pop("SCPN_BENCHMARK_CAMPAIGN_ID", None)
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=60)
    (root / "observation.json").write_text(
        json.dumps(
            {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    latest, manifest = load_verified_latest(root / "records", "e2e-contract-software")
    assert latest["campaign_id"] == "actual-observation"
    assert manifest["status"] == "succeeded" and manifest["evidence_class"] == "software_boundary_test"
    assert len(manifest["artifacts"]) == 1
    artifact = manifest["artifacts"][0]
    immutable = root / artifact["immutable_path"]
    assert immutable.read_bytes() == report.read_bytes()
    assert artifact["sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["iterations"] == 3 and payload["warmup"] == 1 and payload["grid"] == "16x16"
    assert payload["production_claim_allowed"] is False and payload["evidence_class"] == "local_regression"
    assert payload["target_hardware"]["id"] == "local-host-unqualified"
    return report


def derived_report(measured: Path, output: Path, field: str, value: object) -> Path:
    """Write an explicitly authored field derivative of an actual observed report.

    Parameters
    ----------
    measured : pathlib.Path
        Original report, never modified.
    output : pathlib.Path
        Temporary derivative destination.
    field : str
        Dot-separated report field; intermediate mappings must already exist.
    value : object
        Authored replacement, including invalid metadata/numeric declarations.

    Returns
    -------
    pathlib.Path
        Destination rebuilt by the public payload builder. This new declaration
        is reader-test input, not a new observation or hardware qualification.
    """
    payload: dict[str, Any] = json.loads(measured.read_text(encoding="utf-8"))
    parent = payload
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value
    output.write_text(json.dumps(build_e2e_latency_evidence_payload(payload)), encoding="utf-8")
    return output
