# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual JAX latency observation.

"""Actual copied JAX benchmark custody and explicitly authored declaration derivatives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scpn_control.benchmark_records import load_verified_latest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def observed_transport_reports(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Run the complete byte-identical public JAX producer with temporary software custody.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Session-owned temporary repository factory.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        Real21-point one-step, four-step rollout and blocked-readiness reports
        from installed CPU JAX, with raw wrapper/status/source/immutable digests.

    Notes
    -----
    The shipped benchmark invokes actual public audited-gradient APIs with one
    warmup/five timings; no helpers are called directly by tests. Its copied
    module resolves output/formal-report paths inside the temporary repository,
    so no canonical science report is overwritten. No external reference,
    source authentication, controlled comparison or hardware admission follows.
    """
    pytest.importorskip("jax", reason="the producer runs on installed JAX; the coverage lane installs it")
    root = tmp_path_factory.mktemp("differentiable-observation")
    source = root / "validation/benchmark_differentiable_transport_latency.py"
    source.parent.mkdir(parents=True)
    original = ROOT / "validation/benchmark_differentiable_transport_latency.py"
    shutil.copyfile(original, source)
    assert source.read_bytes() == original.read_bytes()
    names = [
        "differentiable_transport_latency",
        "differentiable_transport_rollout_latency",
        "differentiable_transport_full_fidelity_readiness",
    ]
    argv = [
        sys.executable,
        str(ROOT / "tools/run_recorded_benchmark.py"),
        "--repository-root",
        str(root),
        "--records-root",
        "records",
        "--family",
        "differentiable-reader-software",
        "--campaign-id",
        "actual-observation",
        "--evidence-class",
        "software_boundary_test",
    ]
    for role, name in zip(["one", "rollout", "readiness"], names, strict=True):
        argv += ["--artifact", f"{role}=validation/reports/{name}.json"]
    argv += ["--", sys.executable, str(source)]
    env = dict(
        os.environ, PYTHONPATH=str(ROOT / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu"
    )
    env.pop("SCPN_BENCHMARK_CAMPAIGN_ID", None)
    result = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True, check=False, timeout=90)
    (root / "observation.json").write_text(
        json.dumps(
            {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    latest, manifest = load_verified_latest(root / "records", "differentiable-reader-software")
    assert latest["campaign_id"] == "actual-observation" and manifest["status"] == "succeeded"
    assert manifest["evidence_class"] == "software_boundary_test" and len(manifest["artifacts"]) == 3
    for artifact in manifest["artifacts"]:
        actual = root / artifact["source_path"]
        immutable = root / artifact["immutable_path"]
        assert actual.read_bytes() == immutable.read_bytes()
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == artifact["sha256"]
    one, rollout, readiness = (root / "validation/reports" / (name + ".json") for name in names)
    observed = json.loads(one.read_text(encoding="utf-8"))
    assert observed["backend"] == "jax" and observed["audit"]["passed"] is True
    assert observed["runtime_metadata"]["jax_default_backend"] == "cpu"
    assert observed["n_rho"] == 21 and observed["timed_runs"] == 5
    assert json.loads(readiness.read_text())["full_fidelity_claim_admissible"] is False
    assert source.read_bytes() == original.read_bytes()
    return one, rollout, readiness


def write_derivative(observed: Path, destination: Path, field: str, value: object) -> Path:
    """Write authored metadata/metric input without changing the actual observation.

    Parameters
    ----------
    observed : pathlib.Path
        Actual source report, read only.
    destination : pathlib.Path
        Isolated derivative destination.
    field : str
        Dot-separated field; existing intermediate objects are required.
    value : object
        Authored replacement. No new measurement, audit or qualification is implied.

    Returns
    -------
    pathlib.Path
        JSON destination written with the standard serialiser.
    """
    payload: dict[str, Any] = json.loads(observed.read_text(encoding="utf-8"))
    parent = payload
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination
