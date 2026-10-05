# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Manufactured mesh public command tests.

"""Actual public manufactured mesh count and complete fixed-study command checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_control.benchmark_records import load_verified_latest
from validation.mesh_convergence_study import run_solovev_benchmark

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("cap", [0, -1, 1])
def test_actual_completed_sweep_count(cap: int) -> None:
    """A zero/negative cap completes zero sweeps and one actual sweep reports one."""
    result = run_solovev_benchmark(5, 7, max_iter=cap)
    assert result["iterations"] == max(0, cap)
    assert result["nr"] == 5 and result["nz"] == 7 and result["h"] == 0.5
    assert result["rmse"] > 0 and result["nrmse"] > 0 and result["wall_time_s"] >= 0


def test_actual_complete_fixed_mesh_study_uses_temporary_software_custody(tmp_path: Path) -> None:
    """Execute all four actual default grids without overwriting canonical scientific reports."""
    argv = [
        sys.executable,
        str(ROOT / "tools/run_recorded_benchmark.py"),
        "--repository-root",
        str(tmp_path),
        "--records-root",
        "records",
        "--family",
        "mesh-reader-software",
        "--campaign-id",
        "actual-study",
        "--evidence-class",
        "software_boundary_test",
        "--artifact",
        "report=validation/reports/mesh_convergence.json",
        "--artifact",
        "markdown=validation/reports/mesh_convergence.md",
        "--",
        sys.executable,
        str(ROOT / "validation/mesh_convergence_study.py"),
    ]
    result = subprocess.run(
        argv,
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    (tmp_path / "command.json").write_text(
        json.dumps(
            {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    latest, manifest = load_verified_latest(tmp_path / "records", "mesh-reader-software")
    assert latest["campaign_id"] == "actual-study" and manifest["status"] == "succeeded"
    assert manifest["evidence_class"] == "software_boundary_test"
    rows = json.loads((tmp_path / "validation/reports/mesh_convergence.json").read_text())
    assert [r["nr"] for r in rows] == [17, 33, 65, 129]
    assert all(r["nr"] == r["nz"] and 0 < r["iterations"] <= 25000 for r in rows)
    assert "convergence_rate" not in rows[0]
    assert all(1.8 <= r["convergence_rate"] <= 2.2 for r in rows[1:])
    assert all(rows[i]["nrmse"] > rows[i + 1]["nrmse"] for i in range(3))
    assert "Results saved" in result.stdout
