# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual vacuum diagnostic report command

"""Exercise actual vacuum diagnostics and prevent an off-axis display from becoming a report PASS."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_control.benchmark_records import load_verified_latest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "validation/benchmark_free_boundary.py"


def _run(argv: list[str], repository: Path) -> subprocess.CompletedProcess[str]:
    """Run the actual source/wrapper with real temporary config files and no inherited grant."""
    configs = repository / "configs"
    configs.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "src"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        TMPDIR=str(configs),
    )
    env.pop("SCPN_BENCHMARK_CAMPAIGN_ID", None)
    result = subprocess.run(argv, cwd=repository, env=env, capture_output=True, text=True, check=False, timeout=60)
    assert not list(configs.iterdir())
    return result


def test_actual_public_vacuum_diagnostics_preserve_original_numerics(tmp_path: Path) -> None:
    """The real public diagnostic API retains its explicitly legacy qualitative marker."""
    code = (
        "import json;from validation.benchmark_free_boundary import run_free_boundary_benchmark;"
        "print(json.dumps(run_free_boundary_benchmark(),allow_nan=False))"
    )
    result = _run([sys.executable, "-c", code], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["single_coil"]["calculated"] == data["single_coil"]["reference"]
    assert data["single_coil"]["error_rel"] == 0.0
    assert data["helmholtz"]["pass"] is True
    assert not math.isclose(data["helmholtz"]["bz_at_min_r"], data["helmholtz"]["bz_axis_ref"], rel_tol=0.01)
    assert data["x_point"]["detected_r"] > data["x_point"]["expected_r"] == 0.0


def test_actual_unrecorded_copied_script_refuses_persistent_reports(tmp_path: Path) -> None:
    """The real fixed-path script refuses before writing unrecorded copied-root reports."""
    source = tmp_path / "validation" / SOURCE.name
    source.parent.mkdir()
    shutil.copyfile(SOURCE, source)
    result = _run([sys.executable, str(source)], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "persistent benchmark output requires" in result.stderr
    assert not (tmp_path / "validation/reports").exists()
    assert source.read_bytes() == SOURCE.read_bytes()


@pytest.mark.parametrize("entry", ["script", "api"])
def test_actual_public_writer_does_not_admit_off_axis_field(tmp_path: Path, entry: str) -> None:
    """Real script and canonical imported main record an unassessed Helmholtz diagnostic."""
    source = tmp_path / "validation" / SOURCE.name
    source.parent.mkdir()
    shutil.copyfile(SOURCE, source)
    command = (
        [sys.executable, str(source)]
        if entry == "script"
        else [sys.executable, "-c", "from validation.benchmark_free_boundary import main;main()"]
    )
    result = _run(
        [
            sys.executable,
            str(ROOT / "tools/run_recorded_benchmark.py"),
            "--repository-root",
            str(tmp_path),
            "--records-root",
            "records",
            "--family",
            "vacuum-diagnostics",
            "--campaign-id",
            f"actual-{entry}",
            "--evidence-class",
            "software_boundary_test",
            "--artifact",
            "report=validation/reports/free_boundary_benchmark.json",
            "--artifact",
            "markdown=validation/reports/free_boundary_benchmark.md",
            "--",
            *command,
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _, manifest = load_verified_latest(tmp_path / "records", "vacuum-diagnostics")
    assert manifest["status"] == "succeeded"
    for artifact in manifest["artifacts"]:
        current = tmp_path / artifact["source_path"]
        immutable = tmp_path / artifact["immutable_path"]
        assert current.read_bytes() == immutable.read_bytes()
        assert hashlib.sha256(immutable.read_bytes()).hexdigest() == artifact["sha256"]
    payload = json.loads((tmp_path / "validation/reports/free_boundary_benchmark.json").read_text())
    assert payload["helmholtz"]["pass"] is None
    assert payload["helmholtz"]["assessment"] == "diagnostic_only_off_axis_sample"
    assert not math.isclose(payload["helmholtz"]["bz_at_min_r"], payload["helmholtz"]["bz_axis_ref"], rel_tol=0.01)
    markdown = (tmp_path / "validation/reports/free_boundary_benchmark.md").read_text()
    assert "| Helmholtz |" in markdown and "| N/A |" in markdown
    assert source.read_bytes() == SOURCE.read_bytes()
