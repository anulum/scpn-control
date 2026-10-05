# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual bounded particle claim producer commands

"""Exercise identical copied public producers with actual software-only custody."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scpn_control.benchmark_records import load_verified_latest

ROOT = Path(__file__).resolve().parents[1]
PRODUCERS = ["benchmark_density_control_claims", "benchmark_current_drive_claims"]


def _copied_producer(tmp_path: Path, producer: str) -> tuple[Path, Path]:
    """Copy the complete unchanged producer into a real temporary output root."""
    repository = tmp_path / "repository"
    source = repository / "validation" / f"{producer}.py"
    source.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "validation" / source.name, source)
    assert source.read_bytes() == (ROOT / "validation" / source.name).read_bytes()
    return repository, source


def _run(argv: list[str], repository: Path) -> subprocess.CompletedProcess[str]:
    """Run a real producer/wrapper process without an inherited campaign grant."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.pop("SCPN_BENCHMARK_CAMPAIGN_ID", None)
    return subprocess.run(argv, cwd=repository, env=env, capture_output=True, text=True, check=False, timeout=60)


@pytest.mark.parametrize("producer", PRODUCERS)
def test_actual_unrecorded_producer_refuses_before_report_creation(tmp_path: Path, producer: str) -> None:
    """The actual script cannot create persistent copied reports without custody."""
    repository, source = _copied_producer(tmp_path, producer)
    result = _run([sys.executable, str(source)], repository)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "persistent benchmark output requires tools/run_recorded_benchmark.py" in result.stderr
    assert not (repository / "validation/reports").exists()
    assert source.read_bytes() == (ROOT / "validation" / source.name).read_bytes()


@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("entry", ["script", "api"])
def test_actual_recorded_public_producer_preserves_bounded_claims(tmp_path: Path, producer: str, entry: str) -> None:
    """Real script/imported main reports are retained by the actual custody runner."""
    repository, source = _copied_producer(tmp_path, producer)
    stem = producer.removeprefix("benchmark_")
    if entry == "script":
        command = [sys.executable, str(source)]
    else:
        code = (
            "import importlib.util,sys;"
            "s=importlib.util.spec_from_file_location('actual_copied_producer',sys.argv[1]);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main()"
        )
        command = [sys.executable, "-c", code, str(source)]
    result = _run(
        [
            sys.executable,
            str(ROOT / "tools/run_recorded_benchmark.py"),
            "--repository-root",
            str(repository),
            "--records-root",
            "records",
            "--family",
            stem,
            "--campaign-id",
            f"actual-{entry}",
            "--evidence-class",
            "software_boundary_test",
            "--artifact",
            f"report=validation/reports/{stem}.json",
            "--artifact",
            f"markdown=validation/reports/{stem}.md",
            "--",
            *command,
        ],
        repository,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    latest, manifest = load_verified_latest(repository / "records", stem)
    assert latest["campaign_id"] == f"actual-{entry}"
    assert manifest["status"] == "succeeded"
    assert manifest["evidence_class"] == "software_boundary_test"
    assert {a["role"] for a in manifest["artifacts"]} == {"report", "markdown"}
    for artifact in manifest["artifacts"]:
        current = repository / artifact["source_path"]
        immutable = repository / artifact["immutable_path"]
        assert current.read_bytes() == immutable.read_bytes()
        assert hashlib.sha256(immutable.read_bytes()).hexdigest() == artifact["sha256"]
    report = json.loads((repository / f"validation/reports/{stem}.json").read_text(encoding="utf-8"))
    markdown = (repository / f"validation/reports/{stem}.md").read_text(encoding="utf-8")
    if producer == "benchmark_density_control_claims":
        assert report["source"] == "synthetic_regression_reference"
        assert report["source_id"] == "density-control-bounded-regression-v1"
        assert report["n_rho"] == 12 and report["R0_m"] == 6.2 and report["a_m"] == 2.0
        assert report["dt_requested_s"] == 1.0
        assert report["total_source_particles_per_s"] == pytest.approx(1.0e20)
        assert report["facility_density_claim_allowed"] is False
        assert report["reference_comparison_passed"] is False
        assert "not facility-calibrated" in markdown
    else:
        assert report["source"] == "repository_current_drive_regression"
        assert report["source_id"] == "current-drive-claim-benchmark-v1"
        assert report["profile_points"] == 80
        assert report["rho_min"] == 0.0 and report["rho_max"] == 1.0
        assert report["total_absorbed_power_W"] == pytest.approx(25.0e6)
        assert report["grid_normalised_power"] is True
        assert report["nbi_beam_energy_keV"] == 1000.0
        assert report["external_claim_allowed"] is False
        assert report["reference_artifact_sha256"] is None
        assert "not ray-traced, Fokker-Planck, or facility" in markdown
    assert source.read_bytes() == (ROOT / "validation" / source.name).read_bytes()
