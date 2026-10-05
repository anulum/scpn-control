# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual bounded uncertainty and equilibrium claim producer commands

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
PRODUCERS = ["benchmark_uq_claims", "benchmark_kinetic_efit_claims", "benchmark_free_boundary_tracking_claims"]


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
    if producer == "benchmark_uq_claims":
        assert report["source"] == "synthetic_regression_reference"
        assert report["source_id"] == "uq-bounded-regression-v1"
        assert report["seed"] == 31 and report["n_samples"] == 256
        assert report["finite_outputs"] is True
        assert report["tau_E_percentiles_ordered"] is True
        assert report["P_fusion_percentiles_ordered"] is True
        assert report["Q_percentiles_ordered"] is True
        assert report["fuel_ion_fraction"] == 1.0
        assert report["tau_E_s"] > 0.0 and report["P_fusion_MW"] > 0.0 and report["Q"] > 0.0
        assert report["calibrated_uq_claim_allowed"] is False
        assert report["tau_E_relative_error"] is None and report["sigma_relative_error"] is None
        assert "not calibrated facility predictive uncertainty" in markdown
    elif producer == "benchmark_kinetic_efit_claims":
        assert report["source"] == "synthetic_regression_reference"
        assert report["source_id"] == "kinetic-efit-bounded-regression-v1"
        assert report["interpolation_geometry"] == "normalised_elliptic_rho"
        assert report["n_te_points"] == report["n_ne_points"] == report["n_ti_points"] == 2
        assert report["n_mse_points"] == 1
        assert report["fast_ion_energy_keV"] == 100.0
        assert report["fast_ion_density_fraction"] == 0.1 and report["anisotropy_sigma"] == 0.2
        assert report["q_axis"] == pytest.approx(1.0 + 5.0 / 90.0)
        assert report["q_edge"] == pytest.approx(report["q_axis"] + 2.0)
        assert report["facility_claim_allowed"] is False
        assert report["pressure_relative_error"] is None and report["q_profile_relative_error"] is None
        assert "not a facility EFIT or P-EFIT validation" in markdown
    else:
        assert report["source"] == "repository_free_boundary_regression"
        assert report["source_id"] == "free-boundary-tracking-claim-benchmark-v1"
        assert report["config_path"] == "validation/free_boundary_tracking_claims_fixture.json"
        assert report["steps"] == 5
        assert report["boundary_variant"] == "free_boundary"
        assert report["min_response_rank"] == 4
        assert report["max_abs_coil_current"] <= 3.0
        assert report["measurement_latency_enabled"] is True and report["latency_compensation_enabled"] is True
        assert report["facility_claim_allowed"] is False and report["reference_artifact_sha256"] is None
        assert report["claim_status"] == "bounded_free_boundary_tracking_evidence"
        assert "not commissioned facility-control validation" in markdown
    assert source.read_bytes() == (ROOT / "validation" / source.name).read_bytes()
