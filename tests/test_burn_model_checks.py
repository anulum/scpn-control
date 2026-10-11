# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real burn diagnostics and analytical pressure units.
"""Exercise the public physical conversion, refusal and actual provider surfaces."""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from validation.burn_model_checks import BurnPoint, check_history, evidence_failures
from validation.burn_model_provider import SOURCE_HASHES, evaluate_burn_model

ROOT = Path(__file__).resolve().parents[1]


def point() -> BurnPoint:
    """Return a declared uniform input, with no fitted design beta target."""
    return BurnPoint(6.2, 2.0, 5.3, 15.0, 1.7, 1.01, 50.0)


def test_total_pressure_matches_independent_ideal_gas_definition() -> None:
    """Total electron+ion energy yields twice the electron-only pressure conversion."""
    p = point()
    temperature_j = 8e3 * 1.602e-19
    density = p.ne20 * 1e20
    stored_mj = 3 * density * temperature_j * p.volume_m3 / 1e6
    expected_pressure = 2 * density * temperature_j
    expected = 100 * 2 * (4 * math.pi * 1e-7) * expected_pressure * p.a_m / (p.b_t * p.ip_ma)
    assert p.beta_n(stored_mj) == pytest.approx(expected, rel=1e-14)
    assert replace(p, ip_ma=2 * p.ip_ma).beta_n(stored_mj) == pytest.approx(expected / 2)
    assert p.beta_n(0.0) == 0.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_invalid_physical_input_and_stored_energy_refuse(value: float) -> None:
    """The public physical conversion does not coerce invalid inputs into pressure."""
    with pytest.raises(ValueError):
        replace(point(), b_t=value)
    with pytest.raises(ValueError):
        point().beta_n(value)


def test_negative_impurity_radiation_is_not_a_supported_operating_point() -> None:
    """A subunit effective charge cannot silently produce negative line radiation."""
    with pytest.raises(ValueError, match="at least one"):
        replace(point(), z_eff=0.5)


@pytest.mark.parametrize("history", [[], [float("nan")] * 10, [1.0] * 9, [[1.0] * 10]])
def test_incomplete_or_nonfinite_history_cannot_claim_numerical_evidence(history: object) -> None:
    """Actual public history admission refuses malformed first-channel observations."""
    with pytest.raises(ValueError, match="finite samples"):
        check_history({"time_s": history}, point(), 1.0, 0.1)


@pytest.mark.parametrize("reference_name", ["iter_reference.json", "sparc_reference.json"])
def test_real_source_bound_provider_or_explicit_unavailability(reference_name: str) -> None:
    """Execute actual reference inputs; the hosted provider job requires real success.

    Dependency-light platform jobs exercise the explicit optional-provider
    refusal. The dedicated pinned-provider job sets the requirement environment
    variable and exercises original coarse/refined numerical histories.
    """
    reference = json.loads((ROOT / "validation/reference_data" / reference_name).read_text())
    try:
        beta, evidence = evaluate_burn_model(reference)
    except RuntimeError as exc:
        if os.environ.get("SCPN_REQUIRE_BURN_PROVIDER") == "1":
            raise
        assert "reference values cannot replace model predictions" in str(exc)
        return
    assert beta > 0 and math.isfinite(beta)
    assert evidence["model_provenance"]["source_files_sha256"] == SOURCE_HASHES
    assert evidence["physical_validation_claim"] is False
    assert evidence_failures(evidence["checks"]) == []
    coarse, refined = evidence["native_histories"]
    assert len(coarse["W_MJ"]) == 20_000 and len(refined["W_MJ"]) == 40_000
    assert evidence["W_MJ"] == refined["W_MJ"][-1]
    assert evidence["run_checks"][1]["beta_energy_time_s"] == 200.0
    assert evidence["run_checks"][1]["power_sample_time_s"] == pytest.approx(199.995)
    assert evidence["Q_unclipped"] == pytest.approx(refined["P_fus_MW"][-1] / reference["P_aux_MW"])


def test_reference_beta_is_never_a_model_input() -> None:
    """Invalid physical fields refuse before a reference beta could supply a prediction."""
    reference = json.loads((ROOT / "validation/reference_data/iter_reference.json").read_text())
    reference.update(beta_N=0.0, B_t_T=0.0)
    with pytest.raises(ValueError, match="B_t_T"):
        evaluate_burn_model(reference)


@pytest.mark.parametrize("duration,dt", [(0.0, 0.1), (1.0, 0.0), (float("nan"), 0.1), (0.05, 0.1)])
def test_invalid_history_horizon_never_admits_empty_evidence(duration: float, dt: float) -> None:
    """Zero, nonfinite and undersampled intervals refuse at the public history boundary."""
    with pytest.raises(ValueError, match="time steps"):
        check_history({}, point(), duration, dt)


@pytest.mark.parametrize("field,value", [("B_t_T", True), ("B_t_T", "5.3"), ("n_e_1e19", float("nan"))])
def test_reference_units_require_actual_positive_numbers(field: str, value: object) -> None:
    """A malformed design input cannot be coerced or replaced by a target."""
    reference = json.loads((ROOT / "validation/reference_data/iter_reference.json").read_text())
    reference[field] = value
    with pytest.raises(ValueError, match=field):
        evaluate_burn_model(reference)


@pytest.mark.parametrize("value", [None, True, -1.0, float("nan"), float("inf"), 10**400])
def test_invalid_numerical_observation_cannot_pass_fidelity(value: object) -> None:
    """Public evidence admission refuses malformed or nonrepresentable error observations."""
    from validation.burn_model_checks import CLOSURE_COUNTS, NUMERICAL_LIMITS

    checks = {**NUMERICAL_LIMITS, **dict.fromkeys(CLOSURE_COUNTS, 0)}
    checks["energy_balance_relative_error"] = value
    assert any("energy_balance_relative_error" in error for error in evidence_failures(checks))
    assert evidence_failures(None)


@pytest.mark.parametrize("defect", ["version", "source"])
def test_real_provider_copy_refuses_wrong_version_or_changed_source(tmp_path: Path, defect: str) -> None:
    """A real copied distribution with changed identity/source refuses at the public API.

    The dedicated hosted job has the provider installed. Dependency-light jobs
    assert its actual absence contract elsewhere; no modules or APIs are mocked.
    Subprocess coverage follows this real filesystem/import boundary.
    """
    import importlib.util
    import shutil
    import subprocess
    import sys
    from importlib.metadata import distribution

    spec = importlib.util.find_spec("scpn_fusion")
    if spec is None:
        assert os.environ.get("SCPN_REQUIRE_BURN_PROVIDER") != "1"
        return
    assert spec.origin is not None
    site = tmp_path / "site"
    site.mkdir()
    shutil.copytree(Path(spec.origin).parent, site / "scpn_fusion")
    dist = distribution("scpn-fusion")
    metadata = next(
        Path(dist.locate_file(file)).parent for file in dist.files or [] if str(file).endswith(".dist-info/METADATA")
    )
    shutil.copytree(metadata, site / metadata.name)
    if defect == "version":
        path = site / metadata.name / "METADATA"
        original = path.read_text()
        assert "Version: 4.0.0" in original
        path.write_text(original.replace("Version: 4.0.0", "Version: 0.0.0"))
        expected = "unsupported FUSION burn provider version/source"
    else:
        path = site / "scpn_fusion/core/dynamic_burn_model.py"
        path.write_bytes(path.read_bytes() + b"\n# distinct source identity\n")
        expected = "does not match the immutable pin"
    script = (
        "import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
        "from validation.burn_model_provider import evaluate_burn_model; "
        "from tools.ci_rmse_gate import main; "
        "r=json.loads(Path(sys.argv[2]).read_text()); "
        "evaluate_burn_model(r)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(site), str(ROOT / "validation/reference_data/iter_reference.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0 and expected in result.stderr
    assert "reference values cannot replace model predictions" in result.stderr
