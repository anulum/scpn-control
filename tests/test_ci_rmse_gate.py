# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CI RMSE gate tests.
"""Exercise required carrier/threshold contracts through the real RMSE CLI.

Real producer calls on tracked reference inputs supply the JSON shape. The
positive consumer boundary case inserts the exact beta threshold explicitly;
it does not claim an available model prediction. Separate producer tests keep
unavailable models from regenerating reference-as-prediction values. Mutation
tests establish consumer refusal, not physics or facility validity.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci_rmse_gate import THRESHOLDS, main
from validation.rmse_dashboard import beta_rmse_iter_sparc, confinement_rmse_itpa, sparc_axis_rmse
from validation.validate_real_shots import validate_disruption

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def disruption_report() -> dict[str, object]:
    """Run the actual predictor over the repository synthetic disruption files."""
    result: dict[str, object] = validate_disruption(ROOT / "validation/reference_data/diiid/disruption_shots")
    n_safe = result["n_safe"]
    assert isinstance(n_safe, int) and n_safe > 0
    return {"schema": "scpn-control.reference-evidence-validation.v1", "lanes": {"disruption_synthetic": result}}


@pytest.fixture
def carriers(tmp_path: Path, disruption_report: dict[str, object]) -> Path:
    """Construct real producer lanes and an explicit threshold-bound consumer case."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    report = {
        "schema": "scpn-control.rmse-dashboard.v1",
        "confinement_itpa": confinement_rmse_itpa(ROOT / "validation/reference_data/itpa/hmode_confinement.csv"),
        "sparc_axis": sparc_axis_rmse(ROOT / "validation/reference_data/sparc"),
        "beta_iter_sparc": beta_rmse_iter_sparc(ROOT / "validation/reference_data", ROOT / "validation"),
    }
    # This consumer boundary case substitutes a supplied threshold value for
    # the unavailable lane; it is not evidence for a model prediction.
    report["beta_iter_sparc"] = {"count": 2, "beta_n_rmse": THRESHOLDS["beta_iter_sparc_beta_n_rmse"], "rows": []}
    (artifacts / "rmse_dashboard_ci.json").write_text(json.dumps(report))
    (artifacts / "reference_evidence_validation.json").write_text(json.dumps(disruption_report))
    return artifacts


def _cli(artifacts: Path) -> subprocess.CompletedProcess[str]:
    """Execute the production script against the exact selected artifact directory."""
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/ci_rmse_gate.py"), "--artifact-dir", str(artifacts)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _dashboard(artifacts: Path) -> dict[str, object]:
    """Read copied carrier data for explicit consumer-contract corruption cases."""
    result: dict[str, object] = json.loads((artifacts / "rmse_dashboard_ci.json").read_text())
    return result


def _write_dashboard(artifacts: Path, report: dict[str, object]) -> None:
    """Write the copied JSON carrier, preserving the canonical evidence bytes."""
    (artifacts / "rmse_dashboard_ci.json").write_text(json.dumps(report))


def test_gate_reads_bounded_disruption_lane(carriers: Path) -> None:
    """Complete supplied carrier values pass only bounded numerical comparisons."""
    result = _cli(carriers)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  disruption FPR" in result.stdout and "no facility validation admitted" in result.stdout


def test_gate_default_cwd_contract(carriers: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The no-flag CI invocation consumes artifacts relative to its working directory."""
    monkeypatch.chdir(carriers.parent)
    assert main([]) == 0


def test_gate_requires_rmse_dashboard(tmp_path: Path) -> None:
    """Missing primary evidence fails explicitly instead of finding another root's carrier."""
    assert main(["--artifact-dir", str(tmp_path)]) == 1


@pytest.mark.parametrize("which", ["rmse_dashboard_ci.json", "reference_evidence_validation.json"])
def test_gate_requires_both_carriers(carriers: Path, which: str) -> None:
    """Losing either required CI input cannot silently retire its regression lane."""
    (carriers / which).rename(carriers / (which + ".saved"))
    assert _cli(carriers).returncode == 1


@pytest.mark.parametrize("lane", ["confinement_itpa", "sparc_axis", "beta_iter_sparc"])
@pytest.mark.parametrize("defect", ["absent", "empty", "zero_count", "bool_count", "skipped", "error"])
def test_gate_refuses_incomplete_dashboard_lane(carriers: Path, lane: str, defect: str) -> None:
    """Missing, unpopulated or unavailable producer lanes never become zero-error passes."""
    report = _dashboard(carriers)
    selected = report[lane]
    assert isinstance(selected, dict)
    if defect == "absent":
        report.pop(lane)
    elif defect == "empty":
        report[lane] = {}
    elif defect == "zero_count":
        selected["count"] = 0
    elif defect == "bool_count":
        selected["count"] = True
    elif defect == "skipped":
        selected["skipped"] = True
    else:
        selected["error"] = "producer failure"
    _write_dashboard(carriers, report)
    result = _cli(carriers)
    assert result.returncode == 1 and lane in result.stdout


@pytest.mark.parametrize("value", [None, True, "0", -0.1, float("nan"), float("inf"), 10**400])
def test_gate_refuses_nonfinite_or_nonphysical_metric(carriers: Path, value: object) -> None:
    """Null/coerced/negative/nonfinite/overflow RMSE values cannot admit a carrier."""
    report = _dashboard(carriers)
    selected = report["confinement_itpa"]
    assert isinstance(selected, dict)
    selected["tau_rmse_s"] = value
    _write_dashboard(carriers, report)
    assert _cli(carriers).returncode == 1


@pytest.mark.parametrize("defect", ["schema", "lanes", "missing_lane", "missing_fpr", "zero_safe", "range"])
def test_gate_refuses_invalid_reference_contract(carriers: Path, defect: str) -> None:
    """An absent safe-shot denominator or malformed bounded lane cannot manufacture FPR zero."""
    path = carriers / "reference_evidence_validation.json"
    report = json.loads(path.read_text())
    lane = report["lanes"]["disruption_synthetic"]
    if defect == "schema":
        report["schema"] = "unknown"
    elif defect == "lanes":
        report["lanes"] = []
    elif defect == "missing_lane":
        report["lanes"].pop("disruption_synthetic")
    elif defect == "missing_fpr":
        lane.pop("false_positive_rate")
    elif defect == "zero_safe":
        lane["n_safe"] = 0
    else:
        lane["false_positive_rate"] = 1.1
    path.write_text(json.dumps(report))
    assert _cli(carriers).returncode == 1


def test_gate_reports_every_bounded_rmse_regression(carriers: Path) -> None:
    """All four independent threshold regressions are reported in one completed scan."""
    report = _dashboard(carriers)
    for lane, metric, threshold in (
        ("confinement_itpa", "tau_rmse_s", "confinement_itpa_tau_rmse_s"),
        ("sparc_axis", "axis_rmse_m", "sparc_axis_rmse_m"),
        ("beta_iter_sparc", "beta_n_rmse", "beta_iter_sparc_beta_n_rmse"),
    ):
        selected = report[lane]
        assert isinstance(selected, dict)
        selected[metric] = THRESHOLDS[threshold] * 2
    _write_dashboard(carriers, report)
    path = carriers / "reference_evidence_validation.json"
    reference = json.loads(path.read_text())
    reference["lanes"]["disruption_synthetic"]["false_positive_rate"] = 0.5
    path.write_text(json.dumps(reference))
    result = _cli(carriers)
    assert result.returncode == 1
    assert result.stdout.count("  FAIL  ") == 4


@pytest.mark.parametrize(
    "raw", ["{}", "[]", "not JSON", '{"beta_iter_sparc": {}, "beta_iter_sparc": {}}', '{"metadata": 1e9999}']
)
def test_gate_refuses_invalid_json_content(carriers: Path, raw: str) -> None:
    """Root type, duplicate keys and nonfinite tokens are refused before comparison."""
    (carriers / "rmse_dashboard_ci.json").write_text(raw)
    assert _cli(carriers).returncode == 1


def test_gate_refuses_unversioned_legacy_dashboard(carriers: Path) -> None:
    """Legacy producers could return reference-as-prediction values without marking unavailability."""
    report = _dashboard(carriers)
    report.pop("schema")
    _write_dashboard(carriers, report)
    result = _cli(carriers)
    assert result.returncode == 1 and "regenerate" in result.stdout
