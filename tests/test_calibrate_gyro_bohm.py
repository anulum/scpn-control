# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — IPB98(y,2)-form confinement prefactor fit.
"""The prefactor fit reproduces its recorded result and never overwrites the solvers' c_gB file."""

from __future__ import annotations

import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import pytest

from tools import calibrate_gyro_bohm as fit

HEADER = "machine,shot,Ip_MA,BT_T,ne19_1e19m3,Ploss_MW,R_m,a_m,kappa,delta,M_AMU,tau_E_s,H98y2,source\n"
JET_ROW = "JET,92436,2.5,2.7,7.0,10.0,2.96,0.95,1.68,0.24,2.0,{tau},0.95,Challis_NF_2015\n"


def _csv(tmp_path: Path, *rows: str) -> Path:
    """Write a CSV with the real header and the given data rows."""
    path = tmp_path / "rows.csv"
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def _jet() -> fit.ShotRecord:
    """Return the first JET row of the shipped CSV."""
    return next(shot for shot in fit.load_itpa_csv() if shot.shot == "92436")


def test_shipped_csv_fit_reproduces_the_recorded_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The 20-row fit gives the recorded prefactor, uncertainty and errors, and writes exactly the record."""
    output = tmp_path / "out" / "fit.json"
    record = fit.calibrate(output_path=output, verbose=True)

    assert record["ipb98_prefactor"] == 0.077848
    assert record["ipb98_prefactor_uncertainty"] == 0.005036
    assert record["rmse_s"] == 0.058244
    assert record["rmse_percent"] == 12.9532
    assert record["log_rmse"] == 0.1742
    assert record["mape_percent"] == 30.83
    assert record["n_shots"] == 20
    assert record["exponents"] == fit.IPB98_EXPONENTS
    assert "not the local gyro-Bohm coefficient c_gB" in str(record["claim_boundary"])
    assert output.read_text(encoding="utf-8") == json.dumps(record, indent=2) + "\n"
    lines = capsys.readouterr().out.splitlines()
    rows = lines[lines.index("-" * 90) + 1 : lines.index("=" * 90, lines.index("-" * 90))]
    assert len(rows) == 20
    assert rows[0].split()[:2] == ["ITER", "design"]


def test_solver_coefficient_file_is_never_written(tmp_path: Path) -> None:
    """The dimensional prefactor cannot replace the solvers' local c_gB, by default or by request."""
    before = hashlib.sha256(fit.SOLVER_COEFFICIENT_JSON.read_bytes()).hexdigest()
    assert fit.OUTPUT_JSON != fit.SOLVER_COEFFICIENT_JSON
    with pytest.raises(ValueError, match="not the local gyro-Bohm coefficient c_gB"):
        fit.calibrate(output_path=fit.SOLVER_COEFFICIENT_JSON, verbose=False)
    assert fit.main(["--quiet", "--output", str(fit.SOLVER_COEFFICIENT_JSON)]) == 2
    assert hashlib.sha256(fit.SOLVER_COEFFICIENT_JSON.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    ("row", "message"),
    [
        pytest.param(JET_ROW.format(tau="x"), "line 2: column tau_E_s is missing or not a number", id="text"),
        pytest.param(JET_ROW.format(tau="nan"), "line 2: column tau_E_s is not finite", id="nan"),
        pytest.param(JET_ROW.format(tau="0.0"), "line 2: column tau_E_s must be positive", id="zero"),
        pytest.param("JET,92436,2.5\n", "line 2: column", id="short-row"),
    ],
)
def test_loader_refuses_unusable_cells(tmp_path: Path, row: str, message: str) -> None:
    """Every numeric cell must be finite, and positive where the form takes a power of it."""
    with pytest.raises(ValueError, match=message):
        fit.load_itpa_csv(_csv(tmp_path, row))


def test_loader_refuses_a_file_without_rows_and_keeps_negative_triangularity(tmp_path: Path) -> None:
    """A header-only file is refused; triangularity, which the form does not use, may be negative."""
    with pytest.raises(ValueError, match="no data rows"):
        fit.load_itpa_csv(_csv(tmp_path))
    shots = fit.load_itpa_csv(_csv(tmp_path, JET_ROW.format(tau="0.41").replace(",0.24,", ",-0.3,")))
    assert shots[0].delta == -0.3
    assert shots[0].source == "Challis_NF_2015"


def test_shape_is_the_published_power_law() -> None:
    """The shape factor is the IPB98(y,2) product of the row, without a prefactor."""
    shot = _jet()
    expected = (
        2.5**0.93 * 2.7**0.15 * 7.0**0.41 * 10.0**-0.69 * 2.96**1.97 * 1.68**0.78 * (0.95 / 2.96) ** 0.58 * 2.0**0.19
    )
    assert fit.ipb98_shape(shot) == pytest.approx(expected, rel=1e-14)


def test_neoclassical_factor_is_mostly_at_its_floor_for_the_shipped_rows() -> None:
    """At the fitted prefactor, 16 of the 20 rows sit at the 0.5 floor; the other four lie inside (0.5, 1]."""
    factors = {
        (shot.machine, shot.shot): fit.neoclassical_factor(shot, 0.077848 * fit.ipb98_shape(shot))
        for shot in fit.load_itpa_csv()
    }
    inside = {key for key, value in factors.items() if value > 0.5}
    assert inside == {("ITER", "design"), ("JET", "DTE2"), ("JT-60U", "48158"), ("SPARC", "design")}
    assert all(0.5 < factors[key] <= 1.0 for key in inside)
    assert sum(value == 0.5 for value in factors.values()) == 16
    shot = _jet()
    tau = 0.077848 * fit.ipb98_shape(shot)
    assert fit.predict_tau_e(shot, 0.077848) == pytest.approx(0.5 * tau)
    assert fit.predict_tau_e(shot, 1e-12) == 1e-4


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param((0.0, 5.0, 5.0, 2.0, 3.0, 1.0, 5.0), id="radius"),
        pytest.param((0.5, 0.0, 5.0, 2.0, 3.0, 1.0, 5.0), id="temperature"),
        pytest.param((0.5, 5.0, 0.0, 2.0, 3.0, 1.0, 5.0), id="density"),
        pytest.param((0.5, 5.0, 5.0, 0.0, 3.0, 1.0, 5.0), id="safety-factor"),
        pytest.param((0.5, 5.0, 5.0, 2.0, 3.0, 1.0, 0.0), id="field"),
        pytest.param((1e-7, 5.0, 5.0, 2.0, 3.0, 1.0, 5.0), id="tiny-aspect-ratio"),
        pytest.param((0.5, 5.0, 1e300, 2.0, 3.0, 1.0, 5.0), id="overflow"),
    ],
)
def test_chang_hinton_guards_return_the_floor(arguments: tuple[float, ...]) -> None:
    """Unphysical inputs, a vanishing inverse aspect ratio and overflow all give the 0.01 floor."""
    assert fit.chang_hinton_chi_scalar(*arguments) == 0.01


def test_chang_hinton_matches_its_closed_form() -> None:
    """At a regular point the helper returns the documented Chang-Hinton expression."""
    m_i = 2.0 * fit.M_PROTON
    t_j = 2.0 * fit.E_KEV_J
    v_ti = math.sqrt(2.0 * t_j / m_i)
    rho_i = m_i * v_ti / (fit.E_CHARGE * 2.0)
    eps = 0.5 * 0.6 / 1.7
    nu_ii = 3e19 * 1.5**2 * fit.E_CHARGE**4 * 17.0 / (12.0 * math.pi**1.5 * fit.EPS_0**2 * m_i**0.5 * t_j**1.5)
    nu_star = nu_ii * 3.0 * 1.7 / (eps**1.5 * v_ti)
    chi = 0.66 * (1 + 1.54 * eps) * 9.0 * rho_i**2 * nu_ii / (eps**1.5 * (1 + 0.74 * nu_star ** (2 / 3)))
    assert fit.chang_hinton_chi_scalar(0.5, 2.0, 3.0, 3.0, 1.7, 0.6, 2.0) == pytest.approx(max(chi, 0.01), rel=1e-12)


def test_metrics_and_bootstrap_are_deterministic() -> None:
    """The objective is the RMSE in log10 space and equal seeds give equal bootstrap spreads."""
    shots = fit.load_itpa_csv()
    assert fit.objective(math.log10(0.05), shots) == pytest.approx(fit.compute_rmse(0.05, shots))
    assert fit.compute_log_rmse(0.077848, shots) == pytest.approx(0.1742, abs=5e-5)
    first = fit.bootstrap_uncertainty(shots, 0.077848, n_bootstrap=5, rng_seed=3)
    assert first == fit.bootstrap_uncertainty(shots, 0.077848, n_bootstrap=5, rng_seed=3)
    assert first > 0.0


def test_command_reports_pass_fail_and_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exit 0 below the 15 % target, 1 above it, 2 for a refused CSV."""
    assert fit.main(["--quiet", "--output", str(tmp_path / "pass.json")]) == 0
    assert "PASS: relative RMSE 12.95 % < 15 %" in capsys.readouterr().out

    inconsistent = _csv(tmp_path, JET_ROW.format(tau="0.05"), JET_ROW.format(tau="2.0"))
    assert fit.main(["--quiet", "--csv", str(inconsistent), "--output", str(tmp_path / "fail.json")]) == 1
    assert "FAIL: relative RMSE" in capsys.readouterr().out

    assert fit.main(["--quiet", "--csv", str(tmp_path / "missing.csv"), "--output", str(tmp_path / "x.json")]) == 2
    assert capsys.readouterr().err.startswith("refused: ")


def test_script_entry_exits_with_the_status_of_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Executed as a script, the module exits with main's status."""
    script = Path(fit.__file__)
    monkeypatch.setattr(sys, "argv", [str(script), "--quiet", "--output", str(tmp_path / "fit.json")])
    with pytest.raises(SystemExit) as script_exit:
        runpy.run_path(str(script), run_name="__main__")
    assert script_exit.value.code == 0
    assert json.loads((tmp_path / "fit.json").read_text(encoding="utf-8"))["n_shots"] == 20
