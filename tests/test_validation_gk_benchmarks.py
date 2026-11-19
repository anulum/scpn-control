# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Linear GK and hybrid-accuracy benchmark producers.
"""The linear GK suite and the hybrid surrogate+GK benchmark run and report what they compute.

Each run moves the report directory into a temporary directory, which the
recorded-campaign guard treats as temporary output; without a campaign the
script entries refuse the committed report location.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scpn_control.benchmark_records import CAMPAIGN_ENV

gk_linear = importlib.import_module("validation.benchmark_gk_linear")
hybrid = importlib.import_module("validation.benchmark_hybrid_accuracy")


def test_linear_suite_writes_the_four_cases_it_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite report holds the CBC, scan, model comparison and SPARC/ITER cases, as returned."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    monkeypatch.setattr(gk_linear, "_REPORT_DIR", tmp_path / "reports")

    report = gk_linear.run_all()

    assert json.loads((tmp_path / "reports" / "gk_linear_benchmark.json").read_text(encoding="utf-8")) == report
    cbc = report["cyclone_base"]
    assert cbc["gamma_max"] > 0.0
    assert cbc["gamma_max"] == max(cbc["gamma"])
    assert cbc["k_y_max"] == cbc["k_y"][cbc["gamma"].index(cbc["gamma_max"])]
    assert cbc["dominant_mode"] != "stable"
    assert len(cbc["k_y"]) == cbc["n_ky"] == 12
    scan = report["critical_gradient"]
    assert len(scan["R_L_Ti"]) == len(scan["gamma_max"]) == 12
    assert set(report["multi_code"]) == {"case", "native_gk", "quasilinear"}
    for device in ("SPARC", "ITER"):
        assert all(math.isfinite(value) for value in report["sparc_iter"]["results"][device].values())


def test_stable_spectrum_is_labelled_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no positive growth rate the CBC case reports a stable dominant mode and no transport."""
    real_solve = gk_linear.solve_linear_gk

    def damped_solve(*args: Any, **kwargs: Any) -> Any:
        """Return the real spectrum with every growth rate set to zero."""
        result = real_solve(*args, **kwargs)
        return dataclasses.replace(result, gamma=np.zeros_like(result.gamma))

    monkeypatch.setattr(gk_linear, "solve_linear_gk", damped_solve)
    case = gk_linear.run_cyclone_base_case(n_ky=2, n_theta=16)
    assert case["dominant_mode"] == "stable"
    assert case["gamma_max"] == 0.0
    assert case["chi_i_m2s"] == 0.0


def test_hybrid_benchmark_spot_checks_on_the_scheduled_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Over three steps the periodic scheduler verifies only the third, at its three anchor surfaces."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    monkeypatch.setattr(hybrid, "_REPORT_DIR", tmp_path / "reports")

    result = hybrid.run_hybrid_accuracy_benchmark(n_steps=3)

    written = json.loads((tmp_path / "reports" / "hybrid_accuracy_benchmark.json").read_text(encoding="utf-8"))
    assert written == result
    verification = result["verification_report"]
    assert (verification["total_steps"], verification["steps_verified"]) == (3, 1)
    assert verification["total_spot_checks"] == verification["n_correction_records"] == 3
    for key in ("mean_surrogate_error", "mean_corrected_error", "final_surrogate_error", "final_corrected_error"):
        assert math.isfinite(result[key]) and result[key] >= 0.0
    surrogate = result["final_surrogate_error"]
    assert result["mean_surrogate_error"] == pytest.approx(surrogate, rel=1e-12)
    # Before the first spot-check the correction is the identity, so steps 1 and 2 keep the surrogate error.
    assert result["mean_corrected_error"] == pytest.approx((2 * surrogate + result["final_corrected_error"]) / 3)


@pytest.mark.parametrize(
    ("module", "report_name"),
    [(gk_linear, "gk_linear_benchmark.json"), (hybrid, "hybrid_accuracy_benchmark.json")],
)
def test_script_entry_refuses_the_committed_report_without_a_campaign(
    module: Any, report_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Executed as a script, each benchmark stops at the recorded-campaign guard before computing."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    report = Path(str(module.__file__)).parent / "reports" / report_name
    before = report.read_bytes() if report.exists() else None
    with pytest.raises(RuntimeError, match="requires tools/run_recorded_benchmark.py"):
        runpy.run_path(str(module.__file__), run_name="__main__")
    assert (report.read_bytes() if report.exists() else None) == before
