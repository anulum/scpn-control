# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Transport Benchmark.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Transport Validation Benchmark Tests
# © 1996–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Regression tests for the transport validation benchmark."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

import pytest

import validation.benchmark_transport as bench
from scpn_control.benchmark_records import CAMPAIGN_ENV
from scpn_control.core.scaling_laws import ipb98y2_with_uncertainty
from validation.benchmark_transport import (
    PURE_DIFFUSION_MAX_RELATIVE_ERROR,
    run_pure_diffusion_benchmark,
    run_threshold_benchmark,
)


def test_transport_pure_diffusion() -> None:
    """The ion channel matches its half-power cylindrical diffusion reference."""
    result = run_pure_diffusion_benchmark(nr=50)
    assert result["max_relative_error"] < PURE_DIFFUSION_MAX_RELATIVE_ERROR


def test_transport_threshold() -> None:
    """Verify that the transport model respects critical gradients."""
    result = run_threshold_benchmark()
    assert result["pass"]
    assert result["chi_sub"] == 0.0
    assert result["chi_super"] > 0.0


def test_iter_scaling_case_reports_the_ipb98_time_it_evaluates() -> None:
    """The ITER-like case returns the IPB98(y,2) time and uncertainty for its fixed inputs."""
    tau, sigma = ipb98y2_with_uncertainty(15.0, 5.3, 10.0, 50.0, 6.2, 1.7, 2.0 / 6.2, 2.5)
    result = bench.run_iter_scaling_benchmark()
    assert result == {"tau_predicted": tau, "uncertainty_sigma": sigma, "pass": True}


@pytest.mark.parametrize(("tau", "sigma"), [(math.nan, 0.1), (0.0, 0.1), (3.0, -0.1), (3.0, math.inf)])
def test_iter_scaling_case_fails_when_the_scaling_is_unusable(
    tau: float, sigma: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-finite or non-positive time, or an unusable uncertainty, no longer passes."""
    monkeypatch.setattr(bench, "ipb98y2_with_uncertainty", lambda *args: (tau, sigma))
    assert bench.run_iter_scaling_benchmark()["pass"] is False


def test_report_paths_are_anchored_to_the_module() -> None:
    """The report pair lives in this module's reports directory, independent of the working directory."""
    assert bench.REPORT_DIR == Path(bench.__file__).resolve().parent / "reports"
    assert bench.JSON_REPORT.is_absolute() and bench.JSON_REPORT.parent == bench.REPORT_DIR
    assert bench.MARKDOWN_REPORT.parent == bench.REPORT_DIR


def test_main_writes_all_three_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The report pair carries the three cases and the Markdown table rows."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    out = tmp_path / "reports"
    monkeypatch.setattr(bench, "REPORT_DIR", out)
    monkeypatch.setattr(bench, "JSON_REPORT", out / "transport_benchmark.json")
    monkeypatch.setattr(bench, "MARKDOWN_REPORT", out / "transport_benchmark.md")
    monkeypatch.setattr(bench, "run_pure_diffusion_benchmark", lambda: run_pure_diffusion_benchmark(nr=50))

    bench.main()

    report = json.loads((out / "transport_benchmark.json").read_text(encoding="utf-8"))
    assert set(report) == {"pure_diffusion", "threshold", "iter_scaling"}
    assert report["pure_diffusion"]["pass"] and report["threshold"]["pass"] and report["iter_scaling"]["pass"]
    table = (out / "transport_benchmark.md").read_text(encoding="utf-8").splitlines()
    assert table[0] == "# Transport Validation Benchmark"
    assert [row.split("|")[1].strip() for row in table[4:7]] == ["Pure Diffusion", "Threshold", "ITER Scaling"]


def test_script_entry_from_another_directory_still_needs_a_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run from a foreign working directory, the benchmark refuses the committed location and writes nothing there."""
    monkeypatch.delenv(CAMPAIGN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="requires tools/run_recorded_benchmark.py"):
        runpy.run_path(str(bench.__file__), run_name="__main__")
    assert not (tmp_path / "validation").exists()
