# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — RMSE report rendering contracts.
"""Exercise supplied real dashboard carriers through Markdown/PNG/native surfaces."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from validation import rmse_dashboard, rmse_dashboard_rendering

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def actual_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Generate a real canonical CLI report once with caller-owned output paths."""
    directory = tmp_path_factory.mktemp("actual-dashboard-rendering-input")
    output = directory / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "validation/rmse_dashboard.py"),
            "--output-json",
            str(output),
            "--output-md",
            str(directory / "report.md"),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report: dict[str, Any] = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "scpn-control.rmse-dashboard.v2"
    beta = report["beta_iter_sparc"]
    if beta.get("skipped", False):
        assert beta["rows"] == [] and beta["beta_n_rmse"] is None
    else:
        assert beta["count"] == 2 and beta["physical_validation_claim"] is False
    return report


def test_facade_and_owner_render_the_same_actual_carrier(actual_report: dict[str, Any]) -> None:
    """Public facade/owner emit identical Markdown with explicit unavailable beta status."""
    text = rmse_dashboard.render_markdown(actual_report, plot_dir="figures")
    assert text == rmse_dashboard_rendering.render_markdown(actual_report, plot_dir="figures")
    if actual_report["beta_iter_sparc"].get("skipped", False):
        assert "No beta_N model prediction was available" in text
        assert "beta_n_scatter.png" not in text
    else:
        assert "indicative target" in text
    assert "figures/tau_e_scatter.png" in text
    assert f"{actual_report['sparc_axis']['axis_rmse_m']:.6f}" in text
    assert f"{actual_report['confinement_itpa']['tau_rmse_s']:.4f} s" in text


def test_plot_owner_writes_actual_reference_charts_and_closes_figures(
    actual_report: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Render real confinement/axis rows into PNGs and preserve other live figures."""
    import matplotlib.image as image
    import matplotlib.pyplot as plt

    caller_figure = plt.figure()
    try:
        before = plt.get_fignums()
        paths = rmse_dashboard_rendering.render_plots(actual_report, tmp_path / "figures")
        expected = ["tau_e_scatter.png", "sparc_axis_error.png"]
        if not actual_report["beta_iter_sparc"].get("skipped", False):
            expected.insert(1, "beta_n_scatter.png")
        assert [p.name for p in paths] == expected
        for path in paths:
            shape = (600, 1200) if path.name == "sparc_axis_error.png" else (900, 900)
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            pixels = image.imread(path)
            assert pixels.shape[:2] == shape
            assert float(pixels.min()) < float(pixels.max())
        assert plt.get_fignums() == before
        assert (tmp_path / "figures/beta_n_scatter.png").exists() is (
            not actual_report["beta_iter_sparc"].get("skipped", False)
        )
    finally:
        plt.close(caller_figure)


def test_renderer_without_matplotlib_uses_real_dependency_absence(
    actual_report: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Run the actual presentation owner under -S, where optional Matplotlib is absent."""
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(actual_report), encoding="utf-8")
    code = (
        "import json,sys; from pathlib import Path; "
        "from validation.rmse_dashboard_rendering import render_plots,render_markdown; "
        "r=json.loads(Path(sys.argv[1]).read_text()); "
        "print(render_plots(r,Path(sys.argv[2]))); print(render_markdown(r))"
    )
    output_dir = tmp_path / "plots"
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(report_path), str(output_dir)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("[]\n# SCPN RMSE Dashboard")
    assert (
        "No beta_N model prediction was available" in result.stdout
        if actual_report["beta_iter_sparc"].get("skipped", False)
        else "indicative target" in result.stdout
    )
    assert not output_dir.exists()


def test_native_facade_exposes_imported_owning_contracts() -> None:
    """Actual pydoc rendering exposes moved public functions through the facade."""
    import html
    import pydoc
    import re

    rendered = pydoc.HTMLDoc().docmodule(rmse_dashboard)
    text = " ".join(html.unescape(re.sub("<[^>]+>", " ", rendered)).split())
    for expected in ("ipb98_tau_e", "render_plots", "parse_args", "main", "closed even", "reference"):
        assert expected in text


def test_markdown_refuses_missing_required_carrier_without_mutating_input(actual_report: dict[str, Any]) -> None:
    """Removing the real required confinement lane raises the owning public error."""
    report = copy.deepcopy(actual_report)
    del report["confinement_itpa"]
    before = copy.deepcopy(report)
    with pytest.raises(KeyError, match="confinement_itpa"):
        rmse_dashboard_rendering.render_markdown(report)
    assert report == before


def _section(text: str, heading: str) -> str:
    """Return the Markdown body under one level-2 heading, up to the next one."""
    return text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0].strip()


def test_display_flags_follow_each_threshold_form() -> None:
    """Unknown keys have no flag; ranges pass inclusively; lower-better values pass, warn, then fail."""
    flag = rmse_dashboard_rendering._flag
    assert flag(1.0, "no_such_metric") == ""
    assert [flag(value, "tbr") for value in (1.0, 1.2, 1.4, 0.99, 1.41)] == ["PASS", "PASS", "PASS", "FAIL", "FAIL"]
    assert [flag(value, "sparc_axis_rmse_m") for value in (0.08, 0.09, 0.10, 0.11)] == ["PASS", "WARN", "WARN", "FAIL"]
    assert flag(float("nan"), "tbr") == "FAIL" and flag(float("nan"), "sparc_axis_rmse_m") == "FAIL"


def test_unavailable_lanes_render_their_reasons_and_no_values(actual_report: dict[str, Any]) -> None:
    """Skipped beta, point-wise psi and forward lanes print their reasons, never zero errors."""
    report = copy.deepcopy(actual_report)
    report["beta_iter_sparc"] = {
        "count": 0,
        "beta_n_rmse": None,
        "rows": [],
        "skipped": True,
        "reason": "burn provider refused the run",
    }
    report["sparc_psi_rmse"] = {"skipped": True, "reason": "point-wise psi module absent"}
    report["forward_diagnostics"] = {
        "count_interferometer_channels": 0,
        "phase_rmse_rad": None,
        "neutron_rate_rel_error_pct": None,
        "skipped": True,
        "reason": "forward diagnostics module unavailable",
    }
    text = rmse_dashboard_rendering.render_markdown(report, plot_dir="figures")
    beta = _section(text, "Beta_N RMSE (ITER + SPARC references)")
    assert beta == "- Samples: `0`\n- Unavailable: burn provider refused the run"
    assert "beta_n_scatter.png" not in text
    assert "No beta_N model prediction was available" in _section(text, "Notes")
    assert _section(text, "SPARC Point-wise psi(R,Z) RMSE") == "- Unavailable: point-wise psi module absent"
    assert _section(text, "Forward Diagnostics RMSE") == "- Unavailable: forward diagnostics module unavailable"


def test_supplied_psi_lane_renders_and_an_absent_forward_lane_is_left_out(actual_report: dict[str, Any]) -> None:
    """A point-wise psi report renders its five values; a report without a forward lane has no forward section."""
    report = copy.deepcopy(actual_report)
    report["sparc_psi_rmse"] = {
        "count": 2,
        "mean_psi_rmse_norm": 0.0125,
        "mean_psi_relative_l2": 0.25,
        "mean_gs_residual_l2": 0.5,
        "worst_file": "lmode_vh.geqdsk",
        "worst_psi_rmse_norm": 0.02,
    }
    del report["forward_diagnostics"]
    text = rmse_dashboard_rendering.render_markdown(report)
    assert _section(text, "SPARC Point-wise psi(R,Z) RMSE").split("\n") == [
        "- Files: `2`",
        "- Mean normalised psi RMSE: `0.012500`",
        "- Mean relative L2: `0.250000`",
        "- Mean GS residual (rel L2): `0.5000`",
        "- Worst file: `lmode_vh.geqdsk` (psi_N RMSE = `0.020000`)",
    ]
    assert "## Forward Diagnostics RMSE" not in text


def test_forward_lane_without_thomson_fields_renders_its_three_values(actual_report: dict[str, Any]) -> None:
    """A supplied forward lane without Thomson fields renders the interferometer and neutron values only."""
    report = copy.deepcopy(actual_report)
    report["forward_diagnostics"] = {
        "count_interferometer_channels": 3,
        "phase_rmse_rad": 1.5e-3,
        "neutron_rate_rel_error_pct": 3.0,
    }
    section = _section(rmse_dashboard_rendering.render_markdown(report), "Forward Diagnostics RMSE")
    assert section.split("\n") == [
        "- Interferometer channels: `3`",
        "- Interferometer phase RMSE: `1.500000e-03 rad`",
        "- Neutron-count relative error: `3.000%`",
    ]


def test_real_forward_lane_renders_its_values_or_its_reason(actual_report: dict[str, Any]) -> None:
    """The generated report's forward lane renders its computed errors, or its reason when skipped."""
    forward = actual_report["forward_diagnostics"]
    section = _section(rmse_dashboard_rendering.render_markdown(actual_report), "Forward Diagnostics RMSE")
    if forward.get("skipped", False):
        assert section == f"- Unavailable: {forward['reason']}"
        return
    assert f"- Interferometer phase RMSE: `{forward['phase_rmse_rad']:.6e} rad`" in section
    assert f"- Neutron-count relative error: `{forward['neutron_rate_rel_error_pct']:.3f}%`" in section
    assert f"- Thomson voltage RMSE: `{forward['thomson_voltage_rmse_v']:.6e} V`" in section


def test_plots_skip_every_lane_without_rows(actual_report: dict[str, Any], tmp_path: Path) -> None:
    """Lanes without rows draw nothing, so the created directory stays empty."""
    report = copy.deepcopy(actual_report)
    for lane in ("confinement_itpa", "beta_iter_sparc", "sparc_axis"):
        report[lane]["rows"] = []
    assert rmse_dashboard_rendering.render_plots(report, tmp_path / "figures") == []
    assert list((tmp_path / "figures").iterdir()) == []
