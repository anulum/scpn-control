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
    assert report["schema"] == "scpn-control.rmse-dashboard.v1"
    assert report["beta_iter_sparc"]["skipped"] and report["beta_iter_sparc"]["rows"] == []
    return report


def test_facade_and_owner_render_the_same_actual_carrier(actual_report: dict[str, Any]) -> None:
    """Public facade/owner emit identical Markdown with explicit unavailable beta status."""
    text = rmse_dashboard.render_markdown(actual_report, plot_dir="figures")
    assert text == rmse_dashboard_rendering.render_markdown(actual_report, plot_dir="figures")
    assert "No beta_N model prediction was available" in text
    assert "beta_n_scatter.png" not in text
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
        assert [p.name for p in paths] == ["tau_e_scatter.png", "sparc_axis_error.png"]
        for path, shape in zip(paths, [(900, 900), (600, 1200)]):
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            pixels = image.imread(path)
            assert pixels.shape[:2] == shape
            assert float(pixels.min()) < float(pixels.max())
        assert plt.get_fignums() == before
        assert not (tmp_path / "figures/beta_n_scatter.png").exists()
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
    assert "Unavailable: burn model unavailable" in result.stdout
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
