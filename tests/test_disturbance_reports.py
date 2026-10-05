# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disturbance report figure and output lifecycle tests.

"""Exercise real plotting failures and partial outputs from actual PID traces."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from validation.benchmark_disturbance_rejection import SCENARIOS, PIDController, run_scenario
from validation.disturbance_reports import save_overlay_plots
from validation.disturbance_runtime import ScenarioMetrics, TraceData


def _observations() -> tuple[list[ScenarioMetrics], dict[tuple[str, str], TraceData]]:
    """Run two finite public PID scenarios with distinct real output filenames."""
    metrics: list[ScenarioMetrics] = []
    traces: dict[tuple[str, str], TraceData] = {}
    for name in ("VDE", "ELM pacing"):
        configuration = dict(SCENARIOS[name])
        configuration["duration_s"] = 0.002
        row, trace = run_scenario("PID", PIDController(), name, configuration)
        assert row.stable and row.completed_steps == 20
        metrics.append(row)
        traces[("PID", name)] = trace
    return metrics, traces


@pytest.mark.parametrize("blocked_name", ["benchmark_vde.png", "benchmark_elm_pacing.png"])
def test_plot_write_failure_closes_figures_and_retains_earlier_outputs(tmp_path: Path, blocked_name: str) -> None:
    """A genuine directory collision fails without leaked figures or rollback of earlier plots."""
    metrics, traces = _observations()
    blocked = tmp_path / blocked_name
    blocked.mkdir()
    before = plt.get_fignums()
    for _ in range(2):
        with pytest.raises(OSError):
            save_overlay_plots(traces, metrics, tmp_path)
        assert plt.get_fignums() == before
        assert blocked.is_dir() and not list(blocked.iterdir())
        if blocked_name == "benchmark_elm_pacing.png":
            assert (tmp_path / "benchmark_vde.png").read_bytes().startswith(b"\x89PNG")
        else:
            assert not (tmp_path / "benchmark_elm_pacing.png").exists()


def test_real_overlay_images_are_readable_and_release_figures(tmp_path: Path) -> None:
    """Actual healthy reports return ordered, nonblank PNGs and preserve other figure ownership."""
    metrics, traces = _observations()
    retained = plt.figure()
    try:
        before = plt.get_fignums()
        paths = save_overlay_plots(traces, metrics, tmp_path)
        assert [Path(p).name for p in paths] == ["benchmark_vde.png", "benchmark_elm_pacing.png"]
        assert plt.get_fignums() == before
        for path in paths:
            image = plt.imread(path)
            assert image.ndim == 3 and image.shape[0] > 0 and image.shape[1] > 0
            assert np.isfinite(image).all() and np.ptp(image[:, :, :3]) > 0.0
    finally:
        plt.close(retained)
