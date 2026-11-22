# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — PID and H-infinity comparison tables.
"""The comparison runs the campaign's registered controllers and renders their aggregates.

Shots run through the real flight simulator with a deterministic
equilibrium stand-in passed through its kernel-factory seam; a second
stand-in without an interior axis exercises the refusal.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import validation.controller_comparison as comparison
import validation.stress_test_campaign as campaign


class _CentredPeakKernel:
    """Equilibrium stand-in with one interior flux peak that follows the radial coil."""

    def __init__(self, _config_file: str) -> None:
        """Start from a peak near R = 6.1 m, Z = 0 on a 9 x 9 grid."""
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R = np.linspace(5.8, 6.4, 9)
        self.Z = np.linspace(-0.3, 0.3, 9)
        self.Psi = np.zeros((9, 9))
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        """Move the peak radially with the radial coil current."""
        drive = float(self.cfg["coils"][2]["current"])
        ir = int(np.argmin(np.abs(self.R - (6.1 + 0.05 * np.tanh(drive / 10.0)))))
        self.Psi.fill(-1.0)
        self.Psi[4, ir] = 1.0

    def find_x_point(self, _psi: npt.NDArray[np.float64]) -> tuple[tuple[float, float], float]:
        """Return a fixed X-point."""
        return (float(self.R[-2]), float(self.Z[1])), 0.0


class _EdgePeakKernel(_CentredPeakKernel):
    """Stand-in whose flux maximum lies on the box edge."""

    def solve_equilibrium(self) -> None:
        """Put the maximum in the first radial column."""
        self.Psi.fill(-1.0)
        self.Psi[4, 0] = 1.0


def test_comparison_uses_the_campaign_registry() -> None:
    """Controllers are registered once, in the stress campaign."""
    assert comparison.CONTROLLERS is campaign.CONTROLLERS


def test_comparison_aggregates_both_controllers_and_renders_both_tables() -> None:
    """Both controllers run; the Markdown and LaTeX tables carry one row each."""
    results = comparison.run_comparison(n_episodes=2, shot_duration=6, kernel_factory=_CentredPeakKernel)
    assert list(results) == ["PID", "H-infinity"] and all(m.n_episodes == 2 for m in results.values())
    markdown = comparison.generate_comparison_table(results).splitlines()
    assert len(markdown) == 4 and markdown[2].startswith("| PID") and markdown[3].startswith("| H-infinity")
    latex = comparison.generate_latex_table(results)
    assert latex.startswith(r"\begin{table}[htbp]") and latex.endswith(r"\end{table}")
    rows = [line for line in latex.splitlines() if line.endswith(r"\\") and line.startswith("  ")]
    assert [row.split(" & ")[0].strip() for row in rows] == ["PID", "H-infinity"]
    assert rows[0].split(" & ")[2] == f"{results['PID'].mean_reward:.4f}"


def test_command_prints_the_markdown_table(capsys: pytest.CaptureFixture[str]) -> None:
    """The quick command runs five episodes per controller and prints the table."""
    assert comparison.main(["--quick"], kernel_factory=_CentredPeakKernel) == 0
    printed = capsys.readouterr().out
    assert "Episodes: 5" in printed and "| PID" in printed and "| H-infinity" in printed


def test_command_refuses_when_an_episode_has_no_interior_axis(capsys: pytest.CaptureFixture[str]) -> None:
    """A refused equilibrium makes the command exit 1 and print no table."""
    assert comparison.main(["--episodes", "1"], kernel_factory=_EdgePeakKernel) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Controller comparison refused: PID episode 0 failed: MagneticAxisError")
    assert "| PID" not in captured.out


def test_script_run_refuses_the_repository_test_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as a script with the real kernel and the reduced ITER configuration, the comparison exits 1."""
    root = Path(__file__).resolve().parents[1]
    script = root / "validation" / "controller_comparison.py"
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys, "argv", [str(script), "--episodes", "1", "--config", str(root / "tests" / "iter_config_temp.json")]
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 1
    assert "Controller comparison refused: PID episode 0 failed: MagneticAxisError" in capsys.readouterr().err
