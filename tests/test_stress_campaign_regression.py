# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Stress Campaign Regression.
"""The stress campaign against the real equilibrium kernel and a repository ITER configuration.

With the repository's reduced ITER test configuration the flight simulator's
solved flux maximum leaves the interior after the first control step, so the
checked magnetic-axis reader refuses the shot. The campaign must then exit 1,
name the refusal and write no report, instead of the zero-filled metrics it
used to write. When the equilibrium is repaired this test fails and must be
replaced by a gate on the measured controller metrics.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.benchmark
def test_stress_campaign_refuses_the_repository_config_without_an_interior_axis(tmp_path: Path) -> None:
    """Two real episodes per controller: the first PID episode is refused and nothing is written."""
    report_path = tmp_path / "stress_report.json"
    command = [
        sys.executable,
        str(ROOT / "validation" / "stress_test_campaign.py"),
        "--episodes",
        "2",
        "--duration",
        "30",
        "--config",
        str(ROOT / "tests" / "iter_config_temp.json"),
        "--output",
        str(report_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Stress campaign refused: PID episode 0 failed: MagneticAxisError" in result.stderr
    assert "is not an interior O-point" in result.stderr
    assert not report_path.exists()
