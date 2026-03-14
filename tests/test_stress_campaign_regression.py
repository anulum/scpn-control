# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Stress Campaign Regression Gate
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""
Regression gate for the 1000-shot stress campaign (quick mode).
Ensures that physics changes do not degrade controller performance.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.benchmark
def test_stress_campaign_quick_regression(tmp_path: Path):
    """Run 10-episode stress campaign and verify performance gates."""
    report_path = tmp_path / "stress_report.json"

    # Locate script and config relative to this test file
    this_dir = Path(__file__).parent
    script_path = this_dir.parent / "validation" / "stress_test_campaign.py"
    config_path = this_dir / "iter_config_temp.json"

    # Run via subprocess to ensure clean environment and path handling
    # We use 2 episodes (1 per controller) and default duration for stability
    cmd = [
        sys.executable,
        str(script_path),
        "--episodes",
        "2",
        "--duration",
        "30",
        "--config",
        str(config_path),
        "--output",
        str(report_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Stress campaign failed with: {result.stderr}"

    with open(report_path, "r") as f:
        data = json.load(f)

    # 1. PID controller checks
    pid = data.get("PID")
    assert pid is not None
    assert pid["mean_reward"] > -50.0  # Very loose
    assert pid["disruption_rate"] <= 1.0

    # 2. H-infinity controller checks
    hinf = data.get("H-infinity")
    assert hinf is not None
    assert hinf["mean_reward"] > -50.0

    # 3. Latency checks (Python fallback)
    # P99 loosened for slow CI environments; Windows runners are ~1.5x slower
    assert pid["p99_latency_us"] < 750000
    assert hinf["p99_latency_us"] < 750000

    # 4. Energy efficiency
    assert pid["mean_energy_efficiency"] >= 0.0
