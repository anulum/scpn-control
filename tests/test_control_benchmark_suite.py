# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Control Benchmark Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path


from validation.control_benchmark_suite import (
    BenchmarkRunner,
    PIDWrapper,
    setpoint_tracking,
)


def test_benchmark_runner(tmp_path: Path):
    scenarios = [setpoint_tracking()]
    controllers = [PIDWrapper(Kp=2.0, Ki=1.0), PIDWrapper(Kp=0.5, Ki=0.1)]

    # Differentiate names for the test
    controllers[0].name = lambda: "PID_Aggressive"
    controllers[1].name = lambda: "PID_Sluggish"

    runner = BenchmarkRunner(controllers, scenarios)
    results = runner.run()

    assert len(results) == 2

    # Aggressive should have lower IAE but maybe higher overshoot
    res_agg = results[0]
    res_slug = results[1]

    assert res_agg.iae >= 0.0
    assert res_slug.iae >= 0.0
    assert res_agg.computation_time_us > 0.0
    assert res_agg.violations >= 0

    # Test JSON save
    json_path = tmp_path / "results.json"
    runner.save_json(json_path)
    assert json_path.exists()

    with open(json_path) as f:
        data = json.load(f)
        assert len(data) == 2
        assert "iae" in data[0]

    # Test Markdown save
    md_path = tmp_path / "results.md"
    runner.save_markdown(md_path)
    assert md_path.exists()
    with open(md_path) as f:
        content = f.read()
        assert "PID_Aggressive" in content
        assert "setpoint_tracking" in content
