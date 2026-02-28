# ──────────────────────────────────────────────────────────────────────
# SCPN Control — CLI Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""Tests for the scpn-control Click CLI entry point."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_demo_json_out(runner):
    result = runner.invoke(main, ["demo", "--steps", "10", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in ("scenario", "steps", "final_state", "final_error", "converged"):
        assert key in data
    assert data["steps"] == 10
    assert isinstance(data["final_state"], float)
    assert isinstance(data["converged"], bool)


def test_demo_pid_text(runner):
    result = runner.invoke(main, ["demo", "--scenario", "pid", "--steps", "5"])
    assert result.exit_code == 0
    assert "Scenario: pid" in result.output
    assert "Steps: 5" in result.output


def test_benchmark_json_out(runner):
    result = runner.invoke(main, ["benchmark", "--n-bench", "100", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in ("n_bench", "pid_us_per_step", "snn_us_per_step", "speedup_ratio"):
        assert key in data
    assert data["n_bench"] == 100
    assert data["pid_us_per_step"] > 0


def test_validate_json_out(runner):
    result = runner.invoke(main, ["validate", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "status" in data
    assert data["status"] in ("pass", "fail")


def test_hil_test_nonexistent_dir(runner):
    result = runner.invoke(
        main, ["hil-test", "--shots-dir", "nonexistent_dir_12345", "--json-out"]
    )
    assert result.exit_code != 0


def test_live_help(runner):
    result = runner.invoke(main, ["live", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--zeta" in result.output


def test_benchmark_text_output(runner):
    result = runner.invoke(main, ["benchmark", "--n-bench", "50"])
    assert result.exit_code == 0
    assert "PID:" in result.output
    assert "SNN:" in result.output
    assert "Ratio:" in result.output


def test_validate_text_output(runner):
    result = runner.invoke(main, ["validate"])
    assert result.exit_code == 0
    assert "Transport solver:" in result.output
    assert "Import clean:" in result.output
    assert "Status:" in result.output


def test_info_json_out(runner):
    result = runner.invoke(main, ["info", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "version" in data
    assert "python" in data
    assert "numpy" in data
    assert "rust_backend" in data


def test_info_text_output(runner):
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "scpn-control" in result.output
    assert "Rust backend:" in result.output
    assert "Python:" in result.output
    assert "NumPy:" in result.output


def test_hil_test_with_mock_shots(runner, tmp_path):
    """Cover hil-test loading NPZ files (lines 222-243)."""
    rng = np.random.default_rng(0)
    for name in ("shot_001", "shot_002"):
        np.savez(
            tmp_path / f"{name}.npz",
            psi=rng.standard_normal((10, 10)),
            ip=np.array([15e6]),
        )
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path), "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["n_shots"] == 2
    assert len(data["shots"]) == 2
    assert data["shots"][0]["status"] == "loaded"
    assert "psi" in data["shots"][0]["keys"]


def test_hil_test_text_output(runner, tmp_path):
    np.savez(tmp_path / "s42.npz", plasma=np.zeros(5))
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "1 shots" in result.output
    assert "s42" in result.output


def test_hil_test_empty_dir(runner, tmp_path):
    result = runner.invoke(main, ["hil-test", "--shots-dir", str(tmp_path), "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["n_shots"] == 0


def test_info_json_includes_weights_list(runner):
    result = runner.invoke(main, ["info", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data["weights"], list)


def test_demo_combined_json(runner):
    result = runner.invoke(main, ["demo", "--scenario", "combined", "--steps", "5", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["scenario"] == "combined"


def test_demo_snn_text(runner):
    result = runner.invoke(main, ["demo", "--scenario", "snn", "--steps", "3"])
    assert result.exit_code == 0
    assert "Scenario: snn" in result.output
