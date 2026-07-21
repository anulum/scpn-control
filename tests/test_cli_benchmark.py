# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI benchmark command tests
"""Tests for the ``benchmark`` CLI command (PID vs SNN per-step timing)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_benchmark_json_out(runner: CliRunner) -> None:
    result = runner.invoke(main, ["benchmark", "--n-bench", "100", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in ("n_bench", "pid_us_per_step", "snn_us_per_step", "speedup_ratio"):
        assert key in data
    assert data["n_bench"] == 100
    assert data["pid_us_per_step"] > 0


def test_benchmark_text_output(runner: CliRunner) -> None:
    result = runner.invoke(main, ["benchmark", "--n-bench", "50"])
    assert result.exit_code == 0
    assert "PID:" in result.output
    assert "SNN:" in result.output
    assert "Ratio:" in result.output
