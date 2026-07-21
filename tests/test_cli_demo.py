# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI demo command tests
"""Tests for the ``demo`` CLI command (scenario control demonstrations)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_demo_json_out(runner: CliRunner) -> None:
    result = runner.invoke(main, ["demo", "--steps", "10", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in ("scenario", "steps", "final_state", "final_error", "converged"):
        assert key in data
    assert data["steps"] == 10
    assert isinstance(data["final_state"], float)
    assert isinstance(data["converged"], bool)


def test_demo_pid_text(runner: CliRunner) -> None:
    result = runner.invoke(main, ["demo", "--scenario", "pid", "--steps", "5"])
    assert result.exit_code == 0
    assert "Scenario: pid" in result.output
    assert "Steps: 5" in result.output


def test_demo_combined_json(runner: CliRunner) -> None:
    result = runner.invoke(main, ["demo", "--scenario", "combined", "--steps", "5", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["scenario"] == "combined"


def test_demo_snn_text(runner: CliRunner) -> None:
    result = runner.invoke(main, ["demo", "--scenario", "snn", "--steps", "3"])
    assert result.exit_code == 0
    assert "Scenario: snn" in result.output
