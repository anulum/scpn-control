# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI info command tests
"""Tests for the ``info`` CLI command (environment + backend + weights report)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_info_json_out(runner: CliRunner) -> None:
    result = runner.invoke(main, ["info", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "version" in data
    assert "python" in data
    assert "numpy" in data
    assert "rust_backend" in data


def test_info_text_output(runner: CliRunner) -> None:
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "scpn-control" in result.output
    assert "Rust backend:" in result.output
    assert "Python:" in result.output
    assert "NumPy:" in result.output
    assert "neural_equilibrium_sparc.npz" in result.output


def test_info_json_includes_weights_list(runner: CliRunner) -> None:
    result = runner.invoke(main, ["info", "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data["weights"], list)
