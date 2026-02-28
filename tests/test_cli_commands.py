# ──────────────────────────────────────────────────────────────────────
# SCPN Control — CLI Command Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests for click CLI commands: validate, version, info."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


class TestValidateCommand:
    def test_validate_text(self, runner):
        result = runner.invoke(main, ["validate"])
        assert result.exit_code == 0
        assert "Transport solver:" in result.output
        assert "Status:" in result.output

    def test_validate_json(self, runner):
        result = runner.invoke(main, ["validate", "--json-out"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "transport_solver_available" in data
        assert "status" in data


class TestVersionFlag:
    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "scpn-control" in result.output or "version" in result.output.lower()


class TestInfoCommand:
    def test_info(self, runner):
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
