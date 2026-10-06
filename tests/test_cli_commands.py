# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test Cli Commands.

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — CLI Command Tests
# © 1996–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for click CLI commands: validate, version, info."""

from __future__ import annotations

import json
import sys

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def clean_optional_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the optional plotting and ML modules that earlier tests may have loaded.

    The validate command fails when one of them is loaded. These tests are
    about its output for an interpreter that has not loaded them, whatever ran
    before in the same session.
    """
    for name in ("matplotlib", "torch", "streamlit"):
        monkeypatch.delitem(sys.modules, name, raising=False)


class TestValidateCommand:
    def test_validate_text(self, runner, clean_optional_imports):
        result = runner.invoke(main, ["validate"])
        assert result.exit_code == 0
        assert "Transport solver:" in result.output
        assert "Status:" in result.output

    def test_validate_json(self, runner, clean_optional_imports):
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
