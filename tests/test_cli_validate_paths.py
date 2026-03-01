# ──────────────────────────────────────────────────────────────────────
# SCPN Control — CLI Validate Command Edge Path Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Coverage for cli.py validate command: contamination check (lines 149-150,
161-164), weight file iteration (278), and info --json-out path."""
from __future__ import annotations

import json
import sys
import types

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


class TestValidateCommand:
    def test_validate_json_structure(self):
        """validate --json-out returns valid JSON with expected keys."""
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--json-out"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "transport_solver_available" in data
        assert "import_clean" in data
        assert "status" in data

    def test_validate_text_output(self):
        """validate without --json-out produces text summary."""
        runner = CliRunner()
        result = runner.invoke(main, ["validate"])
        assert result.exit_code == 0
        assert "Transport solver:" in result.output
        assert "Import clean:" in result.output

    def test_validate_contaminated_module(self):
        """validate detects contaminated sys.modules (lines 161-164).

        matplotlib is already loaded by test infra, so the contamination
        check fires for it. We just verify the detection logic works.
        """
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--json-out"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # matplotlib is typically loaded → contamination detected
        if "matplotlib" in sys.modules:
            assert data["import_clean"] is False
            assert data["contaminated_module"] == "matplotlib"
        else:
            assert data["import_clean"] is True


class TestInfoCommand:
    def test_info_json(self):
        """info --json-out returns structured JSON."""
        runner = CliRunner()
        result = runner.invoke(main, ["info", "--json-out"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "version" in data
        assert "numpy" in data

    def test_info_text(self):
        """info text output prints version and numpy."""
        runner = CliRunner()
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "scpn-control" in result.output
        assert "NumPy" in result.output
