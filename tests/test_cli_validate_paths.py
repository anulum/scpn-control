# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI Validate Command Edge Path Tests

"""Regression tests for cli.py validate command: contamination check (lines 149-150,
161-164), weight file iteration (278), and info --json-out path.
"""

from __future__ import annotations

import json
import sys

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


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
    def test_validate_json_structure(self, clean_optional_imports):
        """Validate --json-out returns valid JSON with expected keys."""
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--json-out"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "transport_solver_available" in data
        assert "import_clean" in data
        assert "status" in data

    def test_validate_text_output(self, clean_optional_imports):
        """Validate without --json-out produces text summary."""
        runner = CliRunner()
        result = runner.invoke(main, ["validate"])
        assert result.exit_code == 0
        assert "Transport solver:" in result.output
        assert "Import clean:" in result.output

    def test_validate_text_names_every_skipped_gate(self, clean_optional_imports):
        """Each gate that an explicit flag skips is named as skipped in the text summary."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "validate",
                "--no-data-manifests",
                "--no-jax-gk-parity",
                "--no-physics-traceability",
                "--no-multi-shot-campaign-evidence",
                "--no-runtime-admission-evidence",
                "--no-native-formal-certificate",
            ],
        )
        assert result.exit_code == 0, result.output
        for gate in (
            "Data manifests",
            "JAX GK parity",
            "Physics traceability",
            "Multi-shot campaign evidence",
            "Runtime admission evidence",
            "Native formal certificate",
        ):
            assert f"{gate}: SKIPPED" in result.output
        assert "Import clean: OK" in result.output
        assert result.output.rstrip().endswith("Status: pass")

    def test_validate_contaminated_module(self):
        """Validate detects contaminated sys.modules (lines 161-164).

        The test runner may have optional plotting or ML packages loaded from
        earlier tests. Verify the ordered contamination contract rather than an
        incidental dependency.
        """
        runner = CliRunner()
        contaminated = next((mod for mod in ("matplotlib", "torch", "streamlit") if mod in sys.modules), None)
        result = runner.invoke(main, ["validate", "--json-out"])
        # A prohibited module fails the command; the JSON result is printed first.
        assert result.exit_code == (0 if contaminated is None else 1)
        data = json.loads(result.output)
        if contaminated is not None:
            assert data["import_clean"] is False
            assert data["contaminated_module"] == contaminated
        else:
            assert data["import_clean"] is True


class TestInfoCommand:
    def test_info_json(self):
        """Info --json-out returns structured JSON."""
        runner = CliRunner()
        result = runner.invoke(main, ["info", "--json-out"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "version" in data
        assert "numpy" in data

    def test_info_text(self):
        """Info text output prints version and numpy."""
        runner = CliRunner()
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "scpn-control" in result.output
        assert "NumPy" in result.output
