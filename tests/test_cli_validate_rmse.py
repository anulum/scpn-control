# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI validate-rmse tests
"""Tests for the ``validate-rmse`` CLI command and its report-echo behaviour."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_validate_rmse_command(runner, tmp_path, monkeypatch):
    """Exercise cli.py lines 238-248: validate-rmse imports rmse_dashboard."""
    called = {}

    def fake_rmse_main():
        import json as _json
        from pathlib import Path as _P

        out = _P(tmp_path) / "rmse_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps({"status": "pass"}))
        called["invoked"] = True
        return 0

    monkeypatch.setitem(
        __import__("sys").modules,
        "validation.rmse_dashboard",
        type("M", (), {"main": staticmethod(fake_rmse_main)})(),
    )

    result = runner.invoke(
        main,
        [
            "validate-rmse",
            "--json-out",
            "--output-json",
            str(tmp_path / "rmse_report.json"),
            "--output-md",
            str(tmp_path / "rmse_report.md"),
        ],
    )
    assert called.get("invoked")
    assert "pass" in result.output


def test_validate_rmse_without_json_out_skips_report_echo(runner, tmp_path, monkeypatch):
    """Without ``--json-out`` the report file is never echoed (arc 452->458)."""

    def fake_rmse_main() -> int:
        return 0

    monkeypatch.setitem(
        __import__("sys").modules,
        "validation.rmse_dashboard",
        type("M", (), {"main": staticmethod(fake_rmse_main)})(),
    )

    result = runner.invoke(
        main,
        [
            "validate-rmse",
            "--output-json",
            str(tmp_path / "rmse_report.json"),
            "--output-md",
            str(tmp_path / "rmse_report.md"),
        ],
    )

    assert result.exit_code == 0
    assert "Report SHA-256" not in result.output


def test_validate_rmse_json_out_skips_echo_when_report_absent(runner, tmp_path, monkeypatch):
    """With ``--json-out`` but no written report, the missing-file guard skips echo (arc 456->458)."""

    def fake_rmse_main() -> int:
        return 0  # deliberately does not write the output JSON

    monkeypatch.setitem(
        __import__("sys").modules,
        "validation.rmse_dashboard",
        type("M", (), {"main": staticmethod(fake_rmse_main)})(),
    )

    missing_report = tmp_path / "never_written.json"
    result = runner.invoke(
        main,
        [
            "validate-rmse",
            "--json-out",
            "--output-json",
            str(missing_report),
            "--output-md",
            str(tmp_path / "rmse_report.md"),
        ],
    )

    assert result.exit_code == 0
    assert not missing_report.exists()
    assert result.output.strip() == ""
