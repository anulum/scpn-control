# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CLI validate-rmse tests
"""Run the registered RMSE command on repository inputs with real report exports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_control.cli import main


@pytest.mark.parametrize("json_out", [False, True])
def test_validate_rmse_command_generates_real_bounded_report(tmp_path: Path, json_out: bool) -> None:
    """The registered CLI exports unavailable beta truthfully and never rewrites process argv."""
    report = tmp_path / "rmse_report.json"
    markdown = tmp_path / "rmse_report.md"
    argv_before = sys.argv[:]
    args = ["validate-rmse", "--output-json", str(report), "--output-md", str(markdown)]
    if json_out:
        args.append("--json-out")
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    payload = json.loads(report.read_text())
    assert payload["schema"] == "scpn-control.rmse-dashboard.v2"
    beta = payload["beta_iter_sparc"]
    if beta.get("skipped", False):
        assert beta["beta_n_rmse"] is None and beta["rows"] == []
    else:
        assert beta["count"] == 2 and beta["physical_validation_claim"] is False
        assert beta["design_rmse_is_binding"] is False
    assert (
        '"schema": "scpn-control.rmse-dashboard.v2"' in result.output if json_out else '"schema"' not in result.output
    )
    assert (
        "Unavailable:" in markdown.read_text()
        if beta.get("skipped", False)
        else "indicative target" in markdown.read_text()
    )
    assert sys.argv == argv_before


def test_validate_rmse_output_failure_preserves_process_arguments(tmp_path: Path) -> None:
    """An actual output-path error becomes a Click failure without a forged success carrier."""
    destination = tmp_path / "file_as_directory"
    destination.write_bytes(b"preserve existing destination")
    argv_before = sys.argv[:]
    result = CliRunner().invoke(
        main,
        [
            "validate-rmse",
            "--output-json",
            str(destination / "report.json"),
            "--output-md",
            str(tmp_path / "report.md"),
            "--json-out",
        ],
    )
    assert result.exit_code == 1 and "Error:" in result.output
    assert destination.read_bytes() == b"preserve existing destination"
    assert not (tmp_path / "report.md").exists()
    assert sys.argv == argv_before
