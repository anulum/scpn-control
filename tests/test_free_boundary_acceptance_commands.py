# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real free-boundary acceptance command contracts.

"""Exercise the actual complete kernel campaign, public renderer and CLI refusals."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scpn_control.benchmark_records import CAMPAIGN_ENV
from validation.free_boundary_tracking_acceptance import NOMINAL_THRESHOLDS, main, render_markdown

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation/free_boundary_tracking_acceptance.py"


def _command(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the real source-checkout script without inherited import/custody/tracer flags."""
    env = os.environ.copy()
    for key in ("PYTHONPATH", CAMPAIGN_ENV, "COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments], cwd=cwd, env=env, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def observed_report(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """Run the full strict cohort outside the repository and read its actual output."""
    out = tmp_path_factory.mktemp("actual-free-boundary")
    result = _command(
        "--json-out", str(out / "report.json"), "--md-out", str(out / "report.md"), "--require-thresholds", cwd=out
    )
    assert result.returncode == 0, result.stderr
    return out, json.loads((out / "report.json").read_text())


def test_full_real_command_reports_cohort_and_model_limits(observed_report: tuple[Path, dict[str, Any]]) -> None:
    """Verify completed real cases retain their fixed cohort and explicit scientific limits."""
    out, report = observed_report
    assert report["schema_version"] == "scpn-control.free-boundary-tracking-acceptance.v2"
    assert report["runtime_seconds"] > 0
    assert report["physical_reference_admitted"] is False
    contract = report["model_contract"]
    assert contract["grid_shape"] == [12, 12] and contract["vacuum_permeability"] == 1.0
    assert contract["plasma_current_target"] == 1.0 and contract["physical_reference_admitted"] is False
    assert "same FusionKernel" in contract["targets"]
    assert "exact supplied" in contract["measurement_correction"]
    campaign = report["free_boundary_tracking_acceptance"]
    assert campaign["passes_thresholds"] is True
    assert len(campaign["scenarios"]) == 18 and len(campaign["sweeps"]) == 12
    for case in campaign["scenarios"].values():
        assert case["passes_thresholds"] and all(case["checks"].values())
        assert case["summary"]["steps"] == 4
    for sweep in campaign["sweeps"].values():
        assert sweep["entries"] and sweep["passes_thresholds"] and all(sweep["checks"].values())
    assert (out / "report.json").read_bytes().endswith(b"\n")


def test_public_markdown_matches_actual_output_and_limits(observed_report: tuple[Path, dict[str, Any]]) -> None:
    """Render the genuine observed report through the compatible public API."""
    out, report = observed_report
    text = render_markdown(report)
    assert text == (out / "report.md").read_text() and text.endswith("\n")
    assert "exact injected bias/drift" in text and "not calibrated SI flux" in text
    assert "excluding rendering and writes" in text and "No independent physical-reference" in text
    for name in report["free_boundary_tracking_acceptance"]["scenarios"]:
        assert name in text
    for name in report["free_boundary_tracking_acceptance"]["sweeps"]:
        assert name in text


@pytest.mark.parametrize("arguments,expected", [(("--help",), 0), (("--unknown-option",), 2), (("--json-out",), 2)])
def test_real_parser_help_and_usage(tmp_path: Path, arguments: tuple[str, ...], expected: int) -> None:
    """Preserve argparse help/usage exits in a real outside-checkout process."""
    result = _command(*arguments, cwd=tmp_path)
    assert result.returncode == expected
    assert "usage:" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "alias",
    ["same", "output_symlink", "output_hardlink", "source", "source_symlink", "source_hardlink", "leaf", "core"],
)
def test_real_cli_refuses_output_and_source_aliases_before_campaign(tmp_path: Path, alias: str) -> None:
    """Reject actual filesystem aliases before creating the unrelated report directory."""
    destination = tmp_path / "selected.json"
    partner = tmp_path / "untouched" / "report.md"
    if alias.startswith("output") or alias == "same":
        destination.write_text("preserve these original output bytes")
        partner = destination if alias == "same" else tmp_path / "partner.md"
        if alias == "output_symlink":
            partner.symlink_to(destination)
        elif alias == "output_hardlink":
            os.link(destination, partner)
        protected = destination
    else:
        protected = SCRIPT
        if alias == "leaf":
            protected = ROOT / "validation/free_boundary_acceptance_presets.py"
        elif alias == "core":
            protected = ROOT / "src/scpn_control/core/fusion_kernel.py"
        if alias == "source_symlink":
            destination.symlink_to(protected)
        elif alias == "source_hardlink":
            os.link(protected, destination)
        else:
            destination = protected
    before = protected.read_bytes()
    result = _command("--json-out", str(destination), "--md-out", str(partner), cwd=tmp_path)
    assert result.returncode == 2
    assert "input, custody or execution refused" in result.stderr
    assert protected.read_bytes() == before
    if alias not in ("same", "output_symlink", "output_hardlink"):
        assert not partner.parent.exists()


def test_default_persistent_outputs_refuse_without_custody(tmp_path: Path) -> None:
    """Leave both actual historical reports untouched when no recorded campaign is supplied."""
    paths = [
        ROOT / "validation/reports/free_boundary_tracking_acceptance.json",
        ROOT / "validation/reports/free_boundary_tracking_acceptance.md",
    ]
    before = [p.read_bytes() if p.is_file() else None for p in paths]
    result = _command(cwd=tmp_path)
    assert result.returncode == 2 and "custody" in result.stderr
    assert [p.read_bytes() if p.is_file() else None for p in paths] == before


def test_imported_main_refuses_aliases(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exercise the compatible imported main before any genuine campaign work."""
    output = tmp_path / "report"
    output.write_text("original")
    assert main(["--json-out", str(output), "--md-out", str(output)]) == 2
    assert output.read_text() == "original" and "refused" in capsys.readouterr().err


def test_real_output_io_failure_returns_authored_refusal(tmp_path: Path) -> None:
    """Run the actual cohort then refuse a destination whose parent is a regular file."""
    obstruction = tmp_path / "regular-file"
    obstruction.write_text("original file")
    result = _command(
        "--json-out", str(obstruction / "report.json"), "--md-out", str(tmp_path / "report.md"), cwd=tmp_path
    )
    assert result.returncode == 2 and "execution refused" in result.stderr
    assert obstruction.read_text() == "original file" and not (tmp_path / "report.md").exists()


def test_public_strict_main_reports_actual_failure_under_tighter_declared_limit(tmp_path: Path) -> None:
    """Tighten the shared public nominal limit and exercise a genuine failed kernel cohort."""
    original = NOMINAL_THRESHOLDS["max_final_tracking_error_norm"]
    NOMINAL_THRESHOLDS["max_final_tracking_error_norm"] = 0.0
    report_path = tmp_path / "strict.json"
    markdown_path = tmp_path / "strict.md"
    try:
        assert main(["--json-out", str(report_path), "--md-out", str(markdown_path), "--require-thresholds"]) == 1
    finally:
        NOMINAL_THRESHOLDS["max_final_tracking_error_norm"] = original
    report = json.loads(report_path.read_text())
    case = report["free_boundary_tracking_acceptance"]["scenarios"]["nominal"]
    assert case["summary"]["final_tracking_error_norm"] > 0
    assert case["thresholds"]["max_final_tracking_error_norm"] == 0.0
    assert case["checks"]["final_tracking_error_norm"] is False
    assert report["free_boundary_tracking_acceptance"]["passes_thresholds"] is False
    assert report["physical_reference_admitted"] is False and markdown_path.is_file()
