# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual disturbance public surface tests.

"""Observe genuine script, imported CLI and compatible imported-main execution."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from campaign_command_observation import ROOT, observe_command

from scpn_control.control.neuro_cybernetic_controller import SC_NEUROCORE_AVAILABLE

MODULE = "validation.benchmark_disturbance_rejection"


def _command(entry: str, output: Path, *, scale: float = 0.001, strict: bool = False) -> list[str]:
    """Construct real argv or a direct public-main call without replacing runtime behavior."""
    if entry == "imported-main":
        code = (
            "from "
            + MODULE
            + " import main;main("
            + repr(str(output))
            + ",duration_scale="
            + repr(scale)
            + ",require_complete="
            + repr(strict)
            + ")"
        )
        return [sys.executable, "-c", code]
    argv = (
        [sys.executable, str(ROOT / "validation/benchmark_disturbance_rejection.py")]
        if entry == "script"
        else [sys.executable, "-c", "from " + MODULE + " import cli;raise SystemExit(cli())"]
    )
    argv += ["--output-dir", str(output), "--duration-scale", str(scale)]
    return argv + (["--require-complete"] if strict else [])


@pytest.mark.parametrize("entry", ["script", "imported-cli", "imported-main"])
@pytest.mark.parametrize("strict", [False, True])
def test_actual_bounded_cohort_and_all_written_artifacts(tmp_path: Path, entry: str, strict: bool) -> None:
    """All four real providers produce 12 finite bounded rows and three real plots."""
    out = tmp_path / "run"
    result = observe_command(tmp_path, _command(entry, out, strict=strict), include_repository=True)
    assert result.returncode == (2 if strict and not SC_NEUROCORE_AVAILABLE else 0), result.stdout + result.stderr
    report = json.loads((out / "benchmark_disturbance_rejection.json").read_text())
    expected = ["PID", "H-infinity", "MPC"] + (["SNN"] if SC_NEUROCORE_AVAILABLE else [])
    assert report["schema_version"] == 2 and len(report["results"]) == 3 * len(expected)
    assert report["actual_controllers"] == expected
    if SC_NEUROCORE_AVAILABLE:
        assert report["controller_runtime"]["SNN"]["backend"] == "sc_neurocore"
    else:
        assert report["missing_controllers"] == ["SNN"]
    assert report["complete_controller_cohort"] == SC_NEUROCORE_AVAILABLE
    assert report["all_runs_completed_bounded"]
    assert not report["model_contract"]["physical_reference_admitted"]
    expected_steps = {"VDE": 20, "Density ramp": 40, "ELM pacing": 30}
    for row in report["results"]:
        assert row["completed_steps"] == row["requested_steps"] == expected_steps[row["scenario"]]
        assert row["completed_duration_s"] == row["requested_duration_s"]
    assert (out / "benchmark_disturbance_rejection.md").read_text().endswith("\n")
    assert len(list(out.glob("*.png"))) == 3
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in out.glob("*.png"))


def test_actual_full_duration_strict_exit_follows_truthful_report_writes(tmp_path: Path) -> None:
    """The original-duration cohort reports real instability and exits two after all artifacts."""
    out = tmp_path / "full"
    result = observe_command(tmp_path, _command("script", out, scale=1, strict=True))
    assert result.returncode == 2
    report = json.loads((out / "benchmark_disturbance_rejection.json").read_text())
    assert report["complete_controller_cohort"] == SC_NEUROCORE_AVAILABLE
    assert not report["all_runs_completed_bounded"]
    for row in report["results"]:
        if row["controller"] == "H-infinity":
            assert row["stable"]
        if not row["stable"]:
            assert row["completed_steps"] < row["requested_steps"]
            assert row["completed_duration_s"] < row["requested_duration_s"]
    assert (out / "benchmark_disturbance_rejection.md").is_file() and len(list(out.glob("*.png"))) == 3
    assert "Reports written to:" in result.stdout


@pytest.mark.parametrize(
    "flags,expected", [(["--help"], 0), (["--invalid-option"], 2), (["--duration-scale", "wrong"], 2)]
)
def test_actual_parser_help_and_syntax(tmp_path: Path, flags: list[str], expected: int) -> None:
    """Real argparse help/syntax exits before report generation."""
    result = observe_command(
        tmp_path, [sys.executable, str(ROOT / "validation/benchmark_disturbance_rejection.py"), *flags]
    )
    assert result.returncode == expected and not list(tmp_path.glob("*.png"))


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "0.000001", "0.000075"])
def test_actual_duration_domain_refusal_writes_no_report(tmp_path: Path, value: str) -> None:
    """Invalid actual CLI durations cannot silently round or produce successful output."""
    out = tmp_path / "bad"
    argv = [
        sys.executable,
        str(ROOT / "validation/benchmark_disturbance_rejection.py"),
        "--output-dir",
        str(out),
        "--duration-scale",
        value,
    ]
    result = observe_command(tmp_path, argv)
    assert result.returncode == 2 and "was refused" in result.stderr
    assert not (out / "benchmark_disturbance_rejection.json").exists()


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_actual_report_alias_refusal_preserves_bytes(tmp_path: Path, alias: str) -> None:
    """Actual JSON/Markdown inode aliases refuse before any controller/artifact write."""
    out = tmp_path / "alias"
    out.mkdir()
    first = out / "benchmark_disturbance_rejection.json"
    second = out / "benchmark_disturbance_rejection.md"
    first.write_bytes(b"owner report")
    if alias == "symlink":
        second.symlink_to(first)
    else:
        os.link(first, second)
    result = observe_command(tmp_path, _command("script", out))
    assert result.returncode == 2 and first.read_bytes() == second.read_bytes() == b"owner report"
    assert not list(out.glob("*.png"))


def test_actual_source_plot_alias_and_default_custody_refuse(tmp_path: Path) -> None:
    """Plot/source aliases and unrecorded canonical defaults preserve all source bytes."""
    source = ROOT / "validation/disturbance_runtime.py"
    before = source.read_bytes()
    out = tmp_path / "source"
    out.mkdir()
    (out / "benchmark_vde.png").symlink_to(source)
    result = observe_command(tmp_path, _command("script", out))
    assert result.returncode == 2 and source.read_bytes() == before
    assert not (out / "benchmark_disturbance_rejection.json").exists()
    result = observe_command(tmp_path, [sys.executable, str(ROOT / "validation/benchmark_disturbance_rejection.py")])
    assert result.returncode == 2 and "was refused" in result.stderr
