# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — SOL CLI report custody and real entry points.
"""Exercise real parsing, output paths, report IO and checkout entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from validation.sol_two_point_contracts.command import ROOT, main, render_markdown
from validation.sol_two_point_contracts.evidence import build_evidence, validate_evidence_payload
from validation.sol_two_point_contracts.models import validate_sol_two_point


def test_report_writes_complete_failed_json_and_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Strict failure still publishes internally consistent false content to scratch."""
    report = tmp_path / "nested/report.json"
    assert main(["--report", str(report), "--json-out", "--exact-tol", "1e-30"]) == 1
    printed = json.loads(capsys.readouterr().out)
    stored = json.loads(report.read_text())
    assert printed == stored and validate_evidence_payload(stored) is False
    markdown = report.with_suffix(".md").read_text()
    assert "**fail**" in markdown and "admission is not supplied" in markdown


@pytest.mark.parametrize("args", [["--exact-tol", "0"], ["--exact-tol", "inf"], ["--target-id", " "]])
def test_bad_cli_inputs_refuse_with_authored_status(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """Declared input refusals return two without success stdout or a traceback."""
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "refused" in captured.err and "Traceback" not in captured.err


@pytest.mark.parametrize("argument,code", [("--help", 0), ("--unknown-option", 2)])
def test_parser_retains_standard_help_and_usage_exits(argument: str, code: int) -> None:
    """Real argparse preserves its help and invalid-argument control flow."""
    with pytest.raises(SystemExit) as raised:
        main([argument])
    assert raised.value.code == code


def test_json_and_markdown_same_path_refuse_before_creation(tmp_path: Path) -> None:
    """The .md report spelling cannot destroy JSON with its sibling write."""
    report = tmp_path / "report.md"
    assert main(["--report", str(report)]) == 2 and not report.exists()


@pytest.mark.parametrize("alias", ["direct", "symbolic", "hard"])
def test_selected_source_aliases_are_refused_without_byte_changes(tmp_path: Path, alias: str) -> None:
    """Direct, symbolic and inode aliases cannot overwrite production model source."""
    source = ROOT / "src/scpn_control/core/sol_model.py"
    original = source.read_bytes()
    report = source
    if alias == "symbolic":
        report = tmp_path / "alias.json"
        report.symlink_to(source)
    elif alias == "hard":
        report = tmp_path / "alias.json"
        report.hardlink_to(source)
    assert main(["--report", str(report)]) == 2
    assert source.read_bytes() == original and not report.with_suffix(".md").exists()


def test_markdown_alias_is_checked_before_json_write(tmp_path: Path) -> None:
    """The second selected output is independently checked before either write."""
    source = ROOT / "validation/validate_sol_two_point.py"
    original = source.read_bytes()
    report = tmp_path / "report.json"
    report.with_suffix(".md").symlink_to(source)
    assert main(["--report", str(report)]) == 2
    assert source.read_bytes() == original and not report.exists()


def test_persistent_evidence_requires_recorded_custody(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ordinary invocation cannot silently replace scientific evidence files."""
    monkeypatch.delenv("SCPN_BENCHMARK_CAMPAIGN_ID", raising=False)
    report = ROOT / "validation/reports/sol_two_point.json"
    before = report.read_bytes() if report.exists() else None
    assert main(["--report", str(report)]) == 2
    assert (report.read_bytes() if report.exists() else None) == before


def test_first_and_second_io_failures_are_reported(tmp_path: Path) -> None:
    """Actual directory destinations preserve sequential-write limitations."""
    first = tmp_path / "first.json"
    first.mkdir()
    assert main(["--report", str(first)]) == 2
    assert not first.with_suffix(".md").exists()
    second = tmp_path / "second.json"
    second.with_suffix(".md").mkdir()
    assert main(["--report", str(second)]) == 2
    assert validate_evidence_payload(json.loads(second.read_text())) is True


def test_renderer_refuses_tampered_content_and_escapes_target_lines() -> None:
    """Readable output preserves literal line breaks in the supplied target string."""
    evidence = build_evidence(validate_sol_two_point(), target_id="first\nsecond")
    assert '"first\\nsecond"' in render_markdown(evidence)
    evidence["passed"] = False
    with pytest.raises(ValueError, match="does not match"):
        render_markdown(evidence)


@pytest.mark.parametrize("entry", ["module", "script"])
def test_real_checkout_entry_points_bootstrap_without_pythonpath(tmp_path: Path, entry: str) -> None:
    """Python module and direct-script entry points run without inherited source paths."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable]
    argv += (
        ["-m", "validation.validate_sol_two_point"]
        if entry == "module"
        else [str(ROOT / "validation/validate_sol_two_point.py")]
    )
    report = tmp_path / (entry + ".json")
    child = subprocess.run(
        [*argv, "--json-out", "--report", str(report)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert validate_evidence_payload(json.loads(child.stdout)) is True
    assert json.loads(report.read_text()) == json.loads(child.stdout)
