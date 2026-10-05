# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test quality policy guard tests

"""Exercise filename, wording and scan-root admission through the policy API."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_test_quality_policy import Violation, collect_violations, main


def test_test_quality_policy_rejects_bucket_filename(tmp_path: Path) -> None:
    """Reject a file named as a generic test bucket."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_cov_final_gaps.py").write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )

    violations = collect_violations(tests)

    assert len(violations) == 1
    assert "forbidden generic bucket test filename" in violations[0].reason


@pytest.mark.parametrize(
    "filename",
    (
        "test_batch.py",
        "test_controller_round.py",
        "test_transport_final.py",
        "test_physics_remaining.py",
        "test_adapter_push.py",
        "test_module_misc.py",
        "test_new_modules.py",
    ),
)
def test_test_quality_policy_rejects_forbidden_bucket_name_families(tmp_path: Path, filename: str) -> None:
    """Reject every explicitly prohibited bucket filename family."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / filename).write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )

    violations = collect_violations(tests)

    assert len(violations) == 1
    assert "forbidden generic bucket test filename" in violations[0].reason


def test_test_quality_policy_rejects_slop_intent_markers(tmp_path: Path) -> None:
    """Report both distinct forbidden intent markers at their source line."""
    tests = tmp_path / "tests"
    tests.mkdir()
    forbidden_text = '"""coverage ' + "gaps for uncovered " + 'lines."""\n'
    (tests / "test_named_module.py").write_text(forbidden_text, encoding="utf-8")

    violations = collect_violations(tests)

    assert len(violations) == 2
    assert all("forbidden coverage-slop wording" in violation.reason for violation in violations)


def test_test_quality_policy_accepts_named_contract_surface(tmp_path: Path) -> None:
    """Accept a named contract file with no forbidden intent markers."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_named_module.py").write_text(
        '"""Boundary contract tests for named module behaviour."""\n',
        encoding="utf-8",
    )

    assert collect_violations(tests) == []


@pytest.mark.parametrize("scope", ["missing", "file", "empty"])
def test_scan_refuses_uninspectable_scope(tmp_path: Path, scope: str) -> None:
    """Refuse missing, non-directory and empty scopes through API and CLI."""
    root = tmp_path / scope
    if scope == "file":
        root.write_text("ordinary file\n", encoding="utf-8")
    elif scope == "empty":
        root.mkdir()
    with pytest.raises(ValueError, match="test scope"):
        collect_violations(root)
    assert main(["--test-root", str(root)]) == 1


def test_scan_refuses_invalid_utf8(tmp_path: Path) -> None:
    """Refuse source bytes whose unreadable text could conceal intent markers."""
    (tmp_path / "test_channel.py").write_bytes(b"\xff\n")
    with pytest.raises(UnicodeDecodeError):
        collect_violations(tmp_path)
    assert main(["--test-root", str(tmp_path)]) == 1


def test_real_cli_scans_named_contract_file(tmp_path: Path) -> None:
    """Run the real script on a named test file and a refused scope."""
    (tmp_path / "test_channel.py").write_text('"""Tests of channel admission."""\n', encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "tools/check_test_quality_policy.py"
    good = subprocess.run(
        [sys.executable, str(script), "--test-root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert good.returncode == 0
    assert "guard passed" in good.stdout
    bad = subprocess.run(
        [sys.executable, str(script), "--test-root", str(tmp_path / "absent")],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert bad.returncode == 1
    assert "guard failed" in bad.stdout


def test_cli_reports_detected_rules_and_source_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Expose filename and wording findings as diagnostics with one-based lines."""
    file = tmp_path / "test_bucket.py"
    file.write_text('"""coverage ' + 'closure."""\n', encoding="utf-8")
    assert main(["--test-root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "violations: 2" in output
    assert f"{file}:1: forbidden generic bucket test filename" in output
    assert f"{file}:1: forbidden coverage-slop wording" in output


def test_relative_collector_root_keeps_relative_diagnostic(tmp_path: Path) -> None:
    """Preserve the caller's relative path in API findings outside the repository."""
    (tmp_path / "test_bucket.py").write_text("pass\n", encoding="utf-8")
    relative = Path(os.path.relpath(tmp_path))
    findings = collect_violations(relative)
    assert len(findings) == 1
    assert findings[0].format().startswith(f"{relative / 'test_bucket.py'}:1:")


def test_repository_diagnostic_uses_portable_relative_path() -> None:
    """Render an in-repository source diagnostic without its installation prefix."""
    finding = Violation(Path(__file__).resolve(), 1, "source diagnostic")
    assert finding.format() == "tests/test_check_test_quality_policy.py:1: source diagnostic"


def test_cli_repository_relative_root_scans_real_test_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    """Exercise the configured CI scope through a repository-relative CLI root."""
    assert main(["--test-root", "tests"]) == 0
    assert "guard passed" in capsys.readouterr().out
