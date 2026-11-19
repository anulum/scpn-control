# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Python preflight wrapper tests.
"""Regression tests for the fast Python preflight command list."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import run_python_preflight
from tools.run_python_preflight import _build_checks

ROOT = Path(__file__).resolve().parents[1]


def _commands_by_name(*, skip_version_metadata: bool = False) -> dict[str, list[str]]:
    """Return preflight command lists keyed by check name."""
    return dict(
        _build_checks(
            skip_version_metadata=skip_version_metadata,
            skip_notebook_quality=True,
            skip_threshold_smoke=True,
            skip_mypy=True,
        )
    )


def test_preflight_uses_live_project_metadata_test() -> None:
    """The version metadata gate points at the live project metadata test."""
    commands = _commands_by_name()
    command = commands["Version metadata consistency"]

    assert "tests/test_project_metadata.py" in command
    assert "tests/test_version_metadata.py" not in command


def test_preflight_skip_version_metadata_removes_project_metadata_gate() -> None:
    """The skip flag removes the metadata test from the command list."""
    commands = _commands_by_name(skip_version_metadata=True)

    assert "Version metadata consistency" not in commands


@pytest.mark.parametrize("metadata", [False, True])
def test_preflight_main_executes_selected_real_checks(
    metadata: bool, capfd: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the actual metadata selection and mandatory doc gate outside repo cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_ADDOPTS", f"-o cache_dir={tmp_path / 'cache'} --basetemp={tmp_path / 'children'}")
    argv = ["--skip-notebook-quality", "--skip-threshold-smoke", "--skip-mypy"]
    if not metadata:
        argv.append("--skip-version-metadata")

    assert run_python_preflight.main(argv) == 0
    output = capfd.readouterr()
    assert "[preflight] docstring coverage:" in output.out
    assert "[preflight] All selected checks passed." in output.out
    assert ("[preflight] Version metadata consistency:" in output.out) is metadata
    assert "[preflight] FAILED" not in output.err


def test_preflight_cli_executes_mandatory_doc_gate_from_other_cwd(tmp_path: Path) -> None:
    """Run the installed-interpreter command with every optional check omitted."""
    argv = [
        sys.executable,
        str(ROOT / "tools/run_python_preflight.py"),
        "--skip-version-metadata",
        "--skip-notebook-quality",
        "--skip-threshold-smoke",
        "--skip-mypy",
    ]
    result = subprocess.run(argv, cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "[preflight] docstring coverage:" in result.stdout
    assert "[preflight] All selected checks passed." in result.stdout
    assert "[preflight] FAILED" not in result.stderr


def test_preflight_cli_rejects_unknown_selection_flag(tmp_path: Path) -> None:
    """Reject an unsupported flag before executing any child check."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/run_python_preflight.py"), "--skip-docstrings"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "unrecognized arguments: --skip-docstrings" in result.stderr


def test_full_profile_orders_every_check_with_the_docstring_gate_last() -> None:
    """Without skip flags the profile runs metadata, notebook, threshold smoke, mypy, then the docstring gate."""
    checks = _build_checks(
        skip_version_metadata=False, skip_notebook_quality=False, skip_threshold_smoke=False, skip_mypy=False
    )
    assert [name for name, _ in checks] == [
        "Version metadata consistency",
        "Golden notebook quality gate",
        "Task 5/6 threshold smoke",
        "mypy strict",
        "docstring coverage",
    ]
    commands = dict(checks)
    assert commands["Golden notebook quality gate"][-2] == "tests/test_neuro_symbolic_control_demo_notebook.py"
    assert all("::test_task" in node for node in commands["Task 5/6 threshold smoke"][3:5])
    assert commands["mypy strict"] == [sys.executable, "tools/run_mypy_strict.py"]


def test_first_failing_check_stops_the_run_with_its_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first nonzero child status is returned and no later check runs."""
    seen: list[str] = []

    def fake_check(name: str, _cmd: list[str]) -> int:
        """Record the check and fail the mypy step."""
        seen.append(name)
        return 3 if name == "mypy strict" else 0

    monkeypatch.setattr(run_python_preflight, "_run_check", fake_check)
    status = run_python_preflight.main(["--skip-version-metadata", "--skip-notebook-quality", "--skip-threshold-smoke"])
    assert status == 3
    assert seen == ["mypy strict"]
    assert "[preflight] FAILED at 'mypy strict' with exit code 3." in capsys.readouterr().err
