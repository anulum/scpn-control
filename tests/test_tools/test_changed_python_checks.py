# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native changed-Python static-profile qualification.
"""Exercise real Ruff, mypy, Git and physical CLI failure propagation."""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_tools.test_source_quality_scope import ROOT, _git, _write
from test_tools.test_source_quality_scope import git_repo as git_repo

from tools.changed_python_checks import changed_python_errors, main
from tools.preflight import GATES, run_gate
from tools.source_quality_scope import _git as production_git

_VALID = '"""Describe a native profile control."""\n\nVALUE: int = 1\n'


@pytest.fixture
def python_repo(git_repo: tuple[Path, str]) -> tuple[Path, str]:
    """Materialise only the authentic native configuration and one small owner."""
    repo, base = git_repo
    _write(repo, "pyproject.toml", _git(repo, "show", f"{base}:pyproject.toml"))
    _write(repo, "new_root/owner.py", _VALID.encode())
    return repo, base


def _cli(repo: Path, base: str, *, isolated: bool = False) -> subprocess.CompletedProcess[str]:
    """Execute the physical CLI with real installed or deliberately absent tools."""
    command = [sys.executable, "-B"]
    coverage_rc = os.environ.get("SCPN_CHANGED_PYTHON_COVERAGE_RC")
    if isolated:
        command += ["-I", "-S"]
        if coverage_rc:
            bootstrap = (
                "import sys,runpy;"
                f"sys.path.insert(0,{sysconfig.get_path('purelib')!r});"
                "import coverage;"
                f"c=coverage.Coverage(config_file={coverage_rc!r},data_suffix=True);c.start();"
                "sys.argv=sys.argv[1:]\n"
                "try:\n runpy.run_path(sys.argv[0],run_name='__main__')\n"
                "finally:\n c.stop();c.save()\n"
            )
            command += ["-c", bootstrap]
    elif coverage_rc:
        command += ["-m", "coverage", "run", "--parallel-mode", f"--rcfile={coverage_rc}"]
    return subprocess.run(
        [*command, str(ROOT / "tools/changed_python_checks.py"), "--repo", str(repo), "--base", base],
        capture_output=True,
        text=True,
        check=False,
    )


def test_new_root_owner_passes_actual_native_tools(python_repo: tuple[Path, str]) -> None:
    """A complete typed owner reaches real Ruff, format and mypy outside old roots."""
    repo, base = python_repo
    assert changed_python_errors(repo, base) == []
    result = _cli(repo, base)
    assert result.returncode == 0, result.stderr
    assert "static profile passed" in result.stdout


@pytest.mark.parametrize(
    "content,diagnostic",
    [
        (_VALID + "\nimport os  # noqa: F401\n", "Ruff failed"),
        (_VALID.replace("VALUE: int = 1", "VALUE: int = 'wrong'"), "mypy failed"),
        (_VALID.replace("VALUE: int = 1", "VALUE: int=1"), "format failed"),
        (_VALID + "\n\ndef _hidden() -> int:\n    return VALUE\n", "_hidden"),
        ('# ruff: noqa\n"""Document a whole-file suppression control."""\n\nimport os\n', "Ruff failed"),
    ],
)
def test_complete_file_native_defects_cannot_hide_behind_directory_defaults(
    python_repo: tuple[Path, str], content: str, diagnostic: str
) -> None:
    """Each real native checker and private-definition documentation can reject."""
    repo, base = python_repo
    _write(repo, "new_root/owner.py", content.encode())
    errors = changed_python_errors(repo, base)
    assert any(diagnostic in item for item in errors), errors
    result = _cli(repo, base)
    assert result.returncode == 1 and diagnostic in result.stderr


@pytest.mark.parametrize(
    "directive", ["# type: ignore[assignment]", "# mypy: ignore-errors", "# fmt: off", "# fmt: skip"]
)
def test_actual_comment_waivers_cannot_certify_selected_source(python_repo: tuple[Path, str], directive: str) -> None:
    """Native tools may honour a waiver, but whole-file admission still refuses it."""
    repo, base = python_repo
    _write(repo, "new_root/owner.py", (_VALID + directive + "\n").encode())
    assert any("waiver requires explicit review" in item for item in changed_python_errors(repo, base))


def test_fixture_strings_do_not_masquerade_as_active_waivers(python_repo: tuple[Path, str]) -> None:
    """A source string can describe invalid controls without granting exceptions."""
    repo, base = python_repo
    _write(
        repo, "new_root/owner.py", (_VALID + 'TEXT = "# type: ignore; # mypy: ignore-errors; # fmt: off"\n').encode()
    )
    assert changed_python_errors(repo, base) == []


def test_nested_configuration_cannot_weaken_explicit_original_profile(python_repo: tuple[Path, str]) -> None:
    """Nested Ruff discovery cannot remove selected rules for a new directory."""
    repo, base = python_repo
    _write(repo, "new_root/ruff.toml", b"[lint]\nselect = []\n")
    _write(repo, "new_root/owner.py", (_VALID + "\nimport os\n").encode())
    assert any("F401" in item for item in changed_python_errors(repo, base))
    config = repo / "pyproject.toml"
    config.write_text(config.read_text().replace('select = ["E",', 'select = ["E000",'), encoding="utf-8")
    with pytest.raises(RuntimeError, match="original native configuration"):
        changed_python_errors(repo, base)


def test_syntax_error_and_real_native_timeout_refuse_measurement(python_repo: tuple[Path, str]) -> None:
    """Unavailable parsing or unfinished native execution cannot pass a gate."""
    repo, base = python_repo
    _write(repo, "new_root/owner.py", b"def invalid(:\n")
    with pytest.raises(RuntimeError, match="readable sources"):
        changed_python_errors(repo, base)
    result = _cli(repo, base)
    assert result.returncode == 2 and "refused" in result.stderr
    _write(repo, "new_root/owner.py", _VALID.encode())
    with pytest.raises(RuntimeError):
        changed_python_errors(repo, base, timeout_s=0.05)


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_invalid_native_timeout_refuses(python_repo: tuple[Path, str], timeout: float) -> None:
    """Invalid command budgets do not silently disable bounds."""
    repo, base = python_repo
    with pytest.raises(RuntimeError, match="finite and positive"):
        changed_python_errors(repo, base, timeout_s=timeout)


def test_absent_native_distributions_are_real_failures(python_repo: tuple[Path, str]) -> None:
    """The real isolated interpreter lacks Ruff/mypy and cannot emit success."""
    repo, base = python_repo
    result = _cli(repo, base, isolated=True)
    assert result.returncode == 1 and "No module named" in result.stderr
    assert "static profile passed" not in result.stdout


def test_no_changed_python_and_unavailable_default_ledger(
    git_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Other-language selection makes no Python claims; a missing baseline refuses."""
    repo, base = git_repo
    _write(repo, "new/main.rs")
    assert changed_python_errors(repo, base) == []
    assert main(["--repo", str(repo), "--base", base]) == 0
    assert "other native profiles remain separate" in capsys.readouterr().out
    assert main(["--repo", str(repo)]) == 2
    assert "ledger" in capsys.readouterr().err


def test_real_ledger_default_is_resolved_before_native_checks(python_repo: tuple[Path, str]) -> None:
    """The default CLI cannot replace its original source anchor with HEAD."""
    repo, _base = python_repo
    _write(repo, "tools/docstring_debt_ceiling.json", (ROOT / "tools/docstring_debt_ceiling.json").read_bytes())
    assert main(["--repo", str(repo)]) == 0


def test_source_mutation_after_selection_refuses(
    python_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutate real bytes after the actual Git read, preserving every native result."""
    repo, base = python_repo

    def read_then_change(root: Path, args: Sequence[str], timeout_s: float) -> bytes:
        """Return real Git bytes and alter the physical owner at the known boundary."""
        result = production_git(root, args, timeout_s)
        _write(repo, "new_root/owner.py", _VALID.replace("= 1", "= 2").encode())
        return result

    monkeypatch.setattr("tools.changed_python_checks._git", read_then_change)
    with pytest.raises(RuntimeError, match="during measurement"):
        changed_python_errors(repo, base)


@pytest.mark.parametrize("new_owner", [False, True])
def test_source_mutation_after_real_native_checks_refuses(
    python_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, new_owner: bool
) -> None:
    """Changed final bytes cannot reuse actual successful lint/type subprocesses."""
    repo, base = python_repo
    actual_run = subprocess.run

    def run_then_change(
        argv: list[str],
        *,
        cwd: Path | None = None,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        """Preserve the real command result and then alter its measured source."""
        result = actual_run(
            argv, cwd=cwd, capture_output=capture_output, check=check, timeout=timeout, text=text, env=env
        )
        if "mypy" in argv:
            name = "late_added.py" if new_owner else "new_root/owner.py"
            _write(repo, name, _VALID.replace("= 1", "= 2").encode())
        return result

    monkeypatch.setattr("tools.changed_python_checks.subprocess.run", run_then_change)
    with pytest.raises(RuntimeError, match="during native checks"):
        changed_python_errors(repo, base)


def test_registered_preflight_gate_propagates_native_success_and_failure(python_repo: tuple[Path, str]) -> None:
    """Execute the registered production command through the actual gate runner."""
    repo, base = python_repo
    name, command, cwd = next(gate for gate in GATES if gate[0] == "changed-python-static")
    command = [*command, "--repo", str(repo), "--base", base]
    assert run_gate(name, command, cwd)
    _write(repo, "new_root/owner.py", _VALID.replace("= 1", '= "wrong"').encode())
    assert not run_gate(name, command, cwd)
