# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Python lint-scope consistency tests.
"""Exercise the real lint-scope checker over live and malformed surfaces."""

from __future__ import annotations

import doctest
import importlib.util
import os
import pydoc
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools import check_python_lint_contract as gate
from tools.ci_workflow_inventory import ci_workflow_paths


def _write_contract_surfaces(root: Path) -> None:
    """Copy the actual four maintained surfaces without manufacturing command-only fragments."""
    for contract in gate.CONTRACTS:
        path = root / contract.path
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gate.ROOT / contract.path, path)


def test_live_repository_contract_passes() -> None:
    """Require the checked-in repository surfaces to agree."""
    assert gate.lint_contract_errors(gate.ROOT) == []


def test_missing_surface_is_rejected(tmp_path: Path) -> None:
    """Reject a repository that omits any governed surface."""
    errors = gate.lint_contract_errors(tmp_path)
    assert len(errors) == len(gate.CONTRACTS)
    assert errors[0].startswith("missing contract surface:")


@pytest.mark.parametrize("contract_index", range(len(gate.CONTRACTS)))
def test_missing_required_fragment_is_rejected(tmp_path: Path, contract_index: int) -> None:
    """Reject removal of a required scope fragment from each surface."""
    _write_contract_surfaces(tmp_path)
    contract = gate.CONTRACTS[contract_index]
    path = tmp_path / contract.path
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(contract.required[0], ""), encoding="utf-8")
    assert any("missing required fragment" in error for error in gate.lint_contract_errors(tmp_path))


@pytest.mark.parametrize(
    "contract",
    [contract for contract in gate.CONTRACTS if contract.forbidden],
)
def test_forbidden_broad_scope_is_rejected(tmp_path: Path, contract: gate.SurfaceContract) -> None:
    """Reject the historical broad test-lint forms explicitly."""
    _write_contract_surfaces(tmp_path)
    path = tmp_path / contract.path
    with path.open("a", encoding="utf-8") as stream:
        stream.write(contract.forbidden[0])
    assert any("forbidden broad lint scope" in error for error in gate.lint_contract_errors(tmp_path))


def test_generated_ledger_spellcheck_exclusion_is_required(tmp_path: Path) -> None:
    """Reject a spellcheck hook that can rewrite deterministic ledger IDs."""
    _write_contract_surfaces(tmp_path)
    path = tmp_path / ".pre-commit-config.yaml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(gate.TYPO_LEDGER_EXCLUSION, ""), encoding="utf-8")
    assert any("coverage_exception_ledger" in error for error in gate.lint_contract_errors(tmp_path))


def test_cli_reports_success_and_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Expose deterministic CLI status and diagnostics for automation."""
    _write_contract_surfaces(tmp_path)
    assert gate.main(["--repo", str(tmp_path)]) == 0
    assert "PASS:" in capsys.readouterr().out
    (tmp_path / "Makefile").unlink()
    assert gate.main(["--repo", str(tmp_path)]) == 1
    assert "FAIL:" in capsys.readouterr().out


def _cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Inspect copied production surfaces through the actual stdlib-only command."""
    return subprocess.run(
        [sys.executable, "-S", str(gate.ROOT / "tools/check_python_lint_contract.py"), "--repo", str(root), *args],
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.parametrize("index", range(len(gate.CONTRACTS)))
def test_actual_cli_refuses_each_missing_surface(tmp_path: Path, index: int) -> None:
    """Removing each actual configuration file makes the public API and command fail deterministically."""
    _write_contract_surfaces(tmp_path)
    contract = gate.CONTRACTS[index]
    (tmp_path / contract.path).unlink()
    errors = gate.lint_contract_errors(tmp_path)
    assert errors == [f"missing contract surface: {contract.path.as_posix()}"]
    result = _cli(tmp_path)
    assert result.returncode == 1 and result.stderr == ""
    assert errors[0] in result.stdout


@pytest.mark.parametrize("index", range(len(gate.CONTRACTS)))
def test_invalid_utf8_is_an_authored_surface_refusal(tmp_path: Path, index: int) -> None:
    """Invalid bytes in an actual copied surface cannot expose a Unicode decoder exception."""
    _write_contract_surfaces(tmp_path)
    contract = gate.CONTRACTS[index]
    target = tmp_path / contract.path
    target.write_bytes(target.read_bytes() + b"\xff")
    errors = gate.lint_contract_errors(tmp_path)
    assert errors == [f"could not read UTF-8 contract surface: {contract.path.as_posix()}"]
    result = _cli(tmp_path)
    assert result.returncode == 1 and result.stderr == "" and errors[0] in result.stdout


def test_unreadable_or_undecodable_surface_has_fixed_text(tmp_path: Path) -> None:
    """POSIX read denial and platforms retaining read access both produce the same authored read refusal."""
    _write_contract_surfaces(tmp_path)
    target = tmp_path / "Makefile"
    mode = stat.S_IMODE(target.stat().st_mode)
    target.write_bytes(target.read_bytes() + b"\xff")
    target.chmod(0)
    try:
        errors = gate.lint_contract_errors(tmp_path)
        assert errors == ["could not read UTF-8 contract surface: Makefile"]
        result = _cli(tmp_path)
        assert result.returncode == 1 and result.stderr == "" and errors[0] in result.stdout
    finally:
        target.chmod(mode)


def test_root_resolution_loop_has_fixed_cli_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A real looping root symlink fails resolution in both public command entrypoints."""
    root = tmp_path / "loop"
    root.symlink_to(root)
    assert gate.main(["--repo", str(root)]) == 2
    assert capsys.readouterr().out == "FAIL: could not resolve Python lint contract repository\n"
    result = _cli(root)
    assert result.returncode == 2 and result.stderr == ""
    assert result.stdout == "FAIL: could not resolve Python lint contract repository\n"


def test_actual_crlf_text_normalises_before_matching(tmp_path: Path) -> None:
    """Windows newlines in copied declarations retain the documented text-level contract."""
    _write_contract_surfaces(tmp_path)
    for contract in gate.CONTRACTS:
        target = tmp_path / contract.path
        target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    assert gate.lint_contract_errors(tmp_path) == []
    result = _cli(tmp_path)
    assert result.returncode == 0 and result.stderr == ""
    assert result.stdout == "PASS: required Python lint declarations match repository fragments\n"


def test_comments_are_not_executable_lint_evidence(tmp_path: Path) -> None:
    """Exact fragments inside Make comments pass only declaration inspection, as documented."""
    _write_contract_surfaces(tmp_path)
    target = tmp_path / "Makefile"
    text = target.read_text()
    for fragment in gate.CONTRACTS[0].required:
        text = text.replace(fragment, "#" + fragment)
    target.write_text(text)
    assert gate.lint_contract_errors(tmp_path) == []
    result = _cli(tmp_path)
    assert result.returncode == 0 and "declarations match repository fragments" in result.stdout


def test_native_actual_repository_examples_and_html(tmp_path: Path) -> None:
    """Execute the real repository example and render the owning-language API contract."""
    result = doctest.testmod(gate)
    assert result.failed == 0 and result.attempted == 1
    html = pydoc.HTMLDoc().docmodule(gate)
    (tmp_path / "python_lint_contract.html").write_text(html, encoding="utf-8")
    assert "SurfaceContract" in html and "lint_contract_errors" in html and "fragment" in html


def _live_hook() -> dict[str, object]:
    """Load the unchanged selected hook declaration from the actual production YAML."""
    config = yaml.safe_load((gate.ROOT / ".pre-commit-config.yaml").read_text())
    for repo in config["repos"]:
        if repo["repo"] == "local":
            for hook in repo["hooks"]:
                if hook["id"] == "check-python-lint-contract":
                    return {str(key): value for key, value in hook.items()}
    raise AssertionError("actual Python lint hook is missing")


def _run_actual_hook(
    root: Path, filename: str, *, remove_ci_fragment: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run the production local hook through pre-commit without unrelated remote hooks or canonical Git writes.

    The calling test is skipped where pre-commit is not installed; among the
    hosted jobs only static governance installs it.
    """
    if importlib.util.find_spec("pre_commit") is None:
        pytest.skip("pre-commit is not installed here; among the hosted jobs only static governance installs it")
    _write_contract_surfaces(root)
    shutil.copy2(gate.ROOT / "tools/check_python_lint_contract.py", root / "tools/check_python_lint_contract.py")
    if remove_ci_fragment:
        target = root / gate.CONTRACTS[1].path
        target.write_text(target.read_text().replace(gate.CONTRACTS[1].required[0], ""))
    selected = root / "selected_hook.yaml"
    selected.write_text(yaml.safe_dump({"repos": [{"repo": "local", "hooks": [_live_hook()]}]}, sort_keys=False))
    subprocess.run(["git", "init", "--quiet", str(root)], capture_output=True, check=True)
    environment = dict(os.environ, PRE_COMMIT_HOME=str(root / "precommit_cache"))
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pre_commit",
            "run",
            "check-python-lint-contract",
            "--config",
            str(selected),
            "--files",
            filename,
            "--verbose",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize(
    "filename",
    [ci_workflow_paths()[0].relative_to(gate.ROOT).as_posix(), ".github/workflows/ci-static-governance.yml"],
)
def test_actual_hook_selects_both_governance_workflow_inputs(tmp_path: Path, filename: str) -> None:
    """The production local hook reaches its real command for entry and static-governance YAML changes."""
    if filename == ci_workflow_paths()[0].relative_to(gate.ROOT).as_posix():
        target = tmp_path / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ci_workflow_paths()[0], target)
    result = _run_actual_hook(tmp_path, filename)
    assert result.returncode == 0 and result.stderr == ""
    assert "Skipped" not in result.stdout and "Passed" in result.stdout
    assert "PASS: required Python lint declarations match repository fragments" in result.stdout


def test_actual_hook_refuses_static_governance_drift(tmp_path: Path) -> None:
    """The real hook runs and refuses removed CI declarations instead of skipping the changed workflow."""
    result = _run_actual_hook(tmp_path, ".github/workflows/ci-static-governance.yml", remove_ci_fragment=True)
    assert result.returncode == 1 and result.stderr == ""
    assert "Skipped" not in result.stdout and "Failed" in result.stdout
    assert "missing required fragment" in result.stdout and "ci-static-governance.yml" in result.stdout


def test_stale_hook_filter_is_a_contract_refusal(tmp_path: Path) -> None:
    """Returning the hook filter to its stale workflow spelling fails the strengthened self-contract."""
    _write_contract_surfaces(tmp_path)
    target = tmp_path / ".pre-commit-config.yaml"
    target.write_text(target.read_text().replace("ci(?:-static-governance)?", "ci"))
    errors = gate.lint_contract_errors(tmp_path)
    assert len(errors) == 1 and "check-python-lint-contract" in errors[0]
    result = _cli(tmp_path)
    assert result.returncode == 1 and result.stderr == "" and errors[0] in result.stdout
