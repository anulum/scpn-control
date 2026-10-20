# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real Bash parser, formatter and ShellCheck profile contracts.
"""Exercise native static tools without invoking script operational commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_tools.test_source_quality_scope import ROOT, _write
from test_tools.test_source_quality_scope import git_repo as git_repo

from tools.preflight import GATES, run_gate
from tools.shell_source_checks import _VERSIONS, _ast_findings, changed_shell_errors, main
from tools.shell_source_checks import _native as actual_native

_VALID = (
    "#!/usr/bin/env bash\nset -euo pipefail\n\n"
    "# Print a fixed label without modifying files or retaining state.\n"
    "print_label() {\n    printf '%s\\n' 'ok'\n}\n\nprint_label\n"
)


@pytest.fixture
def shell_repo(git_repo: tuple[Path, str]) -> tuple[Path, str]:
    """Supply a standalone script in a new root using real original Git objects."""
    repo, base = git_repo
    _write(repo, "new-root/control.sh", _VALID.encode())
    return repo, base


def _cli(repo: Path, base: str) -> subprocess.CompletedProcess[str]:
    """Execute the physical script with optional original-process coverage."""
    argv = [sys.executable, "-B"]
    coverage_rc = os.environ.get("SCPN_SHELL_SOURCE_COVERAGE_RC")
    if coverage_rc:
        argv += ["-m", "coverage", "run", "--parallel-mode", f"--rcfile={coverage_rc}"]
    return subprocess.run(
        [*argv, str(ROOT / "tools/shell_source_checks.py"), "--repo", str(repo), "--base", base],
        capture_output=True,
        text=True,
        check=False,
    )


def test_actual_native_profile_and_cli_accept_valid_new_root(shell_repo: tuple[Path, str]) -> None:
    """All actual native static tools measure the complete source through public entrypoints."""
    repo, base = shell_repo
    assert changed_shell_errors(repo, base) == []
    result = _cli(repo, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "static profile passed" in result.stdout


@pytest.mark.parametrize(
    "content,finding",
    [
        ("#!/usr/bin/env bash\nset -euo pipefail\nif then\n", "Bash syntax failed"),
        (_VALID.replace("'ok'", "$label"), "ShellCheck failed"),
        (_VALID.replace("    printf", " printf"), "shfmt failed"),
        (_VALID.replace("set -euo pipefail\n", ""), "first statement"),
        (_VALID.replace("-euo", "-eu"), "first statement"),
        (_VALID + "\nset +e\n", "shell mode changes"),
        (
            _VALID.replace("# Print a fixed label without modifying files or retaining state.\n", ""),
            "adjacent contract",
        ),
    ],
)
def test_actual_native_quality_failures_propagate(shell_repo: tuple[Path, str], content: str, finding: str) -> None:
    """Syntax, lint, format, strict-mode and documentation each reject actual defects."""
    repo, base = shell_repo
    _write(repo, "new-root/control.sh", content.encode())
    result = _cli(repo, base)
    assert result.returncode == 1 and finding in result.stderr


@pytest.mark.parametrize(
    "directive",
    ["# shellcheck disable=SC2154", "# shellcheck source=/dev/null", "# shfmt: skip"],
)
def test_actual_comment_directives_cannot_silence_the_profile(shell_repo: tuple[Path, str], directive: str) -> None:
    """Use actual parser comment nodes, including directives on otherwise valid code."""
    repo, base = shell_repo
    _write(repo, "new-root/control.sh", (_VALID + directive + "\n").encode())
    assert any("inline shell-tool" in item for item in changed_shell_errors(repo, base))


def test_quoted_and_heredoc_data_are_not_tool_directives(shell_repo: tuple[Path, str]) -> None:
    """The actual parser distinguishes string/heredoc content from real comments."""
    repo, base = shell_repo
    content = (
        _VALID + "\nprintf '%s\\n' '# shellcheck disable=SC2154'\n"
        "cat <<'END'\n# shellcheck disable=SC2154\n# shfmt: skip\nEND\n"
    )
    _write(repo, "new-root/control.sh", content.encode())
    assert changed_shell_errors(repo, base) == []


def test_static_tools_do_not_execute_commands_or_startup_files(
    shell_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real operational and injected-startup side effects remain absent after checks."""
    repo, base = shell_repo
    sentinel = repo / "executed"
    _write(repo, "new-root/control.sh", (_VALID + f"\ntouch -- '{sentinel}'\n").encode())
    startup = tmp_path.parent / (tmp_path.name + "-startup.sh")
    startup.write_text(f"touch -- '{sentinel}'\n", encoding="utf-8")
    monkeypatch.setenv("BASH_ENV", str(startup))
    monkeypatch.setenv("ENV", str(startup))
    assert changed_shell_errors(repo, base) == []
    assert not sentinel.exists()


def test_environment_and_editor_configuration_cannot_weaken_checks(
    shell_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinned command flags ignore per-directory format and process-wide lint overrides."""
    repo, base = shell_repo
    _write(repo, ".editorconfig", b"root = true\n[*]\nindent_style = tab\nindent_size = 8\nignore = true\n")
    assert changed_shell_errors(repo, base) == []
    _write(repo, "new-root/control.sh", _VALID.replace("'ok'", "$label").encode())
    monkeypatch.setenv("SHELLCHECK_OPTS", "--exclude=SC2154,SC2086")
    assert any("ShellCheck failed" in item for item in changed_shell_errors(repo, base))


@pytest.mark.parametrize("shebang", ["#!/bin/sh", "#!/usr/bin/env python3", "#!/bin/bash\r", ""])
def test_other_or_unavailable_interpreter_contracts_refuse(shell_repo: tuple[Path, str], shebang: str) -> None:
    """A Bash-only static pass cannot certify a different runtime interpreter."""
    repo, base = shell_repo
    content = _VALID.replace("#!/usr/bin/env bash", shebang)
    _write(repo, "new-root/control.sh", content.encode())
    with pytest.raises(RuntimeError, match="interpreter requires"):
        changed_shell_errors(repo, base)


def test_empty_source_is_not_a_completed_implementation(shell_repo: tuple[Path, str]) -> None:
    """A comment-only file cannot replace the selected source with an empty measurement."""
    repo, base = shell_repo
    _write(repo, "new-root/control.sh", b"#!/usr/bin/env bash\n# No statements.\n")
    with pytest.raises(RuntimeError, match="syntax graph|contain statements"):
        changed_shell_errors(repo, base)


@pytest.mark.parametrize("tree", [None, [], {"Type": "File"}, {"Type": "File", "Stmts": []}])
def test_invalid_parser_envelopes_refuse(tree: object) -> None:
    """Negative protocol controls cannot masquerade as a complete native AST."""
    with pytest.raises(RuntimeError):
        _ast_findings(tree, "invalid.sh")


@pytest.mark.parametrize("tool", ["bash", "shellcheck", "shfmt"])
def test_native_versions_are_compared_to_the_actual_qualified_declaration(
    shell_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """Real installed version output rejects an intentionally invalid profile declaration."""
    repo, base = shell_repo
    monkeypatch.setitem(_VERSIONS, tool, "invalid-profile-version")
    with pytest.raises(RuntimeError, match="qualified"):
        changed_shell_errors(repo, base)


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_invalid_native_timeout_refuses(shell_repo: tuple[Path, str], timeout: float) -> None:
    """An unavailable resource bound cannot silently disable native limits."""
    repo, base = shell_repo
    with pytest.raises(RuntimeError, match="finite and positive"):
        changed_shell_errors(repo, base, timeout_s=timeout)


def test_real_missing_executable_and_process_timeout_refuse(
    shell_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep real Git available while the actual native tools are absent or timed out."""
    repo, base = shell_repo
    with pytest.raises(RuntimeError, match="completed native"):
        changed_shell_errors(repo, base, timeout_s=1e-12)
    tools = tmp_path.parent / (tmp_path.name + "-git-only")
    tools.mkdir()
    git = shutil.which("git")
    assert git is not None
    (tools / "git").symlink_to(git)
    monkeypatch.setenv("PATH", str(tools))
    with pytest.raises(RuntimeError, match="completed native"):
        changed_shell_errors(repo, base)


def test_absent_shell_cohort_and_default_origin_refusal(
    git_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """A different language is outside this verdict, not a missing-baseline fallback."""
    repo, base = git_repo
    _write(repo, "new/owner.py")
    assert changed_shell_errors(repo, base) == []
    assert main(["--repo", str(repo), "--base", base]) == 0
    assert "other native profiles remain separate" in capsys.readouterr().out
    assert main(["--repo", str(repo)]) == 2
    assert "ledger" in capsys.readouterr().err


def test_actual_owner_ledger_supplies_the_default_origin(shell_repo: tuple[Path, str]) -> None:
    """The mandatory original commit reaches real native checks through the default CLI."""
    repo, _base = shell_repo
    _write(repo, "tools/docstring_debt_ceiling.json", (ROOT / "tools/docstring_debt_ceiling.json").read_bytes())
    assert main(["--repo", str(repo)]) == 0


def test_literal_arguments_and_positional_assignment_preserve_strict_mode(shell_repo: tuple[Path, str]) -> None:
    """Real expansions and set -- change positional arguments without weakening options."""
    repo, base = shell_repo
    content = _VALID + '\nset -- "one two"\nreadonly label=ok\nprintf \'%s\\n\' prefix"${label}" "$1"\n'
    _write(repo, "new-root/control.sh", content.encode())
    assert changed_shell_errors(repo, base) == []


def test_license_header_cannot_replace_a_function_contract(shell_repo: tuple[Path, str]) -> None:
    """An adjacent project identity line is not the function's native documentation."""
    repo, base = shell_repo
    content = _VALID.replace(
        "# Print a fixed label without modifying files or retaining state.", "# SCPN Control — Example."
    )
    _write(repo, "new-root/control.sh", content.encode())
    assert any("adjacent contract" in item for item in changed_shell_errors(repo, base))


def test_invalid_source_encoding_refuses(shell_repo: tuple[Path, str]) -> None:
    """Unreadable complete source cannot be treated as an empty native measurement."""
    repo, base = shell_repo
    _write(repo, "new-root/control.sh", _VALID.encode() + b"\xff")
    with pytest.raises(RuntimeError, match="readable sources"):
        changed_shell_errors(repo, base)


@pytest.mark.parametrize("new_owner", [False, True])
def test_final_source_changes_refuse_after_real_native_results(
    shell_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, new_owner: bool
) -> None:
    """Keep actual native outcomes and change a physical input only after measurement."""
    repo, base = shell_repo

    def run_then_change(
        root: Path, argv: Sequence[str], timeout_s: float, *, text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Return the genuine native result before altering its measured source set."""
        result = actual_native(root, argv, timeout_s, text=text)
        if "--diff" in argv:
            path = "new-root/late.sh" if new_owner else "new-root/control.sh"
            _write(repo, path, _VALID.replace("'ok'", "'changed'").encode())
        return result

    monkeypatch.setattr("tools.shell_source_checks._native", run_then_change)
    with pytest.raises(RuntimeError, match="selection changed"):
        changed_shell_errors(repo, base)


def test_corrupted_native_protocol_is_a_refusal_not_positive_evidence(
    shell_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject only a negative framing fault after an actual parser process completed."""
    repo, base = shell_repo

    def corrupt_protocol(
        root: Path, argv: Sequence[str], timeout_s: float, *, text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Preserve native execution and damage the returned protocol envelope."""
        result = actual_native(root, argv, timeout_s, text=text)
        if "--to-json" in argv:
            result.stdout = json.dumps({"Type": "File", "Stmts": [{"Cmd": {"Type": "CallExpr", "Args": "invalid"}}]})
        return result

    monkeypatch.setattr("tools.shell_source_checks._native", corrupt_protocol)
    with pytest.raises(RuntimeError, match="complete native syntax object"):
        changed_shell_errors(repo, base)


def test_registered_preflight_native_gate_propagates_both_outcomes(shell_repo: tuple[Path, str]) -> None:
    """Execute the registered public command through the actual preflight gate runner."""
    repo, base = shell_repo
    name, command, cwd = next(gate for gate in GATES if gate[0] == "changed-shell-static")
    command = [*command, "--repo", str(repo), "--base", base]
    assert run_gate(name, command, cwd)
    _write(repo, "new-root/control.sh", _VALID.replace("'ok'", "$label").encode())
    assert not run_gate(name, command, cwd)
