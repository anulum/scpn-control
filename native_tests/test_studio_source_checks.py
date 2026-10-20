# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real Node and Studio native-profile qualification.
"""Exercise native compiler enrollment, lint, formatting and CLI refusals.

The explicit native-test job supplies Node/pnpm and a frozen Studio install.
These tests reuse dependency packages read-only and create no second install.
"""

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
from tools.studio_source_checks import _TOOL_VERSIONS, changed_studio_errors, main
from tools.studio_source_checks import _native as actual_native

_VALID = "export const value: number = 1;\n"


@pytest.fixture
def studio_repo(git_repo: tuple[Path, str]) -> tuple[Path, str]:
    """Reuse actual qualified configurations and the existing dependency graph."""
    repo, base = git_repo
    for name in [
        "eslint.config.mjs",
        "tsconfig.json",
        "package.json",
        "pnpm-lock.yaml",
        ".prettierrc.json",
        ".gitignore",
        "vitest.config.ts",
    ]:
        _write(repo, "studio-web/" + name, (ROOT / "studio-web" / name).read_bytes())
    # Git treats a reused dependency-directory symlink as a file, not a directory.
    (repo / ".git/info/exclude").write_text("studio-web/node_modules\n", encoding="utf-8")
    (repo / "studio-web/node_modules").symlink_to(ROOT / "studio-web/node_modules", target_is_directory=True)
    _write(repo, "studio-web/src/control.ts", _VALID.encode())
    return repo, base


def _cli(repo: Path, base: str) -> subprocess.CompletedProcess[str]:
    """Run the physical production CLI, retaining optional real-process coverage."""
    argv = [sys.executable, "-B"]
    coverage_rc = os.environ.get("SCPN_STUDIO_SOURCE_COVERAGE_RC")
    if coverage_rc:
        argv += ["-m", "coverage", "run", "--parallel-mode", f"--rcfile={coverage_rc}"]
    return subprocess.run(
        [*argv, str(ROOT / "tools/studio_source_checks.py"), "--repo", str(repo), "--base", base],
        capture_output=True,
        text=True,
        check=False,
    )


def _package_metadata(repo: Path, content: str) -> None:
    """Replace only fixture metadata, leaving the shared real packages untouched."""
    modules = repo / "studio-web/node_modules"
    modules.unlink()
    target = modules / "react/package.json"
    target.parent.mkdir(parents=True)
    target.write_text(content, encoding="utf-8")


def test_actual_studio_profile_and_physical_cli_accept_enrolled_source(studio_repo: tuple[Path, str]) -> None:
    """Every native tool executes on the actual configuration and typed source."""
    repo, base = studio_repo
    assert changed_studio_errors(repo, base) == []
    result = _cli(repo, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "static profile passed" in result.stdout


@pytest.mark.parametrize(
    "content,diagnostic",
    [
        ("export const value: number = 'wrong';\n", "TypeScript failed"),
        (_VALID + "const unused = 2;\n", "ESLint failed"),
        ('export const value = "valid";\n', "Prettier failed"),
    ],
)
def test_each_real_native_checker_propagates_its_failure(
    studio_repo: tuple[Path, str], content: str, diagnostic: str
) -> None:
    """Native findings cannot become a successful first-code static verdict."""
    repo, base = studio_repo
    _write(repo, "studio-web/src/control.ts", content.encode())
    result = _cli(repo, base)
    assert result.returncode == 1 and diagnostic in result.stderr


@pytest.mark.parametrize("directive", ["// @ts-ignore", "// @ts-expect-error", "// @ts-nocheck", "// prettier-ignore"])
def test_actual_source_comment_waivers_cannot_silence_checks(studio_repo: tuple[Path, str], directive: str) -> None:
    """Use the real ESLint comment parser rather than scanning fixture strings."""
    repo, base = studio_repo
    _write(repo, "studio-web/src/control.ts", (directive + "\n" + _VALID).encode())
    errors = changed_studio_errors(repo, base)
    assert any("no-warning-comments" in item for item in errors), errors


def test_literal_strings_are_not_active_waivers(studio_repo: tuple[Path, str]) -> None:
    """Describing an invalid control in a string grants no suppression."""
    repo, base = studio_repo
    content = "export const example = '@ts-ignore @ts-expect-error @ts-nocheck prettier-ignore';\n"
    _write(repo, "studio-web/src/control.ts", content.encode())
    assert changed_studio_errors(repo, base) == []


def test_inline_lint_directives_cannot_hide_unused_variables(studio_repo: tuple[Path, str]) -> None:
    """Disabling inline configuration preserves the actual selected lint rules."""
    repo, base = studio_repo
    content = "/* eslint-disable */\n" + _VALID + "const unused = 2;\n"
    _write(repo, "studio-web/src/control.ts", content.encode())
    assert any("ESLint failed" in item for item in changed_studio_errors(repo, base))


@pytest.mark.parametrize("suffix", ["ts", "js", "mjs"])
def test_new_source_outside_actual_compiler_program_refuses(studio_repo: tuple[Path, str], suffix: str) -> None:
    """A successful compiler cannot conceal a source omitted by its includes."""
    repo, base = studio_repo
    _write(repo, "studio-web/unregistered/new." + suffix, b"export const value = 1;\n")
    with pytest.raises(RuntimeError, match="absent from the actual compiler program"):
        changed_studio_errors(repo, base)


def test_new_javascript_root_requires_a_native_profile(git_repo: tuple[Path, str]) -> None:
    """Directory discovery cannot quietly exclude a newly introduced language root."""
    repo, base = git_repo
    _write(repo, "new-root/control.js")
    result = _cli(repo, base)
    assert result.returncode == 2 and "own qualified native profile" in result.stderr


@pytest.mark.parametrize(
    "name", ["eslint.config.mjs", "tsconfig.json", "package.json", "pnpm-lock.yaml", ".prettierrc.json"]
)
def test_configuration_changes_require_requalification(studio_repo: tuple[Path, str], name: str) -> None:
    """A candidate cannot weaken its compiler, formatter, lint or tool declarations."""
    repo, base = studio_repo
    path = repo / "studio-web" / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="native-profile requalification"):
        changed_studio_errors(repo, base)


@pytest.mark.parametrize(
    "content",
    ['{"name":"react","version":"0.0.0"}', '{"name":"other","version":"19.3.0"}', "[]", "{", "[" * 2000],
)
def test_unavailable_or_wrong_package_identity_refuses(studio_repo: tuple[Path, str], content: str) -> None:
    """Actual metadata files cannot silently substitute different or malformed packages."""
    repo, base = studio_repo
    _package_metadata(repo, content)
    with pytest.raises(RuntimeError):
        changed_studio_errors(repo, base)


def test_missing_package_metadata_refuses(studio_repo: tuple[Path, str]) -> None:
    """Missing installation data cannot count as an empty successful measurement."""
    repo, base = studio_repo
    (repo / "studio-web/node_modules").unlink()
    with pytest.raises(RuntimeError, match="readable qualified"):
        changed_studio_errors(repo, base)


def test_candidate_cannot_lower_the_existing_native_coverage_floor(studio_repo: tuple[Path, str]) -> None:
    """The qualified Vitest contract cannot be replaced with a weaker threshold."""
    repo, base = studio_repo
    path = repo / "studio-web/vitest.config.ts"
    before = path.read_text()
    assert "statements: 100" in before
    path.write_text(before.replace("statements: 100", "statements: 99"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="vitest.config.ts requires explicit"):
        changed_studio_errors(repo, base)


def test_real_node_startup_failure_refuses(studio_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """A real Node process rejecting an unsupported option cannot certify a profile."""
    repo, base = studio_repo
    monkeypatch.setenv("NODE_OPTIONS", "--unsupported-studio-profile-option")
    with pytest.raises(RuntimeError, match="requires qualified node"):
        changed_studio_errors(repo, base)


@pytest.mark.parametrize("tool", ["node", "pnpm"])
def test_native_version_must_match_the_declared_qualified_profile(
    studio_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """Compare real native version output against an intentionally invalid declaration."""
    repo, base = studio_repo
    monkeypatch.setitem(_TOOL_VERSIONS, tool, "0.0.0-invalid")
    with pytest.raises(RuntimeError, match="requires qualified"):
        changed_studio_errors(repo, base)


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_time_budgets_refuse(studio_repo: tuple[Path, str], timeout: float) -> None:
    """A Boolean or unavailable bound cannot disable native process limits."""
    repo, base = studio_repo
    with pytest.raises(RuntimeError, match="finite and positive"):
        changed_studio_errors(repo, base, timeout_s=timeout)


def test_real_missing_tool_and_process_timeout_refuse(
    studio_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep Git available while Node is genuinely absent, then time out real Node."""
    repo, base = studio_repo
    with pytest.raises(RuntimeError, match="completed native"):
        changed_studio_errors(repo, base, timeout_s=1e-12)
    tool_dir = tmp_path.parent / (tmp_path.name + "-git-only")
    tool_dir.mkdir()
    actual_git = shutil.which("git")
    assert actual_git is not None
    tool_dir.joinpath("git").symlink_to(actual_git)
    monkeypatch.setenv("PATH", str(tool_dir))
    with pytest.raises(RuntimeError, match="completed native"):
        changed_studio_errors(repo, base)


def test_no_changed_studio_and_missing_default_ledger(
    git_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty Studio cohort makes no other-language claim or baseline fallback."""
    repo, base = git_repo
    _write(repo, "new/owner.py")
    assert changed_studio_errors(repo, base) == []
    assert main(["--repo", str(repo), "--base", base]) == 0
    assert "other native profiles remain separate" in capsys.readouterr().out
    assert main(["--repo", str(repo)]) == 2
    assert "ledger" in capsys.readouterr().err


def test_actual_owner_ledger_supplies_the_default_original_revision(studio_repo: tuple[Path, str]) -> None:
    """The mandatory default origin reaches the native checks without guessing HEAD."""
    repo, _base = studio_repo
    _write(repo, "tools/docstring_debt_ceiling.json", (ROOT / "tools/docstring_debt_ceiling.json").read_bytes())
    assert main(["--repo", str(repo)]) == 0


def test_missing_local_compiler_entrypoint_cannot_fall_back_to_a_global_tool(studio_repo: tuple[Path, str]) -> None:
    """Declare actual package identities while removing only the fixture compiler entrypoint."""
    repo, base = studio_repo
    modules = repo / "studio-web/node_modules"
    modules.unlink()
    modules.mkdir()
    package = json.loads((ROOT / "studio-web/package.json").read_text())
    for name in package["dependencies"] | package["devDependencies"]:
        target = modules / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original = ROOT / "studio-web/node_modules" / name
        if name == "typescript":
            target.mkdir()
            (target / "package.json").write_bytes((original / "package.json").read_bytes())
        else:
            target.symlink_to(original, target_is_directory=True)
    with pytest.raises(RuntimeError, match="compiler enrollment failed"):
        changed_studio_errors(repo, base)


@pytest.mark.parametrize("suffix", ["js", "mjs"])
def test_real_javascript_is_strict_checked_alongside_typescript(studio_repo: tuple[Path, str], suffix: str) -> None:
    """A real JSDoc type error cannot bypass the same native compiler profile."""
    repo, base = studio_repo
    content = "/** @type {number} */\nexport const value = 'wrong';\n"
    _write(repo, "studio-web/src/native_control." + suffix, content.encode())
    assert any("TS2322" in item for item in changed_studio_errors(repo, base))


@pytest.mark.parametrize("mutation", ["existing", "new", "configuration"])
def test_real_native_results_cannot_clear_changed_final_inputs(
    studio_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Mutate real bytes only after actual native results, without faking any tool outcome."""
    repo, base = studio_repo

    def run_then_mutate(cwd: Path, argv: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        """Preserve each real native result and change its measured input afterwards."""
        result = actual_native(cwd, argv, timeout_s)
        if "node_modules/prettier/bin/prettier.cjs" in argv:
            if mutation == "configuration":
                path = repo / "studio-web/.prettierrc.json"
                path.write_bytes(path.read_bytes() + b"\n")
            else:
                name = "late_added.ts" if mutation == "new" else "control.ts"
                _write(repo, "studio-web/src/" + name, _VALID.replace("= 1", "= 2").encode())
        return result

    monkeypatch.setattr("tools.studio_source_checks._native", run_then_mutate)
    with pytest.raises(RuntimeError, match="requalification|selection changed"):
        changed_studio_errors(repo, base)


def test_registered_preflight_command_runs_native_acceptance_and_rejection(studio_repo: tuple[Path, str]) -> None:
    """Execute the production registered command through its actual gate runner."""
    repo, base = studio_repo
    name, command, cwd = next(gate for gate in GATES if gate[0] == "changed-studio-static")
    command = [*command, "--repo", str(repo), "--base", base]
    assert run_gate(name, command, cwd)
    _write(repo, "studio-web/src/control.ts", _VALID.replace("= 1", "= 'wrong'").encode())
    assert not run_gate(name, command, cwd)
