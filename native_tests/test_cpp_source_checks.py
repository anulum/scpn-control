# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Actual C++ compiler, formatter and parser contracts.
"""Exercise the native C++ static profile through real API, CLI and preflight commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_tools.test_source_quality_scope import ROOT, _write
from test_tools.test_source_quality_scope import git_repo as git_repo

from tools.cpp_source_checks import _VERSIONS, changed_cpp_errors, main
from tools.cpp_source_checks import _native as actual_native
from tools.preflight import GATES, run_gate

_VALID = "/// Return the fixed test value.\nint value() { return 1; }\n"


@pytest.fixture
def cpp_repo(git_repo: tuple[Path, str]) -> tuple[Path, str]:
    """Supply a complete physical C++ source against actual original Git objects."""
    repo, base = git_repo
    _write(repo, "new-root/control.cpp", _VALID.encode())
    return repo, base


def _cli(repo: Path, base: str) -> subprocess.CompletedProcess[str]:
    """Run the physical production CLI with optional whole-module branch measurement."""
    argv = [sys.executable, "-B"]
    coverage_rc = os.environ.get("SCPN_CPP_SOURCE_COVERAGE_RC")
    if coverage_rc:
        argv += ["-m", "coverage", "run", "--parallel-mode", f"--rcfile={coverage_rc}"]
    return subprocess.run(
        [*argv, str(ROOT / "tools/cpp_source_checks.py"), "--repo", str(repo), "--base", base],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("suffix", [".cpp", ".h", ".hpp"])
def test_actual_compiler_formatter_and_ast_accept_each_qualified_format(
    git_repo: tuple[Path, str], suffix: str
) -> None:
    """Every enrolled native format passes actual independent compiler and parser entrypoints."""
    repo, base = git_repo
    _write(
        repo,
        "new-root/control" + suffix,
        (_VALID.replace("value()", "value(void)") if suffix == ".h" else _VALID).encode(),
    )
    assert changed_cpp_errors(repo, base) == []
    result = _cli(repo, base)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "content,finding",
    [
        (_VALID.replace("return 1;", "return ;"), "failed"),
        (_VALID.replace("return 1;", "int unused; return 1;"), "g++-13 c++17 failed"),
        (_VALID.replace("return 1;", "return 1.5;"), "clang++-18 c++17 failed"),
        (_VALID.replace("int value()", "int  value()"), "clang-format failed"),
        ("int value() { return 1; }\n", "attached contract"),
        (_VALID.replace("Return the fixed test value.", "SCPN Control — Example."), "attached contract"),
        (_VALID.replace("Return the fixed test value.", "\\param absent Invalid parameter."), "clang++-18"),
        (_VALID + "// clang-format off\n", "inline native-tool"),
        (_VALID + "// NOLINT\n", "inline native-tool"),
        (_VALID + '#pragma GCC diagnostic ignored "-Wall"\n', "native pragma"),
        (_VALID + '_Pragma("GCC diagnostic ignored \\"-Wall\\"")\n', "native pragma"),
    ],
)
def test_actual_native_findings_reach_the_cli(cpp_repo: tuple[Path, str], content: str, finding: str) -> None:
    """Genuine source defects cannot become a positive result or disappear behind a waiver."""
    repo, base = cpp_repo
    _write(repo, "new-root/control.cpp", content.encode())
    result = _cli(repo, base)
    assert result.returncode == 1 and finding in result.stderr, result.stdout + result.stderr


def test_header_c_abi_requires_the_real_c_compiler(git_repo: tuple[Path, str]) -> None:
    """A C++-valid overload cannot pass a header advertised as compatible with C11."""
    repo, base = git_repo
    _write(repo, "new/abi.h", (_VALID + "/// Return a different fixed value.\nint value(int) { return 2; }\n").encode())
    errors = changed_cpp_errors(repo, base)
    assert any("gcc-13 c11 failed" in error for error in errors)
    assert any("clang-18 c11 failed" in error for error in errors)


def test_nested_local_headers_and_native_comments_are_enrolled(cpp_repo: tuple[Path, str]) -> None:
    """An actual include chain gets full formatting, compiler, comment and directive checks."""
    repo, base = cpp_repo
    _write(repo, "new-root/control.cpp", ('#include "types.hpp"\n\n' + _VALID).encode())
    _write(repo, "new-root/types.hpp", b'#pragma once\n#include "abi.h"\n')
    _write(repo, "new-root/abi.h", b"/// Return an implementation-provided value.\nint external_value(void);\n")
    # The forwarding header needs an explicit declaration under this standalone profile.
    with pytest.raises(RuntimeError, match="no explicit main-file"):
        changed_cpp_errors(repo, base)
    _write(
        repo,
        "new-root/types.hpp",
        b'#pragma once\n#include "abi.h"\n\n/// Name the C ABI integer.\nusing Number = int;\n',
    )
    assert changed_cpp_errors(repo, base) == []
    _write(repo, "new-root/abi.h", b"int external_value(void);\n")
    assert any("attached contract" in error for error in changed_cpp_errors(repo, base))


def test_quoted_data_and_multiline_raw_strings_are_not_tool_directives(cpp_repo: tuple[Path, str]) -> None:
    """Only actual Clang comments/directive tokens count; quoted payloads keep their literal meaning."""
    repo, base = cpp_repo
    content = (
        _VALID
        + '/// Preserve diagnostic text as data.\nconst char *label = R"(\n// clang-format off\n#pragma GCC diagnostic ignored "-Wall"\n)";\n'
    )
    _write(repo, "new-root/control.cpp", content.encode())
    assert changed_cpp_errors(repo, base) == []


@pytest.mark.parametrize(
    "content,reason",
    [
        ("// No declarations.\n", "no explicit main-file"),
        (_VALID + '#define LOCAL "abi.h"\n#include LOCAL\n', "literal qualified"),
        (_VALID + '#include_next "abi.h"\n', "literal qualified"),
        (_VALID + '#include "../abi.h"\n', "qualified .h/.hpp"),
        (_VALID + '#include "missing.h"\n', "readable sources"),
        (_VALID + '#include "unknown.inc"\n', "qualified .h/.hpp"),
    ],
)
def test_unqualified_source_and_include_contracts_refuse(cpp_repo: tuple[Path, str], content: str, reason: str) -> None:
    """Incomplete implementations and unresolved include custody cannot certify a native profile."""
    repo, base = cpp_repo
    _write(repo, "new-root/control.cpp", content.encode())
    with pytest.raises(RuntimeError, match=reason):
        changed_cpp_errors(repo, base)


def test_source_bodies_and_environment_include_paths_are_never_executed(
    cpp_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compiler syntax-only mode cannot run even a source with observable runtime side effects."""
    repo, base = cpp_repo
    sentinel = repo / "executed"
    content = '#include <cstdio>\n\n/// Create a file only if the program is executed.\nint main() { return std::fopen("executed", "w") == nullptr; }\n'
    _write(repo, "new-root/control.cpp", content.encode())
    monkeypatch.setenv("CPATH", "/unqualified/include")
    monkeypatch.setenv("CCC_OVERRIDE_OPTIONS", "+-w")
    assert changed_cpp_errors(repo, base) == []
    assert not sentinel.exists()


@pytest.mark.parametrize("tool", list(_VERSIONS))
def test_actual_tool_versions_refuse_a_mismatched_qualification(
    cpp_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """Real native version output rejects a deliberately invalid version declaration."""
    repo, base = cpp_repo
    monkeypatch.setitem(_VERSIONS, tool, (_VERSIONS[tool][0], "invalid-version"))
    with pytest.raises(RuntimeError, match="qualified"):
        changed_cpp_errors(repo, base)


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_invalid_native_resource_bound_refuses(cpp_repo: tuple[Path, str], timeout: float) -> None:
    """Native execution always has an actual finite positive time bound."""
    repo, base = cpp_repo
    with pytest.raises(RuntimeError, match="finite and positive"):
        changed_cpp_errors(repo, base, timeout_s=timeout)


def test_real_absent_executable_and_timeout_refuse(
    cpp_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Genuine process absence and timeout cannot become empty positive measurements."""
    repo, base = cpp_repo
    with pytest.raises(RuntimeError, match="completed native"):
        changed_cpp_errors(repo, base, timeout_s=1e-12)
    tools = tmp_path.parent / (tmp_path.name + "-git-only")
    tools.mkdir()
    git = shutil.which("git")
    assert git is not None
    (tools / "git").symlink_to(git)
    monkeypatch.setenv("PATH", str(tools))
    with pytest.raises(RuntimeError, match="completed native"):
        changed_cpp_errors(repo, base)


def test_cli_default_origin_and_unavailable_original_history(
    cpp_repo: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the required ledger supplies a default; absence refuses without guessed history."""
    repo, base = cpp_repo
    assert main(["--repo", str(repo)]) == 2
    assert "ledger" in capsys.readouterr().err
    _write(repo, "tools/docstring_debt_ceiling.json", (ROOT / "tools/docstring_debt_ceiling.json").read_bytes())
    assert main(["--repo", str(repo)]) == 0
    assert main(["--repo", str(repo), "--base", "invalid"]) == 2
    (repo / "new-root/control.cpp").unlink()
    assert changed_cpp_errors(repo, base) == []


@pytest.mark.parametrize("change", ["source", "new-source", "include"])
def test_native_results_are_bound_to_stable_final_inputs(
    cpp_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Keep genuine native results, then alter an actual measured source or include at readback."""
    repo, base = cpp_repo
    _write(repo, "new-root/control.cpp", ('#include "abi.h"\n\n' + _VALID).encode())
    _write(repo, "new-root/abi.h", b"/// Read an external value.\nint external_value(void);\n")

    def run_then_change(
        root: Path, argv: Sequence[str], timeout_s: float, *, text: str = ""
    ) -> subprocess.CompletedProcess[str]:
        """Modify only a physical input after returning the real final compiler result."""
        result = actual_native(root, argv, timeout_s, text=text)
        if "-ast-dump=json" in argv:
            name = {"source": "control.cpp", "new-source": "late.cpp", "include": "abi.h"}[change]
            _write(repo, "new-root/" + name, _VALID.replace("return 1", "return 2").encode())
        return result

    monkeypatch.setattr("tools.cpp_source_checks._native", run_then_change)
    with pytest.raises(RuntimeError, match="changed during"):
        changed_cpp_errors(repo, base)


def test_registered_preflight_gate_runs_the_real_native_command(cpp_repo: tuple[Path, str]) -> None:
    """Actual gate registration propagates passing and failing source outcomes."""
    repo, base = cpp_repo
    name, command, cwd = next(gate for gate in GATES if gate[0] == "changed-cpp-static")
    command = [*command, "--repo", str(repo), "--base", base]
    assert run_gate(name, command, cwd)
    _write(repo, "new-root/control.cpp", _VALID.replace("return 1", "return 1.5").encode())
    assert not run_gate(name, command, cwd)
