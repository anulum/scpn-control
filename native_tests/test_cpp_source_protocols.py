# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native C++ protocol and input custody refusal contracts.
"""Keep actual native execution while testing parser corruption and source custody."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_tools.test_source_quality_scope import _write
from test_tools.test_source_quality_scope import git_repo as git_repo

from native_tests.test_cpp_source_checks import _VALID
from native_tests.test_cpp_source_checks import cpp_repo as cpp_repo
from tools.cpp_source_ast import documentation_errors, raw_tokens
from tools.cpp_source_checks import _VERSIONS, changed_cpp_errors
from tools.cpp_source_checks import _native as actual_native


@pytest.mark.parametrize(
    "tree",
    [
        None,
        [],
        {0: "invalid"},
        {},
        {"kind": "TranslationUnitDecl", "inner": "invalid"},
        {"kind": "TranslationUnitDecl", "inner": [None]},
        {"kind": "TranslationUnitDecl", "inner": [{}]},
        {"kind": "TranslationUnitDecl", "inner": [{"id": "0x0"}]},
    ],
)
def test_incomplete_ast_protocols_refuse(tree: object) -> None:
    """Malformed or absent native envelopes are negative protocol controls only."""
    with pytest.raises(RuntimeError):
        documentation_errors(tree, Path("invalid.cpp"))


@pytest.mark.parametrize("fault", ["empty", "truncated", "location", "json", "children"])
def test_fault_after_actual_native_parser_execution_refuses(
    cpp_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Damage only the envelope of a genuine parser result; no synthetic passing AST is used."""
    repo, base = cpp_repo

    def corrupt(
        root: Path, argv: Sequence[str], timeout_s: float, *, text: str = ""
    ) -> subprocess.CompletedProcess[str]:
        """Preserve native execution before injecting a negative framing fault."""
        result = actual_native(root, argv, timeout_s, text=text)
        if "-dump-raw-tokens" in argv:
            if fault == "empty":
                result.stderr = ""
            elif fault == "truncated":
                result.stderr = result.stderr[:-2]
            elif fault == "location":
                result.stderr = result.stderr.replace(":1:1>", ":0:0>", 1)
        if "-ast-dump=json" in argv:
            if fault == "json":
                result.stdout = result.stdout[:-3]
            elif fault == "children":
                tree = json.loads(result.stdout)
                tree["inner"] = "corrupted"
                result.stdout = json.dumps(tree)
        return result

    monkeypatch.setattr("tools.cpp_source_checks._native", corrupt)
    with pytest.raises(RuntimeError):
        changed_cpp_errors(repo, base)


def test_nonzero_actual_version_probe_refuses(cpp_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """A real compiler invocation with an unavailable flag must not qualify a version."""
    repo, base = cpp_repo
    monkeypatch.setitem(_VERSIONS, "clang++-18", ("--unavailable-version-probe", _VERSIONS["clang++-18"][1]))
    with pytest.raises(RuntimeError, match="qualified"):
        changed_cpp_errors(repo, base)


def test_native_token_failure_cannot_become_a_valid_empty_measurement(
    cpp_repo: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute an actual failed compiler path after the real version checks complete."""
    repo, base = cpp_repo

    def unavailable_input(
        root: Path, argv: Sequence[str], timeout_s: float, *, text: str = ""
    ) -> subprocess.CompletedProcess[str]:
        """Use a genuinely unavailable physical input only for the negative token command."""
        if "-dump-raw-tokens" in argv:
            argv = [*argv[:-1], str(repo / "missing.cpp")]
        return actual_native(root, argv, timeout_s, text=text)

    monkeypatch.setattr("tools.cpp_source_checks._native", unavailable_input)
    with pytest.raises(RuntimeError, match="raw tokens are unavailable"):
        changed_cpp_errors(repo, base)


def test_recursive_include_cycle_is_measured_without_repeated_traversal(cpp_repo: tuple[Path, str]) -> None:
    """Real include guards allow a cycle while every physical header remains checked."""
    repo, base = cpp_repo
    _write(repo, "new-root/control.cpp", ('#include "cycle.hpp"\n\n' + _VALID).encode())
    _write(
        repo,
        "new-root/cycle.hpp",
        b'#pragma once\n#include "cycle.hpp"\n\n/// Name a native integer.\nusing Number = int;\n',
    )
    assert changed_cpp_errors(repo, base) == []


def test_symlinked_include_outside_selected_paths_refuses(cpp_repo: tuple[Path, str]) -> None:
    """An ignored alias cannot evade local include custody checks by escaping Git selection."""
    repo, base = cpp_repo
    _write(repo, ".gitignore", b"/new-root/alias.hpp\n")
    _write(repo, "new-root/original.hpp", b"/// Name a native integer.\nusing Number = int;\n")
    (repo / "new-root/alias.hpp").symlink_to("original.hpp")
    _write(repo, "new-root/control.cpp", ('#include "alias.hpp"\n\n' + _VALID).encode())
    with pytest.raises(RuntimeError, match="repository-owned"):
        changed_cpp_errors(repo, base)


def test_unreadable_encoding_refuses(cpp_repo: tuple[Path, str]) -> None:
    """Complete source bytes must decode before native checking can certify them."""
    repo, base = cpp_repo
    _write(repo, "new-root/control.cpp", _VALID.encode() + b"\xff")
    with pytest.raises(RuntimeError, match="readable sources"):
        changed_cpp_errors(repo, base)


def test_macro_generated_declaration_cannot_evade_documentation(cpp_repo: tuple[Path, str]) -> None:
    """A real macro-expanded declaration requires explicit qualification of its source binding."""
    repo, base = cpp_repo
    content = _VALID + "#define DECL(name) int name() { return 2; }\nDECL(generated)\n"
    _write(repo, "new-root/control.cpp", content.encode())
    with pytest.raises(RuntimeError, match="Macro-generated"):
        changed_cpp_errors(repo, base)


def test_real_formatter_ignore_and_style_files_cannot_suppress_findings(cpp_repo: tuple[Path, str]) -> None:
    """Actual clang-format stdin checking defeats ignored per-directory format controls."""
    repo, base = cpp_repo
    _write(repo, ".gitignore", b".clang-format\n.clang-format-ignore\n")
    _write(repo, "new-root/.clang-format", b"BasedOnStyle: LLVM\nDisableFormat: true\n")
    _write(repo, "new-root/.clang-format-ignore", b"*.cpp\n")
    assert changed_cpp_errors(repo, base) == []
    _write(repo, "new-root/control.cpp", _VALID.replace("int value", "int  value").encode())
    assert any("clang-format failed" in error for error in changed_cpp_errors(repo, base))


def test_actual_for_loop_and_catch_all_parser_slots_are_valid(cpp_repo: tuple[Path, str]) -> None:
    """Native optional statement slots are distinct from a missing declaration object."""
    repo, base = cpp_repo
    content = """#include <vector>

/// Return an accumulated integer or a bounded failure value.
int value() {
    int result = 0;
    const std::vector<int> values{0, 1};
    try {
        for (int index = 0; index < 2; ++index) {
            result += index;
        }
        for (int entry : values) {
            result += entry;
        }
    } catch (...) {
        return -1;
    }
    return result;
}
"""
    _write(repo, "new-root/control.cpp", content.encode())
    assert changed_cpp_errors(repo, base) == []


@pytest.mark.parametrize("directive", ['#line 4 "other.cpp"', '# 4 "other.cpp"'])
def test_source_location_overrides_cannot_evade_own_declaration_checks(
    cpp_repo: tuple[Path, str], directive: str
) -> None:
    """Native raw directives preserve source custody even when presumed AST locations differ."""
    repo, base = cpp_repo
    _write(repo, "new-root/control.cpp", (_VALID + directive + "\n").encode())
    assert any("source-location" in error for error in changed_cpp_errors(repo, base))


@pytest.mark.parametrize("include", ['"/unqualified/abi.h"', "</unqualified/abi.h>", "<../abi.h>", "<missing"])
def test_absolute_or_incomplete_system_include_paths_refuse(cpp_repo: tuple[Path, str], include: str) -> None:
    """System-header spelling cannot smuggle an absolute or parent-relative private source."""
    repo, base = cpp_repo
    _write(repo, "new-root/control.cpp", (_VALID + "#include " + include + "\n").encode())
    with pytest.raises(RuntimeError, match="qualified"):
        changed_cpp_errors(repo, base)


def test_real_tokens_retain_multiline_comment_and_namespace_declarations(cpp_repo: tuple[Path, str]) -> None:
    """Actual native parser data distinguishes local variables from documented global/type members."""
    repo, base = cpp_repo
    content = """namespace state {
/// Store the fixed count.
int count = 1;
/// Hold an integer measurement.
struct Measurement {
    /// Retain the integer value.
    int value;
    /// Read the retained value.
    int read() const { return value; }
};
} // namespace state

/* Return a fixed test value.
 * Keep the local variable outside the public declaration contract. */
int value() {
    int local = state::count;
    return local;
}
"""
    _write(repo, "new-root/control.cpp", content.encode())
    result = actual_native(
        repo, ["clang++-18", "-fsyntax-only", "-Xclang", "-dump-raw-tokens", str(repo / "new-root/control.cpp")], 60.0
    )
    assert any(
        kind == "comment" and "\n" in text for kind, text in raw_tokens(result.stderr, repo / "new-root/control.cpp")
    )
    assert changed_cpp_errors(repo, base) == []
