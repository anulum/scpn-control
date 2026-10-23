# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native complete-file C++ and C ABI header checks.
"""Check changed C++ sources and headers with actual compilers and Clang syntax data.

The standalone profile qualifies .cpp/.hpp as C++17 and .h as C11 plus C++17.
It does not execute source, infer runtime coverage or certify numerical results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.cpp_source_ast import documentation_errors, raw_tokens, token_directives
from tools.docstring_debt_ledger import read_owner_ledger
from tools.source_quality_scope import select_changed_files

ROOT = Path(__file__).resolve().parents[1]
_VERSIONS = {
    "clang++-18": ("--version", "Ubuntu clang version 18.1.3 (1ubuntu1)"),
    "clang-18": ("--version", "Ubuntu clang version 18.1.3 (1ubuntu1)"),
    "g++-13": ("-dumpfullversion", "13.3.0"),
    "gcc-13": ("-dumpfullversion", "13.3.0"),
    "clang-format-18": ("--version", "Ubuntu clang-format version 18.1.3 (1ubuntu1)"),
}
_WARNINGS = ["-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wshadow", "-Werror"]
_GCC_INSTALL = "--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
_STYLE = "{BasedOnStyle: LLVM, IndentWidth: 4, ColumnLimit: 110}"


def _native(repo: Path, argv: Sequence[str], timeout_s: float, *, text: str = "") -> subprocess.CompletedProcess[str]:
    """Run a bounded native static command without environment-injected include/config paths."""
    env = os.environ.copy()
    for name in (
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "OBJC_INCLUDE_PATH",
        "COMPILER_PATH",
        "GCC_EXEC_PREFIX",
        "LIBRARY_PATH",
        "CCC_OVERRIDE_OPTIONS",
        "CLANG_CONFIG_FILE_SYSTEM_DIR",
        "CLANG_CONFIG_FILE_USER_DIR",
    ):
        env.pop(name, None)
    env["LC_ALL"] = "C.UTF-8"
    try:
        return subprocess.run(
            argv, cwd=repo, input=text, capture_output=True, text=True, check=False, timeout=timeout_s, env=env
        )
    except (OSError, UnicodeError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"C++ checks require completed native {argv[0]} execution.") from error


def _qualified_tools(repo: Path, timeout_s: float) -> None:
    """Compare real installed tool outputs with every qualified version declaration."""
    for tool, (flag, expected) in _VERSIONS.items():
        result = _native(repo, [tool, flag], timeout_s)
        if result.returncode or result.stdout.partition("\n")[0] != expected:
            raise RuntimeError(f"C++ checks require the qualified {tool} profile: {expected}.")


def _read_source(path: Path) -> bytes:
    """Read regular source bytes while refusing symlink custody on initial and final readback."""
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError("C++ local includes require regular repository-owned paths.")
    content = path.read_bytes()
    content.decode("utf-8")
    return content


def _inputs(repo: Path, paths: list[Path], timeout_s: float) -> tuple[dict[Path, bytes], list[str]]:
    """Bind selected files and every literal repository-local include before compilation."""
    snapshots: dict[Path, bytes] = {}
    findings: list[str] = []
    pending = paths.copy()
    while pending:
        path = pending.pop()
        if path in snapshots:
            continue
        snapshots[path] = _read_source(path)
        result = _native(
            repo,
            [
                "clang++-18",
                "--no-default-config",
                _GCC_INSTALL,
                "-std=c++17",
                "-fsyntax-only",
                "-Xclang",
                "-dump-raw-tokens",
                "-x",
                "c++-header" if path.suffix.casefold() != ".cpp" else "c++",
                str(path),
            ],
            timeout_s,
        )
        if result.returncode:
            raise RuntimeError(f"{path.name}: complete native raw tokens are unavailable.")
        errors, includes = token_directives(raw_tokens(result.stderr, path), str(path.relative_to(repo)))
        findings.extend(errors)
        for name in includes:
            included = path.parent / name
            if included.suffix not in {".h", ".hpp"} or ".." in Path(name).parts or Path(name).is_absolute():
                raise RuntimeError(f"{path.name}: local include requires a qualified .h/.hpp repository path.")
            pending.append(included)
    return snapshots, findings


def _file_errors(repo: Path, path: Path, timeout_s: float) -> list[str]:
    """Measure one full source with formatting, both compiler families and native AST comments."""
    errors: list[str] = []
    suffix = path.suffix.casefold()
    modes = [("c++17", "c++" if suffix == ".cpp" else "c++-header", "clang++-18", "g++-13")]
    if suffix == ".h":
        modes.append(("c11", "c-header", "clang-18", "gcc-13"))
    result = _native(
        repo,
        ["clang-format-18", f"--style={_STYLE}", "--dry-run", "--Werror", f"--assume-filename={path}"],
        timeout_s,
        text=path.read_text(encoding="utf-8"),
    )
    if result.returncode:
        errors.append(f"{path.name}: clang-format failed ({result.returncode}):\n{result.stdout}{result.stderr}")
    for standard, language, clang, gcc in modes:
        flags = [f"-std={standard}", "-x", language, "-fsyntax-only", *_WARNINGS]
        gcc_flags = [f"-std={standard}", "-x", language.removesuffix("-header"), "-fsyntax-only", *_WARNINGS]
        gcc_input = [str(path)] if suffix == ".cpp" else ["-include", str(path), "-"]
        result = _native(repo, [gcc, *gcc_flags, *gcc_input], timeout_s)
        if result.returncode:
            errors.append(f"{path.name}: {gcc} {standard} failed ({result.returncode}):\n{result.stderr}")
        result = _native(
            repo,
            [
                clang,
                "--no-default-config",
                _GCC_INSTALL,
                *flags,
                "-Wdocumentation",
                "-Wdocumentation-unknown-command",
                "-fparse-all-comments",
                "-Xclang",
                "-ast-dump=json",
                str(path),
            ],
            timeout_s,
        )
        if result.returncode:
            errors.append(f"{path.name}: {clang} {standard} failed ({result.returncode}):\n{result.stderr}")
        else:
            errors.extend(documentation_errors(json.loads(result.stdout), path))
    return errors


def changed_cpp_errors(repo: Path, base_revision: str, *, timeout_s: float = 60.0) -> list[str]:
    """Check whole changed C++ files and dual-language C ABI headers in every root.

    Parameters
    ----------
    repo : pathlib.Path
        Real Git working-tree root. No source bodies are executed or edited.
    base_revision : str
        Trusted full original ancestor commit, checked by the shared selector.
    timeout_s : float
        Finite positive timeout for each native process.

    Returns
    -------
    list of str
        Compiler, formatter, directive or declaration-comment findings. Empty
        covers only this native static profile or an absent C++ cohort.

    Raises
    ------
    RuntimeError
        Original history, qualified tools, source/include custody, complete
        native protocols or stable final inputs are unavailable.

    Notes
    -----
    Ubuntu Clang/clang-format 18.1.3 (1ubuntu1) use GNU 13.3.0 on x86_64.
    Fixed LLVM formatting uses four spaces and 110 columns. Named declarations
    need native prose; literal local includes are recursively bound. Runtime,
    build/link/ABI, semantic review and numerical checks retain separate gates.
    """
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise RuntimeError("The C++-check timeout must be finite and positive.")
    repo = repo.resolve()
    scope = select_changed_files(repo, base_revision)
    paths = [repo / item.path for item in scope if item.profile in {"cpp", "c-header", "cpp-header"}]
    if not paths:
        return []
    try:
        _qualified_tools(repo, timeout_s)
        snapshots, errors = _inputs(repo, paths, timeout_s)
        for path in snapshots:
            errors.extend(_file_errors(repo, path, timeout_s))
        if any(_read_source(path) != content for path, content in snapshots.items()):
            raise RuntimeError("C++ source or include bytes changed during native checks.")
        if select_changed_files(repo, base_revision) != scope:
            raise RuntimeError("C++ source selection changed during native checks.")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError) as error:
        raise RuntimeError("C++ checks require readable sources and complete native parser data.") from error
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the native static profile with mandatory original-history binding.

    Parameters
    ----------
    argv : sequence of str or None
        CLI arguments; absent --base reads the required owner-ledger origin.

    Returns
    -------
    int
        Zero for this static profile, one for findings, two for refused
        measurement. Runtime, numerical and other-language gates are separate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--base")
    args = parser.parse_args(argv)
    try:
        base = args.base or read_owner_ledger(args.repo / "tools/docstring_debt_ceiling.json").origin_revision
        errors = changed_cpp_errors(args.repo, base)
    except RuntimeError as error:
        print(f"Changed-C++ checks refused: {error}", file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Changed-C++ static profile passed; runtime and other native profiles remain separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
