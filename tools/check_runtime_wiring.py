# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime-wiring reachability checker

"""Report existing source modules with no static repository import reference.

Inspect UTF-8 Python source below src, tests, benchmarks, examples, tools and
validation. Nested imports count even in code that never executes; dynamic
imports and external consumers are not discovered. Package initializers and
the declared ROOTS are exempt. This is not entrypoint reachability, importability,
test execution, maintenance classification or scientific-readiness evidence.
An orphan is a review candidate, not proof that a public API should be removed.

Usage::

    python tools/check_runtime_wiring.py            # human report, fails on orphans
    python tools/check_runtime_wiring.py --json     # machine-readable, fails on orphans
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

PKG = "scpn_control"
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

# Declared root exemptions; their registration/execution is not checked here.
ROOTS = (f"{PKG}", f"{PKG}.cli", f"{PKG}.physics_debug")
# Directories whose files count as references to a source module.
IMPORTER_DIRS = ("src", "tests", "benchmarks", "examples", "tools", "validation")


def _module_name(path: Path, source_root: Path = SRC) -> str:
    """Map a lexical source path to dotted components, stripping an __init__ suffix.

    Paths must share their relative/absolute basis. Outside-root paths raise
    ValueError; neither identifier validity nor actual importability is checked.
    """
    parts = list(path.relative_to(source_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _all_modules(repo_root: Path = REPO) -> dict[str, Path]:
    """Discover existing Python files, including private/untracked modules and packages.

    Require a package directory with at least one Python file. Directory names
    ending in .py are excluded; missing/file/empty source scopes raise ValueError.
    """
    source_root = repo_root / "src"
    modules = {_module_name(p, source_root): p for p in source_root.glob(f"{PKG}/**/*.py") if p.is_file()}
    if not (source_root / PKG).is_dir() or not modules:
        raise ValueError("source scope must contain Python modules")
    return modules


def _resolve_relative(current: str, level: int, module: str | None, *, is_package: bool) -> str:
    """Resolve lexical relative imports from a module or package initializer context.

    Level one retains the containing package; deeper levels remove parents.
    Excessive levels yield an empty base instead of validating Python's import
    rules. The optional module suffix is appended without importing anything.
    """
    package = current if is_package else current.rpartition(".")[0]
    base = package.split(".") if package else []
    trim = max(level - 1, 0)
    trimmed = base[: len(base) - trim] if trim <= len(base) else []
    if module:
        trimmed = trimmed + module.split(".")
    return ".".join(trimmed)


def _imports_of(current: str, path: Path) -> set[str]:
    """Collect static package-prefixed import targets at every AST depth.

    Include each from-import base and base.alias candidate, even for symbols or
    unreachable branches. Resolve relative imports using current/package context.
    Dynamic import strings are ignored. UTF-8/file/syntax errors propagate.
    """
    out: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    is_package = path.name == "__init__.py"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith(PKG))
        elif isinstance(node, ast.ImportFrom):
            target = (
                _resolve_relative(current, node.level, node.module, is_package=is_package)
                if node.level
                else (node.module or "")
            )
            if not target.startswith(PKG):
                continue
            out.add(target)
            # `from pkg.sub import name` may import a submodule `pkg.sub.name`.
            out.update(f"{target}.{a.name}" for a in node.names)
    return out


def _iter_importer_files(repo_root: Path = REPO) -> list[Path]:
    """List Python files below existing importer directories in declared root order.

    Glob order within each root follows the filesystem. Missing/non-directory
    optional roots and directory names ending in .py contribute no files.
    """
    files: list[Path] = []
    for directory in IMPORTER_DIRS:
        base = repo_root / directory
        if base.is_dir():
            files.extend(p for p in base.glob("**/*.py") if p.is_file())
    return files


def find_orphans(repo_root: Path = REPO) -> tuple[list[str], int]:
    """Return sorted unreferenced module names and the existing source-file count.

    Inspect the selected repository without importing its modules or writing
    files. Relative roots follow cwd; the default is this script's repository.
    Count every source Python file including __init__ containers. Static targets
    map to their longest existing dotted module prefix, except direct self-imports.
    Any other file's reference suffices, even when its importer is itself orphaned.
    ROOTS and package containers are exempt; no transitive reachability is tested.
    Missing/empty source scope raises ValueError; file, UTF-8 and syntax errors
    propagate. Neither an empty orphan list nor an orphan implies execution.

    Examples
    --------
    Inspect the actual default checkout's current static references:

    >>> find_orphans()[0]
    []
    """
    modules = _all_modules(repo_root)
    source_root = repo_root / "src"
    package_inits = {name for name, path in modules.items() if path.name == "__init__.py"}

    referenced: set[str] = set()
    for path in _iter_importer_files(repo_root):
        current = _module_name(path, source_root) if path.is_relative_to(source_root) else ""
        for target in _imports_of(current, path):
            if target == current:
                continue
            parts = target.split(".")
            while parts:  # map `pkg.mod.Symbol` down to the owning module
                candidate = ".".join(parts)
                if candidate in modules and candidate != current:
                    referenced.add(candidate)
                    break
                parts.pop()

    wired = referenced | set(ROOTS) | package_inits
    orphans = sorted(name for name in modules if name not in wired)
    return orphans, len(modules)


def main(argv: list[str] | None = None) -> int:
    """Print a static reference report and return zero only for an inspected orphan-free scope.

    --repo selects a root relative to cwd; default to the script repository.
    --json preserves total_modules/orphans fields; otherwise emit a human report.
    Return one for orphans or inspection failure, with a fixed stderr refusal and
    no report on failure. Unknown flags exit two via argparse. No files are written
    and process argv is unchanged; success does not prove runtime reachability.
    """
    parser = argparse.ArgumentParser(description="Report modules without static repository import references.")
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a human report")
    args = parser.parse_args(argv)

    try:
        orphans, total = find_orphans(args.repo.resolve())
    except (OSError, UnicodeError, SyntaxError):
        print("Wiring inspection refused: could not inspect UTF-8 Python source.", file=sys.stderr)
        return 1
    except ValueError:
        print("Wiring inspection refused: source scope must contain Python modules.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"total_modules": total, "orphans": orphans}, indent=2))
        return 1 if orphans else 0

    print(f"Static wiring check: {total} source modules scanned against {', '.join(IMPORTER_DIRS)}.")
    if not orphans:
        print("Every non-exempt source module has a static repository import reference.")
        return 0
    print(f"\n{len(orphans)} module(s) referenced by nothing in the repository:")
    for name in orphans:
        print(f"  - {name}")
    print("\nEach is a removal-or-cover candidate: confirm it is a deliberate public")
    print("surface awaiting a consumer, or wire/test it, or remove it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
