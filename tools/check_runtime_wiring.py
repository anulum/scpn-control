# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime-wiring reachability checker

"""Report scpn_control modules that nothing in the repository references.

This package is a library plus a CLI, so "reachable from the CLI" is too narrow —
most controllers and solvers are public modules imported directly by consumers,
not pulled in by the ``scpn-control`` entry point. The actionable signal is
therefore the opposite: a source module that *no* file anywhere in the repository
imports (src, tests, benchmarks, examples, tools — module-level or nested, since
imports are often lazy) is referenced only through documentation or is dead code.
Entry points and package ``__init__`` containers are excluded.

A reported module is not necessarily wrong: it may be a brand-new public surface
awaiting a consumer. But it is a removal/cover candidate to confirm.

Usage::

    python tools/check_runtime_wiring.py            # human report
    python tools/check_runtime_wiring.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

PKG = "scpn_control"
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

# Entry points / public API surface that are wired by definition.
ROOTS = (f"{PKG}", f"{PKG}.cli", f"{PKG}.physics_debug")
# Directories whose files count as references to a source module.
IMPORTER_DIRS = ("src", "tests", "benchmarks", "examples", "tools", "validation")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _all_modules() -> dict[str, Path]:
    return {_module_name(p): p for p in SRC.glob(f"{PKG}/**/*.py")}


def _resolve_relative(current: str, level: int, module: str | None) -> str:
    base = current.split(".")
    # `current` is the importing module; level 1 = its package, level 2 = parent, etc.
    trimmed = base[: len(base) - level] if level <= len(base) else []
    if module:
        trimmed = trimmed + module.split(".")
    return ".".join(trimmed)


def _imports_of(current: str, path: Path) -> set[str]:
    """All scpn_control.* dotted targets imported anywhere in the file."""
    out: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith(PKG))
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_relative(current, node.level, node.module) if node.level else (node.module or "")
            if not target.startswith(PKG):
                continue
            out.add(target)
            # `from pkg.sub import name` may import a submodule `pkg.sub.name`.
            out.update(f"{target}.{a.name}" for a in node.names)
    return out


def _iter_importer_files() -> list[Path]:
    files: list[Path] = []
    for directory in IMPORTER_DIRS:
        base = REPO / directory
        if base.exists():
            files.extend(base.glob("**/*.py"))
    return files


def find_orphans() -> tuple[list[str], int]:
    modules = _all_modules()
    package_inits = {name for name, path in modules.items() if path.name == "__init__.py"}

    referenced: set[str] = set()
    for path in _iter_importer_files():
        current = _module_name(path) if path.is_relative_to(SRC) else ""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Report modules unreachable from a runtime entry point.")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a human report")
    args = parser.parse_args()

    orphans, total = find_orphans()
    if args.json:
        print(json.dumps({"total_modules": total, "orphans": orphans}, indent=2))
        return 0

    print(f"Wiring check: {total} source modules scanned against {', '.join(IMPORTER_DIRS)}.")
    if not orphans:
        print("Every source module is referenced somewhere in the repository.")
        return 0
    print(f"\n{len(orphans)} module(s) referenced by nothing in the repository:")
    for name in orphans:
        print(f"  - {name}")
    print("\nEach is a removal-or-cover candidate: confirm it is a deliberate public")
    print("surface awaiting a consumer, or wire/test it, or remove it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
