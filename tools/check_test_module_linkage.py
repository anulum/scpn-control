#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Check Test Module Linkage.

"""Guard source ownership links through called or asserted public test APIs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "src" / "scpn_control"
DEFAULT_TEST_ROOT = REPO_ROOT / "tests"
DEFAULT_ALLOWLIST = REPO_ROOT / "tools" / "untested_module_allowlist.json"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def collect_source_modules(source_root: Path) -> list[Path]:
    """Return Python implementation owners beneath the configured source root."""
    modules: list[Path] = []
    for path in source_root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        modules.append(path)
    return sorted(modules)


def _module_import_path(source_root: Path, module_path: Path) -> str:
    rel = module_path.relative_to(source_root).with_suffix("")
    return "scpn_control." + ".".join(rel.parts)


def _qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _imports(tree: ast.AST, package: str = "") -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*parent, *([base] if base else [])])
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{base}.{alias.name}"
    return bindings


def _owner(name: str, source_root: Path, seen: frozenset[str] = frozenset()) -> str | None:
    if name in seen or not name.startswith("scpn_control."):
        return None
    parts = name.split(".")
    for count in range(len(parts), 1, -1):
        path = source_root.joinpath(*parts[1:count]).with_suffix(".py")
        if path.is_file():
            return ".".join(parts[:count])
        facade = source_root.joinpath(*parts[1:count], "__init__.py")
        if facade.is_file() and count < len(parts):
            package = ".".join(parts[:count])
            bindings = _imports(ast.parse(facade.read_text(encoding="utf-8")), package)
            target = bindings.get(parts[count])
            if target:
                return _owner(".".join([target, *parts[count + 1 :]]), source_root, seen | {name})
    return None


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _scope_nodes(tree: ast.AST) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for child in ast.iter_child_nodes(tree):
        nodes.append(child)
        if not isinstance(child, _SCOPES):
            nodes.extend(_scope_nodes(child))
    return nodes


def _linked_names(tree: ast.AST, inherited: dict[str, str]) -> set[str]:
    nodes = _scope_nodes(tree)
    bindings = inherited.copy()
    imports: dict[str, set[str]] = {}
    shadowed: set[str] = set()
    if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        shadowed.update(arg.arg for arg in ast.walk(tree.args) if isinstance(arg, ast.arg))
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name, target in _imports(node).items():
                imports.setdefault(name, set()).add(target)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            shadowed.add(node.id)
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            or isinstance(node, ast.ExceptHandler)
            and node.name
            or isinstance(node, (ast.MatchAs, ast.MatchStar))
            and node.name
        ):
            shadowed.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            shadowed.add(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            shadowed.update(node.names)
    for name, targets in imports.items():
        if len(targets) == 1:
            bindings[name] = next(iter(targets))
        else:
            shadowed.add(name)
    for name in shadowed:
        bindings.pop(name, None)
    linked: set[str] = set()
    for node in nodes:
        if isinstance(node, _SCOPES):
            # Class namespaces are not enclosing lexical scopes for methods.
            child_bindings = inherited if isinstance(tree, ast.ClassDef) else bindings
            linked.update(_linked_names(node, child_bindings))
        expressions: list[ast.expr] = []
        if isinstance(node, ast.Call):
            expressions.append(node.func)
        elif isinstance(node, ast.Assert):
            expressions.extend(part for part in _scope_nodes(node.test) if isinstance(part, (ast.Name, ast.Attribute)))
            expressions.append(node.test)
        for expression in expressions:
            name = _qualified_name(expression)
            first, _, rest = name.partition(".")
            if first in bindings:
                linked.add(bindings[first] + ("." + rest if rest else ""))
    return linked


def collect_unlinked_modules(*, source_root: Path, test_root: Path) -> list[str]:
    """Find owners without a called or asserted API, resolving facade re-exports.

    This is a static linkage check, not proof of execution or coverage. Comments,
    strings, unused imports and test filenames cannot establish a linkage.
    Rebound names and conflicting imports within a scope are conservatively
    refused; independent function scopes retain their own import bindings.
    """
    linked: set[str] = set()
    for path in sorted(test_root.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for resolved in _linked_names(tree, {}):
            owner = _owner(resolved, source_root)
            if owner:
                linked.add(owner)
    unlinked: list[str] = []
    for path in collect_source_modules(source_root):
        if _module_import_path(source_root, path) not in linked:
            try:
                unlinked.append(path.relative_to(REPO_ROOT).as_posix())
            except ValueError:
                unlinked.append(path.as_posix())
    return sorted(unlinked)


def load_allowlist(path: Path) -> set[str]:
    """Read the explicit path allowlist and reject malformed entries."""
    if not path.exists():
        raise FileNotFoundError(f"Allowlist file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Allowlist must be a JSON object.")
    entries = payload.get("allowlisted_modules")
    if not isinstance(entries, list):
        raise ValueError("allowlisted_modules must be a list.")
    paths: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"allowlisted_modules[{idx}] must be an object.")
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"allowlisted_modules[{idx}].path must be a non-empty string.")
        paths.add(path_value)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Check static API linkage and report missing or stale owner exemptions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
    )
    parser.add_argument(
        "--test-root",
        default=str(DEFAULT_TEST_ROOT),
    )
    parser.add_argument(
        "--allowlist",
        default=str(DEFAULT_ALLOWLIST),
    )
    parser.add_argument(
        "--allow-stale-allowlist",
        action="store_true",
    )
    args = parser.parse_args(argv)

    source_root = _resolve(args.source_root)
    test_root = _resolve(args.test_root)
    allowlist_path = _resolve(args.allowlist)

    unlinked = set(collect_unlinked_modules(source_root=source_root, test_root=test_root))
    allowlisted = load_allowlist(allowlist_path)

    unexpected = sorted(unlinked - allowlisted)
    stale = sorted(allowlisted - unlinked)

    print(f"Unlinked modules detected: {len(unlinked)}")
    print(f"Allowlisted modules: {len(allowlisted)}")
    print(f"Unexpected modules: {len(unexpected)}")
    print(f"Stale allowlist entries: {len(stale)}")

    if unexpected:
        print("Guard FAILED: new modules without direct test linkage:")
        for path in unexpected:
            print(f"  - {path}")
        return 1

    if stale and not args.allow_stale_allowlist:
        print("Guard FAILED: stale allowlist entries should be removed:")
        for path in stale:
            print(f"  - {path}")
        return 1

    print("Untested-module guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
