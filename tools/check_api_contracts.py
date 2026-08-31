# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Cross-language public API ownership and contract gate.

"""Inventory and gate admitted APIs without conflating importability with stability."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tools/api_contract_registry.toml"


@dataclass(frozen=True)
class Candidate:
    """One non-private top-level Python class or callable."""

    qualified_name: str
    kind: str
    path: str
    line: int
    documented: bool


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT / "src").with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _python_candidates() -> list[Candidate]:
    directives = set(
        re.findall(
            r"^:::\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$",
            (ROOT / "docs/api.md").read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    candidates: list[Candidate] = []
    declaration_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for path in sorted((ROOT / "src/scpn_control").rglob("*.py")):
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, declaration_types) and not node.name.startswith("_"):
                qualified = f"{module}.{node.name}"
                candidates.append(
                    Candidate(
                        qualified_name=qualified,
                        kind="class" if isinstance(node, ast.ClassDef) else "callable",
                        path=path.relative_to(ROOT).as_posix(),
                        line=node.lineno,
                        documented=qualified in directives,
                    )
                )
    return candidates


def _root_exports() -> tuple[list[str], dict[str, str]]:
    tree = ast.parse((ROOT / "src/scpn_control/__init__.py").read_text(encoding="utf-8"))
    exports: list[str] | None = None
    owners: dict[str, str] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if "__all__" in targets:
            exports = list(ast.literal_eval(node.value))
        if "_EXPORT_MODULES" in targets:
            owners = dict(ast.literal_eval(node.value))
    if exports is None or owners is None:
        raise ValueError("package root must declare literal __all__ and _EXPORT_MODULES")
    return exports, owners


def _classification(candidate: Candidate, export_names: set[str], owners: dict[str, str]) -> str:
    short_name = candidate.qualified_name.rsplit(".", 1)[-1]
    owner = owners.get(short_name)
    if short_name in export_names and owner is not None and candidate.qualified_name.startswith(f"{owner}."):
        return "stable-root-owner"
    if candidate.documented:
        return "documented-module-reference"
    return "nonstable-module-surface"


def _digest(items: list[str]) -> str:
    payload = "\n".join(sorted(items)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _native_symbols() -> list[str]:
    header = (ROOT / "src/scpn_control/core/solver.h").read_text(encoding="utf-8")
    return re.findall(r"^SCPN_SOLVER_API\s+[^;]*?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*?\);", header, re.MULTILINE)


def _lean_symbols() -> list[str]:
    source = (ROOT / "lean/SCPNControl/PulsedFSM.lean").read_text(encoding="utf-8")
    return re.findall(r"/--.*?-/\s*(?:inductive|def|theorem)\s+([A-Za-z_][A-Za-z0-9_]*)", source, re.DOTALL)


def _rust_exports() -> list[str]:
    symbols: list[str] = []
    pattern = re.compile(
        r"^\s*pub\s+(?:async\s+)?(?:struct|enum|trait|fn|type)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
    )
    for path in sorted((ROOT / "scpn-control-rs/crates").rglob("*.rs")):
        if "target" not in path.parts:
            module = path.relative_to(ROOT).as_posix()
            symbols.extend(f"{module}:{name}" for name in pattern.findall(path.read_text(encoding="utf-8")))
    return symbols


def _typescript_exports() -> list[str]:
    symbols: list[str] = []
    pattern = re.compile(
        r"^export\s+(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|const|enum)\s+([A-Za-z_][A-Za-z0-9_]*)",
        re.MULTILINE,
    )
    for path in sorted((ROOT / "studio-web/src").rglob("*.ts*")):
        module = path.relative_to(ROOT).as_posix()
        symbols.extend(f"{module}:{name}" for name in pattern.findall(path.read_text(encoding="utf-8")))
    return symbols


def build_inventory() -> dict[str, Any]:
    """Return deterministic cross-language ownership and renderer evidence."""
    candidates = _python_candidates()
    exports, owners = _root_exports()
    export_names = set(exports) - {"__version__", "RUST_BACKEND"}
    classes: dict[str, list[str]] = {
        "stable-root-owner": [],
        "documented-module-reference": [],
        "nonstable-module-surface": [],
    }
    for candidate in candidates:
        classes[_classification(candidate, export_names, owners)].append(candidate.qualified_name)
    rust = _rust_exports()
    typescript = _typescript_exports()
    return {
        "python": {
            "candidate_count": len(candidates),
            "candidate_sha256": _digest([candidate.qualified_name for candidate in candidates]),
            "stable_export_count": len(exports),
            "stable_export_sha256": _digest(exports),
            "classifications": {name: len(values) for name, values in classes.items()},
            "classification_sha256": {name: _digest(values) for name, values in classes.items()},
        },
        "c": {"symbols": _native_symbols()},
        "lean": {"symbols": _lean_symbols()},
        "rust": {"export_count": len(rust), "export_sha256": _digest(rust)},
        "typescript": {"export_count": len(typescript), "export_sha256": _digest(typescript)},
    }


def _expected_inventory(registry: dict[str, Any]) -> dict[str, Any]:
    inventory = registry["inventory"]
    return {
        "python": {
            "candidate_count": inventory["python_candidate_count"],
            "candidate_sha256": inventory["python_candidate_sha256"],
            "stable_export_count": inventory["python_stable_export_count"],
            "stable_export_sha256": inventory["python_stable_export_sha256"],
            "classifications": dict(inventory["python_classifications"]),
            "classification_sha256": dict(inventory["python_classification_sha256"]),
        },
        "c": {"symbols": list(inventory["c_symbols"])},
        "lean": {"symbols": list(inventory["lean_symbols"])},
        "rust": {
            "export_count": inventory["rust_export_count"],
            "export_sha256": inventory["rust_export_sha256"],
        },
        "typescript": {
            "export_count": inventory["typescript_export_count"],
            "export_sha256": inventory["typescript_export_sha256"],
        },
    }


def _check_renderer_contracts(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_fragments = registry["renderer_contracts"]
    for relative, fragments in required_fragments.items():
        content = (ROOT / relative).read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in content:
                errors.append(f"{relative} lacks required renderer contract: {fragment}")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Check the committed inventory, or print a fresh snapshot as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-inventory", action="store_true")
    args = parser.parse_args(argv)
    current = build_inventory()
    if args.print_inventory:
        print(json.dumps(current, indent=2, sort_keys=True))
        return 0
    registry = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))
    errors = _check_renderer_contracts(registry)
    expected = _expected_inventory(registry)
    if current != expected:
        errors.append(
            "API inventory changed without classification review\n"
            f"expected={json.dumps(expected, sort_keys=True)}\n"
            f"current={json.dumps(current, sort_keys=True)}"
        )
    if errors:
        print("API contract gate failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "API contracts current: "
        f"Python={current['python']['candidate_count']} classified, "
        f"C={len(current['c']['symbols'])}, Lean={len(current['lean']['symbols'])}, "
        f"Rust={current['rust']['export_count']}, TypeScript={current['typescript']['export_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
