# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Documentation coverage gate
"""Check that public Python modules are represented in docs and documented."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "scpn_control"
API_DOC = ROOT / "docs" / "api.md"


def module_name(path: Path) -> str:
    """Return the import path for a source file under ``src``."""

    return ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)


def iter_public_modules() -> list[Path]:
    """Return tracked package modules that must appear in the API reference."""

    return sorted(path for path in SOURCE_ROOT.rglob("*.py") if path.name != "__init__.py")


def api_module_directives(api_text: str) -> set[str]:
    """Return module paths represented by mkdocstrings directives."""

    directives = set(re.findall(r"^::: +(scpn_control\.[A-Za-z0-9_\.]+)", api_text, flags=re.MULTILINE))
    represented: set[str] = set()
    for directive in directives:
        parts = directive.split(".")
        for end in range(len(parts), 1, -1):
            candidate = ".".join(parts[:end])
            candidate_path = ROOT / "src" / Path(*candidate.split(".")).with_suffix(".py")
            if candidate_path.exists():
                represented.add(candidate)
                break
    return represented


def modules_missing_docstrings(paths: list[Path]) -> list[str]:
    """Return import paths whose module-level docstring is absent."""

    missing: list[str] = []
    for path in paths:
        module = ast.parse(path.read_text())
        if ast.get_docstring(module) is None:
            missing.append(module_name(path))
    return missing


def main() -> int:
    """Run the documentation coverage checks."""

    module_paths = iter_public_modules()
    modules = [module_name(path) for path in module_paths]
    represented = api_module_directives(API_DOC.read_text())
    missing_api = [name for name in modules if name not in represented]
    missing_docs = modules_missing_docstrings(module_paths)

    if missing_api or missing_docs:
        if missing_api:
            print("Modules missing from docs/api.md:", file=sys.stderr)
            for name in missing_api:
                print(f"  - {name}", file=sys.stderr)
        if missing_docs:
            print("Modules missing module docstrings:", file=sys.stderr)
            for name in missing_docs:
                print(f"  - {name}", file=sys.stderr)
        return 1

    print(f"Documentation coverage OK: {len(modules)} Python modules represented and documented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
