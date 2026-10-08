# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Documentation coverage gate
"""Check source-module docstring presence and lexical API-reference coverage.

Inspect every .py beneath src/scpn_control except package __init__.py files,
including private and untracked files. No source module is imported or rendered.
An object directive represents its longest existing source-module prefix; the
remaining object name is not resolved. Passing this local inventory check does
not prove semantic documentation, executable examples or a successful docs build.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "scpn_control"
API_DOC = ROOT / "docs" / "api.md"


def module_name(path: Path, repo_root: Path = ROOT) -> str:
    """Convert a lexical source path beneath the selected src root to a dotted name.

    Remove its last suffix and join relative path components with dots. The
    source file need not exist. Both paths must share the same relative/absolute
    basis; a path outside repo_root/src raises ValueError. No import or Python
    identifier validation occurs.
    """
    return ".".join(path.relative_to(repo_root / "src").with_suffix("").parts)


def iter_public_modules(repo_root: Path = ROOT) -> list[Path]:
    """List existing non-package Python files in the selected scpn_control tree.

    Return sorted paths beneath repo_root/src/scpn_control, including private
    and untracked files. Exclude every __init__.py by filename and directories
    ending in .py. Relative roots follow cwd. Missing/file roots or no selected
    modules raise ValueError: an empty inventory cannot certify coverage.
    """
    source_root = repo_root / "src/scpn_control"
    paths = sorted(path for path in source_root.rglob("*.py") if path.name != "__init__.py" and path.is_file())
    if not source_root.is_dir() or not paths:
        raise ValueError("source scope must contain Python modules")
    return paths


def api_module_directives(api_text: str, repo_root: Path = ROOT) -> set[str]:
    """Find existing module prefixes of column-zero mkdocstrings directives.

    Match ::: followed by one or more literal spaces and a scpn_control dotted
    name using the legacy ASCII alphanumeric/underscore/dot grammar. Select the
    longest prefix backed by a .py file beneath repo_root/src; collapse repeated
    names. Indented, tab-separated and other-package directives do not match.
    Unknown source prefixes are omitted. Trailing attributes are not resolved,
    source text is not imported, and this is not a Markdown/native docs renderer.
    """
    directives = set(re.findall(r"^::: +(scpn_control\.[A-Za-z0-9_\.]+)", api_text, flags=re.MULTILINE))
    represented: set[str] = set()
    for directive in directives:
        parts = directive.split(".")
        for end in range(len(parts), 1, -1):
            candidate = ".".join(parts[:end])
            candidate_path = repo_root / "src" / Path(*candidate.split(".")).with_suffix(".py")
            if candidate_path.is_file():
                represented.add(candidate)
                break
    return represented


def modules_missing_docstrings(paths: list[Path], repo_root: Path = ROOT) -> list[str]:
    """Read selected UTF-8 sources and report missing module docs in input order.

    AST docstring presence is the criterion; an empty literal string still is a
    docstring. Duplicate input paths retain duplicate findings. Filesystem,
    UnicodeDecodeError, SyntaxError and path-relative ValueError propagate.
    No functions/classes are imported, executed or semantically reviewed.

    Examples
    --------
    Inspect the actual documented EQDSK source through the public API:

    >>> modules_missing_docstrings([SOURCE_ROOT / 'core/eqdsk.py'])
    []
    """
    missing: list[str] = []
    for path in paths:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if ast.get_docstring(module) is None:
            missing.append(module_name(path, repo_root))
    return missing


def main(argv: list[str] | None = None) -> int:
    """Run the ordinary hook/CI inventory command without writing files.

    Parameters
    ----------
    argv
        Arguments excluding the executable; None reads process arguments.
        --repo selects a root resolved against cwd. Default to this script's
        repository regardless of cwd. Caller argv and source files stay intact.

    Returns
    -------
    int
        Zero for a nonempty source inventory with module docstrings and matched
        reference prefixes; one for either missing list or a root/source inspection error.
        Print missing-reference then missing-docstring names in sorted source
        order to stderr. Inspection failures have a fixed refusal sentence;
        only success prints to stdout. Argument errors exit two via argparse.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        try:
            args.repo.stat()
        except FileNotFoundError:
            pass  # The nonempty source-scope contract refuses a missing root.
        repo_root = args.repo.resolve()
        module_paths = iter_public_modules(repo_root)
        modules = [module_name(path, repo_root) for path in module_paths]
        represented = api_module_directives((repo_root / "docs/api.md").read_text(encoding="utf-8"), repo_root)
        missing_api = [name for name in modules if name not in represented]
        missing_docs = modules_missing_docstrings(module_paths, repo_root)
    except (OSError, UnicodeError, SyntaxError, RuntimeError):
        print("Documentation coverage refused: could not inspect UTF-8 source and API reference.", file=sys.stderr)
        return 1
    except ValueError:
        print("Documentation coverage refused: source scope must contain Python modules.", file=sys.stderr)
        return 1

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
