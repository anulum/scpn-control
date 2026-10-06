# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static cross-language declaration inventory.

"""Inspect local declarations without importing packages or running renderers.

Python uses top-level AST declarations and standalone ``:::`` directives in
``docs/api.md``. Native families use bounded regular expressions, not compiler
parsers. Digests cover sorted names joined by LF, not source bytes or semantics.
Selected paths follow filesystem symlinks; there is no containment, cache,
locking or coherent snapshot guarantee. All results are fresh caller-owned data.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATIONS = ("stable-root-owner", "documented-module-reference", "nonstable-module-surface")


class ApiContractInspectionError(ValueError):
    """Authored refusal of unreadable or structurally invalid inventory inputs.

    Public inspection APIs raise this type without embedding interpreter or OS
    text. A refusal produces no successful partial inventory.
    """


class PythonInventory(TypedDict):
    """Counts and name digests for top-level declarations and literal exports.

    Classifications partition candidates. Stable ownership is a root-owner
    prefix match, not import/reexport resolution or compatibility verification.
    """

    candidate_count: int
    candidate_sha256: str
    stable_export_count: int
    stable_export_sha256: str
    classifications: dict[str, int]
    classification_sha256: dict[str, str]


class ExportInventory(TypedDict):
    """Count and digest of regex-selected relative-path:name export strings."""

    export_count: int
    export_sha256: str


class SymbolInventory(TypedDict):
    """Regex-selected native names in source order, without proof/type checks."""

    symbols: list[str]


class Inventory(TypedDict):
    """Fresh static declarations for Python, C, Lean, Rust and TypeScript."""

    python: PythonInventory
    c: SymbolInventory
    lean: SymbolInventory
    rust: ExportInventory
    typescript: ExportInventory


@dataclass(frozen=True)
class Candidate:
    """A non-private top-level Python class, function or async function.

    Attributes
    ----------
    qualified_name : str
        Lexical module plus declaration name; package initialisers omit __init__.
    kind : str
        ``class`` or ``callable``; methods/nested declarations are not candidates.
    path : str
        Repository-relative POSIX source path.
    line : int
        One-based AST definition line, without decorator offsets.
    documented : bool
        Exact standalone directive membership, without rendering/import checks.

    Notes
    -----
    Construction does not validate supplied fields; frozen prevents reassignment.
    """

    qualified_name: str
    kind: str
    path: str
    line: int
    documented: bool


def _read(path: Path) -> str:
    """Read UTF-8 source with universal newlines or refuse expected I/O errors."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ApiContractInspectionError("could not read a required API input as UTF-8") from None


def _tree(path: Path) -> ast.Module:
    """Parse Python without executing it and refuse malformed source."""
    try:
        return ast.parse(_read(path))
    except (SyntaxError, ValueError, RecursionError):
        raise ApiContractInspectionError("a required Python API source cannot be parsed") from None


def _files(repo: Path, relative: str, pattern: str) -> list[Path]:
    """Require a nonempty sorted source family, excluding Rust target paths."""
    try:
        # Directory walking raised for an unusable family root before Python
        # 3.13 and yields nothing since, so the root is inspected explicitly.
        try:
            (repo / relative).stat()
        except FileNotFoundError:
            pass
        paths = sorted((repo / relative).rglob(pattern))
        if pattern == "*.rs":
            paths = [p for p in paths if "target" not in p.relative_to(repo).parts]
    except OSError:
        raise ApiContractInspectionError("could not enumerate an API source family") from None
    if not paths:
        raise ApiContractInspectionError("a required API source family is empty or missing")
    return paths


def _python_candidates(repo: Path) -> list[Candidate]:
    """Collect top-level non-private declarations and exact document directives."""
    directives = set(re.findall(r"^:::\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$", _read(repo / "docs/api.md"), re.MULTILINE))
    candidates: list[Candidate] = []
    for path in _files(repo, "src/scpn_control", "*.py"):
        parts = list(path.relative_to(repo / "src").with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        for node in _tree(path).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
                "_"
            ):
                qualified = f"{module}.{node.name}"
                candidates.append(
                    Candidate(
                        qualified,
                        "class" if isinstance(node, ast.ClassDef) else "callable",
                        path.relative_to(repo).as_posix(),
                        node.lineno,
                        qualified in directives,
                    )
                )
    return candidates


def _root_exports(repo: Path) -> tuple[list[str], dict[str, str]]:
    """Validate literal export containers while retaining aggregator ownership."""
    declarations: dict[str, object] = {}
    for node in _tree(repo / "src/scpn_control/__init__.py").body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("__all__", "_EXPORT_MODULES"):
                    try:
                        declarations[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError, RecursionError):
                        raise ApiContractInspectionError("package exports must use literal containers") from None
    raw_exports = declarations.get("__all__")
    raw_owners = declarations.get("_EXPORT_MODULES")
    if not isinstance(raw_exports, list) or not isinstance(raw_owners, dict):
        raise ApiContractInspectionError("package exports require a literal list and owner table")
    exports: list[str] = []
    for name in raw_exports:
        if not isinstance(name, str) or not name.isidentifier() or name in exports:
            raise ApiContractInspectionError("package exports require unique identifier strings")
        exports.append(name)
    owners: dict[str, str] = {}
    for name, module in raw_owners.items():
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(module, str):
            raise ApiContractInspectionError("package owners require identifier names and module strings")
        if not all(part.isidentifier() for part in module.split(".")):
            raise ApiContractInspectionError("package owners require dotted module identifiers")
        owners[name] = module
    return exports, owners


def _classification(candidate: Candidate, names: set[str], owners: dict[str, str]) -> str:
    """Prefer an exported owner prefix, then exact directive membership."""
    short_name = candidate.qualified_name.rsplit(".", 1)[-1]
    owner = owners.get(short_name)
    if short_name in names and owner is not None and candidate.qualified_name.startswith(f"{owner}."):
        return "stable-root-owner"
    return "documented-module-reference" if candidate.documented else "nonstable-module-surface"


def _digest(items: list[str]) -> str:
    """Hash sorted UTF-8 names joined by LF, retaining duplicate declarations."""
    return hashlib.sha256("\n".join(sorted(items)).encode("utf-8")).hexdigest()


def _exports(repo: Path, relative: str, glob: str, expression: str) -> ExportInventory:
    """Inventory line-regex declarations with their relative file owners."""
    pattern = re.compile(expression, re.MULTILINE)
    symbols = [
        f"{path.relative_to(repo).as_posix()}:{name}"
        for path in _files(repo, relative, glob)
        for name in pattern.findall(_read(path))
    ]
    return {"export_count": len(symbols), "export_sha256": _digest(symbols)}


def build_inventory(repo: Path = ROOT) -> Inventory:
    """Return static local declarations without executing any inspected code.

    Parameters
    ----------
    repo : pathlib.Path
        Repository root; relative paths use the caller's working directory.
        Selected inputs follow symlinks. Git tracking is not consulted.

    Returns
    -------
    Inventory
        Fresh name/count digests and C/Lean lists in source order. Python scans
        classes/functions/async functions only at module top level. C matches
        SCPN_SOLVER_API prototypes; Lean matches documented inductive/def/theorem
        ASCII names. Rust matches pub declarations, excluding target paths;
        TypeScript matches column-zero export declarations in *.ts* files.
        Regex families do not resolve comments, cfg, reexports or signatures.

    Raises
    ------
    ApiContractInspectionError
        Required inputs are unreadable, malformed or a selected family is empty.
        Valid files with no regex matches still produce zero declarations.

    Notes
    -----
    Equality with a declared inventory does not prove documentation semantics,
    compatibility, renderer execution, independent review or scientific admission.

    Examples
    --------
    Inspect the maintained repository without importing its optional backends.

    >>> inventory = build_inventory()
    >>> sum(inventory['python']['classifications'].values()) == inventory['python']['candidate_count']
    True
    >>> inventory['c']['symbols'][0]
    'scpn_solver_create_v1'
    """
    candidates = _python_candidates(repo)
    exports, owners = _root_exports(repo)
    names = set(exports) - {"__version__", "RUST_BACKEND"}
    classes: dict[str, list[str]] = {name: [] for name in CLASSIFICATIONS}
    for candidate in candidates:
        classes[_classification(candidate, names, owners)].append(candidate.qualified_name)
    return {
        "python": {
            "candidate_count": len(candidates),
            "candidate_sha256": _digest([c.qualified_name for c in candidates]),
            "stable_export_count": len(exports),
            "stable_export_sha256": _digest(exports),
            "classifications": {name: len(values) for name, values in classes.items()},
            "classification_sha256": {name: _digest(values) for name, values in classes.items()},
        },
        "c": {
            "symbols": re.findall(
                r"^SCPN_SOLVER_API\s+[^;]*?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*?\);",
                _read(repo / "src/scpn_control/core/solver.h"),
                re.MULTILINE,
            )
        },
        "lean": {
            "symbols": re.findall(
                r"/--.*?-/\s*(?:inductive|def|theorem)\s+([A-Za-z_][A-Za-z0-9_]*)",
                _read(repo / "lean/SCPNControl/PulsedFSM.lean"),
                re.DOTALL,
            )
        },
        "rust": _exports(
            repo,
            "scpn-control-rs/crates",
            "*.rs",
            r"^\s*pub\s+(?:async\s+)?(?:struct|enum|trait|fn|type)\s+([A-Za-z_][A-Za-z0-9_]*)",
        ),
        "typescript": _exports(
            repo,
            "studio-web/src",
            "*.ts*",
            r"^export\s+(?:default\s+)?(?:async\s+)?(?:function|class|interface|type|const|enum)\s+([A-Za-z_][A-Za-z0-9_]*)",
        ),
    }
