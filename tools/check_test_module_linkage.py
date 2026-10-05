#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Check Test Module Linkage.

"""Inspect static test-to-implementation links without importing inspected code.

The package name is always ``scpn_control``, including custom source roots.
Implementation files are recursively sorted ``*.py`` files except initializers;
test inputs are recursively sorted ``test_*.py`` files decoded as UTF-8 and
parsed with the native AST. Existing empty directories are supported. Missing
roots and ordinary files are refused; native Path symlink semantics apply.

Only top-level ``test_`` functions and ``test_`` methods of top-level ``Test``
classes are entry points. Named local helpers called from those scopes are
followed with their lexical import environment, including async functions.
Calls and names/attributes inside assertions establish static references.
Visible import re-exports in initializers and implementation files are resolved,
including at the package root. A referenced implementation facade and its selected
export target both count; other imports in the facade do not. Cycles terminate.
Named class references also follow visible imported or locally declared base
class identities. Unreferenced class declarations do not establish links. Selected
unambiguous top-level functions follow named body calls through lexical imports
and module helpers; unused functions, defaults and annotations do not add edges.
Selected plain methods of those classes follow the same lexical body references.
A single direct assignment from a visible constructor name can establish an
instance receiver for subsequent references in that scope. Ordinary data-field
writes preserve that binding. Writes/deletions to the selected member, runtime
type/dictionary or imported constructor hooks conservatively refuse the edge.
Imported member writes are resolved through visible aliases. Plain source methods
that reassign a selected member or customize attribute access remain opaque.
Rebinding, conditional assignment and receiver parameters prevent instance links.

Assignments, parameters, deletion, exception/pattern bindings, global/nonlocal
declarations and conflicting imports conservatively invalidate names for the
whole scope. Conditional flow, assignment order, fixture injection, decorators,
inherited/runtime method dispatch, dynamic imports, argument/return-value flow, subprocesses
and runtime availability are not evaluated. Calls in unreachable branches can
therefore count. Only traversed source files are parsed; attribute existence and
source importability are not validated.
This is a bounded static reference guard, not proof of execution or coverage.

The guard only reads files and prints CLI diagnostics; it never runs inspected
tests or source APIs. Files are not an atomic snapshot, and no thread/process
locking or path confinement is provided. Native decode/parse/filesystem errors
propagate. Relative CLI paths resolve from this script's repository; direct
API Path arguments use normal current-working-directory semantics.
"""

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
    """Resolve a CLI path against this script's repository.

    Parameters
    ----------
    path_value : str
        Native filesystem spelling; absolute paths are preserved.

    Returns
    -------
    Path
        Repository-prefixed relative path, without normalization or confinement.
    """
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def collect_source_modules(source_root: Path) -> list[Path]:
    """List regular Python implementation files in stable native path order.

    Parameters
    ----------
    source_root : Path
        Existing directory searched recursively; relative paths follow the caller's
        working directory. Initializers and directories named ``*.py`` are omitted.

    Returns
    -------
    list of Path
        Sorted paths as supplied by native ``Path.rglob``, without resolving them.

    Raises
    ------
    NotADirectoryError
        The root is absent or is not a directory.

    Notes
    -----
    This inventories names only; implementation syntax and importability are not
    checked. Native directory traversal and symlink behavior apply.
    """
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source root must be a directory: {source_root}")
    modules: list[Path] = []
    for path in source_root.rglob("*.py"):
        if path.name == "__init__.py" or not path.is_file():
            continue
        modules.append(path)
    return sorted(modules)


def _module_import_path(source_root: Path, module_path: Path) -> str:
    """Map an inventoried path to the fixed package import prefix.

    Parameters
    ----------
    source_root, module_path : Path
        Lexically related source directory and Python file.

    Returns
    -------
    str
        ``scpn_control`` plus suffix-free relative path components.

    Raises
    ------
    ValueError
        The module path is not beneath the supplied source root.
    """
    rel = module_path.relative_to(source_root).with_suffix("")
    return "scpn_control." + ".".join(rel.parts)


def _qualified_name(node: ast.expr) -> str:
    """Extract a Name/Attribute chain without evaluating expressions.

    Parameters
    ----------
    node : ast.expr
        Expression inspected recursively through attribute values.

    Returns
    -------
    str
        Dotted spelling, or empty text for calls/subscripts/other expressions.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _imports(tree: ast.AST, package: str = "") -> dict[str, str]:
    """Read syntactic import bindings from an AST subtree.

    Parameters
    ----------
    tree : ast.AST
        Import node or import-only subtree; callers establish lexical boundaries.
    package : str, optional
        Dotted package used to spell relative imports, without importing it.

    Returns
    -------
    dict of str to str
        Local names and dotted targets; later entries replace identical local keys.
        Scope callers separately refuse conflicting targets.
    """
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


_ExportBindings = tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]


def _source_exports(path: Path, module: str, package: str) -> _ExportBindings:
    """Read visible imports, named bases and selected callable body references.

    Parameters
    ----------
    path : Path
        Export file read as UTF-8 and parsed with the native AST.
    module, package : str
        Fixed-package module identity and context for relative import spellings.

    Returns
    -------
    tuple of four dicts
        Import bindings, named class bases, function references and plain methods.
        Rebinding, writes, conflicting declarations, decorators and wildcard imports
        prevent declaration attribution. Class keywords and dynamic bases are ignored.

    Notes
    -----
    Class initialization, decorators, metaclasses and runtime MRO dispatch are not followed.
    Selected function bodies use lexical imports and unambiguous module helpers;
    defaults, annotations and unused function bodies establish no call edges.
    Selected plain methods use module imports, without class-namespace lookup or
    implicit receiver/helper dispatch. Conflicting method names, source class
    attribute writes and explicit first-receiver writes to that member prevent
    attribution. Classes with custom attribute hooks or receiver type/dictionary
    writes are opaque; unrelated state fields do not invalidate methods.
    Native read/decode/parse errors propagate. No inspected code is executed.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bindings = _scope_bindings(tree, {}, package)
    nodes = _scope_nodes(tree)
    declarations: dict[str, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] = {}
    if not any(isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names) for node in nodes):
        for declaration in tree.body:
            if not isinstance(declaration, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if declaration.decorator_list or (isinstance(declaration, ast.ClassDef) and declaration.keywords):
                continue
            others = [node for node in nodes if node is not declaration]
            blocked = _shadowed_names(others)
            blocked.update(node.name for node in others if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
            blocked.update(
                name
                for node in others
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for name in _imports(node, package)
            )
            if declaration.name not in blocked:
                declarations[declaration.name] = declaration
    classes = {name: node for name, node in declarations.items() if isinstance(node, ast.ClassDef)}
    bases: dict[str, set[str]] = {}
    for name, declaration in classes.items():
        bases[name] = set()
        for base in declaration.bases:
            first, _, rest = _qualified_name(base).partition(".")
            target = bindings.get(first)
            if target is None and first in classes:
                target = module + "." + first
            if target:
                bases[name].add(target + ("." + rest if rest else ""))
    local_bindings = {**bindings, **{name: module + "." + name for name in declarations}}
    functions = {
        name: _linked_names(
            ast.Module(body=declaration.body, type_ignores=[]),
            _scope_bindings(declaration, local_bindings, package),
            package,
            defining_scope=declaration,
        )
        for name, declaration in declarations.items()
        if isinstance(declaration, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    methods: dict[str, set[str]] = {}
    for name, declaration in classes.items():
        class_nodes = _scope_nodes(declaration)
        method_writes = _class_method_writes(declaration)
        for method in declaration.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or method.decorator_list:
                continue
            others = [node for node in class_nodes if node is not method]
            blocked = _shadowed_names(others)
            blocked.update(node.name for node in others if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
            blocked.update(
                key
                for node in others
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for key in _imports(node, package)
            )
            mutated = any(
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and _qualified_name(node).split(".")[0] == name
                for node in nodes
            )
            if method.name not in blocked | method_writes and "*" not in method_writes and not mutated:
                methods[name + "." + method.name] = _linked_names(
                    ast.Module(body=method.body, type_ignores=[]),
                    _scope_bindings(method, local_bindings, package),
                    package,
                    defining_scope=method,
                )
    return bindings, bases, functions, methods


def _owners(
    name: str,
    source_root: Path,
    exports: dict[Path, _ExportBindings],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    """Resolve selected exports, named bases and callable bodies without execution.

    Parameters
    ----------
    name : str
        Dotted API reference; other packages are ignored.
    source_root : Path
        Physical implementation root.
    exports : dict of Path to tuple of dicts
        Per-inspection cache of imports, named bases, functions and plain methods.
    seen : frozenset of str, optional
        Selected export/class/function edges already traversed in this reference chain.

    Returns
    -------
    set of str
        Referenced facades, selected exports, bases and function-call owners. Files take
        precedence over initializers; API attribute existence is not checked.

    Notes
    -----
    Only selected exports, class bases and unambiguous callable bodies are followed.
    Base identities establish type references without inherited method dispatch.
    Function-local imports can establish body references, not module exports. Regular
    file facades still count when no export/base resolves. Arbitrary assignment
    exports are not followed. Cycles stop without adding another owner; native
    decode and syntax errors propagate. Files are parsed once per inspection.
    """
    if not name.startswith("scpn_control."):
        return set()
    parts = name.split(".")
    for count in range(len(parts), 0, -1):
        module = ".".join(parts[:count])
        path = source_root.joinpath(*parts[1:count]).with_suffix(".py")
        implementation = count > 1 and path.is_file()
        owners = {module} if implementation else set()
        if implementation and count == len(parts):
            return owners
        if not implementation:
            path = source_root.joinpath(*parts[1:count], "__init__.py")
        if path.is_file() and count < len(parts):
            if path not in exports:
                package = module.rpartition(".")[0] if implementation else module
                exports[path] = _source_exports(path, module, package)
            bindings, bases, functions, methods = exports[path]
            target = bindings.get(parts[count])
            references = {target} if target else bases.get(parts[count], set())
            calls = functions.get(parts[count], set()) if count + 1 == len(parts) else set()
            if count + 2 == len(parts):
                calls = methods.get(".".join(parts[count:]), set())
            if references or calls:
                selector = ".".join(parts[count:]) if calls and count + 2 == len(parts) else parts[count]
                edge = f"{path}:{selector}"
                if edge in seen:
                    return owners
                for call in calls:
                    owners.update(_owners(call, source_root, exports, seen | {edge}))
                for reference in references:
                    owners.update(
                        _owners(
                            ".".join([reference, *(parts[count + 1 :] if target else [])]),
                            source_root,
                            exports,
                            seen | {edge},
                        )
                    )
                return owners
        if implementation:
            return owners
    return set()


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _scope_nodes(tree: ast.AST) -> list[ast.AST]:
    """Traverse one lexical scope while leaving nested scope bodies unopened.

    Parameters
    ----------
    tree : ast.AST
        Root whose children are inspected in native child order.

    Returns
    -------
    list of ast.AST
        Descendants including nested scope declarations, excluding their contents.
        Comprehension targets remain conservative bindings in the enclosing scan.
    """
    nodes: list[ast.AST] = []
    for child in ast.iter_child_nodes(tree):
        nodes.append(child)
        if not isinstance(child, _SCOPES):
            nodes.extend(_scope_nodes(child))
    return nodes


def _shadowed_names(nodes: list[ast.AST]) -> set[str]:
    """Collect non-function declarations and writes invalidating scope names.

    Parameters
    ----------
    nodes : list of ast.AST
        Lexically bounded descendants.

    Returns
    -------
    set of str
        Stored/deleted names, class/exception/pattern names and global/nonlocal
        declarations. Function declarations are handled by the binding resolver.
    """
    shadowed: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            shadowed.add(node.id)
        elif (
            isinstance(
                node,
                (ast.ClassDef, ast.ExceptHandler, ast.MatchAs, ast.MatchStar),
            )
            and node.name is not None
        ):
            shadowed.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            shadowed.add(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            shadowed.update(node.names)
    return shadowed


def _scope_bindings(tree: ast.AST, inherited: dict[str, str], package: str = "") -> dict[str, str]:
    """Merge unambiguous imports with inherited lexical bindings.

    Parameters
    ----------
    tree : ast.AST
        Current scope, without evaluation of its control flow.
    inherited : dict of str to str
        Imported names visible in the defining scope; never mutated.
    package : str, optional
        Package spelling for relative imports in traversed export files.

    Returns
    -------
    dict of str to str
        Fresh binding map after whole-scope shadow and conflicting-import removal.
    """
    nodes = _scope_nodes(tree)
    bindings = inherited.copy()
    imports: dict[str, set[str]] = {}
    shadowed = _shadowed_names(nodes)
    if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        shadowed.update(arg.arg for arg in ast.walk(tree.args) if isinstance(arg, ast.arg))
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name, target in _imports(node, package).items():
                imports.setdefault(name, set()).add(target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            shadowed.add(node.name)
    for name, targets in imports.items():
        if len(targets) == 1:
            bindings[name] = next(iter(targets))
        else:
            shadowed.add(name)
    for name in shadowed:
        bindings.pop(name, None)
    return bindings


def _linked_names(
    tree: ast.AST, inherited: dict[str, str], package: str = "", *, defining_scope: ast.AST | None = None
) -> set[str]:
    """Collect imported API references occurring in scope calls or assertions.

    Parameters
    ----------
    tree : ast.AST
        Current test/helper scope.
    inherited : dict of str to str
        Defining-scope imports, copied before shadow resolution.
    package : str, optional
        Package context for relative imports inside source functions.
    defining_scope : ast.AST or None, optional
        Original function/method when inspecting its body-only module, retaining
        parameter refusal without scanning default or annotation expressions.

    Returns
    -------
    set of str
        Dotted static references; nested bodies are handled only by explicit helper
        traversal, and no expression is executed. Explicit imported member writes
        resolve through aliases and refuse that selected path. Changes to imported
        constructor hooks/type/dictionary refuse inferred instances, while their
        bare class identities remain syntactic references.
    """
    nodes = _scope_nodes(tree)
    bindings = _scope_bindings(tree, inherited, package)
    writes = _attribute_writes(nodes)
    instances = _constructor_bindings(defining_scope if defining_scope is not None else tree, nodes, bindings, writes)
    imported_writes: set[str] = set()
    constructor_writes: set[str] = set()
    for write in writes:
        first, _, rest = write.partition(".")
        if first in bindings:
            resolved = bindings[first] + ("." + rest if rest else "")
            imported_writes.add(resolved)
            components = resolved.split(".")
            for index, member in enumerate(components):
                if member in {"__class__", "__dict__", "__init__", "__new__"}:
                    constructor_writes.add(".".join(components[:index]))
    linked: set[str] = set()
    for node in nodes:
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
                resolved = bindings[first] + ("." + rest if rest else "")
            elif (
                first in instances
                and rest
                and (expression.lineno, expression.col_offset) > instances[first][1]
                and not any(name == write or name.startswith(write + ".") for write in writes)
                and not any(
                    instances[first][0] == write or instances[first][0].startswith(write + ".")
                    for write in constructor_writes
                )
            ):
                resolved = instances[first][0] + "." + rest
            else:
                continue
            if not any(resolved == write or resolved.startswith(write + ".") for write in imported_writes):
                linked.add(resolved)
    return linked


def _constructor_bindings(
    tree: ast.AST, nodes: list[ast.AST], bindings: dict[str, str], writes: set[str]
) -> dict[str, tuple[str, tuple[int, int]]]:
    """Identify single direct constructor assignments without inferring value flow.

    Parameters
    ----------
    tree : ast.AST
        Lexical test/helper or selected source body.
    nodes : list of ast.AST
        Descendants bounded to that scope.
    bindings : dict of str to str
        Its unambiguous visible import/declaration names.
    writes : set of str
        Qualified attribute/subscript write targets in that scope.

    Returns
    -------
    dict of str to tuple of str and tuple of int and int
        Receiver names paired with constructor identities and assignment end positions.
        Only a sole direct Name assignment to a named call qualifies. Parameters,
        imports, other name writes/declarations and runtime type/dictionary writes
        refuse it. Selected-member writes are filtered by the reference collector.

    Notes
    -----
    Conditional assignments, aliases, annotations, factories and return values are
    not inferred. The source owner resolver separately requires a selected plain
    class method; a constructor-shaped call to a function grants no method edges.
    """
    instances: dict[str, tuple[str, tuple[int, int]]] = {}
    parameters = (
        {arg.arg for arg in ast.walk(tree.args) if isinstance(arg, ast.arg)}
        if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef))
        else set()
    )
    for statement in getattr(tree, "body", []):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        receiver = statement.targets[0]
        if not isinstance(receiver, ast.Name) or not isinstance(statement.value, ast.Call):
            continue
        first, _, rest = _qualified_name(statement.value.func).partition(".")
        if first not in bindings:
            continue
        others = [node for node in nodes if node is not receiver]
        blocked = _shadowed_names(others) | parameters
        blocked.update(node.name for node in others if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        blocked.update(
            key for node in others if isinstance(node, (ast.Import, ast.ImportFrom)) for key in _imports(node)
        )
        mutated = any(
            write.partition(".")[0] == receiver.id
            and write.partition(".")[2].split(".")[0] in {"__class__", "__dict__"}
            for write in writes
        )
        if receiver.id not in blocked and not mutated:
            instances[receiver.id] = (
                bindings[first] + ("." + rest if rest else ""),
                (statement.end_lineno or statement.lineno, statement.end_col_offset or statement.col_offset),
            )
    return instances


def _attribute_writes(nodes: list[ast.AST]) -> set[str]:
    """Collect qualified attribute/subscript writes without evaluating mutation.

    Parameters
    ----------
    nodes : list of ast.AST
        Lexically bounded nodes; nested function bodies remain excluded.

    Returns
    -------
    set of str
        Name/attribute paths underlying stores or deletions. Subscript indices are
        discarded, preserving the container path through nested subscriptions.
        Dynamic mutator calls and descriptor execution are not inferred.
    """
    writes: set[str] = set()
    for node in nodes:
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            target: ast.expr = node
            while isinstance(target, ast.Subscript):
                target = target.value
            name = _qualified_name(target)
            if name:
                writes.add(name)
    return writes


def _class_method_writes(declaration: ast.ClassDef) -> set[str]:
    """Refuse callable identities changed by a class's own receiver writes or hooks.

    Parameters
    ----------
    declaration : ast.ClassDef
        Stable plain source class; nested scopes are not followed.

    Returns
    -------
    set of str
        First receiver-member names assigned/deleted in declared method bodies.
        ``*`` refuses all methods for custom attribute hooks or type/dictionary
        writes. Constructor/state writes cannot silently overwrite a selected method.

    Notes
    -----
    Only the first positional receiver parameter is recognized. Aliases, dynamic
    mutators, inherited descriptors and runtime dispatch remain outside this scan.
    """
    writes: set[str] = set()
    for method in declaration.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if method.name in {"__getattribute__", "__getattr__", "__setattr__", "__delattr__"}:
            return {"*"}
        parameters = [*method.args.posonlyargs, *method.args.args]
        if not parameters:
            continue
        for path in _attribute_writes(_scope_nodes(ast.Module(body=method.body, type_ignores=[]))):
            first, _, rest = path.partition(".")
            if first == parameters[0].arg:
                member = rest.split(".")[0]
                writes.add("*" if member in {"__class__", "__dict__"} else member)
    return writes


def _test_links(tree: ast.Module) -> set[str]:
    """Follow named test entry points and statically called local helpers.

    Parameters
    ----------
    tree : ast.Module
        Parsed test module; module-level shadowing affects its visible imports.

    Returns
    -------
    set of str
        Referenced imported names from entry points and reachable helper scopes.
        Recursive helper cycles are visited once. Class methods follow un-rebound
        ``self``/``cls`` parameters only, without inheritance or argument-flow proof.
    """
    module_bindings = _scope_bindings(tree, {})
    module_shadowed = _shadowed_names(_scope_nodes(tree))
    helpers = {
        node.name: (node, module_bindings)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in module_shadowed
    }
    linked: set[str] = set()
    visited: set[int] = set()

    def visit(
        scope: ast.FunctionDef | ast.AsyncFunctionDef,
        available: dict[str, tuple[ast.FunctionDef | ast.AsyncFunctionDef, dict[str, str]]],
        inherited: dict[str, str],
        methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    ) -> None:
        """Visit one helper with its defining imports and terminate call cycles.

        Parameters
        ----------
        scope : ast.FunctionDef or ast.AsyncFunctionDef
            Statically selected test/helper body.
        available : dict
            Named helper nodes paired with their defining lexical import maps.
        inherited : dict of str to str
            Imports visible where this helper was defined, not where it was called.
        methods : dict or None, optional
            Named methods of the selected test class.

        Notes
        -----
        Updates enclosing referenced-name and visited-identity sets. Does not evaluate
        call arguments, assignments or runtime dispatch.
        """
        if id(scope) in visited:
            return
        visited.add(id(scope))
        linked.update(_linked_names(scope, inherited))
        nodes = _scope_nodes(scope)
        bindings = _scope_bindings(scope, inherited)
        nested = {
            node.name: (node, bindings)
            for node in scope.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        parameters = {arg.arg for arg in ast.walk(scope.args) if isinstance(arg, ast.arg)}
        rebound = _shadowed_names(nodes)
        blocked = parameters | rebound
        blocked.update(
            name for node in nodes if isinstance(node, (ast.Import, ast.ImportFrom)) for name in _imports(node)
        )
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
                target = (nested.get(name) or available.get(name)) if name not in blocked else None
                if target is not None:
                    visit(target[0], {**available, **nested}, target[1], methods)
            elif (
                methods is not None
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"self", "cls"}
                and node.func.value.id in parameters
                and node.func.value.id not in rebound
            ):
                method_target = methods.get(node.func.attr)
                if method_target is not None:
                    visit(method_target, available, module_bindings, methods)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            visit(node, helpers, module_bindings)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            methods = {
                method.name: method
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for name, method in methods.items():
                if name.startswith("test_"):
                    visit(method, helpers, module_bindings, methods)
    return linked


def collect_unlinked_modules(*, source_root: Path, test_root: Path) -> list[str]:
    """Find inventoried owners without recognized static test references.

    Parameters
    ----------
    source_root, test_root : Path
        Existing source and recursive test directories, used as supplied. The
        package spelling remains ``scpn_control`` for every source root.

    Returns
    -------
    list of str
        Sorted source paths, repository-relative when lexically beneath this
        script's repository, otherwise native ``Path.as_posix`` spellings.

    Raises
    ------
    NotADirectoryError
        Test or source root is absent or not a directory; tests are checked first.
    SyntaxError
        A test or traversed export file is not valid Python.
    UnicodeDecodeError
        An inspected test/export file is not UTF-8.

    Notes
    -----
    The bounded linkage rules in the module contract apply. Referenced export
    files are parsed once per call; selected function bodies are inspected but never executed.
    Native IO errors propagate where Path exposes them.
    """
    if not test_root.is_dir():
        raise NotADirectoryError(f"Test root must be a directory: {test_root}")
    modules = collect_source_modules(source_root)
    linked: set[str] = set()
    exports: dict[Path, _ExportBindings] = {}
    for path in sorted(test_root.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for resolved in _test_links(tree):
            linked.update(_owners(resolved, source_root, exports))
    unlinked: list[str] = []
    for path in modules:
        if _module_import_path(source_root, path) not in linked:
            try:
                unlinked.append(path.relative_to(REPO_ROOT).as_posix())
            except ValueError:
                unlinked.append(path.as_posix())
    return sorted(unlinked)


def load_allowlist(path: Path) -> set[str]:
    """Decode exact exemption path spellings from a native JSON file.

    Parameters
    ----------
    path : Path
        Read-only UTF-8 JSON object with an ``allowlisted_modules`` list of objects,
        each containing a nonempty string ``path``.

    Returns
    -------
    set of str
        Exact path strings; duplicate list entries collapse. No slash/whitespace
        normalization or metadata review is performed. Extra fields are ignored;
        JSON object duplicate keys retain the native decoder's last-value behavior.

    Raises
    ------
    FileNotFoundError
        The allowlist path does not exist.
    ValueError
        Required value shapes or JSON syntax are invalid.

    Notes
    -----
    Native filesystem and Unicode errors propagate. Reading never changes bytes.
    """
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
    """Print static linkage counts and return the selected CLI verdict.

    Parameters
    ----------
    argv : list of str or None, optional
        Argparse arguments, or native process arguments. Relative roots/allowlist
        resolve from this script's repository, independently of working directory.

    Returns
    -------
    int
        Zero for no unexpected owner and no unpermitted stale exemption; one for
        unexpected owners first, or stale exemptions. ``--allow-stale-allowlist``
        permits stale entries only and never admits an unexpected owner.

    Raises
    ------
    SystemExit
        Argparse help exits zero; malformed arguments exit two before file reads.

    Notes
    -----
    Prints four counts then sorted failure paths or a success line. Inspection and
    allowlist errors propagate before counts; native script invocation normally
    exits one with traceback. No inspected files or policy bytes are written.
    """
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
