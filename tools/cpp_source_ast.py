# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Clang declaration and raw-token protocol readers.
"""Read actual Clang syntax data for documentation presence and tool directives."""

from __future__ import annotations

import re
from pathlib import Path

_DECLARATIONS = {
    "FunctionDecl",
    "CXXMethodDecl",
    "CXXConstructorDecl",
    "CXXDestructorDecl",
    "CXXConversionDecl",
    "RecordDecl",
    "CXXRecordDecl",
    "EnumDecl",
    "TypedefDecl",
    "TypeAliasDecl",
    "FieldDecl",
    "EnumConstantDecl",
}
_GLOBAL_CONTEXT = {"TranslationUnitDecl", "NamespaceDecl", "LinkageSpecDecl", "CXXRecordDecl"}
_HEADER = ("SPDX-", "Commercial license", "©", "ORCID:", "Contact:", "SCPN Control")
_NULL_SLOTS = {
    "ForStmt": {},
    "CXXForRangeStmt": {},
    "CXXFoldExpr": {},
    "StaticAssertDecl": {},
    "AlignedAttr": {},
    "CXXCatchStmt": {"id": "0x0"},
    "UsingShadowDecl": {"id": "0x0"},
}


def _node(value: object) -> dict[str, object]:
    """Require a complete object with string keys at a native AST boundary."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeError("Clang AST requires complete object nodes.")
    return {key: item for key, item in value.items()}


def _children(node: dict[str, object]) -> list[dict[str, object]]:
    """Read direct children, recognising only qualified native optional slots by parent kind."""
    children = node.get("inner", [])
    if not isinstance(children, list):
        raise RuntimeError("Clang AST children require a complete ordered list.")
    result: list[dict[str, object]] = []
    for child in children:
        value = _node(child)
        kind = node.get("kind")
        if isinstance(kind, str) and kind in _NULL_SLOTS and value == _NULL_SLOTS[kind]:
            continue
        result.append(value)
    return result


def _has_contract(node: dict[str, object]) -> bool:
    """Find non-header prose in the declaration's attached native comment tree."""
    pending = [child for child in _children(node) if child.get("kind") == "FullComment"]
    while pending:
        child = pending.pop()
        text = child.get("text")
        if child.get("kind") == "TextComment" and isinstance(text, str):
            if text.strip() and not text.strip().startswith(_HEADER):
                return True
        pending.extend(_children(child))
    return False


def documentation_errors(tree: object, path: Path) -> list[str]:
    """Check attached comments on explicit named declarations in the main source.

    Parameters
    ----------
    tree : object
        JSON decoded from a successful Clang TranslationUnitDecl dump.
    path : pathlib.Path
        Absolute main-file spelling supplied to Clang. Included declarations
        and implicit compiler declarations are not this file's documentation.

    Returns
    -------
    list of str
        Missing comment findings for functions, types, members, enum constants
        and namespace or class variables. Local variables are outside scope.

    Raises
    ------
    RuntimeError
        The AST envelope/children are incomplete or no explicit main-file
        declaration can be measured. Presence does not prove comment semantics.
    """
    root = _node(tree)
    if root.get("kind") != "TranslationUnitDecl":
        raise RuntimeError("Clang AST requires a TranslationUnitDecl root.")
    pending = [(child, "TranslationUnitDecl") for child in _children(root)]
    findings: list[str] = []
    measured = 0
    while pending:
        node, parent = pending.pop()
        kind = node.get("kind")
        if not isinstance(kind, str):
            raise RuntimeError("Clang AST nodes require a native kind.")
        loc = _node(node.get("loc", {}))
        named = isinstance(node.get("name"), str) and bool(node.get("name"))
        own = "offset" in loc and "includedFrom" not in loc and loc.get("file", str(path)) == str(path)
        declaration = kind in _DECLARATIONS or (kind == "VarDecl" and parent in _GLOBAL_CONTEXT)
        expansion = loc.get("expansionLoc")
        if declaration and named and isinstance(expansion, dict):
            if "includedFrom" not in expansion and expansion.get("file", str(path)) == str(path):
                raise RuntimeError("Macro-generated declarations require explicit native profile qualification.")
        if own and named and declaration and not node.get("isImplicit", False):
            measured += 1
            if not _has_contract(node):
                findings.append(f"{path.name}@{loc['offset']}: {node['name']} requires an attached contract comment")
        pending.extend((child, kind) for child in _children(node))
    if not measured:
        raise RuntimeError("Clang AST contains no explicit main-file declarations.")
    return findings


def raw_tokens(dump: str, path: Path) -> list[tuple[str, str]]:
    """Decode every native raw-token record, retaining comments and string boundaries.

    Parameters
    ----------
    dump : str
        Complete stderr from Clang's successful dump-raw-tokens command.
    path : pathlib.Path
        Exact absolute input spelling used by that command.

    Returns
    -------
    list of tuple of str
        Native token kind and spelling in source order, including whitespace.

    Raises
    ------
    RuntimeError
        Any record or source location is absent, malformed or incomplete.
        Multiline comments and raw strings remain single native records.
    """
    record = re.compile(r"(\w+) '(.*?)'\t[^\n]*?Loc=<" + re.escape(str(path)) + r":(\d+):(\d+)>\n", re.S)
    tokens: list[tuple[str, str]] = []
    offset = 0
    previous = (0, 0)
    while offset < len(dump):
        match = record.match(dump, offset)
        if match is None:
            raise RuntimeError("Clang raw-token protocol is incomplete.")
        location = (int(match[3]), int(match[4]))
        if location <= previous or location[1] < 1:
            raise RuntimeError("Clang raw-token locations require increasing source positions.")
        previous = location
        tokens.append((match[1], match[2]))
        offset = match.end()
    if not tokens:
        raise RuntimeError("Clang raw-token protocol is empty.")
    return tokens


def token_directives(tokens: list[tuple[str, str]], name: str) -> tuple[list[str], list[str]]:
    """Find real waiver directives and literal quoted includes, excluding string data.

    Parameters
    ----------
    tokens : list of tuple of str
        Complete actual Clang raw tokens, before macro expansion.
    name : str
        Source name for findings.

    Returns
    -------
    tuple of list of str and list of str
        Waiver findings and relative quoted include paths. Only pragma once
        is admitted; macro include paths refuse this standalone source profile.

    Raises
    ------
    RuntimeError
        A local include cannot be resolved from a literal source spelling.
    """
    significant = [(kind, text.replace("\\\n", "")) for kind, text in tokens if kind != "unknown"]
    findings: list[str] = []
    includes: list[str] = []
    for index, (kind, text) in enumerate(significant):
        if kind == "comment" and re.search(r"\b(?:clang-format|NOLINT)\b", text):
            findings.append(f"{name}: inline native-tool configuration requires profile review")
        if kind == "raw_identifier" and text in {"_Pragma", "__pragma"}:
            findings.append(f"{name}: native pragma requires profile review")
        if kind == "hash" and index + 2 < len(significant):
            directive, target = significant[index + 1][1], significant[index + 2]
            if directive == "pragma" and target[1] != "once":
                findings.append(f"{name}: native pragma requires profile review")
            if directive == "line" or significant[index + 1][0] == "numeric_constant":
                findings.append(f"{name}: source-location directives require profile review")
            if directive in {"include", "include_next", "import"}:
                if directive == "include" and target[0] == "string_literal" and target[1].startswith('"'):
                    includes.append(target[1][1:-1])
                elif directive != "include" or target[0] != "less":
                    raise RuntimeError(f"{name}: include contract requires a literal qualified path.")
                else:
                    end = next((i for i in range(index + 3, len(significant)) if significant[i][0] == "greater"), None)
                    angle = "" if end is None else "".join(word for _, word in significant[index + 3 : end])
                    if not angle or Path(angle).is_absolute() or ".." in Path(angle).parts:
                        raise RuntimeError(f"{name}: angle include requires a qualified system header.")
    return findings, includes
