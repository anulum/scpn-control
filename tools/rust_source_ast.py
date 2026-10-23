# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Qualified native Rust AST documentation and attribute contracts.
"""Read the pinned compiler's pre-expansion AST, including hidden and test items.

Rust source is parsed by rustc. This module reads its qualified debug protocol;
it does not infer declarations or attributes from source-text regular expressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_OPEN = re.compile(r"( *)(?:mac: (?=MacCall))?(Crate|Item|FieldDef|Variant|Attribute|MacCall|MacroDef) \{")
_DOCUMENTED = {"Fn", "Struct", "Enum", "Union", "TyAlias", "Trait", "TraitAlias", "Const", "Static", "Mod", "MacroDef"}
_OTHER = {"Use", "ExternCrate", "Impl", "ForeignMod", "MacCall", "GlobalAsm"}
_DECLARATION_WORDS = {"fn", "struct", "enum", "union", "trait", "type", "mod", "impl", "macro_rules"}
_HEADER = ("SPDX-", "Commercial license", "©", "ORCID:", "Contact:", "SCPN Control")


@dataclass
class _Block:
    """Retain one native record's line extent and closest retained parent."""

    tag: str
    indent: int
    first: int
    last: int
    parent: int | None


def _blocks(lines: list[str]) -> list[_Block]:
    """Require one complete native crate and balanced declaration/attribute records."""
    if not lines or lines[0] != "Crate {" or lines[-1] != "}":
        raise RuntimeError("Rust AST requires one complete native Crate envelope.")
    records: list[_Block] = []
    stack: list[int] = []
    for index, line in enumerate(lines):
        opening = _OPEN.fullmatch(line)
        indent = len(line) - len(line.lstrip())
        if opening:
            if indent % 4 or (stack and indent <= records[stack[-1]].indent):
                raise RuntimeError("Rust AST record indentation is incomplete.")
            records.append(_Block(opening[2], indent, index, -1, stack[-1] if stack else None))
            stack.append(len(records) - 1)
        elif stack and line.strip() in {"}", "},"} and indent <= records[stack[-1]].indent:
            if indent != records[stack[-1]].indent:
                raise RuntimeError("Rust AST record closure is incomplete.")
            records[stack.pop()].last = index
    if (
        stack
        or not records
        or records[0].last != len(lines) - 1
        or sum(record.tag == "Crate" for record in records) != 1
    ):
        raise RuntimeError("Rust AST records must close within one native crate.")
    return records


def _fields(block: _Block, lines: list[str]) -> list[str]:
    """Read only direct scalar/constructor fields of the requested native record."""
    return [
        line.strip()
        for line in lines[block.first + 1 : block.last]
        if len(line) - len(line.lstrip()) == block.indent + 4
    ]


def _kind(block: _Block, lines: list[str]) -> str:
    """Require a known native item constructor without inspecting nested expression kinds."""
    kinds = [line for line in _fields(block, lines) if line.startswith("kind: ")]
    if len(kinds) != 1:
        raise RuntimeError("Rust AST item requires exactly one native kind.")
    match = re.fullmatch(r"kind: ([A-Za-z]+)\(", kinds[0])
    if match is None or match[1] not in _DOCUMENTED | _OTHER:
        raise RuntimeError("Rust AST item kind requires explicit profile qualification.")
    return match[1]


def _string(spelling: str) -> str:
    """Decode a native debug string, including Rust's braced Unicode and NUL escapes."""
    if len(spelling) < 2 or spelling[0] != '"' or spelling[-1] != '"':
        raise RuntimeError("Rust AST documentation requires a native string value.")
    value = ""
    index = 1
    escapes = {"n": "\n", "r": "\r", "t": "\t", "0": "\0", '"': '"', "\\": "\\"}
    while index < len(spelling) - 1:
        char = spelling[index]
        if char == "\\":
            index += 1
            escape = spelling[index]
            if escape in escapes:
                char = escapes[escape]
            elif escape == "u" and spelling[index + 1 : index + 2] == "{":
                end = spelling.find("}", index + 2)
                if end < 0:
                    raise RuntimeError("Rust AST Unicode escape is incomplete.")
                code = int(spelling[index + 2 : end], 16)
                if 0xD800 <= code <= 0xDFFF:
                    raise RuntimeError("Rust AST Unicode value must be a scalar.")
                char = chr(code)
                index = end
            else:
                raise RuntimeError("Rust AST string escape requires profile qualification.")
        elif char == '"' or ord(char) < 32:
            raise RuntimeError("Rust AST strings require complete escaped scalar values.")
        value += char
        index += 1
    return value


def _comment(block: _Block, lines: list[str]) -> str | None:
    """Return native doc-comment prose, distinguishing it from normal doc attributes."""
    kinds = [line for line in _fields(block, lines) if line.startswith("kind: ")]
    if kinds == ["kind: Normal("]:
        return None
    if kinds != ["kind: DocComment("]:
        raise RuntimeError("Rust AST attribute requires a qualified native kind.")
    strings = [
        line.strip().removesuffix(",") for line in lines[block.first + 1 : block.last] if line.lstrip().startswith('"')
    ]
    if len(strings) != 1:
        raise RuntimeError("Rust AST doc comment requires exactly one string.")
    text = _string(strings[0])
    return re.sub(r"(?m)^\s*\* ?", "", text).strip()


def _position(block: _Block, lines: list[str], path: Path) -> str:
    """Bind an explicit declaration span to the exact native main-file spelling."""
    if block.tag == "Crate":
        return f"{path}:1:1"
    spans = [line for line in _fields(block, lines) if line.startswith("span: ")]
    if len(spans) != 1:
        raise RuntimeError("Rust AST declaration requires one complete native span.")
    match = re.fullmatch(r"span: " + re.escape(str(path)) + r":(\d+):(\d+): \d+:\d+ \(#\d+\),", spans[0])
    if match is None or min(int(match[1]), int(match[2])) < 1:
        raise RuntimeError("Rust AST declaration span is outside the qualified main file.")
    return f"{path}:{match[1]}:{match[2]}"


def _attribute_errors(block: _Block, lines: list[str], path: Path) -> list[str]:
    """Reject actual native waiver and formatter/config attributes, excluding quoted data."""
    if _comment(block, lines) is not None:
        return []
    names = [
        match[1]
        for line in lines[block.first + 1 : block.last]
        if (match := re.fullmatch(r"\s*ident: ([^#\s]+)#\d+,", line))
    ]
    if not names:
        raise RuntimeError("Rust AST normal attribute requires a native path.")
    configured = names[0] in {"allow", "expect", "rustfmt"}
    if names[0] == "cfg_attr":
        configured = bool(set(_token_words(block, lines)) & {"allow", "expect", "rustfmt"})
    if configured:
        return [f"{path}: inline Rust profile configuration requires review ({names[0]})"]
    return []


def _token_words(block: _Block, lines: list[str]) -> list[str]:
    """Read native identifier tokens without mistaking string or raw-string payloads for code."""
    body = lines[block.first + 1 : block.last]
    return [
        _string(body[index + 1].strip().removesuffix(","))
        for index, line in enumerate(body[:-1])
        if line.strip() == "kind: Ident("
    ]


def _macro_errors(block: _Block, lines: list[str], path: Path) -> list[str]:
    """Refuse declaration-bearing native macro tokens until their generated source is qualified."""
    if set(_token_words(block, lines)) & _DECLARATION_WORDS:
        return [f"{path}: macro declaration tokens require generated-source qualification"]
    return []


def _contracts(records: list[_Block], lines: list[str], path: Path) -> tuple[dict[int, list[str]], list[str]]:
    """Collect only attached native comments and actual attribute/macro findings."""
    comments: dict[int, list[str]] = {}
    findings: list[str] = []
    for block in records:
        if block.tag == "Attribute":
            findings.extend(_attribute_errors(block, lines, path))
            parent_index = block.parent
            parent = records[parent_index] if parent_index is not None else None
            text = _comment(block, lines)
            if (
                parent_index is not None
                and parent
                and block.indent == parent.indent + 8
                and text
                and not text.startswith(_HEADER)
            ):
                comments.setdefault(parent_index, []).append(text)
        elif block.tag in {"MacCall", "MacroDef"}:
            findings.extend(_macro_errors(block, lines, path))
    return comments, findings


def rust_ast_errors(dump: str, path: Path) -> list[str]:
    """Check genuine doc comments and attributes on every physical native declaration.

    Parameters
    ----------
    dump : str
        Successful ``rustc +nightly-2026-08-21 -Zunpretty=ast-tree`` output.
    path : pathlib.Path
        Exact absolute main-file path passed to the native compiler parser.

    Returns
    -------
    list of str
        Missing doc-comment or native configuration findings, including hidden,
        private, nested, test and inactive-cfg declarations. Item imports, impl
        containers and foreign blocks defer to their contained declarations.

    Raises
    ------
    RuntimeError
        The pinned protocol, nonempty implementation or exact source binding is
        unavailable. Unsupported generated/configured forms cannot certify it.

    Notes
    -----
    File/module, function, type, field and variant contracts use native ``//!``,
    ``///`` or block doc comments. Normal doc attributes do not substitute for
    those comments. Declaration-bearing macro tokens require a generated-source
    profile. Presence is independent of semantic review and Cargo compilation.
    """
    try:
        lines = dump.splitlines()
        records = _blocks(lines)
        if not any(block.tag == "Item" for block in records):
            raise RuntimeError("Rust AST needs a nonempty native implementation.")
        comments, findings = _contracts(records, lines, path)
        for index, block in enumerate(records):
            required = block.tag in {"Crate", "FieldDef", "Variant"}
            if block.tag == "Item":
                required = _kind(block, lines) in _DOCUMENTED
            position = _position(block, lines, path) if block.tag in {"Crate", "Item", "FieldDef", "Variant"} else ""
            if required and not comments.get(index):
                findings.append(f"{position}: {block.tag} requires an attached contract comment")
        return findings
    except (ValueError, IndexError, KeyError, TypeError, UnicodeError, RecursionError) as error:
        raise RuntimeError("Rust AST requires complete native records and strings.") from error
