# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Native syntax, lint and contract checks for changed Bash sources.
"""Check complete changed Bash files without executing their operational bodies.

Bash syntax, shfmt's actual parser/formatter and ShellCheck are independent
checks. Function comments are presence contracts, not semantic review. Other
interpreters, runtime behaviour and scientific acceptance remain separate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.docstring_debt_ledger import read_owner_ledger
from tools.source_quality_scope import select_changed_files

ROOT = Path(__file__).resolve().parents[1]
_VERSIONS = {
    "bash": "GNU bash, version 5.2.21(",
    "shellcheck": "version: 0.9.0",
    "shfmt": "3.8.0",
}
_FORMAT = ["--language-dialect", "bash", "--indent", "4", "--case-indent"]
_HEADER = ("!", "SPDX-", "Commercial license", "©", "ORCID:", "Contact:", "SCPN Control")


def _native(
    repo: Path, argv: Sequence[str], timeout_s: float, *, text: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a native static tool with startup-file injection removed from its environment."""
    env = os.environ.copy()
    for name in ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "SHELLCHECK_OPTS"):
        env.pop(name, None)
    env["SHFMT_NO_EDITORCONFIG"] = "true"
    try:
        return subprocess.run(
            argv, cwd=repo, input=text, capture_output=True, text=True, check=False, timeout=timeout_s, env=env
        )
    except (OSError, UnicodeError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Shell checks require completed native {argv[0]} execution.") from error


def _literal_arguments(command: object) -> list[object]:
    """Retain plain native word values, marking expansion or composition as unknown."""
    if not isinstance(command, dict) or not isinstance(command.get("Args", []), list):
        raise RuntimeError("Shell call arguments require a complete native syntax object.")
    return [word["Parts"][0].get("Value") if len(word["Parts"]) == 1 else None for word in command.get("Args", [])]


def _ast_findings(tree: object, name: str) -> list[str]:
    """Read real shfmt comments and declarations without mistaking quoted data for directives."""
    if (
        not isinstance(tree, dict)
        or tree.get("Type") != "File"
        or not isinstance(tree.get("Stmts"), list)
        or not tree["Stmts"]
    ):
        raise RuntimeError("A complete nonempty shfmt File syntax graph is required.")
    findings: list[str] = []
    first = tree["Stmts"][0].get("Cmd", {})
    words = _literal_arguments(first)
    if first.get("Type") != "CallExpr" or words != ["set", "-euo", "pipefail"]:
        findings.append(f"{name}: first statement must establish set -euo pipefail")
    pending: list[object] = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, dict):
            if node.get("Type") == "CallExpr":
                words = _literal_arguments(node)
                if words and words[0] == "set" and words != ["set", "-euo", "pipefail"] and words[1:2] != ["--"]:
                    findings.append(f"{name}: shell mode changes require profile review")
            if "Hash" in node and isinstance(node.get("Text"), str):
                text = node["Text"].strip()
                if re.match(r"(?:shellcheck|shfmt)(?:\s|:)", text):
                    findings.append(f"{name}: inline shell-tool configuration requires profile review")
            command = node.get("Cmd")
            if isinstance(command, dict) and command.get("Type") == "FuncDecl":
                comments = node.get("Comments", [])
                documented = any(
                    comment["End"]["Line"] == command["Pos"]["Line"] - 1
                    and bool(comment["Text"].strip())
                    and not comment["Text"].strip().startswith(_HEADER)
                    for comment in comments
                )
                if not documented:
                    findings.append(
                        f"{name}: function {command['Name']['Value']} requires an adjacent contract comment"
                    )
            pending.extend(node.values())
    return findings


def changed_shell_errors(repo: Path, base_revision: str, *, timeout_s: float = 60.0) -> list[str]:
    """Run native whole-file Bash checks on every automatically selected shell path.

    Parameters
    ----------
    repo : pathlib.Path
        Git working-tree root. Scripts are read, parsed, linted and formatted
        for comparison; none of their operational commands is executed.
    base_revision : str
        Trusted original full ancestor commit, validated by the shared selector.
    timeout_s : float
        Finite positive bound per native command, including version probes.

    Returns
    -------
    list of str
        Syntax, lint, formatting or function-comment findings. Empty means this
        static profile passed or no changed shell source was selected.

    Raises
    ------
    RuntimeError
        Original history, complete parser data, native tools, the qualified
        interpreter/profile or stable final source selection is unavailable.

    Notes
    -----
    The qualified versions are Bash 5.2.21, ShellCheck 0.9.0 and shfmt 3.8.0.
    Only Bash shebangs and an initial strict-mode statement are admitted.
    ShellCheck ignores rc files and uses its
    complete default severity scope; optional rule families are not silently
    substituted for the existing profile. Actual inline tool directives refuse.
    Shfmt uses four spaces and indented cases with explicit parser/format flags.
    Header prose alone does not document a function. These are native static and
    documentation-presence checks, not runtime coverage or semantic admission.
    """
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise RuntimeError("The shell-check timeout must be finite and positive.")
    repo = repo.resolve()
    scope = select_changed_files(repo, base_revision)
    selected = [item for item in scope if item.profile == "shell"]
    if not selected:
        return []
    try:
        for tool, expected in _VERSIONS.items():
            result = _native(repo, [tool, "--version"], timeout_s)
            lines = result.stdout.splitlines()
            matches = any(line.startswith(expected) for line in lines) if tool == "bash" else expected in lines
            if result.returncode or not matches:
                raise RuntimeError(f"Shell checks require the qualified {tool} profile: {expected}.")
        errors: list[str] = []
        for item in selected:
            path = repo / item.path
            source = path.read_bytes().decode("utf-8")
            if source.partition("\n")[0] not in {"#!/bin/bash", "#!/usr/bin/env bash"}:
                raise RuntimeError(f"{item.path}: interpreter requires its own qualified shell profile.")
            ast = _native(
                repo,
                ["shfmt", "--to-json", "--language-dialect", "bash", "--filename", item.path],
                timeout_s,
                text=source,
            )
            if ast.returncode:
                errors.append(f"{item.path}: shfmt parse failed:\n{ast.stderr}")
            else:
                errors.extend(_ast_findings(json.loads(ast.stdout), item.path))
            commands = (
                ("Bash syntax", ["bash", "--noprofile", "--norc", "-n", "--", str(path)]),
                ("ShellCheck", ["shellcheck", "--norc", "--shell=bash", "--severity=style", str(path)]),
                ("shfmt", ["shfmt", *_FORMAT, "--diff", str(path)]),
            )
            for label, argv in commands:
                result = _native(repo, argv, timeout_s)
                if result.returncode:
                    errors.append(f"{item.path}: {label} failed ({result.returncode}):\n{result.stdout}{result.stderr}")
        if select_changed_files(repo, base_revision) != scope:
            raise RuntimeError("Shell source selection changed during native checks.")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as error:
        raise RuntimeError("Shell checks require readable sources and complete native parser data.") from error
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shell-only static profile with mandatory original-history binding.

    Parameters
    ----------
    argv : sequence of str or None
        CLI arguments; omitted ``--base`` uses the required owner-ledger origin.

    Returns
    -------
    int
        Zero for this profile, one for findings, two for refused measurement.
        Other native languages and operational script execution are separate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--base")
    args = parser.parse_args(argv)
    try:
        base = args.base or read_owner_ledger(args.repo / "tools/docstring_debt_ceiling.json").origin_revision
        errors = changed_shell_errors(args.repo, base)
    except RuntimeError as error:
        print(f"Changed-shell checks refused: {error}", file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Changed-shell static profile passed; operational commands and other native profiles remain separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
