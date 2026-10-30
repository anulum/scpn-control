# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — every package subprocess call is time-bounded

"""Every ``subprocess`` call in the package passes an explicit ``timeout``.

A child process that hangs must not hang the caller. The scan covers every
module under ``src/scpn_control`` and every way of reaching the module:
``subprocess.run(...)``, an aliased ``import subprocess as sp`` and
``from subprocess import run``. ``Popen`` has no ``timeout`` argument, so it
is refused outright; a caller needing one must bound ``communicate``/``wait``
through ``run`` instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "scpn_control"
BOUNDED = frozenset({"run", "call", "check_call", "check_output"})
REFUSED = frozenset({"Popen", "getoutput", "getstatusoutput"})


def unbounded_calls(tree: ast.AST) -> list[int]:
    """Return the line of every subprocess call without a ``timeout`` keyword."""
    modules = {"subprocess"}
    functions: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.asname or alias.name for alias in node.names if alias.name == "subprocess")
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            functions.update({alias.asname or alias.name: alias.name for alias in node.names})
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id in modules:
            name = target.attr
        elif isinstance(target, ast.Name) and target.id in functions:
            name = functions[target.id]
        else:
            continue
        if name in REFUSED or (name in BOUNDED and not any(k.arg == "timeout" for k in node.keywords)):
            lines.append(node.lineno)
    return lines


def test_every_package_subprocess_call_passes_a_timeout() -> None:
    """Refuse any unbounded subprocess call in the shipped package."""
    offenders = [
        f"{path.relative_to(PACKAGE.parent)}:{line}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for line in unbounded_calls(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import subprocess\nsubprocess.run(['x'])\n", [2]),
        ("import subprocess as sp\nsp.check_output(['x'])\n", [2]),
        ("from subprocess import run as go\ngo(['x'])\n", [2]),
        ("import subprocess\nsubprocess.Popen(['x'])\n", [2]),
        ("import subprocess\nsubprocess.run(['x'], timeout=5)\n", []),
        ("from subprocess import CalledProcessError\nCalledProcessError(1, 'x')\n", []),
        ("import os\nos.path.join('a')\nprint('x')\n", []),
    ],
)
def test_scanner_finds_unbounded_calls_in_every_spelling(source: str, expected: list[int]) -> None:
    """Find unbounded calls through module, alias and from-import spellings."""
    assert unbounded_calls(ast.parse(source)) == expected
