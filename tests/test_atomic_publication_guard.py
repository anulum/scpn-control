# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — every package output is published atomically

"""Files the package publishes go through ``scpn_control._atomic``.

The scan finds every direct file-writing call in ``src/scpn_control``:
``write_text``/``write_bytes``, ``open``/``fdopen`` in a writing mode,
``json.dump``, NumPy and Torch savers, and temporary-file creation. The few
remaining direct writers are reviewed below, each with the reason it is not
a publication. A new writer, or a reviewed one that disappears, fails the
test, so the inventory cannot drift silently.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "scpn_control"
SAVERS = frozenset({"dump", "save", "savez", "savez_compressed", "savetxt", "tofile"})
TEMPORARY = frozenset({"NamedTemporaryFile", "mkstemp"})

REVIEWED: dict[tuple[str, str], tuple[int, str]] = {
    ("_atomic.py", "open:xb"): (1, "the atomic helper's own exclusive temporary file"),
    ("_npz.py", "open:w"): (1, "a member of the in-memory zip archive"),
    ("_npz.py", "save"): (1, "numpy.save into that in-memory archive member"),
    ("benchmark_records.py", "open:xb"): (2, "exclusive create of records it then replaces atomically"),
    ("benchmark_output_lease.py", "open:a+b"): (1, "the operating-system lock file of the lease registry"),
    ("benchmark_output_lease.py", "NamedTemporaryFile"): (1, "a lease reservation written, fsynced and linked"),
    ("benchmark_output_lease.py", "dump"): (1, "the reservation body inside that temporary file"),
    ("core/_rust_compat.py", "mkstemp"): (1, "a private temporary solver configuration for the Rust kernel"),
    ("core/_rust_compat.py", "open:w"): (1, "fdopen of that temporary configuration"),
    ("core/_rust_compat.py", "dump"): (1, "the configuration body inside it"),
    ("core/_rust_compat.py", "savez"): (1, "numpy.savez into memory before write_bytes_atomic"),
    ("core/fusion_kernel.py", "savez"): (1, "numpy.savez into memory before write_bytes_atomic"),
    ("core/mdsplus_acquisition.py", "savez"): (1, "numpy.savez into memory before write_bytes_atomic"),
    ("control/disruption_checkpoint.py", "save"): (1, "torch.save into memory before write_bytes_atomic"),
    ("control/fusion_control_room.py", "save"): (1, "the animation writer, called on the temporary path"),
    ("core/integrated_scenario.py", "NamedTemporaryFile"): (1, "a private temporary transport-solver configuration"),
    ("core/integrated_scenario.py", "dump"): (1, "the configuration body inside it"),
    ("scpn/controller.py", "open:a"): (1, "an append-only audit log; atomic replacement would drop records"),
}


def _mode(call: ast.Call, *, method: bool) -> str | None:
    """Return the literal mode of an ``open``/``fdopen`` call, if it has one."""
    candidates = [k.value for k in call.keywords if k.arg == "mode"]
    position = 0 if method else 1
    if len(call.args) > position:
        candidates.append(call.args[position])
    for node in candidates:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
    return None


def writers(tree: ast.AST) -> list[str]:
    """Return the kind of every direct file-writing call in ``tree``."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        if name in {"write_text", "write_bytes"} or name in SAVERS or name in TEMPORARY:
            found.append(name)
        elif name in {"open", "fdopen"}:
            mode = _mode(node, method=isinstance(func, ast.Attribute) and name == "open")
            if mode is not None and any(flag in mode for flag in "wax"):
                found.append(f"open:{mode}")
    return found


def test_every_direct_writer_is_reviewed() -> None:
    """Refuse an unreviewed direct writer, and a reviewed one that no longer exists."""
    observed: Counter[tuple[str, str]] = Counter()
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).as_posix()
        for kind in writers(ast.parse(path.read_text(encoding="utf-8"))):
            observed[(relative, kind)] += 1
    assert dict(observed) == {key: count for key, (count, _reason) in REVIEWED.items()}
    assert all(reason for _count, reason in REVIEWED.values())


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from pathlib import Path\nPath('a').write_text('x')\n", ["write_text"]),
        ("open('a', 'w')\nopen('a')\nopen('a', mode='rb')\n", ["open:w"]),
        ("import os\nos.fdopen(3, 'a', encoding='utf-8')\n", ["open:a"]),
        ("from pathlib import Path\nPath('a').open('xb')\nPath('a').open()\n", ["open:xb"]),
        ("import json\njson.dump({}, f)\njson.dumps({})\n", ["dump"]),
        ("import numpy as np\nnp.savez('a', x=1)\nnp.load('a')\n", ["savez"]),
        ("import tempfile\ntempfile.mkstemp()\n", ["mkstemp"]),
    ],
)
def test_scanner_finds_each_writer_spelling(source: str, expected: list[str]) -> None:
    """Find writing calls and ignore reading ones."""
    assert writers(ast.parse(source)) == expected
