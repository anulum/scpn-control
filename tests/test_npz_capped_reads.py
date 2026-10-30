# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — every package NPZ read goes through the capped reader

"""Every ``.npz`` read in the package goes through ``load_npz_capped``.

The capped reader refuses pickled objects and archives whose declared
decompressed size exceeds its cap, before NumPy materialises anything. Only
``scpn_control._npz`` itself may call ``numpy.load``.
"""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import pytest

from scpn_control._npz import NpzSizeError
from scpn_control.core.neural_turbulence import QLKNNSurrogate

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "scpn_control"
READER = PACKAGE / "_npz.py"


def raw_loads(tree: ast.AST) -> list[int]:
    """Return the line of every ``numpy.load`` call, however NumPy was imported."""
    modules = {"numpy"}
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.asname or alias.name for alias in node.names if alias.name == "numpy")
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
            names.update(alias.asname or alias.name for alias in node.names if alias.name == "load")
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in modules
            )
            or (isinstance(node.func, ast.Name) and node.func.id in names)
        )
    ]


def test_only_the_capped_reader_calls_numpy_load() -> None:
    """Refuse a raw ``numpy.load`` anywhere in the shipped package but the reader."""
    offenders = [
        f"{path.relative_to(PACKAGE.parent)}:{line}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path != READER
        for line in raw_loads(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert offenders == []
    assert raw_loads(ast.parse(READER.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import numpy as np\nnp.load('a.npz')\n", [2]),
        ("import numpy\nnumpy.load('a.npz')\n", [2]),
        ("from numpy import load as read\nread('a.npz')\n", [2]),
        ("import json\njson.load(open('a'))\n", []),
        ("import numpy as np\nnp.save('a', 1)\n", []),
    ],
)
def test_scanner_finds_raw_loads_in_every_spelling(source: str, expected: list[int]) -> None:
    """Find raw loads through module, alias and from-import spellings."""
    assert raw_loads(ast.parse(source)) == expected


def test_turbulence_weights_refuse_an_archive_over_the_cap(tmp_path: Path) -> None:
    """A weights archive that declares more than 512 MiB is refused unread.

    The member is streamed zeros, so the file on disk stays small while its
    central directory declares 513 MiB, the shape of a decompression bomb.
    """
    path = tmp_path / "weights.npz"
    chunk = bytes(8 * 1024 * 1024)
    with (
        zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive,
        archive.open("w0.npy", "w", force_zip64=True) as member,
    ):
        for _ in range(513 // 8 + 1):
            member.write(chunk)
    with pytest.raises(NpzSizeError):
        QLKNNSurrogate().load_weights(str(path))
