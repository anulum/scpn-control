# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Malicious G-EQDSK parser hardening tests.
"""Fail-closed behaviour of :func:`read_geqdsk` on hostile G-EQDSK input.

G-EQDSK files can be attacker-supplied (they are globbed from reference
directories during validation). The parser must reject malformed content and
oversized declared grids with a typed :class:`GEqdskFormatError` instead of
crashing with a bare ``IndexError``/``ValueError`` or exhausting memory on an
unbounded allocation. These tests pin every rejection path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_control.core.eqdsk import (
    _MAX_BOUNDARY_POINTS,
    _MAX_GRID_DIM,
    GEqdskFormatError,
    read_geqdsk,
)


def _valid_tokens(nw: int = 1, nh: int = 1, nbdry: int = 0, nlim: int = 0) -> list[str]:
    """Return an all-zero numeric body for a syntactically valid ``nw × nh`` grid."""
    tokens = ["0.0"] * (20 + nw * (5 + nh))
    tokens += [str(nbdry), str(nlim)]
    tokens += ["0.0"] * (2 * nbdry + 2 * nlim)
    return tokens


def _write(
    path: Path,
    *,
    header: str = "TEST 0 1 1",
    tokens: list[str] | None = None,
) -> str:
    """Write a G-EQDSK document and return its path as a string."""
    body = _valid_tokens() if tokens is None else tokens
    path.write_text(header + "\n" + " ".join(body) + "\n", encoding="utf-8")
    return str(path)


def test_valid_minimal_geqdsk_parses(tmp_path: Path) -> None:
    """A syntactically valid minimal document still parses (positive control)."""
    equilibrium = read_geqdsk(_write(tmp_path / "ok.geqdsk"))

    assert equilibrium.nw == 1
    assert equilibrium.nh == 1


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    """An empty file fails closed rather than indexing an empty line list."""
    empty = tmp_path / "empty.geqdsk"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(GEqdskFormatError, match="empty"):
        read_geqdsk(str(empty))


def test_header_without_grid_sizes_is_rejected(tmp_path: Path) -> None:
    """A header lacking nw/nh cannot index ``parts[-2]`` and is refused."""
    with pytest.raises(GEqdskFormatError, match="nw and nh"):
        read_geqdsk(_write(tmp_path / "short.geqdsk", header="ONLYONE"))


def test_non_integer_grid_size_is_rejected(tmp_path: Path) -> None:
    """A non-integer grid size raises a typed error, not a bare ValueError."""
    with pytest.raises(GEqdskFormatError, match="non-integer grid sizes"):
        read_geqdsk(_write(tmp_path / "bad_dim.geqdsk", header="TEST 0 1 z"))


def test_oversize_grid_dimension_is_rejected(tmp_path: Path) -> None:
    """A grid dimension above the cap is refused before any allocation."""
    huge = _MAX_GRID_DIM + 1
    with pytest.raises(GEqdskFormatError, match="outside"):
        read_geqdsk(_write(tmp_path / "huge.geqdsk", header=f"TEST 0 {huge} 1", tokens=["0.0"]))


def test_zero_grid_dimension_is_rejected(tmp_path: Path) -> None:
    """A non-positive grid dimension is refused (low end of the range)."""
    with pytest.raises(GEqdskFormatError, match="outside"):
        read_geqdsk(_write(tmp_path / "zero.geqdsk", header="TEST 0 0 1", tokens=["0.0"]))


def test_truncated_scalar_block_is_rejected(tmp_path: Path) -> None:
    """A document that ends before the scalar block fails closed."""
    with pytest.raises(GEqdskFormatError, match="ended before"):
        read_geqdsk(_write(tmp_path / "trunc_scalar.geqdsk", tokens=[]))


def test_truncated_array_block_is_rejected(tmp_path: Path) -> None:
    """A document that ends inside an array read fails closed."""
    with pytest.raises(GEqdskFormatError, match="too few values"):
        read_geqdsk(_write(tmp_path / "trunc_array.geqdsk", tokens=["0.0"] * 20))


def test_non_numeric_bytes_between_values_are_ignored(tmp_path: Path) -> None:
    """Non-numeric bytes are dropped by the Fortran token filter, not parsed.

    Injected garbage that does not remove any required value leaves a complete
    numeric body, so the document still parses — the parser never calls
    ``float`` on a non-numeric token.
    """
    body = _valid_tokens()
    laced = ["GARBAGE", *body[:5], "JUNK", *body[5:], "TRAILING"]

    equilibrium = read_geqdsk(_write(tmp_path / "laced.geqdsk", tokens=laced))

    assert equilibrium.nw == 1
    assert equilibrium.nh == 1


def test_oversize_boundary_count_is_rejected(tmp_path: Path) -> None:
    """A boundary-point count above the cap is refused before ``np.zeros``."""
    huge = _MAX_BOUNDARY_POINTS + 1
    tokens = ["0.0"] * (20 + 1 * (5 + 1)) + [str(huge), "0"]
    with pytest.raises(GEqdskFormatError, match="outside"):
        read_geqdsk(_write(tmp_path / "huge_bdry.geqdsk", tokens=tokens))


def test_non_finite_boundary_count_is_rejected(tmp_path: Path) -> None:
    """A count token that overflows to ±inf fails closed, not with ``int(inf)``.

    A Fortran token with a huge exponent (``1e999``) parses to ``+inf``; the
    parser rejects any non-finite token at ``_next``, so the count never reaches
    ``int(inf)`` (which would raise a bare ``OverflowError``).
    """
    body = _valid_tokens()
    body[-2] = "1e999"  # boundary-count slot overflows to +inf
    with pytest.raises(GEqdskFormatError, match="finite"):
        read_geqdsk(_write(tmp_path / "inf_bdry.geqdsk", tokens=body))


def test_non_finite_scalar_value_is_rejected(tmp_path: Path) -> None:
    """A non-finite scalar (huge Fortran exponent) is rejected at parse."""
    body = _valid_tokens()
    body[0] = "1e999"  # the rdim scalar overflows to +inf
    with pytest.raises(GEqdskFormatError, match="non-finite value"):
        read_geqdsk(_write(tmp_path / "inf_scalar.geqdsk", tokens=body))


def test_non_finite_array_value_is_rejected(tmp_path: Path) -> None:
    """A non-finite profile/flux value is rejected before it reaches the equilibrium."""
    body = _valid_tokens()
    body[20] = "1e999"  # the first profile-array element overflows to +inf
    with pytest.raises(GEqdskFormatError, match="array contains a non-finite value"):
        read_geqdsk(_write(tmp_path / "inf_array.geqdsk", tokens=body))


def test_non_utf8_file_is_rejected(tmp_path: Path) -> None:
    """A file that is not valid UTF-8 fails closed, not with ``UnicodeDecodeError``.

    Reference directories are globbed for ``.geqdsk`` files, so a byte sequence
    that cannot be decoded as text must raise the typed error rather than leak a
    decoder exception from ``readlines``.
    """
    bad = tmp_path / "not_utf8.geqdsk"
    bad.write_bytes(b"TEST 0 1 1\n\xff\xfe not text \x80\n")
    with pytest.raises(GEqdskFormatError, match="UTF-8"):
        read_geqdsk(str(bad))
