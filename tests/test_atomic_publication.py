# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — atomic file publication tests

"""A failed publication leaves the previous bytes and no temporary file.

Every failure here is real: an unencodable text, a destination that is a
directory, and a directory without write permission.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from scpn_control._atomic import publish_through_temporary, write_bytes_atomic, write_text_atomic


def _leftovers(directory: Path) -> list[str]:
    """Return the names of temporary files left in ``directory``."""
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))


def test_text_is_published_and_replaced_with_parents_created(tmp_path: Path) -> None:
    """Create missing parents, publish, then replace the whole content."""
    target = tmp_path / "nested" / "report.json"
    assert write_text_atomic(target, '{"a": 1}\n') == target
    assert target.read_text(encoding="utf-8") == '{"a": 1}\n'
    write_text_atomic(str(target), "é\n", encoding="latin-1")
    assert target.read_bytes() == b"\xe9\n"
    assert _leftovers(target.parent) == []


def test_unencodable_text_leaves_the_previous_file_untouched(tmp_path: Path) -> None:
    """Refuse a lone surrogate before the file system is touched."""
    target = tmp_path / "claim.json"
    target.write_bytes(b"previous")
    with pytest.raises(UnicodeEncodeError):
        write_text_atomic(target, "\ud800")
    assert target.read_bytes() == b"previous"
    assert _leftovers(tmp_path) == []


def test_failed_move_removes_the_temporary_file(tmp_path: Path) -> None:
    """A destination that is a directory refuses the move; the temporary goes."""
    target = tmp_path / "evidence"
    target.mkdir()
    (target / "keep.txt").write_text("kept", encoding="utf-8")
    with pytest.raises(OSError):
        write_bytes_atomic(target, b"new")
    assert (target / "keep.txt").read_text(encoding="utf-8") == "kept"
    assert _leftovers(tmp_path) == []


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions without root")
def test_unwritable_directory_keeps_the_previous_file(tmp_path: Path) -> None:
    """A directory without write permission refuses the temporary file."""
    directory = tmp_path / "locked"
    directory.mkdir()
    target = directory / "certificate.json"
    target.write_bytes(b"previous")
    directory.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            write_bytes_atomic(target, b"new")
        assert target.read_bytes() == b"previous"
        assert _leftovers(directory) == []
    finally:
        directory.chmod(0o700)


def test_path_writer_publishes_with_the_destination_suffix(tmp_path: Path) -> None:
    """Hand the writer a sibling path with the same suffix, then publish it."""
    target = tmp_path / "loop.gif"
    seen: list[Path] = []

    def write(path: Path) -> None:
        """Record the path and write a GIF header to it."""
        seen.append(path)
        path.write_bytes(b"GIF89a")

    assert publish_through_temporary(target, write) == target
    assert target.read_bytes() == b"GIF89a"
    assert seen[0].parent == target.parent
    assert seen[0].suffix == ".gif"
    assert [p.name for p in target.parent.iterdir()] == ["loop.gif"]


def test_failing_path_writer_keeps_the_previous_file(tmp_path: Path) -> None:
    """A writer that fails half way leaves the previous bytes and no temporary file."""
    target = tmp_path / "loop.gif"
    target.write_bytes(b"previous")

    def write(path: Path) -> None:
        """Write part of a file, then fail like a broken encoder."""
        path.write_bytes(b"partial")
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        publish_through_temporary(target, write)
    assert target.read_bytes() == b"previous"
    assert [p.name for p in tmp_path.iterdir()] == ["loop.gif"]


def test_path_writer_needs_an_existing_directory(tmp_path: Path) -> None:
    """Leave directory creation to the caller: a missing directory fails the writer."""
    target = tmp_path / "missing" / "loop.gif"

    def write(path: Path) -> None:
        """Write a GIF header the way an animation writer opens its file."""
        path.write_bytes(b"GIF89a")

    with pytest.raises(FileNotFoundError):
        publish_through_temporary(target, write)
    assert not (tmp_path / "missing").exists()
