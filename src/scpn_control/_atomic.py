# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — atomic file publication

"""Publish files so a reader sees the old bytes or the new bytes, never a mix.

The new content is written to a temporary file in the destination's own
directory, flushed to disk, and moved over the destination with
:func:`os.replace`, which is atomic on POSIX and Windows when both paths are
on the same file system. If anything fails, the destination keeps its
previous bytes (or stays absent) and the temporary file is removed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

__all__ = ["publish_through_temporary", "write_bytes_atomic", "write_text_atomic"]


def write_bytes_atomic(path: str | os.PathLike[str], data: bytes) -> Path:
    """Replace ``path`` with ``data`` in one atomic step.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Destination file. Missing parent directories are created.
    data : bytes
        The complete new content.

    Returns
    -------
    pathlib.Path
        The destination path.

    Raises
    ------
    OSError
        If the temporary file cannot be written or moved; the destination is
        then unchanged and no temporary file remains.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_text_atomic(path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8") -> Path:
    """Replace ``path`` with ``text`` in one atomic step.

    The text is encoded before anything touches the file system, so an
    encoding error leaves no trace.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Destination file. Missing parent directories are created.
    text : str
        The complete new content.
    encoding : str
        Text encoding, UTF-8 by default.

    Returns
    -------
    pathlib.Path
        The destination path.

    Raises
    ------
    UnicodeEncodeError
        If ``text`` cannot be encoded; nothing is written.
    OSError
        If the temporary file cannot be written or moved; the destination is
        then unchanged and no temporary file remains.
    """
    return write_bytes_atomic(path, text.encode(encoding))


def publish_through_temporary(path: str | os.PathLike[str], write: Callable[[Path], None]) -> Path:
    """Let ``write`` create a file beside ``path``, then move it over ``path``.

    For writers that need a file name rather than bytes, such as Matplotlib
    animation writers. The temporary name keeps the destination's suffix,
    so writers that infer a format from it still see the right one.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Destination file. Its directory must already exist, as for the
        writer's own ``open``.
    write : callable
        Writes the complete content to the path it is given.

    Returns
    -------
    pathlib.Path
        The destination path.

    Raises
    ------
    Exception
        Whatever ``write`` raises, or ``OSError`` from the move. The
        destination is then unchanged and no temporary file remains.
    """
    destination = Path(path)
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}")
    try:
        write(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
