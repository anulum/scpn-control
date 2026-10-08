# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark artifact tree inspection and digests.
"""Hash regular benchmark files and directory trees without reading special files.

The versioned tree digest orders UTF-8 POSIX relative names case-sensitively,
includes file/directory types and empty directories, and excludes timestamps,
permissions and the root's spelling. File digests remain ordinary SHA-256.
"""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

DIRECTORY_DIGEST_ALGORITHM = "sha256-directory-tree.v2"
LEGACY_DIRECTORY_DIGEST_ALGORITHM = "sha256-directory-files.v1"


def _kind(path: Path) -> str:
    """Classify an actual node, refusing symlinks and special filesystem entries."""
    mode = path.lstat().st_mode
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    raise ValueError(f"benchmark artifacts require regular files/directories and cannot contain symlinks: {path}")


def _entries(path: Path) -> list[Path]:
    """Validate every descendant and order its relative POSIX name case-sensitively."""
    entries = sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
    for entry in entries:
        _kind(entry)
    return entries


def _file_digest(path: Path) -> bytes:
    """Read a regular file in bounded chunks and return its binary SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def sha256_path(path: Path, *, directory_algorithm: str = DIRECTORY_DIGEST_ALGORITHM) -> str:
    """Return a file SHA-256 or a named, versioned directory-content digest.

    Parameters
    ----------
    path : Path
        Existing regular file or directory. Symlinks and special entries refuse
        before content reads. Relative paths use the current working directory.
    directory_algorithm : str
        ``sha256-directory-tree.v2`` includes all relative names, node types and
        file digests in case-sensitive order. ``sha256-directory-files.v1`` reads
        historical file-only digests in native Path order; it makes no portable
        tree or empty-directory integrity claim.

    Returns
    -------
    str
        Lowercase 64-character hexadecimal SHA-256 digest. File content, and
        directory file content/structure for v2, determine the result; metadata
        such as permissions and timestamps does not.

    Raises
    ------
    ValueError
        Unsupported directory algorithm, symlink or special filesystem entry.
    OSError
        Missing/unreadable paths or content read failure. No artifacts mutate.
    """
    if directory_algorithm not in (DIRECTORY_DIGEST_ALGORITHM, LEGACY_DIRECTORY_DIGEST_ALGORITHM):
        raise ValueError("unsupported benchmark directory digest algorithm")
    if _kind(path) == "file":
        return _file_digest(path).hex()
    entries = _entries(path)
    digest = hashlib.sha256()
    if directory_algorithm == LEGACY_DIRECTORY_DIGEST_ALGORITHM:
        for entry in sorted(item for item in entries if _kind(item) == "file"):
            digest.update(entry.relative_to(path).as_posix().encode("utf-8") + b"\0")
            digest.update(_file_digest(entry) + b"\0")
    else:
        digest.update(b"scpn-control.directory-tree.v2\0")
        for entry in entries:
            kind = _kind(entry)
            digest.update(kind.encode("ascii") + b"\0")
            digest.update(entry.relative_to(path).as_posix().encode("utf-8") + b"\0")
            if kind == "file":
                digest.update(_file_digest(entry))
            digest.update(b"\0")
    return digest.hexdigest()


def path_size(path: Path) -> int:
    """Return total regular-file bytes in an artifact, validating every node.

    Parameters
    ----------
    path : Path
        Existing regular file or directory tree. Empty directories contribute
        zero bytes; empty regular files are permitted for inspection.

    Returns
    -------
    int
        Nonnegative sum of file lengths in bytes, including each named hard-link
        entry. It is payload size, not allocated disk space or semantic validity.

    Raises
    ------
    ValueError
        Symlink or special entry. No named pipe or device is opened.
    OSError
        Filesystem inspection fails or an entry disappears during inspection.
    """
    if _kind(path) == "file":
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in _entries(path) if _kind(entry) == "file")
