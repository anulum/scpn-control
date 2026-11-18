# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Source and runtime provenance of recorded benchmark runs.
"""Bind a benchmark run to the exact source content and runtime it measured.

A commit identifier alone cannot tell a clean checkout from an edited one. This
module hashes the declared source inputs themselves, tracked and untracked alike,
and compares them with the Git index, without spawning any command:

* the **source snapshot** digests every declared source file by path and content,
  so two runs whose sources differ never share a digest, whatever the commit says;
* ``worktree_vs_index`` reports ``clean`` when every declared file equals its
  staged blob and no declared file is untracked, deleted or in conflict, ``dirty``
  otherwise, and ``unknown`` when no readable index exists. Files are compared as
  ``git add`` would stage them under the repository's LF normalisation. Staged
  changes are not compared with HEAD; the content digest still separates such runs;
* the **runtime identity** names the Python interpreter by digest and the native
  extension by the digest of its module file, found without importing it.

Internal planning files live outside the declared roots, and secret-like names
and build caches inside them are never read or listed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

SOURCE_DIGEST_ALGORITHM = "sha256-declared-sources.v1"
DECLARED_SOURCE_ROOTS = ("src/scpn_control", "scpn-control-rs/crates", "benchmarks", "tools")
DECLARED_SOURCE_FILES = ("pyproject.toml", "uv.lock", "scpn-control-rs/Cargo.toml", "scpn-control-rs/Cargo.lock")
NATIVE_EXTENSION_MODULE = "scpn_control_rs"

_EXCLUDED_DIRECTORIES = frozenset(
    {"__pycache__", "target", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules", ".git"}
)
_SECRET_NAME = re.compile(r"^\.env|\.(pem|key|p12|pfx)$|credential|secret", re.IGNORECASE)


@dataclass(frozen=True)
class SourceSnapshot:
    """Content digest and index comparison of the declared source inputs."""

    tree_sha256: str
    file_count: int
    worktree_vs_index: str
    modified: int
    untracked: int
    deleted: int

    def to_record(self) -> dict[str, object]:
        """Return the JSON-ready provenance fields of this snapshot.

        Returns
        -------
        dict[str, object]
            The digest algorithm, digest, file count, index status and counts.
        """
        return {
            "algorithm": SOURCE_DIGEST_ALGORITHM,
            "tree_sha256": self.tree_sha256,
            "file_count": self.file_count,
            "worktree_vs_index": self.worktree_vs_index,
            "modified": self.modified,
            "untracked": self.untracked,
            "deleted": self.deleted,
        }


def _excluded(relative: str) -> bool:
    """Return whether a POSIX relative path must never be read or listed."""
    parts = relative.split("/")
    return any(part in _EXCLUDED_DIRECTORIES for part in parts[:-1]) or bool(_SECRET_NAME.search(parts[-1]))


def _declared_paths(repository_root: Path) -> list[str]:
    """Return the sorted POSIX relative paths of every declared source file present."""
    found: set[str] = set()
    for name in DECLARED_SOURCE_FILES:
        if (repository_root / name).is_file() and not _excluded(name):
            found.add(name)
    for root in DECLARED_SOURCE_ROOTS:
        base = repository_root / root
        if not base.is_dir():
            continue
        for directory, subdirectories, files in os.walk(base):
            subdirectories[:] = sorted(name for name in subdirectories if name not in _EXCLUDED_DIRECTORIES)
            for name in files:
                if name.endswith(".pyc"):
                    continue
                relative = Path(directory, name).relative_to(repository_root).as_posix()
                if not _excluded(relative):
                    found.add(relative)
    return sorted(found)


def _within_declared(relative: str) -> bool:
    """Return whether an index path belongs to the declared source set."""
    if _excluded(relative) or relative.endswith(".pyc"):
        return False
    return relative in DECLARED_SOURCE_FILES or any(relative.startswith(root + "/") for root in DECLARED_SOURCE_ROOTS)


def _git_directory(repository_root: Path) -> Path | None:
    """Locate the repository's Git directory, following a worktree ``gitdir:`` file."""
    marker = repository_root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir: "):
            return (repository_root / text.removeprefix("gitdir: ")).resolve()
    return None


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decode the Git index v4 offset varint at ``offset``; return value and next offset."""
    byte = data[offset]
    value = byte & 0x7F
    offset += 1
    while byte & 0x80:
        byte = data[offset]
        value = ((value + 1) << 7) | (byte & 0x7F)
        offset += 1
    return value, offset


def read_index_blobs(index_path: Path) -> dict[str, str | None]:
    """Return the staged blob SHA-1 of every index path; ``None`` marks a conflict.

    Parameters
    ----------
    index_path : Path
        A Git index file of version 2, 3 or 4.

    Returns
    -------
    dict[str, str or None]
        POSIX path to the lowercase hexadecimal blob SHA-1 of stage 0, or ``None``
        when the path has an unmerged (stage 1-3) entry.

    Raises
    ------
    ValueError
        If the file is not a Git index of a supported version or is truncated.
    """
    data = index_path.read_bytes()
    if len(data) < 12 or data[:4] != b"DIRC":
        raise ValueError(f"not a Git index: {index_path}")
    version, count = struct.unpack(">II", data[4:12])
    if version not in (2, 3, 4):
        raise ValueError(f"unsupported Git index version {version}")
    offset, previous = 12, b""
    blobs: dict[str, str | None] = {}
    for _ in range(count):
        start = offset
        if offset + 62 > len(data):
            raise ValueError("truncated Git index entry")
        sha1 = data[offset + 40 : offset + 60].hex()
        flags = struct.unpack(">H", data[offset + 60 : offset + 62])[0]
        offset += 62
        if version >= 3 and flags & 0x4000:
            offset += 2
        if version == 4:
            strip, offset = _read_varint(data, offset)
            end = data.index(b"\0", offset)
            name = previous[: len(previous) - strip] + data[offset:end]
            offset = end + 1
        else:
            end = data.index(b"\0", offset)
            name = data[offset:end]
            offset = start + ((end - start + 8) // 8) * 8
        previous = name
        path = name.decode("utf-8")
        stage = (flags >> 12) & 0x3
        blobs[path] = None if stage or blobs.get(path, "") is None else sha1
    return blobs


def _blob_sha1(content: bytes) -> str:
    """Return the Git blob SHA-1 that ``git add`` would stage for this content.

    The repository's ``.gitattributes`` declares ``* text=auto eol=lf``: content
    without a NUL byte is text and is staged with CRLF line endings normalised to
    LF; content with a NUL byte is binary and is staged unchanged.
    """
    staged = content if b"\0" in content else content.replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(staged) + staged, usedforsecurity=False).hexdigest()


def capture_source_snapshot(repository_root: Path) -> SourceSnapshot:
    """Digest the declared source inputs and compare them with the Git index.

    Parameters
    ----------
    repository_root : Path
        The checkout whose declared sources the benchmark runs from.

    Returns
    -------
    SourceSnapshot
        The content digest, file count, index status and change counts.
    """
    digest = hashlib.sha256()
    contents: dict[str, bytes] = {}
    for relative in _declared_paths(repository_root):
        content = (repository_root / relative).read_bytes()
        contents[relative] = content
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(content).digest())
    git_directory = _git_directory(repository_root)
    index = None if git_directory is None else git_directory / "index"
    if index is None or not index.is_file():
        return SourceSnapshot(digest.hexdigest(), len(contents), "unknown", 0, 0, 0)
    try:
        staged = read_index_blobs(index)
    except (ValueError, UnicodeDecodeError):
        return SourceSnapshot(digest.hexdigest(), len(contents), "unknown", 0, 0, 0)
    modified = sum(1 for path, content in contents.items() if path in staged and staged[path] != _blob_sha1(content))
    untracked = sum(1 for path in contents if path not in staged)
    deleted = sum(1 for path in staged if _within_declared(path) and path not in contents)
    status = "clean" if modified == untracked == deleted == 0 else "dirty"
    return SourceSnapshot(digest.hexdigest(), len(contents), status, modified, untracked, deleted)


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 of a regular file, or ``unavailable`` when it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def runtime_identity() -> dict[str, object]:
    """Return the interpreter and native-extension identity of this process.

    The native extension is located by its import spec and hashed as a file; it is
    never imported here.

    Returns
    -------
    dict[str, object]
        Python implementation, version, platform, machine and executable SHA-256,
        and the native extension's module name, presence and file SHA-256.
    """
    spec = importlib.util.find_spec(NATIVE_EXTENSION_MODULE)
    origin = None if spec is None else spec.origin
    native: dict[str, object] = {"module": NATIVE_EXTENSION_MODULE, "status": "absent"}
    if origin is not None:
        native = {"module": NATIVE_EXTENSION_MODULE, "status": "present", "sha256": _file_sha256(Path(origin))}
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "python_executable_sha256": _file_sha256(Path(sys.executable).resolve()),
        "native_extension": native,
    }


__all__ = [
    "DECLARED_SOURCE_FILES",
    "DECLARED_SOURCE_ROOTS",
    "NATIVE_EXTENSION_MODULE",
    "SOURCE_DIGEST_ALGORITHM",
    "SourceSnapshot",
    "capture_source_snapshot",
    "read_index_blobs",
    "runtime_identity",
]
