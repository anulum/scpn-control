# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark artifact digest contracts.
"""Exercise actual artifact trees, portable ordering and special-file refusals."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from scpn_control.benchmark_artifacts import LEGACY_DIRECTORY_DIGEST_ALGORITHM, path_size, sha256_path


def test_file_digest_and_size_use_actual_bytes(tmp_path: Path) -> None:
    """Regular files retain the standard SHA-256 and payload-byte definitions."""
    file = tmp_path / "report.bin"
    payload = b"\x00\xff" * (1024 * 1024 + 1)
    file.write_bytes(payload)
    assert sha256_path(file) == hashlib.sha256(payload).hexdigest()
    assert path_size(file) == len(payload)
    file.write_bytes(b"")
    assert sha256_path(file) == hashlib.sha256(b"").hexdigest()
    assert path_size(file) == 0


def test_tree_digest_uses_case_sensitive_names_and_empty_directories(tmp_path: Path) -> None:
    """A declared wire vector binds uppercase-before-lowercase order and empty structure."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root, names in [(first, ["a", "Z"]), (second, ["Z", "a"])]:
        for name in names:
            (root / name).write_bytes(name.encode("ascii"))
        (root / "empty").mkdir()
    vector = (
        b"scpn-control.directory-tree.v2\0file\0Z\0"
        + hashlib.sha256(b"Z").digest()
        + b"\0file\0a\0"
        + hashlib.sha256(b"a").digest()
        + b"\0directory\0empty\0\0"
    )
    expected = hashlib.sha256(vector).hexdigest()
    assert sha256_path(first) == sha256_path(second) == expected
    assert path_size(first) == 2
    (second / "empty").rmdir()
    assert sha256_path(first) != sha256_path(second)
    assert sha256_path(first, directory_algorithm=LEGACY_DIRECTORY_DIGEST_ALGORITHM) == sha256_path(
        second, directory_algorithm=LEGACY_DIRECTORY_DIGEST_ALGORITHM
    )


def test_tree_digest_distinguishes_file_directory_and_nested_names(tmp_path: Path) -> None:
    """Tree types and relative names change the digest even when byte totals match."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "node").write_bytes(b"")
    (second / "node").mkdir()
    assert path_size(first) == path_size(second) == 0
    assert sha256_path(first) != sha256_path(second)
    (second / "node/data").write_bytes(b"nested")
    assert path_size(second) == 6
    original = sha256_path(second)
    (second / "node/data").rename(second / "node/renamed")
    assert sha256_path(second) != original


@pytest.mark.parametrize("operation", [sha256_path, path_size])
def test_artifact_inspection_refuses_actual_symlinks(tmp_path: Path, operation: Callable[[Path], str | int]) -> None:
    """Neither root links nor linked tree entries can certify outside bytes."""
    target = tmp_path / "target"
    target.write_bytes(b"original")
    link = tmp_path / "linked"
    link.symlink_to(target)
    root = tmp_path / "tree"
    root.mkdir()
    (root / "linked").symlink_to(target)
    for path in [link, root]:
        with pytest.raises(ValueError, match="cannot contain symlinks"):
            operation(path)
    assert target.read_bytes() == b"original"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Named-pipe filesystem capability requires os.mkfifo")
def test_artifact_inspection_refuses_native_special_entries(tmp_path: Path) -> None:
    """An actual FIFO is refused without a blocking read on platforms that supply it."""
    root = tmp_path / "tree"
    root.mkdir()
    pipe = root / "stream"
    mkfifo: Callable[[Path], None] | None = getattr(os, "mkfifo", None)
    assert mkfifo is not None
    mkfifo(pipe)
    for path in [pipe, root]:
        with pytest.raises(ValueError, match="regular files/directories"):
            sha256_path(path)
        with pytest.raises(ValueError, match="regular files/directories"):
            path_size(path)


def test_artifact_inspection_refuses_missing_paths_and_unknown_protocol(tmp_path: Path) -> None:
    """Missing inputs and unsupported algorithms refuse without creating artifacts."""
    missing = tmp_path / "absent"
    operations: tuple[Callable[[Path], str | int], ...] = (sha256_path, path_size)
    for inspect in operations:
        with pytest.raises(FileNotFoundError):
            inspect(missing)
    with pytest.raises(ValueError, match="unsupported"):
        sha256_path(tmp_path, directory_algorithm="unknown")
    assert list(tmp_path.iterdir()) == []
