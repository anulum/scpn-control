#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Export Zenodo Dataset.

"""Create a local source/data ZIP candidate from the current checkout.

The fixed include patterns below select regular, non-symlink files. Git ignore
rules, excluded directories and docs/internal apply even to tracked files.
Nonignored untracked public files are included, so an uncommitted module split
does not produce an archive with missing imports. This is a working-directory
export, not an export of a committed tree or a complete release distribution.

The prefix uses the version declared in .zenodo.json. No metadata, licence,
scientific, reproducibility or publication admission is inferred from that
declaration. The command uploads nothing; any later upload is a separate action.
ZIP timestamps and permissions come from the selected files. Reads are
sequential, without a lock or a consistent snapshot of concurrent changes.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import zipfile
from collections.abc import Sequence

ROOT = pathlib.Path(__file__).resolve().parent.parent

INCLUDE_PATTERNS = [
    "src/**/*.py",
    "tests/**/*.py",
    "tools/**/*.py",
    "examples/**/*.ipynb",
    "examples/**/*.py",
    "validation/**/*.json",
    "validation/**/*.csv",
    "validation/**/*.py",
    "docs/**/*.md",
    "scpn-control-rs/Cargo.toml",
    "scpn-control-rs/Cargo.lock",
    "scpn-control-rs/crates/**/*.rs",
    "scpn-control-rs/crates/**/*.toml",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "LICENSE-MIT",
    "LICENSE-APACHE",
    "CITATION.cff",
    "CHANGELOG.md",
    ".zenodo.json",
    "mkdocs.yml",
]

EXCLUDE_DIRS = {".venv", "__pycache__", ".mypy_cache", ".ruff_cache", "target", "node_modules", "site", "dist"}


def _eligible(path: pathlib.Path, root: pathlib.Path) -> bool:
    """Select regular in-root files without excluded or symlink path components."""
    relative = path.relative_to(root)
    if relative.parts[:2] == ("docs", "internal") or any(part in EXCLUDE_DIRS for part in relative.parts):
        return False
    if not path.is_file():
        return False
    return not any(root.joinpath(*relative.parts[:i]).is_symlink() for i in range(1, len(relative.parts) + 1))


def _ignored(files: list[pathlib.Path], root: pathlib.Path) -> set[str]:
    """Read actual Git ignore rules without using the index or inherited Git roots."""
    if not files:
        return set()
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        env.pop(key, None)
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), "check-ignore", "--no-index", "--stdin", "-z"],
        input=b"".join(os.fsencode(path.relative_to(root).as_posix()) + b"\0" for path in files),
        capture_output=True,
        env=env,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("Git ignore selection is unavailable")
    return {os.fsdecode(name) for name in result.stdout.split(b"\0") if name}


def collect_files(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    """Return sorted unique public files selected from the live Git checkout.

    Parameters
    ----------
    root : pathlib.Path
        Checkout directory, resolved before applying INCLUDE_PATTERNS. The
        default is the checkout containing this tool, independent of caller cwd.

    Returns
    -------
    list[pathlib.Path]
        Absolute regular files. Ignored files are omitted even when tracked;
        eligible nonignored untracked files remain included. An empty selection
        returns an empty list without invoking Git.

    Raises
    ------
    OSError
        Filesystem or Git process access failed.
    RuntimeError
        Git could not evaluate the nonempty selection's ignore rules.
    """
    root = root.resolve()
    files: set[pathlib.Path] = set()
    for pattern in INCLUDE_PATTERNS:
        files.update(root.glob(pattern))
    # Order by the spelled name: path objects compare without case on Windows,
    # which would order the archive members differently there.
    eligible = sorted((path for path in files if _eligible(path, root)), key=pathlib.Path.as_posix)
    ignored = _ignored(eligible, root)
    return [path for path in eligible if path.relative_to(root).as_posix() not in ignored]


def _version(root: pathlib.Path) -> str:
    """Read a regular metadata file's nonempty ASCII version as one ZIP component."""
    metadata = root / ".zenodo.json"
    if metadata.is_symlink():
        raise ValueError("Metadata must be a regular checkout file")
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    version = meta.get("version") if isinstance(meta, dict) else None
    if not isinstance(version, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+\-]*", version) is None:
        raise ValueError("Metadata version must form one safe archive component")
    return version


def create_archive(output: pathlib.Path | None = None, *, root: pathlib.Path = ROOT) -> tuple[pathlib.Path, int]:
    """Create a new ZIP candidate containing the current public file selection.

    Parameters
    ----------
    output : pathlib.Path or None
        Caller-relative or absolute destination. None uses the caller's cwd and
        scpn-control-{declared_version}.zip. Parent directories must exist.
    root : pathlib.Path
        Live checkout supplying metadata and collect_files selection.

    Returns
    -------
    tuple[pathlib.Path, int]
        Absolute created path and number of members. Names use the prefix
        scpn-control-{declared_version}/ and each checkout-relative POSIX path.

    Raises
    ------
    ValueError
        Metadata/version is invalid, the selection is empty, or the destination
        resolves into root/.git. Metadata file symlinks are refused.
    FileExistsError
        Any destination already exists, including input symlinks/hard links.
        Existing archives and selected inputs are preserved by exclusive create.
    OSError
        Input access, destination creation or writing failed. A write failure
        after creation can leave a partial archive; it is not silently removed.
    RuntimeError
        Git ignore selection is unavailable or path resolution failed.

    Notes
    -----
    Compression is ZIP_DEFLATED with the library's default compression level.
    The operation changes no inputs or Git refs and validates no scientific
    reports or release metadata. Source changes during export are not locked.
    """
    root = root.resolve()
    version = _version(root)
    destination = (output if output is not None else pathlib.Path(f"scpn-control-{version}.zip")).absolute()
    if destination.resolve().is_relative_to((root / ".git").resolve()):
        raise ValueError("Archive destination must not be Git metadata")
    files = collect_files(root)
    if not files:
        raise ValueError("Archive selection is empty")
    prefix = f"scpn-control-{version}"
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"{prefix}/{path.relative_to(root).as_posix()}")
    return destination, len(files)


def main(argv: Sequence[str] | None = None) -> int:
    """Create a local candidate; return 0 on success or caller-safe refusal 2.

    Help exits 0 and argparse usage errors exit 2 before metadata is read.
    Existing output files are never replaced. No upload or release is performed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=pathlib.Path, help="New ZIP path; default uses the declared version in caller cwd"
    )
    args = parser.parse_args(argv)
    try:
        destination, count = create_archive(args.output)
    except (OSError, ValueError, RuntimeError):
        print("Dataset export refused: metadata, source selection or archive output is invalid.", file=sys.stderr)
        return 2
    print(f"Created {destination} ({count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
