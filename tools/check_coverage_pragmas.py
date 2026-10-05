#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Coverage pragma reason checker.

"""Find lexical ``pragma: no cover`` exclusions lacking trailing reason text.

The default CLI scans ``src/scpn_control`` relative to this script's repository.
It does not parse Python comments, judge a reason's scientific validity, or
certify coverage. Files are read as UTF-8 without changing sources or policy.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "src" / "scpn_control"
PRAGMA_PATTERN = re.compile(r"pragma:\s*no cover(?P<tail>.*)$")
REASON_PREFIX_PATTERN = re.compile(r"^[-:;.,#)\]\s–—]*")


@dataclass(frozen=True)
class CoveragePragmaViolation:
    """Immutable diagnostic for one lexical exclusion without trailing text.

    Parameters
    ----------
    path : str
        Resolved POSIX path, relative to the repository when possible and
        absolute otherwise. A diagnostic path does not establish containment.
    line : int
        One-based source line number, counted after ``str.splitlines``.
    text : str
        Entire matching source line with surrounding whitespace removed.

    Attributes
    ----------
    path : str
        Diagnostic filename, without filesystem validation by this class.
    line : int
        Source line number; construction does not validate its range.
    text : str
        Source text, without parsing or justification review.

    Notes
    -----
    The fields are frozen after construction. Native dataclass initialization
    rejects missing or unexpected arguments; supplied values are not coerced.
    """

    path: str
    line: int
    text: str


def _is_reasoned(tail: str) -> bool:
    """Test for text remaining after leading whitespace and separators.

    Parameters
    ----------
    tail : str
        Same-line text following the matched exclusion marker.

    Returns
    -------
    bool
        Whether text remains after whitespace and ``-:;.,#) ]–—`` prefixes.
        Mixed separators alone are insufficient; meaningfulness is not judged.
    """
    return bool(REASON_PREFIX_PATTERN.sub("", tail.strip()))


def iter_python_files(paths: Sequence[Path]) -> list[Path]:
    """Enumerate requested Python files and directory descendants.

    Parameters
    ----------
    paths : Sequence[Path]
        Files ending in case-sensitive ``.py`` or directories. Relative API
        paths resolve through the caller's working directory. Directory search
        uses native ``Path.rglob('*.py')`` and retains file targets only.

    Returns
    -------
    list[Path]
        Sorted paths in their supplied spelling. Overlapping requests retain
        duplicates; an empty sequence or directory produces an empty list.

    Raises
    ------
    FileNotFoundError
        An explicitly requested path is absent or a broken symlink.
    ValueError
        An existing requested path is neither a Python file nor a directory.
    OSError
        Native filesystem operations fail. Enumeration is not an atomic
        snapshot, and traversal follows the active interpreter's Path rules.
    """
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Coverage pragma scan path does not exist: {path}")
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(child for child in path.rglob("*.py") if child.is_file())
        else:
            raise ValueError(f"Coverage pragma scan expects a Python file or directory: {path}")
    return sorted(files)


def find_unreasoned_pragmas(paths: Sequence[Path]) -> list[CoveragePragmaViolation]:
    """Inspect requested UTF-8 Python files for lexical unreasoned markers.

    Parameters
    ----------
    paths : Sequence[Path]
        Paths accepted by :func:`iter_python_files`, relative to the caller's
        working directory when not absolute. Sources are only read.

    Returns
    -------
    list[CoveragePragmaViolation]
        Diagnostics ordered by sorted file path and one-based line number.
        Repeated files produce repeated diagnostics. Marker matching is
        case-sensitive and searches whole lines, including strings/docstrings.
        A separator-only suffix fails; remaining text is not reviewed.

    Raises
    ------
    FileNotFoundError
        A requested path is absent, or a file disappears before reading.
    ValueError
        A requested existing path is not a Python file or directory.
    UnicodeDecodeError
        A selected file cannot be decoded as UTF-8.
    OSError
        Native resolution, enumeration or reading fails.

    Notes
    -----
    No reason ownership, coverage measurement, compiler variant, readiness,
    containment or concurrent-edit guarantee is provided. Empty results apply
    only to the files actually enumerated.
    """
    violations: list[CoveragePragmaViolation] = []
    for path in iter_python_files(paths):
        resolved = path.resolve()
        try:
            rel = resolved.relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            rel = resolved.as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = PRAGMA_PATTERN.search(line)
            if match is not None and not _is_reasoned(match.group("tail")):
                violations.append(CoveragePragmaViolation(path=rel, line=lineno, text=line.strip()))
    return violations


def _resolve_paths(raw_paths: Sequence[str]) -> list[Path]:
    """Resolve CLI spellings against this script's repository root.

    Parameters
    ----------
    raw_paths : Sequence[str]
        CLI path strings, without shell, environment or tilde expansion.

    Returns
    -------
    list[Path]
        Absolute spellings unchanged and relative spellings prefixed by
        ``REPO_ROOT``. Filesystem existence is checked during enumeration.
    """
    paths: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        paths.append(path if path.is_absolute() else REPO_ROOT / path)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Run the local lexical exclusion-reason CLI.

    Parameters
    ----------
    argv : list[str] or None, optional
        Explicit argparse tokens, or process arguments when None. Positional
        paths resolve against the script repository; omission selects its
        ``src/scpn_control`` directory. ``--json`` changes output format only.

    Returns
    -------
    int
        Zero for no unreasoned markers in enumerated files; one for findings.
        JSON contains ``unreasoned`` records with path, line and text fields.
        Human output prints a summary and each diagnostic to standard output.

    Raises
    ------
    SystemExit
        Argparse help exits zero; invalid options exit two.
    FileNotFoundError
        A requested path or the default source directory is absent.
    ValueError
        A requested existing path is not a Python file or directory.
    UnicodeDecodeError
        A selected source is not UTF-8.
    OSError
        Native filesystem operations fail. These failures propagate before
        a success summary or JSON is printed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(DEFAULT_SOURCE_ROOT)],
        help="Python files or directories to scan; defaults to src/scpn_control",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    violations = find_unreasoned_pragmas(_resolve_paths(args.paths))
    if args.json:
        print(json.dumps({"unreasoned": [asdict(item) for item in violations]}, indent=2))
        return 1 if violations else 0

    if not violations:
        print("Coverage pragma reason guard passed: 0 unreasoned pragmas.")
        return 0

    print(f"Coverage pragma reason guard FAILED: {len(violations)} unreasoned pragma(s).")
    for item in violations:
        print(f"  - {item.path}:{item.line}: {item.text}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
