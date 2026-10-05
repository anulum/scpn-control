#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test quality policy guard

"""Reject prohibited test filenames and intent markers in a nonempty test tree.

The scan reads UTF-8 ``test_*.py`` files recursively without executing them.
It enforces explicit filename and wording rules; it does not establish that
assertions exercise production behaviour or that measured coverage is meaningful.
Missing, empty or unreadable scan scopes cannot produce a passing CLI verdict.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_ROOT = REPO_ROOT / "tests"

FORBIDDEN_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)test_cov[^/]*\.py$", re.IGNORECASE),
    re.compile(r"(^|/)test_coverage[^/]*\.py$", re.IGNORECASE),
    re.compile(r"(coverage_closure|final_gaps?|remaining_gaps?)\.py$", re.IGNORECASE),
    re.compile(r"(last_mile|small_gaps|transport_gaps|scpn_gaps)\.py$", re.IGNORECASE),
    re.compile(r"(^|/)test_(?:.*_)?(batch|round|final|remaining)\.py$", re.IGNORECASE),
    re.compile(r"(^|/).*_(push|bucket|misc|new_modules?)\.py$", re.IGNORECASE),
)

FORBIDDEN_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcoverage\s+gaps?\b", re.IGNORECASE),
    re.compile(r"\bcoverage\s+closure\b", re.IGNORECASE),
    re.compile(r"\blast[- ]mile\s+coverage\b", re.IGNORECASE),
    re.compile(r"\bclose\s+the\s+last\b", re.IGNORECASE),
    re.compile(r"\buncovered\s+lines?\b", re.IGNORECASE),
    re.compile(r"\btarget\s+lines?\b", re.IGNORECASE),
    re.compile(r"\b100%\+\s+target\b", re.IGNORECASE),
    re.compile(r"\bdeep\s+coverage\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class Violation:
    """One filename or wording finding with its one-based source line.

    Parameters
    ----------
    path
        Scanned test source. Filename violations refer to line one.
    line
        One-based source line containing the finding.
    reason
        Rule description containing the matched regular expression.
    """

    path: Path
    line: int
    reason: str

    def format(self) -> str:
        """Return ``path:line:reason``, using repository-relative paths where possible."""
        if self.path.is_absolute() and self.path.is_relative_to(REPO_ROOT):
            rel = self.path.relative_to(REPO_ROOT)
        else:
            rel = self.path
        return f"{rel}:{self.line}: {self.reason}"


def _resolve(path_value: str) -> Path:
    """Interpret relative CLI roots from the repository, preserving absolute roots."""
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _test_files(test_root: Path) -> list[Path]:
    """Require a directory containing at least one recursively discovered test file."""
    if not test_root.is_dir():
        raise ValueError(f"test scope must be an existing directory: {test_root}")
    files = sorted(test_root.rglob("test_*.py"))
    if not files:
        raise ValueError(f"test scope contains no test_*.py files: {test_root}")
    return files


def _name_violations(path: Path) -> list[Violation]:
    """Return at most one filename finding, using the first matching name rule."""
    if path.is_absolute() and path.is_relative_to(REPO_ROOT):
        rel = path.relative_to(REPO_ROOT).as_posix()
    else:
        rel = path.as_posix()
    violations: list[Violation] = []
    for pattern in FORBIDDEN_NAME_PATTERNS:
        if pattern.search(rel):
            violations.append(Violation(path, 1, f"forbidden generic bucket test filename: {pattern.pattern}"))
            break
    return violations


def _text_violations(path: Path) -> list[Violation]:
    """Read strict UTF-8 and report every matching wording rule on each source line."""
    violations: list[Violation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern.search(line):
                violations.append(
                    Violation(
                        path,
                        line_number,
                        f"forbidden coverage-slop wording: {pattern.pattern}",
                    )
                )
    return violations


def collect_violations(test_root: Path) -> list[Violation]:
    """Read a nonempty test tree and return ordered lexical policy findings.

    Parameters
    ----------
    test_root
        Directory scanned recursively for ``test_*.py``. Relative paths are
        interpreted by :class:`pathlib.Path` from the process working directory.

    Returns
    -------
    list of Violation
        Findings ordered by sorted file path, with filename findings first and
        wording findings in source-line and configured-rule order. An empty
        list means the inspected files match no prohibited lexical rule.

    Raises
    ------
    ValueError
        If the root is not a directory or contains no matching test files.
    OSError
        If an inspected file cannot be read.
    UnicodeDecodeError
        If a test file does not contain valid UTF-8.
    """
    violations: list[Violation] = []
    for path in _test_files(test_root):
        violations.extend(_name_violations(path))
        violations.extend(_text_violations(path))
    return violations


def main(argv: list[str] | None = None) -> int:
    """Print the scan verdict and return its process exit status.

    Parameters
    ----------
    argv
        CLI arguments; ``None`` reads the process arguments. ``--test-root``
        accepts an absolute path or a repository-relative path and defaults to
        the repository's ``tests`` directory.

    Returns
    -------
    int
        Zero for a completed scan with no findings, one for findings or a
        refused scan. Argument parsing retains argparse's exit status two.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", default=str(DEFAULT_TEST_ROOT))
    args = parser.parse_args(argv)

    test_root = _resolve(args.test_root)
    try:
        violations = collect_violations(test_root)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Test quality policy guard failed: {exc}")
        return 1
    print(f"Test quality policy violations: {len(violations)}")

    if violations:
        for violation in violations:
            print(f"  - {violation.format()}")
        return 1

    print("Test quality policy guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
