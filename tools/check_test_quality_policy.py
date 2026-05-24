#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Test quality policy guard

"""Reject generic bucket tests and coverage-slop intent markers."""

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
    """A concrete test-quality policy violation."""

    path: Path
    line: int
    reason: str

    def format(self) -> str:
        if self.path.is_absolute() and self.path.is_relative_to(REPO_ROOT):
            rel = self.path.relative_to(REPO_ROOT)
        else:
            rel = self.path
        return f"{rel}:{self.line}: {self.reason}"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _test_files(test_root: Path) -> list[Path]:
    return sorted(test_root.rglob("test_*.py"))


def _name_violations(path: Path) -> list[Violation]:
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
    violations: list[Violation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
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
    violations: list[Violation] = []
    for path in _test_files(test_root):
        violations.extend(_name_violations(path))
        violations.extend(_text_violations(path))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", default=str(DEFAULT_TEST_ROOT))
    args = parser.parse_args(argv)

    test_root = _resolve(args.test_root)
    violations = collect_violations(test_root)
    print(f"Test quality policy violations: {len(violations)}")

    if violations:
        for violation in violations:
            print(f"  - {violation.format()}")
        return 1

    print("Test quality policy guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
