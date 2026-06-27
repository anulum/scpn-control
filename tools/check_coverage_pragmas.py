#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Coverage pragma reason checker.

"""Fail closed when source ``pragma: no cover`` comments lack a reason."""

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


@dataclass(frozen=True)
class CoveragePragmaViolation:
    """A source line containing an unreasoned coverage exclusion."""

    path: str
    line: int
    text: str


def _is_reasoned(tail: str) -> bool:
    """Return ``True`` when the text after the pragma contains an actual reason."""
    reason = tail.strip()
    if not reason:
        return False
    reason = reason.lstrip("-:;.,#) ]")
    reason = reason.lstrip("–—")
    return bool(reason.strip())


def iter_python_files(paths: Sequence[Path]) -> list[Path]:
    """Return sorted Python files under the provided paths."""
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(child for child in path.rglob("*.py") if child.is_file())
    return sorted(files)


def find_unreasoned_pragmas(paths: Sequence[Path]) -> list[CoveragePragmaViolation]:
    """Find ``pragma: no cover`` comments without explanatory text."""
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
    paths: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        paths.append(path if path.is_absolute() else REPO_ROOT / path)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Run the coverage-pragma reason gate."""
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
