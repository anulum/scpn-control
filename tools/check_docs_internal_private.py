#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Permanent private-tree guard for docs/internal/
"""Fail closed if ``docs/internal/`` is tracked, re-included, or published.

This tree is the private operational surface (TODO, audits, handovers). It must
never enter the git index, the MkDocs public site, or the public markdown
inventory. The guard is permanent policy, not a one-shot checklist.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNAL_PREFIX = "docs/internal"
GITIGNORE_MARKERS = ("docs/internal/", "docs/internal/**")
MKDOCS_EXCLUDE_MARKER = "internal/**"


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def check_gitignore_rules(gitignore_text: str) -> list[str]:
    """Return errors if permanent ignore rules for docs/internal are missing or negated."""
    errors: list[str] = []
    lines = [line.strip() for line in gitignore_text.splitlines()]
    has_prefix = any(line in GITIGNORE_MARKERS for line in lines if line and not line.startswith("#"))
    if not has_prefix:
        errors.append(".gitignore must contain 'docs/internal/' or 'docs/internal/**' (permanent private tree)")
    for line in lines:
        if not line or line.startswith("#"):
            continue
        # Negation that re-includes the private tree is forbidden.
        if line.startswith("!") and "docs/internal" in line:
            errors.append(f".gitignore must not re-include private tree via negation: {line!r}")
    return errors


def check_no_tracked_internal_paths(tracked_paths: list[str]) -> list[str]:
    """Return errors for any currently tracked path under docs/internal/."""
    bad = sorted(
        path
        for path in tracked_paths
        if path == INTERNAL_PREFIX or path.startswith(f"{INTERNAL_PREFIX}/")
    )
    if not bad:
        return []
    return [
        "tracked docs/internal paths must be removed from the index (git rm --cached):",
        *[f"  - {path}" for path in bad],
    ]


def check_no_history_internal_paths(history_paths: list[str]) -> list[str]:
    """Return errors if docs/internal ever appeared in git history."""
    bad = sorted(
        path
        for path in history_paths
        if path == INTERNAL_PREFIX or path.startswith(f"{INTERNAL_PREFIX}/")
    )
    if not bad:
        return []
    return [
        "docs/internal paths found in git history (must never reappear; history rewrite needs owner GO):",
        *[f"  - {path}" for path in bad[:50]],
        *(["  - ..."] if len(bad) > 50 else []),
    ]


def check_mkdocs_excludes_internal(mkdocs_text: str) -> list[str]:
    """Return errors if MkDocs would publish docs/internal content."""
    if re.search(r"(?m)^\s*exclude_docs\s*:", mkdocs_text) is None:
        return ["mkdocs.yml must declare exclude_docs (to keep docs/internal off the public site)"]
    # exclude_docs is a YAML block scalar; require the private glob somewhere under it.
    if MKDOCS_EXCLUDE_MARKER not in mkdocs_text and "docs/internal" not in mkdocs_text:
        return [f"mkdocs.yml exclude_docs must include {MKDOCS_EXCLUDE_MARKER!r}"]
    return []


def collect_errors(*, check_history: bool = True) -> list[str]:
    """Run all private-tree checks and return a flat error list."""
    errors: list[str] = []
    gitignore_path = REPO_ROOT / ".gitignore"
    if not gitignore_path.is_file():
        errors.append(".gitignore is missing")
    else:
        errors.extend(check_gitignore_rules(gitignore_path.read_text(encoding="utf-8")))

    tracked = [line.strip() for line in _git("ls-files").splitlines() if line.strip()]
    errors.extend(check_no_tracked_internal_paths(tracked))

    if check_history:
        history = [
            line.strip()
            for line in _git("log", "--all", "--name-only", "--pretty=format:").splitlines()
            if line.strip()
        ]
        errors.extend(check_no_history_internal_paths(history))

    mkdocs_path = REPO_ROOT / "mkdocs.yml"
    if not mkdocs_path.is_file():
        errors.append("mkdocs.yml is missing")
    else:
        errors.extend(check_mkdocs_excludes_internal(mkdocs_path.read_text(encoding="utf-8")))

    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entry: exit 0 when docs/internal stays private."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Skip full-history path scan (faster local loops; CI must not skip).",
    )
    args = parser.parse_args(argv)
    errors = collect_errors(check_history=not args.skip_history)
    if errors:
        print("FAIL: docs/internal private-tree policy violated:")
        for error in errors:
            print(error)
        return 1
    print("PASS: docs/internal is gitignored, untracked, off MkDocs, and clean of history paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
