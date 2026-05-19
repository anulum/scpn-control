#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Git History Exposure Guard
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

"""Detect internal or sensitive paths that are tracked anywhere in Git history."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BLOCKED_PATTERNS = (
    r"(^|/)(\.coordination|docs/internal|04_ARCANE_SAPIENCE|ARCHIVE|BACKUP|MODELS)(/|$)",
    r"(^|/)\.env($|\.)",
    r"(^|/)\.coverage($|\.)",
    r"(^|/)coverage\.xml$",
    r"(^|/).*\.(pem|key|log)$",
    r"(^|/)id_(rsa|ed25519)$",
    r"(^|/)(credentials?|secrets?|tokens?)(/|$|[._-])",
)

ALLOWED_PATTERNS = (
    r"(^|/)discord-bot/\.env\.example$",
    r"(^|/)src/.*/secret_sharing\.py$",
    r"(^|/)benchmarks/models/.*/tokenizer(_config)?\.json$",
)


@dataclass(frozen=True)
class Exposure:
    path: str
    first_commit: str


def _compile(patterns: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns))


BLOCKED_RE = _compile(BLOCKED_PATTERNS)
ALLOWED_RE = _compile(ALLOWED_PATTERNS)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _is_blocked(path: str) -> bool:
    return bool(BLOCKED_RE.search(path)) and not bool(ALLOWED_RE.search(path))


def collect_history_paths(repo: Path) -> set[str]:
    output = _git(repo, "log", "--all", "--name-only", "--pretty=format:")
    return {line.strip() for line in output.splitlines() if line.strip()}


def collect_current_paths(repo: Path) -> set[str]:
    output = _git(repo, "ls-files")
    return {line.strip() for line in output.splitlines() if line.strip()}


def find_first_commit(repo: Path, path: str) -> str:
    output = _git(repo, "log", "--all", "--format=%H", "--", path)
    return output.splitlines()[-1] if output.splitlines() else ""


def collect_exposures(repo: Path, *, include_history: bool) -> list[Exposure]:
    paths = collect_current_paths(repo)
    if include_history:
        paths |= collect_history_paths(repo)

    exposures = [
        Exposure(path=path, first_commit=find_first_commit(repo, path))
        for path in sorted(paths)
        if _is_blocked(path)
    ]
    return exposures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=str(REPO_ROOT),
        help="Repository root to audit.",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Audit only the current tracked tree, not historical paths.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable exposure records.",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    exposures = collect_exposures(repo, include_history=not args.current_only)

    if args.json:
        print(json.dumps([exposure.__dict__ for exposure in exposures], indent=2))
    elif exposures:
        scope = "current tree" if args.current_only else "current tree or history"
        print(f"FAIL: blocked internal or sensitive paths found in {scope}:")
        for exposure in exposures:
            commit = f" first_seen={exposure.first_commit}" if exposure.first_commit else ""
            print(f"  - {exposure.path}{commit}")
    else:
        scope = "current tree" if args.current_only else "current tree and history"
        print(f"OK: no blocked internal or sensitive paths found in {scope}.")

    return 1 if exposures else 0


if __name__ == "__main__":
    raise SystemExit(main())
