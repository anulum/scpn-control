#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Git History Exposure Guard.

"""Inspect tracked Git path names against the blocked and allowed expressions.

Current inspection reads the index; history adds names reachable through local
refs, including deleted files. It does not read blobs, find untracked secrets,
inspect reflogs/unreachable objects or certify remote publication. Git filenames
are NUL-delimited and decoded with UTF-8/surrogateescape without stripping.

>>> "README.md" in collect_current_paths(REPO_ROOT)
True
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
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
    """Retain the literal Git name and last ID from its all-ref path log.

    ``first_commit`` is the legacy field name. Its ID is the last result from
    Git's default log ordering, without rename following; this need not be the
    earliest timestamp across branches. An index-only path has no commit ID.
    """

    path: str
    first_commit: str


def _compile(patterns: tuple[str, ...]) -> re.Pattern[str]:
    """Join the unchanged case-sensitive path expressions without normalising names."""
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns))


BLOCKED_RE = _compile(BLOCKED_PATTERNS)
ALLOWED_RE = _compile(ALLOWED_PATTERNS)


def _git(repo: Path, *args: str) -> str:
    """Read actual Git stdout with literal pathspecs; operational errors propagate."""
    completed = subprocess.run(
        ["git", "-C", str(repo), "--literal-pathspecs", *args],
        check=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
    )
    return completed.stdout


def _is_blocked(path: str) -> bool:
    """Apply blocked expressions and explicit allow exceptions to the literal name."""
    return bool(BLOCKED_RE.search(path)) and not bool(ALLOWED_RE.search(path))


def collect_history_paths(repo: Path) -> set[str]:
    """Return exact unique names from all locally reachable ref history, including deletions.

    NUL framing preserves Git quoting-sensitive characters and whitespace.
    Reflogs, unreachable objects, remote-only refs and blob contents are outside
    this observation. Git/IO errors propagate without an empty success fallback.
    """
    output = _git(repo, "log", "--all", "--name-only", "--format=", "-z")
    return {path for path in output.split("\0") if path}


def collect_current_paths(repo: Path) -> set[str]:
    """Return exact index names, including staged additions and files absent from disk.

    This inspects the index rather than HEAD or untracked/ignored working files.
    Reading does not alter Git state; operational failures propagate.
    """
    output = _git(repo, "ls-files", "-z")
    return {path for path in output.split("\0") if path}


def find_first_commit(repo: Path, path: str) -> str:
    """Return the last all-ref log ID for the exact path, or empty for index-only names.

    Pathspec metacharacters are literal. Default Git ordering and no rename
    following preserve the legacy field semantics, not a timestamp guarantee.
    """
    output = _git(repo, "log", "--all", "--format=%H", "--", path)
    return output.splitlines()[-1] if output.splitlines() else ""


def collect_exposures(repo: Path, *, include_history: bool) -> list[Exposure]:
    """Return sorted blocked index/history names for an actual worktree root.

    Calls are sequential Git observations, not a coherent index/ref snapshot.
    The flag uses ordinary Python truth testing. Only names are inspected;
    passing this pattern policy does not certify absence of secret contents.
    A subdirectory refuses rather than silently limiting the index to a prefix.
    """
    top = Path(_git(repo, "rev-parse", "--show-toplevel").removesuffix("\n")).resolve()
    if repo.resolve() != top:
        raise ValueError("repository selection must be the Git worktree root")
    paths = collect_current_paths(repo)
    if include_history:
        paths |= collect_history_paths(repo)

    exposures = [
        Exposure(path=path, first_commit=find_first_commit(repo, path)) for path in sorted(paths) if _is_blocked(path)
    ]
    return exposures


def main(argv: list[str] | None = None) -> int:
    """Inspect explicit/process options with clean0, exposures1 and operational refusal2.

    ``--repo`` selects a Git worktree (default script root); ``--current-only``
    limits names to the index and ``--json`` emits literal names as JSON string
    escapes. Text output escapes nonprintable names onto one line. Git/path/IO
    failures produce fixed stderr without a traceback or partial success JSON.
    Argparse help/usage retain exits0/2. No blob reads, writes, fetch or cleanup.
    """
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

    try:
        repo = Path(args.repo).resolve()
        exposures = collect_exposures(repo, include_history=not args.current_only)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
        print("could not inspect Git paths for history exposure", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([exposure.__dict__ for exposure in exposures], indent=2))
    elif exposures:
        scope = "current tree" if args.current_only else "current tree or history"
        print(f"FAIL: blocked internal or sensitive paths found in {scope}:")
        for exposure in exposures:
            commit = f" first_seen={exposure.first_commit}" if exposure.first_commit else ""
            shown = exposure.path if exposure.path.isprintable() else json.dumps(exposure.path, ensure_ascii=True)
            print(f"  - {shown}{commit}")
    else:
        scope = "current tree" if args.current_only else "current tree and history"
        print(f"OK: no blocked internal or sensitive paths found in {scope}.")

    return 1 if exposures else 0


if __name__ == "__main__":
    raise SystemExit(main())
