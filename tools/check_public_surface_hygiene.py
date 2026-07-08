# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public-surface claim hygiene guard.
"""Guard public repository surfaces against self-applied promotion terms.

The guard scans tracked outward-facing text files and rejects bare promotional
superlatives. Internal planning surfaces may keep aspirational target language,
and bounded negative or candidate terminology remains allowed because it does not
claim achieved superiority.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

REPO_ROOT: Final = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES: Final = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

SKIPPED_PREFIXES: Final = (
    ".git/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".venv/",
    ".coordination/",
    "04_ARCANE_SAPIENCE/",
    "docs/internal/",
    "htmlcov/",
    "site/",
)

SKIPPED_PATHS: Final = {
    "tools/check_public_surface_hygiene.py",
    "tests/test_public_surface_hygiene.py",
}

BANNED_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("world-class", re.compile(r"\bworld[- ]class\b", re.IGNORECASE)),
    ("best-in-class", re.compile(r"\bbest[- ]in[- ]class\b", re.IGNORECASE)),
    ("state-of-the-art", re.compile(r"\bstate[- ]of[- ]the[- ]art\b", re.IGNORECASE)),
    ("SOTA", re.compile(r"\bSOTA\b")),
    ("category of one", re.compile(r"\bcategory of one\b", re.IGNORECASE)),
    ("cutting-edge", re.compile(r"\bcutting[- ]edge\b", re.IGNORECASE)),
    ("revolutionary", re.compile(r"\brevolutionary\b", re.IGNORECASE)),
    ("groundbreaking", re.compile(r"\bgroundbreaking\b", re.IGNORECASE)),
    ("unrivalled", re.compile(r"\bunrival(?:led|ed)\b", re.IGNORECASE)),
)

ALLOWED_CONTEXTS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bnot yet SOTA\b", re.IGNORECASE),
    re.compile(r"\bSOTA[- ]candidate\b", re.IGNORECASE),
    re.compile(r"\bSOTA grade\b", re.IGNORECASE),
    re.compile(r"\bbelow the published state of the art\b", re.IGNORECASE),
    re.compile(r"\bstate[- ]of[- ]the[- ]art methods? as (?:a )?baseline\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class Finding:
    """One outward-facing claim hygiene finding.

    Attributes
    ----------
    path
        Repository-relative path that contains the finding.
    line
        One-based line number in ``path``.
    category
        Stable finding category.
    detail
        The matched source line with surrounding whitespace stripped.
    """

    path: str
    line: int
    category: str
    detail: str


def _git_ls_files(repo: Path) -> list[str]:
    """Return tracked paths in ``repo`` from Git's index."""

    completed = subprocess.run(
        ["git", "-C", str(repo), "ls-files"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _is_scanned_path(path: str) -> bool:
    """Return whether ``path`` is an outward-facing text file for this guard."""

    if path in SKIPPED_PATHS or any(path.startswith(prefix) for prefix in SKIPPED_PREFIXES):
        return False
    return Path(path).suffix in TEXT_SUFFIXES or path.startswith(".github/workflows/")


def iter_scanned_files(repo: Path) -> Iterable[Path]:
    """Yield tracked outward-facing files that should be scanned.

    Parameters
    ----------
    repo
        Repository root to inspect.

    Yields
    ------
    Path
        Absolute paths for tracked text files outside private/internal surfaces.
    """

    for tracked_path in _git_ls_files(repo):
        if _is_scanned_path(tracked_path):
            yield repo / tracked_path


def _is_allowed_context(line: str) -> bool:
    """Return whether ``line`` uses a bounded allowed context."""

    return any(pattern.search(line) is not None for pattern in ALLOWED_CONTEXTS)


def scan_text(path: str, text: str) -> list[Finding]:
    """Scan one text payload for outward-facing promotion terms.

    Parameters
    ----------
    path
        Logical path to report in findings.
    text
        File content to scan.

    Returns
    -------
    list[Finding]
        Promotion-term findings, excluding explicitly bounded contexts.
    """

    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _is_allowed_context(line):
            continue
        for category, pattern in BANNED_PATTERNS:
            if pattern.search(line) is not None:
                findings.append(Finding(path, line_number, category, line.strip()))
                break
    return findings


def scan_repository(repo: Path) -> list[Finding]:
    """Scan tracked outward-facing text files in ``repo``.

    Undecodable tracked files are skipped so binary artifacts do not fail the
    claim-hygiene gate for unrelated encoding reasons.
    """

    findings: list[Finding] = []
    for path in iter_scanned_files(repo):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(scan_text(str(path.relative_to(repo)), text))
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the command-line public-surface hygiene guard."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    findings = scan_repository(args.repo.resolve())
    if not findings:
        print("PASS: public surfaces contain no bare self-applied promotion terms")
        return 0

    print("FAIL: outward-facing promotion terms found")
    for finding in findings:
        print(f"  - {finding.path}:{finding.line}: {finding.category}: {finding.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
