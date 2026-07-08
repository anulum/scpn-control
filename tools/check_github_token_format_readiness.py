# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GitHub installation-token readiness guard.

"""Detect brittle GitHub installation-token format assumptions.

GitHub installation tokens are opaque credentials. Historical integrations often
treated them as fixed-width ``ghs_`` strings, stored them in short columns, or
used the installation-token endpoint without the stateless-token override
header. This guard scans tracked text files and fails on those patterns so CI
cannot reintroduce format-coupled token handling.
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
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SKIPPED_PREFIXES: Final = (
    ".git/",
    ".venv/",
    ".coordination/",
    "04_ARCANE_SAPIENCE/",
    "docs/internal/",
    "tests/",
)

SKIPPED_PATHS: Final = {
    "tools/check_github_token_format_readiness.py",
}

STATELESS_HEADER: Final = "X-GitHub-Stateless-S2S-Token"

TOKEN_PATTERN_RE: Final = re.compile(
    r"ghs_[^\"'\n]*(?:\{36\}|\{40\}|\[A-Za-z0-9\]\{36|\[A-Za-z0-9_\]\{36)",
    re.IGNORECASE,
)
EXACT_LENGTH_RE: Final = re.compile(
    r"(?:len\([^)\n]*(?:github|ghs|installation|token)[^)\n]*\)\s*(?:==|!=|<=|>=)\s*(?:40|255)"
    r"|(?:github|ghs|installation|token)[^\n]{0,48}(?:length|len)[^\n]{0,16}(?:40|255))",
    re.IGNORECASE,
)
SMALL_STORAGE_RE: Final = re.compile(
    r"(?:github|ghs|installation|token)[^\n]{0,80}"
    r"(?:varchar|character varying|String|CharField|max_length)\s*[\(=]\s*(?:40|64|128|255)\b",
    re.IGNORECASE,
)
INSTALLATION_TOKEN_ENDPOINT_RE: Final = re.compile(
    r"/app/installations/[^\"'\s]+/access_tokens|app/installations/.+access_tokens",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    """One token-format readiness finding.

    Attributes
    ----------
    path
        Repository-relative path where the finding was detected.
    line
        One-based line number in ``path``.
    category
        Stable machine-readable finding category.
    detail
        Matched source fragment that triggered the finding.
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
    """Return whether ``path`` is a tracked text file covered by the guard."""

    if path in SKIPPED_PATHS or any(path.startswith(prefix) for prefix in SKIPPED_PREFIXES):
        return False
    return Path(path).suffix in TEXT_SUFFIXES or path.startswith(".github/workflows/")


def iter_scanned_files(repo: Path) -> Iterable[Path]:
    """Yield tracked repository files that should be scanned.

    Parameters
    ----------
    repo
        Repository root to inspect.

    Yields
    ------
    Path
        Absolute paths for tracked files with text-like suffixes or workflow
        paths, excluding private and test fixtures.
    """

    for tracked_path in _git_ls_files(repo):
        if _is_scanned_path(tracked_path):
            yield repo / tracked_path


def _line_number(text: str, offset: int) -> int:
    """Return the one-based line number for ``offset`` in ``text``."""

    return text.count("\n", 0, offset) + 1


def _append_regex_findings(
    findings: list[Finding], path: str, text: str, regex: re.Pattern[str], category: str
) -> None:
    """Append one finding per match of ``regex`` in ``text``."""

    for match in regex.finditer(text):
        findings.append(Finding(path, _line_number(text, match.start()), category, match.group(0).strip()))


def scan_text(path: str, text: str) -> list[Finding]:
    """Scan one text payload for brittle installation-token handling.

    Parameters
    ----------
    path
        Logical path to report in findings.
    text
        File content to scan.

    Returns
    -------
    list[Finding]
        Findings for exact-width token regexes, fixed token-length checks,
        undersized token storage, and missing stateless-token override headers.
    """

    findings: list[Finding] = []
    _append_regex_findings(findings, path, text, TOKEN_PATTERN_RE, "brittle-ghs-regex")
    _append_regex_findings(findings, path, text, EXACT_LENGTH_RE, "fixed-token-length")
    _append_regex_findings(findings, path, text, SMALL_STORAGE_RE, "small-token-storage")

    if INSTALLATION_TOKEN_ENDPOINT_RE.search(text) and STATELESS_HEADER not in text:
        match = INSTALLATION_TOKEN_ENDPOINT_RE.search(text)
        assert match is not None
        findings.append(
            Finding(
                path,
                _line_number(text, match.start()),
                "missing-stateless-token-override-header",
                match.group(0).strip(),
            )
        )

    return findings


def scan_repository(repo: Path) -> list[Finding]:
    """Scan tracked text files in ``repo`` for token-format assumptions.

    Files that cannot be decoded as UTF-8 are skipped so binary artifacts do not
    make the guard fail for reasons unrelated to token-format readiness.
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
    """Run the command-line token-format readiness guard."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    findings = scan_repository(args.repo.resolve())
    if not findings:
        print("PASS: no brittle GitHub installation-token format assumptions found")
        return 0

    print("FAIL: GitHub installation-token format readiness issues found")
    for finding in findings:
        print(f"  - {finding.path}:{finding.line}: {finding.category}: {finding.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
