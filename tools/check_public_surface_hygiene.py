# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Public-surface claim hygiene guard.
"""Guard public repository surfaces against promotion and operational plans.

The guard scans tracked outward-facing text files and rejects bare promotional
superlatives. Internal planning surfaces may keep aspirational target language,
and bounded negative or candidate terminology remains allowed because it does not
claim achieved superiority.

Tracked public Markdown and JSON are also rejected when they expose unchecked
work lists, prioritisation headings, internal task identifiers, or private
operational paths. Narrow path-and-line allowlists preserve benign tutorial and
contribution-template navigation.
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
    ("crown jewel", re.compile(r"\bcrown jewel\b", re.IGNORECASE)),
    ("unsupported uniqueness", re.compile(r"\bdoes not exist elsewhere\b", re.IGNORECASE)),
    ("stale notebook output path", re.compile(r"\bartefacts/notebook-exec\b")),
)

INTERNAL_IDENTIFIER_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "internal task identifier",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:"
            r"L2F-[A-Za-z0-9]+(?:\([A-Za-z0-9]+\))?|"
            r"CTL-G[A-Za-z0-9-]+|"
            r"R[0-9]+-S[0-9]+|"
            r"U-[0-9]{3}|"
            r"SYS-AUDIT-[A-Za-z0-9-]+|"
            r"WCG-[A-Za-z0-9-]+|"
            r"(?:CONTROL|CTRL)-AUD-[A-Za-z0-9-]+"
            r")(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
)

PUBLIC_PLANNING_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("public operational task", re.compile(r"^\s*[-*+]\s+\[\s\]")),
    (
        "public operational heading",
        re.compile(
            r"^\s*#{1,6}\s+.*(?:roadmap|backlog|next\s+steps?|future\s+work|"
            r"implementation\s+plan|action\s+items?|gap\s+resolution|"
            r"priority|priorities|prioritize|prioritized|prioritizes|prioritizing|"
            r"prioritise|prioritised|prioritises|prioritising|prioritization|prioritisation|"
            r"remaining\s+.*work|current\s+support\s+request|active\s+public-data\s+acquisition|"
            r"campaign\s+budget|what\s+support\s+pays\s+for|drive\s+remediation|funding-to|"
            r"release\s+checklist)",
            re.IGNORECASE,
        ),
    ),
    (
        "public unresolved execution plan",
        re.compile(
            r'^\s*"(?:required_actions|action_items|next_steps|remediation_plan|task_priority)"\s*:', re.IGNORECASE
        ),
    ),
    ("private operational path", re.compile(r"(?:^|[^\w])(?:docs/internal/|\.coordination/)", re.IGNORECASE)),
)

PLANNING_CONTEXT_ALLOWLIST: Final[dict[str, tuple[re.Pattern[str], ...]]] = {
    "docs/tutorials/first_steps.md": (re.compile(r"^\s*##\s+Next Steps\s*$", re.IGNORECASE),),
    "docs/benchmarks.md": (re.compile(r"^\s*##\s+Evidence-first execution checklist\s*$", re.IGNORECASE),),
    "docs/onboarding.md": (
        re.compile(r"^\s*##\s+Roles and expected next evidence\s*$", re.IGNORECASE),
        re.compile(r"^\s*##\s+First hour checklist\s*$", re.IGNORECASE),
    ),
    "docs/pricing.md": (
        re.compile(r"^\s*##\s+Practical funding policy\s*$", re.IGNORECASE),
        re.compile(r"^\s*##\s+How to use this page for planning\s*$", re.IGNORECASE),
    ),
    "docs/production_readiness.md": (
        re.compile(r"^\s*##\s+How to use this boundary in release planning\s*$", re.IGNORECASE),
    ),
    "docs/use_cases.md": (re.compile(r"^\s*##\s+How to apply this page to planning\s*$", re.IGNORECASE),),
}

CHANGELOG_INTERNAL_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "public changelog internal AI profile",
        re.compile(r"\bdirector[-_ ]ai\b.*\bprofile\b", re.IGNORECASE),
    ),
    (
        "public changelog internal workstation detail",
        re.compile(r"\b(?:local\s+)?workstation\b", re.IGNORECASE),
    ),
    (
        "public changelog facility gateway detail",
        re.compile(
            r"\bfacility[- ]gateway|\bfacility gateways\b|\blocal gateway protocols\b",
            re.IGNORECASE,
        ),
    ),
)

PUBLIC_PAYMENT_IDENTIFIER_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "public payment bank account detail",
        re.compile(r"\bIBAN\b|\bBIC\s+[A-Z0-9]{8,11}\b|\bCH\d{2}(?:\s*\d{4}){4,5}\b", re.IGNORECASE),
    ),
    (
        "public payment crypto address",
        re.compile(
            r"\bbc1[ac-hj-np-z02-9]{20,}\b|\bltc1[ac-hj-np-z02-9]{20,}\b|\b0x[a-f0-9]{40}\b",
            re.IGNORECASE,
        ),
    ),
)

PATH_BANNED_PATTERNS: Final[dict[str, tuple[tuple[str, re.Pattern[str]], ...]]] = {
    "README.md": PUBLIC_PAYMENT_IDENTIFIER_PATTERNS,
    "CHANGELOG.md": CHANGELOG_INTERNAL_PATTERNS,
    "docs/changelog.md": CHANGELOG_INTERNAL_PATTERNS,
    "docs/pricing.md": PUBLIC_PAYMENT_IDENTIFIER_PATTERNS,
}

RENDERED_MARKDOWN_HEADER_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^\s*<!--\s*SPDX-License-Identifier:", re.IGNORECASE),
    re.compile(r"^\s*<!--\s*Commercial license available", re.IGNORECASE),
    re.compile(r"^\s*SPDX-License-Identifier:", re.IGNORECASE),
    re.compile(r"^\s*Commercial license available", re.IGNORECASE),
    re.compile(r"^\s*©\s+(?:Concepts|Code)\b", re.IGNORECASE),
    re.compile(r"^\s*\(c\)\s+(?:Concepts|Code)\b", re.IGNORECASE),
    re.compile(r"^\s*ORCID:", re.IGNORECASE),
    re.compile(r"^\s*Contact:", re.IGNORECASE),
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
        candidate = repo / tracked_path
        if _is_scanned_path(tracked_path) and candidate.is_file():
            yield candidate


def _is_allowed_context(line: str) -> bool:
    """Return whether ``line`` uses a bounded allowed context."""
    return any(pattern.search(line) is not None for pattern in ALLOWED_CONTEXTS)


def _is_public_planning_surface(path: str) -> bool:
    """Return whether ``path`` is public prose or serialized public metadata."""
    return Path(path).suffix in {".md", ".json"}


def _is_allowed_planning_context(path: str, line: str, category: str) -> bool:
    """Return whether one planning-like line is a reviewed benign context."""
    if category == "public operational task" and path.startswith(".github/"):
        return True
    return any(pattern.search(line) is not None for pattern in PLANNING_CONTEXT_ALLOWLIST.get(path, ()))


def _rendered_markdown_header_finding(path: str, text: str) -> Finding | None:
    """Return a finding when rendered Markdown opens with legal metadata."""
    if not path.endswith(".md"):
        return None
    for line_number, line in enumerate(text.splitlines()[:8], start=1):
        if line.strip() == "":
            continue
        if any(pattern.search(line) is not None for pattern in RENDERED_MARKDOWN_HEADER_PATTERNS):
            return Finding(
                path=path,
                line=line_number,
                category="rendered markdown legal header",
                detail=line.strip(),
            )
        return None
    return None


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
        Promotion-term and path-specific internal-token findings, excluding
        explicitly bounded contexts.
    """
    findings: list[Finding] = []
    rendered_header_finding = _rendered_markdown_header_finding(path, text)
    if rendered_header_finding is not None:
        findings.append(rendered_header_finding)
    in_fenced_code = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if path.endswith(".md") and re.match(r"^\s*(```|~~~)", line):
            in_fenced_code = not in_fenced_code
            continue
        for category, pattern in INTERNAL_IDENTIFIER_PATTERNS:
            if pattern.search(line) is not None:
                findings.append(Finding(path, line_number, category, line.strip()))
                break
        if _is_allowed_context(line):
            continue
        for category, pattern in BANNED_PATTERNS:
            if pattern.search(line) is not None:
                findings.append(Finding(path, line_number, category, line.strip()))
                break
        for category, pattern in PATH_BANNED_PATTERNS.get(path, ()):
            if pattern.search(line) is not None:
                findings.append(Finding(path, line_number, category, line.strip()))
                break
        if _is_public_planning_surface(path) and not in_fenced_code:
            for category, pattern in PUBLIC_PLANNING_PATTERNS:
                if pattern.search(line) is not None and not _is_allowed_planning_context(path, line, category):
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
        findings.extend(scan_text(path.relative_to(repo).as_posix(), text))
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the command-line public-surface hygiene guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    findings = scan_repository(args.repo.resolve())
    if not findings:
        print("PASS: public surfaces contain no bare promotion terms or operational planning")
        return 0

    print("FAIL: outward-facing claim or operational-planning findings found")
    for finding in findings:
        print(f"  - {finding.path}:{finding.line}: {finding.category}: {finding.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
