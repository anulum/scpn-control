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

The repository API enumerates Git's index but reads current worktree bytes,
including unstaged edits. It does not inspect untracked files, staged blobs or
history. Selected files with invalid UTF-8 and tracked missing/non-file paths
are skipped. A pass only covers the resulting inspected text, not publication
readiness, scientific validity or a coherent concurrent repository snapshot.
"""

from __future__ import annotations

import argparse
import errno
import os
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

RENDERED_MARKDOWN_PRODUCER_PATTERN: Final = re.compile(
    r"[\"']<!-- SPDX-License-Identifier:\s*AGPL-3\.0-or-later -->[\"']"
)

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
    (
        "unidentified reactor phase-control claim",
        re.compile(
            r"\b(?:authority over (?:the )?plasma modes|"
            r"entry point a real control loop would call|"
            r"maps directly to SNN or PID output amplitude|"
            r"complete real-time monitoring loop)\b",
            re.IGNORECASE,
        ),
    ),
)

INTERNAL_IDENTIFIER_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "internal queue or workstream identifier",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:"
            r"(?:BL|QWC)[-_][A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*"
            r")(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "internal queue or workstream identifier",
        re.compile(
            r"(?<![A-Za-z0-9_])WS[-_][A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*"
            r"(?![A-Za-z0-9_])",
        ),
    ),
    (
        "internal LIF lane identifier",
        re.compile(r"(?<![A-Za-z0-9_])LIF-FF-[0-9]+(?![A-Za-z0-9_])"),
    ),
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
    (
        "internal CONTROL lane identifier",
        re.compile(r"(?<![A-Za-z0-9_])CONTROL-(?:[A-Z0-9]+-)+[0-9]{3}(?![A-Za-z0-9_])"),
    ),
    (
        "internal coverage campaign identifier",
        re.compile(r"(?<![A-Za-z0-9])COV-?1(?![A-Za-z0-9])", re.IGNORECASE),
    ),
    (
        "internal safety campaign identifier",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:"
            r"SS-[0-9]+(?:\s*[ab])?(?:/F[0-9]+)?|"
            r"CF-5|SP-4|LOCK-4"
            r")(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "internal pulsed-control campaign identifier",
        re.compile(r"(?<![A-Za-z0-9_])CON-C(?:\.[0-9]+)?(?![A-Za-z0-9_])", re.IGNORECASE),
    ),
    (
        "internal parity campaign identifier",
        re.compile(r"(?<![A-Za-z0-9_])PARITY-1(?![A-Za-z0-9_])", re.IGNORECASE),
    ),
    (
        "internal controller workstream identifier",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:GAI[-_]?0+2(?:_TORAX_HYBRID)?|"
            r"GDEP[-_]?0+1(?:_DIGITAL_TWIN)?|GNEU[-_]?0+3(?:_FUELING)?)(?![A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "internal implementation-review identifier",
        re.compile(r"(?<![A-Za-z0-9_])CONTROL[-_]F841[-_]REVIEW(?![A-Za-z0-9_])", re.IGNORECASE),
    ),
    (
        "internal integration-stage identifier",
        re.compile(r"(?<![A-Za-z0-9_])INT-[0-9]+(?![A-Za-z0-9_])", re.IGNORECASE),
    ),
    (
        "internal polyglot work-package identifier",
        re.compile(r"(?<![A-Za-z0-9_])WP[-_]PY[0-9]+(?![A-Za-z0-9_])", re.IGNORECASE),
    ),
    (
        "internal resolved-finding identifier",
        re.compile(r"(?<![A-Za-z0-9])O(?:-|_)?0{2}2(?![A-Za-z0-9])", re.IGNORECASE),
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
        One-based splitlines position, or zero for a path-name finding.
    category
        Stable finding category.
    detail
        Source line with surrounding whitespace stripped, or the original path
        for a path-name finding. Payload text is retained, not redacted.

    Notes
    -----
    Frozen fields hold the supplied values without constructor validation.
    Categories may repeat on a line when independent pattern families match.
    """

    path: str
    line: int
    category: str
    detail: str


class PublicSurfaceScanError(ValueError):
    """Authored refusal from Git enumeration, path inspection or a file read.

    Notes
    -----
    Guard producers supply fixed messages without subprocess or OS exception
    text. Callers can catch this type separately from policy findings; a failed
    repository scan has no successful partial-result return.
    """


def _git_ls_files(repo: Path) -> list[str]:
    """Read NUL-delimited index paths without Git quoting or newline ambiguity.

    Filesystem decoding retains native path spellings. Git startup/exit failures
    become a fixed authored refusal; subprocess diagnostics are not exposed.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublicSurfaceScanError("could not enumerate tracked public files") from exc
    return [os.fsdecode(path) for path in completed.stdout.split(b"\0") if path]


def _is_scanned_path(path: str) -> bool:
    """Return whether ``path`` is an outward-facing text file for this guard."""
    if path in SKIPPED_PATHS or any(path.startswith(prefix) for prefix in SKIPPED_PREFIXES):
        return False
    return Path(path).suffix in TEXT_SUFFIXES or path.startswith(".github/workflows/")


def iter_scanned_files(repo: Path) -> Iterable[Path]:
    """Yield selected regular worktree paths enumerated from Git's index.

    Parameters
    ----------
    repo
        Directory passed to Git -C and joined with its relative index paths.
        Relative spelling uses the caller's working directory; the API does not
        resolve the root. A Git subdirectory scans its own index-relative scope.

    Yields
    ------
    Path
        Paths joined to repo, absolute only when repo is absolute. Selection uses
        case-sensitive suffixes or the workflow prefix, then private/cache and
        guard-fixture exclusions. Missing/non-file tracked paths are skipped.

    Raises
    ------
    PublicSurfaceScanError
        Git cannot enumerate the index or inspect a selected worktree path.

    Notes
    -----
    Enumeration is lazy until iteration and follows Git's order. Current regular
    files and symlinks to regular files are yielded, without containment checks.
    Untracked files, staged blob bytes and history are not visited. The API does
    not read payloads or guarantee a coherent concurrent index/worktree snapshot.
    """
    for tracked_path in _git_ls_files(repo):
        candidate = repo / tracked_path
        if not _is_scanned_path(tracked_path):
            continue
        try:
            is_file = candidate.is_file()
        except OSError as exc:
            raise PublicSurfaceScanError("could not inspect tracked public path") from exc
        if is_file:
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


def _rendered_markdown_producer_finding(path: str, text: str) -> Finding | None:
    """Reject validation code that emits the forbidden Markdown preamble."""
    if not path.startswith("validation/") or not path.endswith(".py"):
        return None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if RENDERED_MARKDOWN_PRODUCER_PATTERN.search(line) is not None:
            return Finding(
                path=path,
                line=line_number,
                category="rendered Markdown legal-header producer",
                detail=line.strip(),
            )
    return None


def scan_text(path: str, text: str) -> list[Finding]:
    """Inspect one caller-supplied payload with path-dependent pattern families.

    Parameters
    ----------
    path
        Logical POSIX spelling used for categories and findings, not a file to
        open. Case-sensitive suffix/prefix checks require the intended spelling;
        the function does not apply repository path-selection exclusions.
    text
        Unicode content scanned without parsing Markdown, JSON or source code.

    Returns
    -------
    list[Finding]
        Fresh ordered findings: first path identity, then first-eight-line
        Markdown preamble and validation producer checks, then per-line rules.
        Each family takes its first match; independent families can both emit.

    Notes
    -----
    A Markdown fence-looking line toggles fence state without matching its
    delimiter or length. Only planning checks are suppressed inside fences;
    identifier/promotion checks still apply. Bounded context matches suppress
    promotion, path-specific and planning rules for that line, after identifier
    inspection. Narrow reviewed planning exceptions depend on the exact path.
    No filesystem access, mutation, cache, rendering or semantic claim proof is
    provided. Findings retain source text for the local operator's report.

    Examples
    --------
    Inspect the actual maintained development guide:

    >>> guide = REPO_ROOT / "docs/development.md"
    >>> scan_text("docs/development.md", guide.read_text(encoding="utf-8"))
    []
    """
    findings: list[Finding] = []
    for category, pattern in INTERNAL_IDENTIFIER_PATTERNS:
        if pattern.search(path) is not None:
            findings.append(Finding(path, 0, category, path))
            break
    rendered_header_finding = _rendered_markdown_header_finding(path, text)
    if rendered_header_finding is not None:
        findings.append(rendered_header_finding)
    rendered_producer_finding = _rendered_markdown_producer_finding(path, text)
    if rendered_producer_finding is not None:
        findings.append(rendered_producer_finding)
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
    """Inspect selected indexed paths using their current UTF-8 worktree text.

    Parameters
    ----------
    repo
        Git directory/root spelling used by iter_scanned_files. Relative paths
        remain caller-relative; symlinks follow without containment enforcement.

    Returns
    -------
    list[Finding]
        Fresh findings in index/path and scan_text order. Paths are relative to
        the supplied root. Invalid UTF-8 selected files are skipped as binary;
        empty therefore means no finding in inspected text, not all-file proof.

    Raises
    ------
    PublicSurfaceScanError
        Git enumeration, selected-path inspection or worktree reading fails.

    Notes
    -----
    Worktree bytes use universal newline handling. Untracked/staged/history
    content is not read. A read failure aborts without returning earlier partial
    findings. No repository writes, subprocess timeout, atomic snapshot,
    deployment check or scientific/security certification is provided.
    """
    findings: list[Finding] = []
    for path in iter_scanned_files(repo):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise PublicSurfaceScanError("could not read tracked public text") from exc
        findings.extend(scan_text(path.relative_to(repo).as_posix(), text))
    return findings


def _refuse_link_loop(path: Path) -> None:
    """Raise for a path whose symbolic links never resolve.

    ``Path.resolve`` raised ``RuntimeError`` for a link loop before Python 3.13
    and returns the path unresolved since. The loop is asked for explicitly, so
    the refusal is the same on every supported interpreter.
    """
    try:
        path.stat()
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError(f"Symlink loop from {str(path)!r}") from exc


def main(argv: list[str] | None = None) -> int:
    """Inspect a resolved Git/worktree scope through the stdlib CLI.

    Parameters
    ----------
    argv
        Arguments without the executable name, or None for process arguments.
        Default root comes from this script; --repo is caller-relative and is
        resolved before scanning.

    Returns
    -------
    int
        0 for no inspected-text findings, 1 for policy findings, or 2 for root,
        Git or read refusal. Reports and fixed authored refusals use stdout.
        Finding detail contains the inspected source text and path spelling.

    Raises
    ------
    SystemExit
        Argparse exits 0 for help and 2 for invalid arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    try:
        _refuse_link_loop(args.repo)
        repo = args.repo.resolve()
    except (OSError, RuntimeError, ValueError):
        print("FAIL: could not resolve public surface repository")
        return 2
    try:
        findings = scan_repository(repo)
    except PublicSurfaceScanError as exc:
        print(f"FAIL: {exc}")
        return 2
    if not findings:
        print(
            "PASS: inspected tracked UTF-8 public text contains no forbidden claims, planning, or internal identifiers"
        )
        return 0

    print("FAIL: outward-facing claim or operational-planning findings found")
    for finding in findings:
        print(f"  - {finding.path}:{finding.line}: {finding.category}: {finding.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
