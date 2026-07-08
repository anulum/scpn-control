#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JOSS submission metadata guard.

"""Validate the local JOSS paper, bibliography, and docs mirror."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_PATH = ROOT / "paper.md"
DOCS_PATH = ROOT / "docs" / "joss_paper.md"
BIB_PATH = ROOT / "paper.bib"

REQUIRED_PAPER_MARKERS = (
    "title: 'SCPN Control:",
    "bibliography: paper.bib",
    "orcid: 0009-0009-3560-0851",
    "# Summary",
    "# Statement of Need",
    "# Implementation",
    "# Validation",
    "# Acknowledgements",
    "# References",
    "quantitative external-code claims remain blocked until real artefacts are admitted",
    "production runtime claims remain subject to runtime-admission evidence",
)

REQUIRED_DOC_MARKERS = (
    "# JOSS Paper: SCPN Control",
    "canonical JOSS-formatted source is [`paper.md`",
    "with bibliography in [`paper.bib`",
    "Use this paper draft as the publication-grade summary",
    "Keep the manuscript aligned with reproducible code and benchmark evidence.",
    "Do not introduce new benchmark claims in this document without upstream evidence updates.",
)

_BIB_KEY_RE = re.compile(r"@\w+\{\s*([^,\s]+)\s*,")
_CITATION_BLOCK_RE = re.compile(r"\[[^\]]*@[^]]+\]")
_CITATION_KEY_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_:-]*)")
_TITLE_RE = re.compile(r"^title:\s*['\"]?(.+?)['\"]?\s*$", re.MULTILINE)


def _relative(path: Path) -> str:
    """Return a repository-relative path for diagnostics."""

    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_text(path: Path, errors: list[str]) -> str:
    """Read a UTF-8 file or append a missing-file error."""

    if not path.exists():
        errors.append(f"MISSING: {_relative(path)}")
        return ""
    return path.read_text(encoding="utf-8")


def _bib_keys(text: str) -> tuple[set[str], list[str]]:
    """Return bibliography keys and duplicate keys from BibTeX text."""

    keys = _BIB_KEY_RE.findall(text)
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    return set(keys), duplicates


def _citation_keys(text: str) -> set[str]:
    """Return citation keys from Pandoc-style bracketed citations."""

    keys: set[str] = set()
    for block in _CITATION_BLOCK_RE.findall(text):
        keys.update(_CITATION_KEY_RE.findall(block))
    return keys


def _missing_markers(label: str, text: str, markers: tuple[str, ...]) -> list[str]:
    """Return marker diagnostics for required editorial text."""

    normalized = _normalize_prose(text)
    return [f"MISMATCH: {label} missing {marker!r}" for marker in markers if _normalize_prose(marker) not in normalized]


def _paper_title(text: str) -> str | None:
    """Extract the JOSS paper title from YAML front matter."""

    match = _TITLE_RE.search(text)
    return match.group(1) if match else None


def _normalize_prose(text: str) -> str:
    """Collapse Markdown line wrapping into a stable comparison string."""

    return " ".join(text.split())


def check_repository() -> list[str]:
    """Return all JOSS submission guard failures for the current repository."""

    errors: list[str] = []
    paper = _read_text(PAPER_PATH, errors)
    docs = _read_text(DOCS_PATH, errors)
    bibliography = _read_text(BIB_PATH, errors)

    if paper:
        errors.extend(_missing_markers(_relative(PAPER_PATH), paper, REQUIRED_PAPER_MARKERS))
    if docs:
        errors.extend(_missing_markers(_relative(DOCS_PATH), docs, REQUIRED_DOC_MARKERS))
    if paper and docs:
        title = _paper_title(paper)
        if title is None:
            errors.append(f"MISMATCH: {_relative(PAPER_PATH)} missing YAML title")
        elif _normalize_prose(title) not in _normalize_prose(docs):
            errors.append(f"MISMATCH: {_relative(DOCS_PATH)} missing paper title {title!r}")

    if bibliography:
        keys, duplicates = _bib_keys(bibliography)
        for duplicate in duplicates:
            errors.append(f"MISMATCH: {_relative(BIB_PATH)} duplicate bibliography key {duplicate!r}")
        for path, text in ((PAPER_PATH, paper), (DOCS_PATH, docs)):
            if not text:
                continue
            missing = sorted(_citation_keys(text) - keys)
            if missing:
                errors.append(f"MISMATCH: {_relative(path)} cites missing bibliography keys: {', '.join(missing)}")

    return errors


def main() -> int:
    """Run the JOSS submission guard and print a compact status report."""

    errors = check_repository()
    if errors:
        for error in errors:
            print(error)
        print(f"\n{len(errors)} JOSS submission issue(s) detected.")
        return 1

    print("OK: JOSS paper, bibliography, and docs mirror are submission-review aligned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
