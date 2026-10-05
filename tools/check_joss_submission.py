#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JOSS submission metadata guard.

"""Inspect local JOSS editorial markers, title placement, and citation keys.

The script-relative repository owns three fixed UTF-8 inputs. Missing or blank
files are findings. The title is a lexical ``title:`` line in the initial
``---``-delimited front matter; required prose is compared after whitespace
normalisation. Bibliography keys and bracketed Pandoc citation keys are regular
expression matches, not a complete YAML, BibTeX, or Markdown parser.

These read-only checks do not render a PDF, resolve hyperlinks, authenticate
references, admit scientific results, submit a manuscript, or establish JOSS
review acceptance. Native filesystem and UTF-8 decoding failures propagate.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_PATH = ROOT / "papers" / "submissions" / "001_neuro_symbolic_tokamak_control_software"
PAPER_PATH = SUBMISSION_PATH / "manuscript.md"
DOCS_PATH = ROOT / "docs" / "joss_paper.md"
BIB_PATH = SUBMISSION_PATH / "references.bib"

REQUIRED_PAPER_MARKERS = (
    "title: 'SCPN Control:",
    "bibliography: references.bib",
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
    "canonical manuscript package",
    "papers/submissions/001_neuro_symbolic_tokamak_control_software/",
    "manuscript.md",
    "references.bib",
    "manuscript.pdf",
    "review draft has not been submitted",
)

_BIB_KEY_RE = re.compile(r"@\w+\{\s*([^,\s]+)\s*,")
_CITATION_BLOCK_RE = re.compile(r"\[[^\]]*@[^]]+\]")
_CITATION_KEY_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_:-]*)")
_TITLE_RE = re.compile(r"^title:\s*['\"]?(.+?)['\"]?\s*$", re.MULTILINE)


def _relative(path: Path) -> str:
    """Format a path relative to the script's repository when possible.

    Parameters
    ----------
    path : pathlib.Path
        Diagnostic path; no symlink resolution is performed here.

    Returns
    -------
    str
        POSIX repository-relative spelling, or the original POSIX spelling
        when the path is outside the repository.
    """
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_text(path: Path, errors: list[str]) -> str:
    """Read one required UTF-8 input, recording missing and blank-file findings.

    Parameters
    ----------
    path : pathlib.Path
        Required input. Symlinks retain normal pathlib read behaviour.
    errors : list[str]
        Caller-owned list receiving one diagnostic for a missing or blank file.

    Returns
    -------
    str
        File text, or an empty string for a missing or whitespace-only input.

    Raises
    ------
    OSError
        A present input cannot be read, including a directory or a read race.
    UnicodeError
        The bytes are not valid UTF-8.
    """
    if not path.exists():
        errors.append(f"MISSING: {_relative(path)}")
        return ""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        errors.append(f"EMPTY: {_relative(path)}")
        return ""
    return text


def _bib_keys(text: str) -> tuple[set[str], list[str]]:
    """Collect lexical entry keys without validating complete BibTeX syntax.

    Parameters
    ----------
    text : str
        Bibliography text. The pattern recognises ``@word{key,`` occurrences.

    Returns
    -------
    tuple[set[str], list[str]]
        Case-sensitive keys and sorted keys appearing more than once. Comments
        and malformed entry bodies are not interpreted as BibTeX structure.
    """
    keys = _BIB_KEY_RE.findall(text)
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    return set(keys), duplicates


def _citation_keys(text: str) -> set[str]:
    """Collect case-sensitive keys inside lexical Pandoc citation brackets.

    Parameters
    ----------
    text : str
        Manuscript text. Code blocks and comments receive no special treatment.

    Returns
    -------
    set[str]
        Keys starting with an ASCII letter and continuing with letters,
        numbers, underscores, colons, or hyphens. Bare narrative citations
        outside brackets are not included.
    """
    keys: set[str] = set()
    for block in _CITATION_BLOCK_RE.findall(text):
        keys.update(_CITATION_KEY_RE.findall(block))
    return keys


def _missing_markers(label: str, text: str, markers: tuple[str, ...]) -> list[str]:
    """Check case-sensitive editorial substrings after whitespace normalisation.

    Parameters
    ----------
    label : str
        Input path used in diagnostics.
    text : str
        Entire input text, including comments and code fences.
    markers : tuple[str, ...]
        Required marker strings in diagnostic order.

    Returns
    -------
    list[str]
        One diagnostic per absent normalized marker; this is not a semantic
        assessment of the claims surrounding a present marker.
    """
    normalized = _normalize_prose(text)
    return [f"MISMATCH: {label} missing {marker!r}" for marker in markers if _normalize_prose(marker) not in normalized]


def _paper_title(text: str) -> str | None:
    """Extract the first lexical title line in the initial front-matter block.

    Parameters
    ----------
    text : str
        Manuscript beginning with a literal ``---`` line and containing a
        closing literal ``---`` line.

    Returns
    -------
    str or None
        First regex-matched title with optional surrounding quotes removed,
        or None if delimiters or a title line are absent. YAML escaping,
        duplicate metadata keys, types, and the full JOSS schema are not parsed.
    """
    if not text.startswith("---\n"):
        return None
    lines = text.splitlines()
    try:
        end = lines.index("---", 1)
    except ValueError:
        return None
    match = _TITLE_RE.search("\n".join(lines[1:end]))
    return match.group(1) if match else None


def _normalize_prose(text: str) -> str:
    """Collapse Unicode whitespace without changing case or Markdown syntax.

    Parameters
    ----------
    text : str
        Editorial marker, manuscript, title, or documentation text.

    Returns
    -------
    str
        Whitespace-delimited words joined by one ASCII space.
    """
    return " ".join(text.split())


def check_repository() -> list[str]:
    """Inspect the fixed local manuscript, documentation pointer, and bibliography.

    Returns
    -------
    list[str]
        Findings in input, marker, title, then bibliography/citation order.
        An empty fresh list means only the documented local lexical checks
        passed. All three inputs must be nonblank, the bibliography must contain
        a recognised entry key, and manuscript citation keys must exist in it.
        Documentation citations are outside the canonical manuscript scan.

    Raises
    ------
    OSError
        A present input cannot be read. No file is written or modified.
    UnicodeError
        A required input is not UTF-8.

    Notes
    -----
    Paths are derived from this script's resolved location, independent of
    caller cwd. There are no numeric units, array shapes, simulation clocks,
    controller state, network requests, or publication operations in this API.
    """
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
        if not keys:
            errors.append(f"MISMATCH: {_relative(BIB_PATH)} contains no bibliography entry keys")
        for duplicate in duplicates:
            errors.append(f"MISMATCH: {_relative(BIB_PATH)} duplicate bibliography key {duplicate!r}")
        for path, text in ((PAPER_PATH, paper),):
            if not text:
                continue
            missing = sorted(_citation_keys(text) - keys)
            if missing:
                errors.append(f"MISMATCH: {_relative(path)} cites missing bibliography keys: {', '.join(missing)}")

    return errors


def main() -> int:
    """Print local findings and return the canonical command's status.

    Returns
    -------
    int
        Zero when local editorial/citation checks pass; one when findings are
        printed to standard output. Native read/decode exceptions propagate
        and the standalone interpreter prints its normal traceback.

    Notes
    -----
    The public function takes no arguments and uses the same fixed paths as
    ``check_repository``. The standalone script supplies no option parser.
    A successful status is not manuscript submission or review acceptance.
    """
    errors = check_repository()
    if errors:
        for error in errors:
            print(error)
        print(f"\n{len(errors)} JOSS submission issue(s) detected.")
        return 1

    print("OK: canonical JOSS package and documentation pointer satisfy local editorial and citation checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
