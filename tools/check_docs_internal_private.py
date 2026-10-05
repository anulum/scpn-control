#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Permanent private-tree guard for docs/internal/
"""Check the repository's Git and MkDocs declarations for ``docs/internal/``.

The CLI reads repository-relative UTF-8 configuration and actual Git index and
history paths. YAML is composed into nodes without constructing custom tags.
This declaration guard does not inspect plugin output, distribution archives,
remote publication, or alternate MkDocs configurations. PyYAML is a development
tool dependency; no installed-package API is supplied by this script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNAL_PREFIX = "docs/internal"
GITIGNORE_MARKERS = ("docs/internal/", "docs/internal/**")
MKDOCS_EXCLUDE_MARKER = "internal/**"


def _git(*args: str) -> str:
    """Run Git at the physical script root and retain native process failures.

    Parameters
    ----------
    *args : str
        Git subcommand arguments without shell interpretation.

    Returns
    -------
    str
        UTF-8 standard output with literal path whitespace and NULs retained.

    Raises
    ------
    OSError
        Process launch fails.
    UnicodeError
        Process output cannot be decoded.
    subprocess.CalledProcessError
        Git returns a nonzero status.
    """
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    return completed.stdout


def check_gitignore_rules(gitignore_text: str) -> list[str]:
    """Inspect root ignore declarations without evaluating a Git repository.

    Parameters
    ----------
    gitignore_text : str
        Root ``.gitignore`` contents. Leading whitespace is significant.

    Returns
    -------
    list[str]
        Ordered findings, or an empty list for an accepted declaration. An
        exact private-root rule must follow every negation; literal negations
        mentioning ``docs/internal`` are forbidden anywhere. This conservative
        policy avoids trying to prove arbitrary glob intersections.
    """
    errors: list[str] = []
    lines = [line.rstrip(" ") for line in gitignore_text.splitlines()]
    protective_rules = [index for index, line in enumerate(lines) if line in GITIGNORE_MARKERS]
    if not protective_rules:
        errors.append(".gitignore must contain 'docs/internal/' or 'docs/internal/**' (permanent private tree)")
    for index, line in enumerate(lines):
        if line.startswith("!") and "docs/internal" in line:
            errors.append(f".gitignore must not re-include private tree via negation: {line!r}")
        elif line.startswith("!") and protective_rules and index > protective_rules[-1]:
            errors.append(f".gitignore negations must precede the final private-root rule: {line!r}")
    return errors


def check_no_tracked_internal_paths(tracked_paths: list[str]) -> list[str]:
    """Inspect exact repository-relative index paths without changing the index.

    Parameters
    ----------
    tracked_paths : list[str]
        Decoded paths, including any whitespace or Unicode in their names.

    Returns
    -------
    list[str]
        A heading and sorted private paths, retaining duplicates, or an empty
        list. ``docs/internal-extra`` is outside this literal path prefix.
    """
    bad = sorted(path for path in tracked_paths if path == INTERNAL_PREFIX or path.startswith(f"{INTERNAL_PREFIX}/"))
    if not bad:
        return []
    return [
        "tracked docs/internal paths must be removed from the index (git rm --cached):",
        *[f"  - {path}" for path in bad],
    ]


def check_no_history_internal_paths(history_paths: list[str]) -> list[str]:
    """Inspect exact paths observed across all local Git history references.

    Parameters
    ----------
    history_paths : list[str]
        Repository-relative path names; the caller supplies history scope.

    Returns
    -------
    list[str]
        A heading and at most 50 sorted private paths, with an ellipsis if
        truncated, or an empty list. No history rewrite or deduplication occurs.
    """
    bad = sorted(path for path in history_paths if path == INTERNAL_PREFIX or path.startswith(f"{INTERNAL_PREFIX}/"))
    if not bad:
        return []
    return [
        "docs/internal paths found in git history (must never reappear; history rewrite needs owner GO):",
        *[f"  - {path}" for path in bad[:50]],
        *(["  - ..."] if len(bad) > 50 else []),
    ]


def check_mkdocs_excludes_internal(mkdocs_text: str) -> list[str]:
    """Inspect the actual top-level MkDocs exclusion scalar without tag execution.

    Parameters
    ----------
    mkdocs_text : str
        Single-document YAML configuration text.

    Returns
    -------
    list[str]
        Findings for malformed or ambiguous top-level mappings, missing or
        non-string exclusions, absent exact ``internal/**`` patterns, or any
        negation after the final protective pattern. Patterns are relative to
        MkDocs' ``docs_dir``. Comments and unrelated values do not count.

    Notes
    -----
    Top-level keys must be unique ordinary strings; YAML merges are refused.
    Custom tags in unrelated values are parsed into nodes but never constructed.
    This conservative declaration policy is not a site or plugin-output audit.
    It has no numeric units, clock, or persistent state.
    """
    try:
        document = yaml.compose(mkdocs_text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return ["mkdocs.yml must contain valid single-document YAML"]
    if not isinstance(document, MappingNode) or document.tag != "tag:yaml.org,2002:map":
        return ["mkdocs.yml must contain an ordinary top-level mapping"]
    keys: set[str] = set()
    exclusion: ScalarNode | None = None
    for key, value in document.value:
        if not isinstance(key, ScalarNode) or key.tag != "tag:yaml.org,2002:str":
            return ["mkdocs.yml top-level keys must be ordinary strings; YAML merges are not allowed"]
        if key.value in keys:
            return [f"mkdocs.yml contains a duplicate top-level key: {key.value!r}"]
        keys.add(key.value)
        if key.value == "exclude_docs":
            if not isinstance(value, ScalarNode) or value.tag != "tag:yaml.org,2002:str":
                return ["mkdocs.yml exclude_docs must be an ordinary string"]
            exclusion = value
    if exclusion is None:
        return ["mkdocs.yml must declare exclude_docs (to keep docs/internal off the public site)"]
    patterns = [line.strip() for line in exclusion.value.splitlines()]
    protective_rules = [index for index, pattern in enumerate(patterns) if pattern == MKDOCS_EXCLUDE_MARKER]
    if not protective_rules:
        return [f"mkdocs.yml exclude_docs must include {MKDOCS_EXCLUDE_MARKER!r}"]
    if any(pattern.startswith("!") for pattern in patterns[protective_rules[-1] + 1 :]):
        return ["mkdocs.yml exclude_docs negations must precede the final 'internal/**' pattern"]
    return []


def collect_errors(*, check_history: bool = True) -> list[str]:
    """Read the script's repository and collect declaration and actual path findings.

    Parameters
    ----------
    check_history : bool, default True
        Scan every locally available Git reference as well as the current index.

    Returns
    -------
    list[str]
        Findings in ignore, index, history, and MkDocs order. Missing configuration
        files are findings. No file, index, history, or repository state is changed.

    Raises
    ------
    OSError
        A configuration read or Git process launch fails.
    UnicodeError
        Configuration or Git output cannot be decoded.
    subprocess.CalledProcessError
        Git fails, including an absent repository or unavailable commit history.
        These native errors propagate and cannot produce a PASS result.
    """
    errors: list[str] = []
    gitignore_path = REPO_ROOT / ".gitignore"
    if not gitignore_path.is_file():
        errors.append(".gitignore is missing")
    else:
        errors.extend(check_gitignore_rules(gitignore_path.read_text(encoding="utf-8")))

    tracked = [path for path in _git("ls-files", "-z").split("\0") if path]
    errors.extend(check_no_tracked_internal_paths(tracked))

    if check_history:
        history = [path for path in _git("log", "--all", "--name-only", "--format=", "-z").split("\0") if path]
        errors.extend(check_no_history_internal_paths(history))

    mkdocs_path = REPO_ROOT / "mkdocs.yml"
    if not mkdocs_path.is_file():
        errors.append("mkdocs.yml is missing")
    else:
        errors.extend(check_mkdocs_excludes_internal(mkdocs_path.read_text(encoding="utf-8")))

    return errors


def main(argv: list[str] | None = None) -> int:
    """Print findings or the exact checked scope and return a CLI status.

    Parameters
    ----------
    argv : list[str] or None, default None
        Arguments for argparse; ``None`` consumes process arguments.

    Returns
    -------
    int
        Zero for accepted declarations and paths, or one for findings.

    Raises
    ------
    SystemExit
        Argparse handles help (zero) or invalid arguments (two).
    OSError
        Native file, process, or output failures propagate.
    UnicodeError
        A UTF-8 read or Git output decode fails.
    subprocess.CalledProcessError
        Git fails. ``--skip-history`` never claims history was checked.
    """
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
    if args.skip_history:
        print("PASS: docs/internal ignore and MkDocs declarations accepted; index clean; history not checked")
    else:
        print("PASS: docs/internal ignore and MkDocs declarations accepted; index and local history clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
