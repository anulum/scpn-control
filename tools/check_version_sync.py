#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Check Version Sync.

# SCPN Control — Version sync guard
# Asserts pyproject.toml, CITATION.cff, .zenodo.json, and docs/api.md share the same version.
# © 1996–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available

"""Compare required local release text without querying external registries.

The no-argument hook checks the repository containing this script; ``--repo``
selects another root. The canonical version is the first matching double-quoted
version assignment in pyproject.toml. CITATION.cff and .zenodo.json use their
first matching version field. README badges/release examples, docs/api.md and
version-named release notes must contain the declared literal strings. These
are bounded text checks, not TOML/CFF/JSON schema validation, URL resolution,
package publication evidence or agreement with a remote registry. Missing or
unreadable required inputs fail; source files are never rewritten.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT_SLUG = "scpn-control"


def _extract(path: Path, pattern: str) -> str | None:
    """Read a required text carrier and return the first multiline regex capture.

    Missing files or unmatched fields return None. Other filesystem errors and
    invalid UTF-8 propagate; no structural or duplicate-field validation is
    inferred from this first-match text probe.
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1) if m else None


def _require_contains(path: Path, substring: str, label: str, root: Path) -> str | None:
    """Return a labelled missing-file or missing-literal error relative to root.

    Reads strict UTF-8, performs a case-sensitive substring test and otherwise
    returns None. Read/decoding failures propagate rather than imply agreement.
    """
    if not path.exists():
        return f"MISSING: {label} file {path.relative_to(root).as_posix()} does not exist"
    if substring not in path.read_text(encoding="utf-8"):
        return f"MISMATCH: {label} missing {substring!r}"
    return None


def _release_notes_path(version: str, root: Path) -> Path:
    """Select docs/release_notes_v{version}.md under the supplied checkout root."""
    return root / "docs" / f"release_notes_v{version}.md"


def _metadata_badge_errors(version: str, root: Path) -> list[str]:
    """Collect missing README and release-note literals without external requests.

    Checks four exact badge/link URLs, the package-version table and tag
    example, plus release-note heading and three publication-boundary phrases.
    Repeated errors for the same file remain separately labelled; counts are
    failing checks rather than distinct files.
    """
    readme = root / "README.md"
    release_notes = _release_notes_path(version, root)
    checks = [
        (readme, f"https://img.shields.io/pypi/v/{PROJECT_SLUG}", "README PyPI version badge"),
        (readme, f"https://img.shields.io/pypi/pyversions/{PROJECT_SLUG}", "README Python-version badge"),
        (readme, f"https://pepy.tech/project/{PROJECT_SLUG}", "README Pepy downloads link"),
        (readme, f"https://static.pepy.tech/badge/{PROJECT_SLUG}", "README all-time downloads badge"),
        (readme, f"| Package version | {version} |", "README package-version table"),
        (readme, f"git tag v{version}", "README release tag example"),
        (release_notes, f"# SCPN Control v{version} Release Notes", "release-note heading"),
        (release_notes, "## Publication boundary", "release-note publication boundary"),
        (release_notes, "external mutable state", "release-note external-state boundary"),
        (release_notes, "source-level release history", "release-note source-history boundary"),
    ]
    return [error for path, substring, label in checks if (error := _require_contains(path, substring, label, root))]


def _check(root: Path) -> int:
    """Compare all required version and publication-boundary text in one root.

    Print each mismatch and return one on failure. Return zero only after
    all local literal checks pass; errors cannot retire a required version.
    Filesystem/decoding errors propagate to the public command boundary.
    """
    canonical = _extract(root / "pyproject.toml", r'^version\s*=\s*"([^"]+)"')
    if not canonical:
        print("FAIL: could not extract version from pyproject.toml")
        return 1

    if canonical in {".", ".."} or "/" in canonical or "\\" in canonical:
        print("FAIL: canonical version must be a safe filename component")
        return 1

    versions = {
        "CITATION.cff": _extract(root / "CITATION.cff", r'^version:\s*"?([^"\s]+)"?'),
        ".zenodo.json": _extract(root / ".zenodo.json", r'"version":\s*"([^"]+)"'),
    }

    messages: list[str] = []
    for name, ver in versions.items():
        if ver is None:
            messages.append(f"MISSING: could not extract version from {name}")
        elif ver != canonical:
            messages.append(f"MISMATCH: {name} has {ver!r}, expected {canonical!r}")

    if api_error := _require_contains(root / "docs" / "api.md", canonical, "docs/api.md version marker", root):
        messages.append(api_error)

    messages.extend(_metadata_badge_errors(canonical, root))

    errors = len(messages)
    if errors:
        for message in messages:
            print(message)
        print(f"\nCanonical version (pyproject.toml): {canonical}")
        print(f"{errors} check(s) out of sync.")
        return 1

    print(f"OK: all versions and release metadata = {canonical}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the local release-text guard against an explicit or owning checkout.

    Parameters
    ----------
    argv
        Arguments excluding the executable. None retains the historical
        no-argument API. The script forwards process arguments explicitly.
        --repo paths resolve against the caller's working directory.

    Returns
    -------
    int
        Zero only when all required local text checks pass; one for missing,
        unreadable, undecodable or mismatched input, or an unresolved/non-directory root.
        Invalid command arguments exit two via argparse. No files are written
        and no external registry state is admitted.

    Examples
    --------
    Run the real owning checkout without exposing its progress output:

    >>> import contextlib, io
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     status = main()
    >>> status
    0
    """
    parser = argparse.ArgumentParser(description="Check local release version and metadata text.")
    parser.add_argument("--repo", type=Path, default=ROOT, help="Checkout root; defaults to the script's repository.")
    args = parser.parse_args([] if argv is None else argv)
    try:
        try:
            args.repo.stat()
        except FileNotFoundError:
            pass  # The existing non-directory refusal handles a missing root.
        root = args.repo.resolve()
    except (OSError, ValueError, RuntimeError):
        print("FAIL: repository root could not be resolved")
        return 1
    if not root.is_dir():
        print(f"FAIL: repository root is not a directory: {root}")
        return 1
    try:
        return _check(root)
    except (OSError, UnicodeError) as exc:
        print(f"FAIL: required release metadata could not be read: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
