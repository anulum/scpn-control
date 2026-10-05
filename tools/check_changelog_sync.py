#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Changelog mirror sync guard.
"""Compare the root changelog and documentation mirror without rewriting them.

Hook, CI and preflight consumers use byte equality: encoding, line endings and
whitespace remain significant. This checks two local files, without rendering
Markdown, validating release history or checking published documentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
ROOT_CHANGELOG: Final = Path("CHANGELOG.md")
DOCS_CHANGELOG: Final = Path("docs/changelog.md")


def changelog_sync_errors(repo: Path) -> list[str]:
    """Return changelog mirror drift errors for ``repo``.

    Parameters
    ----------
    repo
        Repository root containing ``CHANGELOG.md`` and ``docs/changelog.md``.
        Relative roots follow the caller's working directory.

    Returns
    -------
    list[str]
        Human-readable validation errors. An empty list means the root
        changelog and docs mirror are byte-identical, including line endings
        and whitespace. Missing root and mirror errors are ordered that way;
        if either is missing, no content is read. Any matching bytes, including
        non-UTF-8 content, pass. No Markdown or version-history parsing occurs.

    Raises
    ------
    OSError
        An existing carrier cannot be read, including a directory at either
        file location. The CLI translates this into a fixed refusal sentence.

    Examples
    --------
    Compare the actual repository files through the public API:

    >>> changelog_sync_errors(ROOT)
    []
    """
    root_changelog = repo / ROOT_CHANGELOG
    docs_changelog = repo / DOCS_CHANGELOG
    errors: list[str] = []

    if not root_changelog.exists():
        errors.append(f"missing {ROOT_CHANGELOG.as_posix()}")
    if not docs_changelog.exists():
        errors.append(f"missing {DOCS_CHANGELOG.as_posix()}")
    if errors:
        return errors

    root_bytes = root_changelog.read_bytes()
    docs_bytes = docs_changelog.read_bytes()
    if root_bytes != docs_bytes:
        errors.append(
            f"{DOCS_CHANGELOG.as_posix()} differs from {ROOT_CHANGELOG.as_posix()}; "
            f"sync the rendered mirror from the root changelog"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    """Check both local changelog carriers through the hook/CI command.

    Parameters
    ----------
    argv
        Arguments excluding the executable; None reads process arguments.
        --repo selects a root, resolved against cwd. Without it use the root
        containing this script, regardless of cwd. Leave caller argv unchanged.

    Returns
    -------
    int
        Zero for byte equality; one for drift, missing files or an OS read
        failure. Print a PASS or FAIL diagnostic to stdout; argument errors
        exit two through argparse. No files, Git state or releases are changed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    try:
        errors = changelog_sync_errors(repo)
    except OSError:
        print("FAIL: could not read changelog files")
        return 1
    if not errors:
        print("PASS: docs/changelog.md matches CHANGELOG.md")
        return 0

    print("FAIL: changelog mirror drift detected")
    for error in errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
