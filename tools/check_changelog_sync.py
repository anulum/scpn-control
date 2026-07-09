#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Changelog mirror sync guard.
"""Fail closed when the rendered changelog mirror drifts from the root file."""

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

    Returns
    -------
    list[str]
        Human-readable validation errors. An empty list means the root
        changelog and rendered docs mirror are byte-identical.
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
    """Run the changelog mirror sync guard."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    errors = changelog_sync_errors(repo)
    if not errors:
        print("PASS: docs/changelog.md matches CHANGELOG.md")
        return 0

    print("FAIL: changelog mirror drift detected")
    for error in errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
