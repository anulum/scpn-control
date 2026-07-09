#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Check Version Sync
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# SCPN Control — Version sync guard
# Asserts pyproject.toml, CITATION.cff, .zenodo.json, and docs/api.md share the same version.
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT_SLUG = "scpn-control"


def _extract(path: Path, pattern: str) -> str | None:
    """Extract the first regex capture group from a UTF-8 text file."""

    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1) if m else None


def _require_contains(path: Path, substring: str, label: str) -> str | None:
    """Return an error message when ``path`` does not contain ``substring``."""

    if not path.exists():
        return f"MISSING: {label} file {path.relative_to(ROOT).as_posix()} does not exist"
    if substring not in path.read_text(encoding="utf-8"):
        return f"MISMATCH: {label} missing {substring!r}"
    return None


def _release_notes_path(version: str) -> Path:
    """Return the expected release-notes path for a package version."""

    return ROOT / "docs" / f"release_notes_v{version}.md"


def _metadata_badge_errors(version: str) -> list[str]:
    """Validate README badges and release-note metadata tied to public releases."""

    readme = ROOT / "README.md"
    release_notes = _release_notes_path(version)
    checks = [
        (readme, f"https://img.shields.io/pypi/v/{PROJECT_SLUG}", "README PyPI version badge"),
        (readme, f"https://img.shields.io/pypi/pyversions/{PROJECT_SLUG}", "README Python-version badge"),
        (readme, f"https://pepy.tech/project/{PROJECT_SLUG}", "README Pepy downloads link"),
        (readme, f"https://static.pepy.tech/badge/{PROJECT_SLUG}", "README all-time downloads badge"),
        (readme, f"| Package version | {version} |", "README package-version table"),
        (readme, f"git tag v{version}", "README release tag example"),
        (release_notes, f"# SCPN Control v{version} Release Notes", "release-note heading"),
        (release_notes, f"`v{version}` commit/tag", "release-note CI checklist"),
        (release_notes, f"`scpn-control=={version}`", "release-note PyPI checklist"),
    ]
    return [error for path, substring, label in checks if (error := _require_contains(path, substring, label))]


def main() -> int:
    """Return success only when repository version and release metadata agree."""

    canonical = _extract(ROOT / "pyproject.toml", r'^version\s*=\s*"([^"]+)"')
    if not canonical:
        print("FAIL: could not extract version from pyproject.toml")
        return 1

    versions = {
        "CITATION.cff": _extract(ROOT / "CITATION.cff", r'^version:\s*"?([^"\s]+)"?'),
        ".zenodo.json": _extract(ROOT / ".zenodo.json", r'"version":\s*"([^"]+)"'),
    }

    errors = 0
    messages: list[str] = []
    for name, ver in versions.items():
        if ver is None:
            print(f"WARN: could not extract version from {name}")
        elif ver != canonical:
            messages.append(f"MISMATCH: {name} has {ver!r}, expected {canonical!r}")

    if api_error := _require_contains(ROOT / "docs" / "api.md", canonical, "docs/api.md version marker"):
        messages.append(api_error)

    messages.extend(_metadata_badge_errors(canonical))

    errors = len(messages)
    if errors:
        for message in messages:
            print(message)
        print(f"\nCanonical version (pyproject.toml): {canonical}")
        print(f"{errors} file(s) out of sync.")
        return 1

    print(f"OK: all versions and release metadata = {canonical}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
