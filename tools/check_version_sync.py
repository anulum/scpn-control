#!/usr/bin/env python3
# SCPN Control — Version sync guard
# Asserts pyproject.toml, CITATION.cff, .zenodo.json, and docs/api.md share the same version.
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: MIT OR Apache-2.0

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _extract(path: Path, pattern: str) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1) if m else None


def _contains(path: Path, substring: str) -> bool:
    if not path.exists():
        return True  # skip missing optional files
    return substring in path.read_text(encoding="utf-8")


def main() -> int:
    canonical = _extract(ROOT / "pyproject.toml", r'^version\s*=\s*"([^"]+)"')
    if not canonical:
        print("FAIL: could not extract version from pyproject.toml")
        return 1

    versions = {
        "CITATION.cff": _extract(ROOT / "CITATION.cff", r'^version:\s*"?([^"\s]+)"?'),
        ".zenodo.json": _extract(ROOT / ".zenodo.json", r'"version":\s*"([^"]+)"'),
    }

    errors = 0
    for name, ver in versions.items():
        if ver is None:
            print(f"WARN: could not extract version from {name}")
        elif ver != canonical:
            print(f"MISMATCH: {name} has {ver!r}, expected {canonical!r}")
            errors += 1

    if not _contains(ROOT / "docs" / "api.md", canonical):
        print(f"MISMATCH: docs/api.md does not contain {canonical!r}")
        errors += 1

    if errors:
        print(f"\nCanonical version (pyproject.toml): {canonical}")
        print(f"{errors} file(s) out of sync.")
        return 1

    print(f"OK: all versions = {canonical}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
