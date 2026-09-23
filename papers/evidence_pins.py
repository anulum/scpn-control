# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — immutable manuscript evidence pins.

"""Verify that manuscript evidence bytes exist at the declared Git revision."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def verify_evidence_pins(metadata_path: Path, repo_root: Path) -> None:
    """Refuse missing, escaped, or changed evidence at an immutable commit."""
    root = repo_root.resolve(strict=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("submission metadata must be an object")
    revision = metadata.get("evidence_revision")
    files = metadata.get("evidence_files")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("evidence_revision must be a full Git commit SHA")
    if not isinstance(files, list) or not files or any(not isinstance(item, str) or not item for item in files):
        raise ValueError("evidence_files must be a nonempty list of paths")
    if len(files) != len(set(files)):
        raise ValueError("evidence_files contains duplicate paths")
    kind = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-t", revision],
        capture_output=True,
        text=True,
        check=False,
    )
    if kind.returncode != 0 or kind.stdout.strip() != "commit":
        raise ValueError(f"evidence_revision is not an available commit: {revision}")
    for relative in files:
        path = (metadata_path.parent / relative).resolve(strict=True)
        try:
            repo_path = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"evidence path escapes repository: {relative}") from exc
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{repo_path.as_posix()}"],
            capture_output=True,
            check=False,
        )
        if blob.returncode != 0:
            raise ValueError(f"evidence missing at declared revision: {repo_path}")
        if blob.stdout != path.read_bytes():
            raise ValueError(f"evidence bytes differ from declared revision: {repo_path}")


def main() -> int:
    """Check one submission package from the paper verification script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()
    verify_evidence_pins(args.metadata, args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
