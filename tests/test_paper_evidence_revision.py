# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — immutable manuscript evidence regression tests.

"""Exercise manuscript evidence admission against real Git objects."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from papers.evidence_pins import verify_evidence_pins


def _git(root: Path, *args: str) -> str:
    """Run a deterministic command in the temporary source repository."""
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _submission(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a real committed evidence blob and a package metadata file."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    evidence = root / "evidence.bin"
    evidence.write_bytes(b"measured-source-bytes\x00")
    _git(root, "add", "evidence.bin")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "evidence")
    metadata = root / "papers" / "submissions" / "001" / "submission_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"evidence_revision": _git(root, "rev-parse", "HEAD"), "evidence_files": ["../../../evidence.bin"]}),
        encoding="utf-8",
    )
    return root, metadata, evidence


def test_accepts_exact_committed_evidence_and_refuses_changed_bytes(tmp_path: Path) -> None:
    """A later file edit cannot inherit an older manuscript evidence pin."""
    root, metadata, evidence = _submission(tmp_path)
    verify_evidence_pins(metadata, root)
    evidence.write_bytes(b"different evidence")
    with pytest.raises(ValueError, match="bytes differ"):
        verify_evidence_pins(metadata, root)


def test_refuses_evidence_missing_at_declared_revision(tmp_path: Path) -> None:
    """A worktree file alone cannot substantiate the declared Git revision."""
    root, metadata, _ = _submission(tmp_path)
    late = root / "late.bin"
    late.write_bytes(b"late")
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document["evidence_files"].append("../../../late.bin")
    metadata.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="missing at declared revision"):
        verify_evidence_pins(metadata, root)


def test_refuses_evidence_outside_repository(tmp_path: Path) -> None:
    """A relative path cannot escape the immutable source repository."""
    root, metadata, _ = _submission(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document["evidence_files"] = ["../../../../outside.bin"]
    metadata.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes repository"):
        verify_evidence_pins(metadata, root)
