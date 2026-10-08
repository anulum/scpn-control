# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Evidence gap registry input contracts
"""Exercise strict metadata parsing on the complete maintained registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tools.evidence_gap_matrix import ROOT, build_evidence_gap_matrix
from tools.evidence_gap_registry import EvidenceGapRegistryError, load_gap_registry


def _copy_registry(path: Path) -> dict[str, object]:
    """Copy the full maintained registry and return its declared JSON object."""
    payload = cast(dict[str, object], json.loads((ROOT / "validation/physics_traceability.json").read_bytes()))
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_registry_keeps_complete_entries_and_unresolved_diagnostics(tmp_path: Path) -> None:
    """An unresolved positive tracker remains a planning gap without inventing admission."""
    path = tmp_path / "registry.json"
    payload = _copy_registry(path)
    rows = cast(list[dict[str, object]], payload["entries"])
    open_row = next(row for row in rows if row["fidelity_status"] == "validation_gap")
    original_claim = open_row["public_claim_allowed"]
    open_row["external_validation_tracker_issue"] = 987654321
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    entries, trackers = load_gap_registry(path)
    matrix = build_evidence_gap_matrix(path)
    assert matrix.entries == entries and matrix.trackers == trackers
    assert len(entries) == len(rows) == 76
    assert len(trackers) == 8
    assert matrix.untracked_open_entries == 1
    assert (
        next(entry for entry in entries if entry.component == open_row["component"]).public_claim_allowed
        == original_claim
    )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "kind",
    [
        "root-list",
        "entries-missing",
        "entries-object",
        "entry-scalar",
        "trackers-missing",
        "trackers-object",
        "tracker-scalar",
        "duplicate-issue",
        "issue-bool",
        "issue-zero",
        "issue-negative",
        "issue-string",
        "issue-float",
        "component-missing",
        "component-blank",
        "module-path-nonstring",
        "status-unknown",
        "status-blank",
        "claim-not-bool",
        "requirements-object",
        "requirement-blank",
        "requirement-number",
        "tracker-link-bool",
        "tracker-link-zero",
        "tracker-title-blank",
        "tracker-url-missing",
        "tracker-scope-number",
        "duplicate-json",
        "nan",
        "infinity",
        "overflow",
    ],
)
def test_registry_refuses_ambiguous_or_invalid_metadata(tmp_path: Path, kind: str) -> None:
    """Actual malformed registry bytes are refused through the new public loader."""
    path = tmp_path / "registry.json"
    payload = _copy_registry(path)
    rows = cast(list[object], payload["entries"])
    trackers = cast(list[object], payload["external_validation_trackers"])
    entry = cast(dict[str, object], rows[0])
    tracker = cast(dict[str, object], trackers[0])
    if kind == "root-list":
        text = "[]"
    elif kind == "duplicate-json":
        text = json.dumps(payload)[:-1] + ', "entries": []}'
    elif kind in {"nan", "infinity", "overflow"}:
        text = (
            json.dumps(payload)[:-1]
            + ', "nonplanning_number": '
            + {"nan": "NaN", "infinity": "Infinity", "overflow": "1e999"}[kind]
            + "}"
        )
    else:
        if kind == "entries-missing":
            del payload["entries"]
        elif kind == "entries-object":
            payload["entries"] = {}
        elif kind == "entry-scalar":
            rows[0] = 7
        elif kind == "trackers-missing":
            del payload["external_validation_trackers"]
        elif kind == "trackers-object":
            payload["external_validation_trackers"] = {}
        elif kind == "tracker-scalar":
            trackers[0] = 7
        elif kind == "duplicate-issue":
            trackers.append(dict(tracker, title="conflicting declared tracker"))
        elif kind.startswith("issue-"):
            tracker["issue"] = {"bool": True, "zero": 0, "negative": -1, "string": "47", "float": 47.0}[kind[6:]]
        elif kind == "component-missing":
            del entry["component"]
        elif kind == "component-blank":
            entry["component"] = " "
        elif kind == "module-path-nonstring":
            entry["module_path"] = 7
        elif kind == "status-unknown":
            entry["fidelity_status"] = "unknown"
        elif kind == "status-blank":
            entry["fidelity_status"] = " "
        elif kind == "claim-not-bool":
            entry["public_claim_allowed"] = 1
        elif kind == "requirements-object":
            entry["claim_admission_requirements"] = {}
        elif kind == "requirement-blank":
            entry["claim_admission_requirements"] = [" "]
        elif kind == "requirement-number":
            entry["claim_admission_requirements"] = [1]
        elif kind == "tracker-link-bool":
            entry["external_validation_tracker_issue"] = True
        elif kind == "tracker-link-zero":
            entry["external_validation_tracker_issue"] = 0
        elif kind == "tracker-title-blank":
            tracker["title"] = " "
        elif kind == "tracker-url-missing":
            del tracker["url"]
        elif kind == "tracker-scope-number":
            tracker["scope"] = 1
        else:
            raise AssertionError(kind)
        text = json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(EvidenceGapRegistryError):
        load_gap_registry(path)
    with pytest.raises(EvidenceGapRegistryError):
        build_evidence_gap_matrix(path)
    assert path.read_bytes() == before


def test_registry_accepts_finite_unused_metadata_and_absent_optional_link(tmp_path: Path) -> None:
    """Finite unused metadata does not force full validator/source admission on planning."""
    path = tmp_path / "registry.json"
    payload = _copy_registry(path)
    payload["nonplanning_number"] = 1.25
    rows = cast(list[dict[str, object]], payload["entries"])
    open_row = next(row for row in rows if row["fidelity_status"] == "validation_gap")
    del open_row["external_validation_tracker_issue"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    matrix = build_evidence_gap_matrix(path)
    assert len(matrix.entries) == 76 and matrix.untracked_open_entries == 1
