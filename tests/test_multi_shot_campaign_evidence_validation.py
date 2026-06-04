# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-shot campaign evidence validation tests.
"""Tests for multi-shot campaign release-evidence admission."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from validation.validate_multi_shot_campaign_evidence import (
    DEFAULT_PYTHON_REPORT,
    DEFAULT_RUST_REPORT,
    validate_multi_shot_campaign_evidence,
)


def _canonical_payload_digest(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned["payload_sha256"] = ""
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _copy_report(source: Path, destination: Path) -> Path:
    shutil.copyfile(source, destination)
    return destination


def _load_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, object], *, refresh_digest: bool = True) -> None:
    if refresh_digest:
        payload["payload_sha256"] = _canonical_payload_digest(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_multi_shot_campaign_evidence_admits_repository_reports():
    """Repository Python/PyO3 and Rust reports admit the multi-shot campaign evidence gate."""
    result = validate_multi_shot_campaign_evidence()

    assert result.status == "pass"
    assert result.errors == ()
    assert result.admitted_surfaces == ("python", "pyo3", "rust")
    assert result.pyo3_status == "ok"
    assert result.python_report_sha256 is not None
    assert result.rust_report_sha256 is not None


def test_multi_shot_campaign_evidence_rejects_missing_pyo3_surface(tmp_path):
    """The release gate fails closed when the Python report lacks PyO3 campaign evidence."""
    python_report = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust_report = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")
    payload = _load_report(python_report)
    payload["pyo3_status"] = "unavailable"
    payload["pyo3_result"] = None
    _write_report(python_report, payload)

    result = validate_multi_shot_campaign_evidence(python_report, rust_report)

    assert result.status == "fail"
    assert "multi_shot_campaign.pyo3_status must be 'ok'" in result.errors
    assert "multi_shot_campaign.pyo3_result must be an object" in result.errors


def test_multi_shot_campaign_evidence_rejects_under_counted_digest_chain(tmp_path):
    """Each admitted surface must preserve the configured number of pulsed-MPC decision digests."""
    python_report = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust_report = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")

    result = validate_multi_shot_campaign_evidence(python_report, rust_report, minimum_digest_count=3)

    assert result.status == "fail"
    assert any("last_pulsed_mpc_admission_digest_count must be at least 3" in error for error in result.errors)


def test_multi_shot_campaign_evidence_rejects_rust_report_without_context(tmp_path):
    """Rust campaign evidence must record CPU affinity and load context before release admission."""
    python_report = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust_report = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")
    payload = _load_report(rust_report)
    payload["context"] = {}
    _write_report(rust_report, payload)

    result = validate_multi_shot_campaign_evidence(python_report, rust_report)

    assert result.status == "fail"
    assert "multi_shot_campaign.rust.context.cpu_affinity must be recorded" in result.errors
    assert "multi_shot_campaign.rust.context must record loadavg_start and loadavg_end" in result.errors


def test_multi_shot_campaign_evidence_rejects_python_payload_tampering(tmp_path):
    """Canonical payload digests prevent silent mutation of persisted benchmark claims."""
    python_report = _copy_report(DEFAULT_PYTHON_REPORT, tmp_path / "python.json")
    rust_report = _copy_report(DEFAULT_RUST_REPORT, tmp_path / "rust.json")
    payload = _load_report(python_report)
    payload["steps"] = int(payload["steps"]) + 1
    _write_report(python_report, payload, refresh_digest=False)

    result = validate_multi_shot_campaign_evidence(python_report, rust_report)

    assert result.status == "fail"
    assert "multi_shot_campaign.python.payload_sha256 does not match canonical payload" in result.errors
