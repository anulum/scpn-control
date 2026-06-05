# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Runtime admission evidence validation tests.
"""Tests for runtime-admission benchmark release evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from validation.validate_runtime_admission_evidence import DEFAULT_REPORT, validate_runtime_admission_evidence


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


def test_runtime_admission_evidence_admits_repository_report() -> None:
    """Repository runtime-admission evidence is admitted as local fail-closed regression evidence."""
    result = validate_runtime_admission_evidence()

    assert result.status == "pass"
    assert result.errors == ()
    assert result.report_sha256 is not None
    assert result.payload_sha256 is not None
    assert result.benchmark_evidence_class == "local_regression"
    assert result.production_claim_allowed is False
    assert result.admission_status == "fail"
    assert result.admission_error_count is not None
    assert result.admission_error_count > 0


def test_runtime_admission_evidence_rejects_payload_tampering(tmp_path: Path) -> None:
    """Canonical payload digests prevent silent runtime-admission report mutation."""
    report = _copy_report(DEFAULT_REPORT, tmp_path / "runtime.json")
    payload = _load_report(report)
    payload["last_admission_status"] = "pass"
    _write_report(report, payload, refresh_digest=False)

    result = validate_runtime_admission_evidence(report)

    assert result.status == "fail"
    assert "runtime_admission.payload_sha256 does not match canonical payload" in result.errors


def test_runtime_admission_evidence_rejects_local_production_claim(tmp_path: Path) -> None:
    """Local regression runtime-admission evidence cannot be promoted to a production timing claim."""
    report = _copy_report(DEFAULT_REPORT, tmp_path / "runtime.json")
    payload = _load_report(report)
    payload["production_claim_allowed"] = True
    _write_report(report, payload)

    result = validate_runtime_admission_evidence(report)

    assert result.status == "fail"
    assert "local runtime admission evidence must not allow production benchmark claims" in result.errors


def test_runtime_admission_evidence_rejects_missing_context(tmp_path: Path) -> None:
    """Runtime-admission evidence must preserve CPU affinity, load, and isolation context."""
    report = _copy_report(DEFAULT_REPORT, tmp_path / "runtime.json")
    payload = _load_report(report)
    payload["context"] = {}
    _write_report(report, payload)

    result = validate_runtime_admission_evidence(report)

    assert result.status == "fail"
    assert "runtime_admission.context.cpu_affinity must be a non-empty sequence" in result.errors
    assert "runtime_admission.context must record loadavg_start and loadavg_end" in result.errors


def test_runtime_admission_evidence_rejects_non_monotonic_percentiles(tmp_path: Path) -> None:
    """Latency percentile fields must remain internally consistent."""
    report = _copy_report(DEFAULT_REPORT, tmp_path / "runtime.json")
    payload = _load_report(report)
    stats = payload["stats"]
    assert isinstance(stats, dict)
    stats["p99_us"] = 1.0
    stats["p95_us"] = 2.0
    _write_report(report, payload)

    result = validate_runtime_admission_evidence(report)

    assert result.status == "fail"
    assert "runtime_admission.stats percentiles must be monotonic" in result.errors
