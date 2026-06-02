# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Tests for benchmark regression admission gates.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from validation.validate_benchmark_regression_gates import (
    SCHEMA_VERSION,
    main,
    validate_benchmark_regression_gates,
)


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _report() -> dict[str, object]:
    return {
        "claim_status": "local evidence only; not a real-time control-loop guarantee",
        "runtime_metadata": {"machine": "x86_64", "platform": "Linux"},
        "timed_runs": 5,
        "p95_ms": 3.5,
    }


def _manifest(report_sha256: str, **entry_overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "benchmark_id": "transport.p95_ms",
        "report_uri": "report.json",
        "report_sha256": report_sha256,
        "metric_path": "p95_ms",
        "observed": 3.5,
        "unit": "ms",
        "max_threshold": 4.0,
        "sample_count_path": "timed_runs",
        "sample_count": 5,
        "min_sample_count": 5,
        "claim_status_path": "claim_status",
        "required_claim_substring": "not a real-time control-loop guarantee",
        "hardware_context_path": "runtime_metadata",
        "status": "pass",
    }
    entry.update(entry_overrides)
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_set_id": "test-benchmark-gates",
        "claim_boundary": "bounded persisted benchmark evidence only",
        "generated_utc": "2026-06-02T00:00:00Z",
        "entries": [entry],
    }


def test_benchmark_regression_gate_admits_complete_manifest(tmp_path: Path) -> None:
    report_sha256 = _write_json(tmp_path / "report.json", _report())
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "pass"
    assert result.admitted_gates == ("transport.p95_ms",)
    assert result.errors == ()


def test_benchmark_regression_gate_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"' + SCHEMA_VERSION + '","schema_version":"' + SCHEMA_VERSION + '"}',
        encoding="utf-8",
    )

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("duplicate JSON key" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_report_digest_mismatch(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "report.json", _report())
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest("0" * 64))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("report_sha256 mismatch" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_threshold_regression(
    tmp_path: Path,
) -> None:
    report_sha256 = _write_json(tmp_path / "report.json", _report())
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256, max_threshold=3.0))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("observed exceeds max_threshold" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_metric_replay_mismatch(
    tmp_path: Path,
) -> None:
    report_sha256 = _write_json(tmp_path / "report.json", _report())
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256, observed=3.0))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("observed metric does not match report" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_unsafe_report_uri(
    tmp_path: Path,
) -> None:
    report_sha256 = _write_json(tmp_path / "report.json", _report())
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256, report_uri="../report.json"))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("report_uri is not repository-relative" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_overclaimed_report(
    tmp_path: Path,
) -> None:
    report = _report()
    report["claim_status"] = "hardware-in-the-loop real-time guarantee"
    report_sha256 = _write_json(tmp_path / "report.json", report)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("required claim boundary substring missing" in error for error in result.errors)


def test_repository_benchmark_regression_manifest_is_admitted() -> None:
    result = validate_benchmark_regression_gates()

    assert result.status == "pass"
    assert result.admitted_gates == (
        "e2e_control_latency.kernel_only_p95_us",
        "e2e_control_latency.e2e_p95_us",
        "differentiable_transport.one_step_p95_ms",
        "differentiable_transport.rollout_p95_ms",
    )


def test_benchmark_regression_gate_cli_returns_nonzero_for_invalid_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"schema_version": "wrong"})

    exit_code = main([str(manifest)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert '"status": "fail"' in captured.out
