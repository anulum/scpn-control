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
    _report_path,
    _value_at_path,
    main,
    validate_benchmark_regression_gates,
)


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _self_digest_payload(payload: dict[str, object], field: str = "payload_sha256") -> dict[str, object]:
    """Return ``payload`` with a canonical self-digest field."""

    unsigned = {key: value for key, value in payload.items() if key != field}
    encoded = json.dumps(unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {**unsigned, field: hashlib.sha256(encoded).hexdigest()}


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


def test_benchmark_regression_gate_rejects_embedded_self_digest_mismatch(
    tmp_path: Path,
) -> None:
    report = _self_digest_payload(_report())
    report["p95_ms"] = 3.6
    report_sha256 = _write_json(tmp_path / "report.json", report)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256, observed=3.6))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("payload_sha256 self-digest mismatch" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_invalid_embedded_self_digest(
    tmp_path: Path,
) -> None:
    report = {**_report(), "samples": [{"payload_sha256": "z" * 64}]}
    report_sha256 = _write_json(tmp_path / "report.json", report)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("samples[0].payload_sha256 must be a SHA-256 hex digest" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_malformed_report_json(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text("{", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(hashlib.sha256(report.read_bytes()).hexdigest()))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("malformed JSON" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_non_object_report(
    tmp_path: Path,
) -> None:
    report_sha256 = _write_json(tmp_path / "report.json", ["not", "an", "object"])
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("root must be a JSON object" in error for error in result.errors)


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


def test_report_path_rejects_escape_after_resolution(tmp_path: Path) -> None:
    """The report resolver rejects unsafe paths before touching the filesystem."""

    with pytest.raises(ValueError, match="unsafe report path escapes report root"):
        _report_path("file:/tmp/report.json", tmp_path)


def test_report_path_rejects_symlink_escape_after_resolution(tmp_path: Path) -> None:
    """The report resolver rejects paths that resolve outside the report root."""

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe report path escapes report root"):
        _report_path("linked/report.json", tmp_path)


def test_value_at_path_rejects_empty_segments() -> None:
    """Dotted report paths must not contain empty path segments."""

    with pytest.raises(KeyError):
        _value_at_path({}, "")


def test_benchmark_regression_gate_rejects_missing_report(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest("0" * 64))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("report does not exist" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_invalid_entry_schema(
    tmp_path: Path,
) -> None:
    report_sha256 = _write_json(tmp_path / "report.json", _report())
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest(
            report_sha256,
            status="fail",
            observed="bad",
            unit="days",
            max_threshold=0.0,
            sample_count=True,
            min_sample_count=0,
        ),
    )

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("gate status must be pass" in error for error in result.errors)
    assert any("observed must be a finite number" in error for error in result.errors)
    assert any("unsupported unit" in error for error in result.errors)
    assert any("max_threshold must be positive" in error for error in result.errors)
    assert any("sample_count must be a positive integer" in error for error in result.errors)
    assert any("min_sample_count must be positive" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_nonfinite_observed_metric(
    tmp_path: Path,
) -> None:
    report_sha256 = _write_json(tmp_path / "report.json", _report())
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256, observed=float("inf")))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("observed must be finite" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_report_path_and_type_mismatches(
    tmp_path: Path,
) -> None:
    report = {
        "claim_status": 7,
        "runtime_metadata": [],
        "timed_runs": "five",
        "p95_ms": "slow",
    }
    report_sha256 = _write_json(tmp_path / "report.json", report)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("report_metric must be a finite number" in error for error in result.errors)
    assert any("report_sample_count must be a positive integer" in error for error in result.errors)
    assert any("sample_count does not match report" in error for error in result.errors)
    assert any("claim_status must be a string" in error for error in result.errors)
    assert any("hardware context must be an object" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_missing_dotted_paths(
    tmp_path: Path,
) -> None:
    report = {"runtime_metadata": {"machine": "", "rt_kernel": ""}, "claim_status": "local evidence only"}
    report_sha256 = _write_json(tmp_path / "report.json", report)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        _manifest(
            report_sha256,
            metric_path="bad..path",
            sample_count_path="missing.count",
            claim_status_path="missing.claim",
            required_claim_substring="not a real-time control-loop guarantee",
            hardware_context_path="missing.hardware",
        ),
    )

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("metric_path missing" in error for error in result.errors)
    assert any("sample_count_path missing" in error for error in result.errors)
    assert any("claim_status_path missing" in error for error in result.errors)
    assert any("hardware context path missing" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_sample_count_below_minimum(
    tmp_path: Path,
) -> None:
    report_sha256 = _write_json(tmp_path / "report.json", _report())
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256, min_sample_count=6))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("sample_count below min_sample_count" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_hardware_context_without_identity(
    tmp_path: Path,
) -> None:
    report = {**_report(), "runtime_metadata": {"machine": "", "platform": "", "rt_kernel": ""}}
    report_sha256 = _write_json(tmp_path / "report.json", report)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _manifest(report_sha256))

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("hardware context must include machine" in error for error in result.errors)
    assert any("hardware context must include platform or rt_kernel" in error for error in result.errors)


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


def test_benchmark_regression_gate_rejects_duplicate_benchmark_id(
    tmp_path: Path,
) -> None:
    report_sha256 = _write_json(tmp_path / "report.json", _report())
    manifest_payload = _manifest(report_sha256)
    entries = manifest_payload["entries"]
    assert isinstance(entries, list)
    entries.append(dict(entries[0]))
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, manifest_payload)

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("duplicate benchmark_id" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_bad_manifest_shape(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "wrong",
            "gate_set_id": "shape-test",
            "claim_boundary": "unbounded evidence",
            "generated_utc": "",
            "entries": ["bad-entry"],
        },
    )

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("schema_version must be" in error for error in result.errors)
    assert any("claim_boundary must be bounded" in error for error in result.errors)
    assert any("generated_utc must be a non-empty string" in error for error in result.errors)
    assert any("entries[0] must be an object" in error for error in result.errors)
    assert any("no benchmark gates admitted" in error for error in result.errors)


def test_benchmark_regression_gate_rejects_empty_entries(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "gate_set_id": "empty-test",
            "claim_boundary": "bounded evidence",
            "generated_utc": "2026-06-02T00:00:00Z",
            "entries": [],
        },
    )

    result = validate_benchmark_regression_gates(manifest)

    assert result.status == "fail"
    assert any("entries must be a non-empty list" in error for error in result.errors)


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
