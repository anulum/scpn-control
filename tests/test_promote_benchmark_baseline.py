# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Benchmark baseline promotion tests
"""Exercise explicit baseline promotion from immutable run evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import tools.promote_benchmark_baseline as promoter
from scpn_control.benchmark_records import BenchmarkOutput, BenchmarkRun
from tools.promote_benchmark_baseline import BASELINE_SCHEMA, PROMOTION_SCHEMA, promote


def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scpn-control.benchmark-regression.v1",
        "generated_utc": "2026-08-28T00:00:00Z",
        "evidence_class": "local_regression",
        "production_claim_allowed": False,
        "provenance": {"commit": "abc123", "cpu_model": "test-cpu"},
        "settings": {"steps": 5, "warmup": 1},
        "benchmarks": {
            "capacitor_bank_discharge": {
                "languages": {"python": {"p50_us": 10.0, "p95_us": 11.0, "p99_us": 12.0, "throughput_ops_s": 100.0}}
            }
        },
    }
    payload["payload_sha256"] = _digest(payload)
    return payload


def _immutable_report(repository: Path) -> tuple[Path, str]:
    output = repository / "reports" / "suite.json"
    output.parent.mkdir(parents=True)
    run = BenchmarkRun.begin(
        repository_root=repository,
        records_root=repository / "benchmark-records",
        family="polyglot-suite",
        outputs=[BenchmarkOutput("report", output)],
        command=["python", "tools/run_benchmark_suite.py"],
        campaign_id="promotion-source",
    )
    output.write_text(json.dumps(_report(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = run.finish(exit_code=0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path, str(manifest["artifacts"][0]["sha256"])


def test_explicit_promotion_binds_source_and_archives_previous_baseline(tmp_path: Path) -> None:
    """Promotion verifies the run digest and retains the superseded baseline."""
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest, source_digest = _immutable_report(repository)
    baseline = repository / "benchmarks" / "baselines" / "capacitor_bank.json"
    baseline.parent.mkdir(parents=True)
    previous = b'{"old":true}\n'
    baseline.write_bytes(previous)

    receipt_path = promote(
        source_manifest=manifest,
        artifact_role="report",
        expected_source_sha256=source_digest,
        baseline_path=baseline,
        suite="capacitor-bank",
        authority_ref="owner-review-2026-08-28",
        hardware_compatibility="matched",
        promotion_id="promotion-one",
        repository_root=repository,
    )

    promoted = json.loads(baseline.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    previous_digest = hashlib.sha256(previous).hexdigest()
    archived = (
        repository / "benchmarks" / "baseline_history" / "capacitor-bank" / "baselines" / f"{previous_digest}.json"
    )
    assert promoted["schema_version"] == BASELINE_SCHEMA
    assert promoted["promotion"]["source_artifact_sha256"] == source_digest
    assert promoted["promotion"]["authority_ref"] == "owner-review-2026-08-28"
    assert receipt["schema_version"] == PROMOTION_SCHEMA
    assert receipt["source_artifact_sha256"] == source_digest
    assert archived.read_bytes() == previous


def test_digest_mismatch_cannot_change_baseline(tmp_path: Path) -> None:
    """A caller must present the exact immutable source artifact digest."""
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest, _ = _immutable_report(repository)
    baseline = repository / "benchmarks" / "baselines" / "capacitor_bank.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("unchanged", encoding="utf-8")

    with pytest.raises(ValueError, match="expected source digest"):
        promote(
            source_manifest=manifest,
            artifact_role="report",
            expected_source_sha256="0" * 64,
            baseline_path=baseline,
            suite="capacitor-bank",
            authority_ref="owner-review",
            hardware_compatibility="matched",
            promotion_id="rejected",
            repository_root=repository,
        )
    assert baseline.read_text(encoding="utf-8") == "unchanged"
    assert not (repository / "benchmarks" / "baseline_history").exists()


def test_promotion_identifier_collision_cannot_reapply(tmp_path: Path) -> None:
    """Promotion receipts are immutable and their identifiers cannot be reused."""
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest, source_digest = _immutable_report(repository)
    baseline = repository / "benchmarks" / "baselines" / "capacitor_bank.json"
    promote(
        source_manifest=manifest,
        artifact_role="report",
        expected_source_sha256=source_digest,
        baseline_path=baseline,
        suite="capacitor-bank",
        authority_ref="owner-review",
        hardware_compatibility="initial-baseline",
        promotion_id="one-use-id",
        repository_root=repository,
    )
    first_bytes = baseline.read_bytes()

    with pytest.raises(FileExistsError, match="promotion identifier"):
        promote(
            source_manifest=manifest,
            artifact_role="report",
            expected_source_sha256=source_digest,
            baseline_path=baseline,
            suite="capacitor-bank",
            authority_ref="owner-review",
            hardware_compatibility="initial-baseline",
            promotion_id="one-use-id",
            repository_root=repository,
        )
    assert baseline.read_bytes() == first_bytes


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"suite": "bad suite"}, "filesystem-safe"),
        ({"promotion_id": "bad/promotion"}, "filesystem-safe"),
        ({"authority_ref": "  "}, "authority reference"),
        ({"hardware_compatibility": "guessed"}, "hardware compatibility"),
        ({"expected_source_sha256": "ABC"}, "lowercase SHA-256"),
    ],
)
def test_promotion_api_rejects_invalid_authority_metadata(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    """Direct API use enforces the same promotion metadata contract as the CLI."""
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest, source_digest = _immutable_report(repository)
    arguments: dict[str, Any] = {
        "source_manifest": manifest,
        "artifact_role": "report",
        "expected_source_sha256": source_digest,
        "baseline_path": repository / "benchmarks" / "baselines" / "baseline.json",
        "suite": "suite",
        "authority_ref": "review",
        "hardware_compatibility": "matched",
        "promotion_id": "promotion",
        "repository_root": repository,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        promote(**arguments)


def test_repository_paths_and_manifest_shape_fail_closed(tmp_path: Path) -> None:
    """Promotion accepts only repository-local successful immutable manifests."""
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(ValueError, match="inside the repository"):
        promoter._repository_path(tmp_path / "outside.json", "source", repository)

    invalid_location = repository / "manifest.json"
    invalid_location.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable runs"):
        promoter._artifact_from_manifest(invalid_location, "report", repository)

    invalid_run = repository / "records" / "runs" / "suite" / "run" / "manifest.json"
    invalid_run.parent.mkdir(parents=True)
    invalid_run.write_text(json.dumps({"schema_version": "wrong", "status": "failed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="successful benchmark run"):
        promoter._artifact_from_manifest(invalid_run, "report", repository)


def test_manifest_digest_role_kind_path_and_artifact_digest_are_verified(tmp_path: Path) -> None:
    """Every manifest-to-artifact binding fails closed under tampering."""
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest_path, _ = _immutable_report(repository)
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tampered = dict(original_manifest)
    tampered["campaign_id"] = "edited"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest payload digest"):
        promoter._artifact_from_manifest(manifest_path, "report", repository)

    manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        promoter._artifact_from_manifest(manifest_path, "missing-role", repository)

    def _write_manifest(manifest: dict[str, object]) -> None:
        unsigned = {key: value for key, value in manifest.items() if key != "payload_sha256"}
        manifest["payload_sha256"] = hashlib.sha256(
            (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    wrong_kind = json.loads(json.dumps(original_manifest))
    wrong_kind["artifacts"][0]["kind"] = "directory"
    _write_manifest(wrong_kind)
    with pytest.raises(ValueError, match="must be a file"):
        promoter._artifact_from_manifest(manifest_path, "report", repository)

    escaped = json.loads(json.dumps(original_manifest))
    escaped["artifacts"][0]["immutable_path"] = str(repository / "outside.json")
    _write_manifest(escaped)
    with pytest.raises(ValueError, match="escapes"):
        promoter._artifact_from_manifest(manifest_path, "report", repository)

    _write_manifest(original_manifest)
    artifact_path = repository / original_manifest["artifacts"][0]["immutable_path"]
    artifact_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact digest"):
        promoter._artifact_from_manifest(manifest_path, "report", repository)


@pytest.mark.parametrize(
    "report, message",
    [
        ({"schema_version": "wrong"}, "schema"),
        ({"schema_version": promoter.REPORT_SCHEMA, "benchmarks": {}}, "no benchmark metrics"),
        (
            {"schema_version": promoter.REPORT_SCHEMA, "benchmarks": {"suite": {}}, "payload_sha256": "0" * 64},
            "payload digest",
        ),
    ],
)
def test_report_validation_rejects_schema_metrics_and_digest(
    tmp_path: Path, report: dict[str, object], message: str
) -> None:
    """Only self-digesting regression reports with metrics can be promoted."""
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        promoter._validated_report(path)


def test_existing_identical_baseline_archive_is_reused(tmp_path: Path) -> None:
    """A digest-identical previous baseline archive is never overwritten."""
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest, source_digest = _immutable_report(repository)
    baseline = repository / "benchmarks" / "baselines" / "baseline.json"
    baseline.parent.mkdir(parents=True)
    previous = b'{"existing":true}\n'
    baseline.write_bytes(previous)
    digest = hashlib.sha256(previous).hexdigest()
    archive = repository / "benchmarks" / "baseline_history" / "suite" / "baselines" / f"{digest}.json"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(previous)

    promote(
        source_manifest=manifest,
        artifact_role="report",
        expected_source_sha256=source_digest,
        baseline_path=baseline,
        suite="suite",
        authority_ref="review",
        hardware_compatibility="matched",
        promotion_id="reuse-archive",
        repository_root=repository,
    )
    assert archive.read_bytes() == previous


def test_promotion_cli_reports_success_and_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The imported CLI emits a receipt on success and a bounded error on failure."""
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest, source_digest = _immutable_report(repository)
    arguments = [
        "--repository-root",
        str(repository),
        "--source-manifest",
        str(manifest),
        "--expected-source-sha256",
        source_digest,
        "--baseline",
        "benchmarks/baselines/baseline.json",
        "--suite",
        "suite",
        "--authority-ref",
        "review",
        "--hardware-compatibility",
        "matched",
        "--promotion-id",
        "cli-promotion",
    ]
    assert promoter.main(arguments) == 0
    assert "baseline promotion receipt" in capsys.readouterr().out
    assert promoter.main(arguments) == 1
    assert "baseline promotion FAILED" in capsys.readouterr().err
