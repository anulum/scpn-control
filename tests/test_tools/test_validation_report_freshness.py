# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Validation report freshness tests.
"""Regression tests for digest-bound validation report lifecycle metadata."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import pytest
from pytest import CaptureFixture

from tools.validation_report_freshness import (
    DEFAULT_LIFECYCLE_REGISTRY,
    ROOT,
    LifecycleRegistryError,
    build_validation_report_freshness_matrix,
    main,
    parse_datetime,
)

AUDIT_AS_OF = datetime(2026, 8, 30, 5, 31, 18, tzinfo=timezone.utc)


def test_validation_report_freshness_inventory_finds_live_stale_reports() -> None:
    """The frozen audit corpus has exact counts and no missing registry boundary."""
    matrix = build_validation_report_freshness_matrix(
        ROOT / "validation" / "reports",
        as_of=AUDIT_AS_OF,
        max_age_days=21,
    )

    stale_paths = {report.path.relative_to(ROOT).as_posix() for report in matrix.stale_reports}
    assert len(matrix.reports) == 127
    assert len(matrix.stale_reports) == 114
    assert "validation/reports/pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json" not in stale_paths
    assert matrix.source_counts["lifecycle_refresh"] == 10
    assert matrix.source_counts["generated_at_utc"] >= 1
    assert matrix.bucket_counts == {
        "external_artifact_blocked": 75,
        "historical_only": 41,
        "rerunnable_local": 11,
    }
    assert matrix.claim_boundary_missing == 0
    assert matrix.source_claim_boundary_missing == 40
    assert matrix.current_admitted_reports == ()


def test_validation_report_freshness_can_render_markdown_summary() -> None:
    """The human-readable inventory exposes lifecycle and source limitations."""
    matrix = build_validation_report_freshness_matrix(
        ROOT / "validation" / "reports",
        as_of=AUDIT_AS_OF,
        max_age_days=21,
    )
    rendered = matrix.to_markdown()

    assert "# SCPN Control Validation Report Freshness" in rendered
    assert "validation/reports/pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json" in rendered
    assert "Reports missing registry claim boundary" in rendered
    assert "Immutable source reports missing embedded claim metadata" in rendered
    assert "## Classification Buckets" in rendered
    assert "rerunnable_local" in rendered


def test_markdown_reports_absent_local_lineages(tmp_path: Path) -> None:
    """An external-only registry renders the empty local-lineage state."""
    reports_root, registry_path, _ = _write_single_report_registry(tmp_path)

    rendered = build_validation_report_freshness_matrix(
        reports_root,
        as_of=AUDIT_AS_OF,
        max_age_days=21,
        registry_path=registry_path,
    ).to_markdown()

    assert "No rerunnable-local validation report lineages were found." in rendered


def test_validation_report_freshness_cli_writes_json_and_markdown(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The CLI writes both supported deterministic output forms."""
    output_json = tmp_path / "freshness.json"
    output_md = tmp_path / "freshness.md"

    assert (
        main(
            [
                "--as-of",
                "2026-08-30T05:31:18Z",
                "--max-age-days",
                "21",
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )

    assert "Validation report freshness:" in capsys.readouterr().out
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "scpn-control.validation-report-freshness.v2"
    assert payload["summary"]["stale_report_count"] > 0
    assert payload["summary"]["bucket_counts"]["rerunnable_local"] > 0
    assert payload["stale_reports"][0]["classification"]["bucket"] in {
        "rerunnable_local",
        "external_artifact_blocked",
        "historical_only",
    }
    assert "Stale Reports" in output_md.read_text(encoding="utf-8")


def test_validation_report_freshness_classifies_known_live_reports() -> None:
    """Known reports retain their audited declarative lifecycle buckets."""
    matrix = build_validation_report_freshness_matrix(
        ROOT / "validation" / "reports",
        as_of=AUDIT_AS_OF,
        max_age_days=21,
    )
    reports = {report.path.relative_to(ROOT).as_posix(): report for report in matrix.reports}

    assert (
        reports[
            "validation/reports/pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json"
        ].classification.bucket
        == "rerunnable_local"
    )
    assert (
        reports["validation/reports/gk_interface_artifacts.json"].classification.bucket == "external_artifact_blocked"
    )
    assert (
        reports["validation/reports/mast_efm_neural_equilibrium_dataset.json"].classification.bucket
        == "historical_only"
    )


def test_validation_report_freshness_exposes_rerunnable_local_refresh_plan() -> None:
    """Every rerunnable local report exposes a bounded refresh plan."""
    matrix = build_validation_report_freshness_matrix(
        ROOT / "validation" / "reports",
        as_of=AUDIT_AS_OF,
        max_age_days=21,
    )
    refresh_plans = {
        report.path.relative_to(ROOT).as_posix(): report.refresh_plan for report in matrix.rerunnable_local_reports
    }

    assert set(refresh_plans) == {
        "validation/reports/aer_observation_admission_20260604T162953Z.json",
        "validation/reports/aer_observation_soft_isolated_20260604T121529Z.json",
        "validation/reports/differentiable_scenario_readiness.json",
        "validation/reports/e2e_control_latency.json",
        "validation/reports/e2e_control_latency_hardening_20260603T0010.json",
        "validation/reports/grad_shafranov_solovev.json",
        "validation/reports/h_infinity_control.json",
        "validation/reports/pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json",
        "validation/reports/runtime_admission_release_20260605T000000Z.json",
        "validation/reports/runtime_admission_soft_isolated_20260604T132240Z.json",
        "validation/reports/static_mu_analysis_claims.json",
    }
    assert (
        refresh_plans["validation/reports/aer_observation_admission_20260604T162953Z.json"].status
        == "ready_exact_command"
    )
    assert (
        "bench_aer_observation.py"
        in refresh_plans["validation/reports/aer_observation_admission_20260604T162953Z.json"].commands[0]
    )
    assert (
        refresh_plans["validation/reports/e2e_control_latency_hardening_20260603T0010.json"].status
        == "ready_exact_command"
    )
    assert (
        "benchmarks/e2e_control_latency.py"
        in refresh_plans["validation/reports/e2e_control_latency_hardening_20260603T0010.json"].commands[0]
    )
    assert (
        refresh_plans["validation/reports/pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json"].status
        == "ready_exact_command"
    )


def test_validation_report_freshness_cli_can_fail_on_stale_reports(capsys: CaptureFixture[str]) -> None:
    """Strict freshness mode rejects the stale audited corpus."""
    assert main(["--as-of", "2026-08-30T05:31:18Z", "--max-age-days", "21", "--fail-on-stale"]) == 1
    assert "Stale validation reports detected:" in capsys.readouterr().err


def test_validation_report_freshness_cli_accepts_current_window(capsys: CaptureFixture[str]) -> None:
    """A caller-selected broad window can make freshness advisory-only."""
    assert main(["--as-of", "2026-08-30T05:31:18Z", "--max-age-days", "10000", "--fail-on-stale"]) == 0
    assert "Validation report freshness:" in capsys.readouterr().out


def test_validation_report_freshness_docs_include_entrypoint() -> None:
    """Public validation documentation names the current schemas and CLI."""
    validation_docs = (ROOT / "docs" / "validation.md").read_text(encoding="utf-8")

    assert (
        "python tools/validation_report_freshness.py "
        "--output-json artifacts/validation_report_freshness.json" in validation_docs
    )
    assert "scpn-control.validation-report-freshness.v2" in validation_docs
    assert "Rerunnable local reports also include a refresh plan" in validation_docs
    assert "scpn-control.validation-report-refresh.v1" in validation_docs


def test_lifecycle_registry_is_digest_bound_and_complete() -> None:
    """The registry covers every audited report with explicit fail-closed state."""
    registry = json.loads(DEFAULT_LIFECYCLE_REGISTRY.read_text(encoding="utf-8"))

    assert registry["schema_version"] == "scpn-control.validation-report-lifecycle.v1"
    assert registry["expected_bucket_counts"] == {
        "external_artifact_blocked": 75,
        "historical_only": 41,
        "rerunnable_local": 11,
    }
    assert len(registry["reports"]) == 127
    assert len({report["path"] for report in registry["reports"]}) == 127
    assert {report["storage_class"] for report in registry["reports"]} == {
        "git_tracked",
        "owner_local_untracked",
    }
    assert sum(report["claim_boundary"]["current_evidence"] for report in registry["reports"]) == 11
    assert sum(report["refresh"]["status"] == "refreshed" for report in registry["reports"]) == 10
    assert all(not report["claim_boundary"]["public_claim_allowed"] for report in registry["reports"])


def _write_single_report_registry(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    source_registry = json.loads(DEFAULT_LIFECYCLE_REGISTRY.read_text(encoding="utf-8"))
    source_record = next(
        report
        for report in source_registry["reports"]
        if report["path"] == "validation/reports/gk_interface_artifacts.json"
    )
    reports_root = tmp_path / "validation" / "reports"
    report_path = reports_root / "gk_interface_artifacts.json"
    reports_root.mkdir(parents=True)
    report_path.write_bytes((ROOT / source_record["path"]).read_bytes())
    registry = deepcopy(source_registry)
    registry["expected_bucket_counts"] = {
        "external_artifact_blocked": 1,
        "historical_only": 0,
        "rerunnable_local": 0,
    }
    registry["reports"] = [deepcopy(source_record)]
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return reports_root, registry_path, registry


def test_lifecycle_registry_rejects_report_digest_drift(tmp_path: Path) -> None:
    """Report-byte changes cannot silently retain lifecycle admission metadata."""
    reports_root, registry_path, _ = _write_single_report_registry(tmp_path)
    (reports_root / "gk_interface_artifacts.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(LifecycleRegistryError, match="report digest drift"):
        build_validation_report_freshness_matrix(
            reports_root,
            as_of=AUDIT_AS_OF,
            max_age_days=21,
            registry_path=registry_path,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report.__setitem__("evidence_class", "local_proxy"), "evidence-class promotion drift"),
        (
            lambda report: report["claim_boundary"].__setitem__("current_evidence", True),
            "blocked or historical report promoted",
        ),
        (lambda report: report.__setitem__("evidence_time_utc", "2027-01-01T00:00:00Z"), "future evidence"),
        (lambda report: report.__setitem__("unexpected", True), "fields drift"),
    ],
)
def test_lifecycle_registry_rejects_promotion_and_schema_drift(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    """Evidence, admission, timestamp, and field drift fail closed."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    report = cast(list[object], registry["reports"])[0]
    assert isinstance(report, dict)
    mutate(report)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(LifecycleRegistryError, match=message):
        build_validation_report_freshness_matrix(
            reports_root,
            as_of=AUDIT_AS_OF,
            max_age_days=21,
            registry_path=registry_path,
        )


def _persist_registry(registry_path: Path, registry: dict[str, object]) -> None:
    registry_path.write_text(json.dumps(registry), encoding="utf-8")


def _build_fixture(reports_root: Path, registry_path: Path, *, max_age_days: int = 21) -> None:
    build_validation_report_freshness_matrix(
        reports_root,
        as_of=AUDIT_AS_OF,
        max_age_days=max_age_days,
        registry_path=registry_path,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda registry: registry.__setitem__("schema_version", "v0"), "unsupported lifecycle"),
        (lambda registry: registry.__setitem__("inventory_as_of_utc", "2027-01-01T00:00:00Z"), "inventory timestamp"),
        (lambda registry: registry.__setitem__("freshness_max_age_days", 20), "audited 21-day"),
        (lambda registry: registry.__setitem__("reports_root", "reports"), "reports_root"),
        (lambda registry: registry.__setitem__("registry_source_commit", "bad"), "full Git SHA"),
        (lambda registry: registry.__setitem__("expected_bucket_counts", []), "must be an object"),
        (
            lambda registry: registry["expected_bucket_counts"].pop("historical_only"),
            "expected_bucket_counts fields drift",
        ),
        (
            lambda registry: registry["expected_bucket_counts"].__setitem__("historical_only", -1),
            "integer >= 0",
        ),
        (lambda registry: registry.__setitem__("reports", {}), "reports must be an array"),
        (lambda registry: registry["reports"].__setitem__(0, "bad"), r"reports\[0\] must be an object"),
        (lambda registry: registry.__setitem__("unexpected", True), "lifecycle registry fields drift"),
    ],
)
def test_lifecycle_registry_rejects_header_drift(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Registry identity, policy, counts, and container types are strict."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    mutate(cast(dict[str, Any], registry))
    _persist_registry(registry_path, registry)

    with pytest.raises(LifecycleRegistryError, match=message):
        _build_fixture(reports_root, registry_path)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("path",), "elsewhere/report.json", "JSON path below"),
        (("path",), "validation/reports/report.txt", "JSON path below"),
        (("path",), "validation/reports/../escape.json", "escapes"),
        (("path",), "validation/reports//absolute.json", "escapes"),
        (("path",), "validation/reports/./ambiguous.json", "escapes"),
        (("path",), "validation/reports/directory\\escape.json", "escapes"),
        (("storage_class",), "network", "unknown storage class"),
        (("report_sha256",), "bad", "lowercase SHA-256"),
        (("report_commit",), "bad", "lowercase full Git SHA"),
        (("lifecycle_bucket",), "unclassified", "unknown lifecycle bucket"),
        (("source_claim_boundary_present",), "yes", "must be a boolean"),
        (("claim_boundary",), [], "must be an object"),
        (("claim_boundary", "current_evidence"), "yes", "must be a boolean"),
        (("claim_boundary", "rationale"), "", "non-empty string"),
        (("refresh",), [], "must be an object"),
        (("refresh", "locally_rerunnable"), "yes", "must be a boolean"),
        (("refresh", "status"), "promoted", "refresh-status promotion drift"),
        (("refresh", "commands"), {}, "must be an array"),
        (("refresh", "commands"), [""], "non-empty string"),
        (("provenance",), [], "must be an object"),
        (("provenance", "source_commit"), 7, "invalid digest syntax"),
        (("provenance", "source_commit"), "bad", "invalid digest syntax"),
        (("provenance", "dependency_lock_sha256"), "bad", "invalid digest syntax"),
        (("provenance", "host_id"), 7, "string or null"),
        (("provenance", "host_load"), "busy", "object or null"),
        (("provenance", "samples"), True, "integer >= 1"),
        (("provenance", "samples"), 0, "integer >= 1"),
        (("provenance", "samples"), "many", "integer >= 1"),
        (("provenance", "artifacts"), {}, "must be an array"),
        (("provenance", "artifacts"), [""], "non-empty string"),
        (("provenance", "artifacts"), ["other.json"], "must include the report path"),
        (("provenance", "failures"), {}, "must be an array"),
        (("provenance", "failures"), [""], "non-empty string"),
    ],
)
def test_lifecycle_registry_rejects_record_type_and_value_drift(
    tmp_path: Path,
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    """Every lifecycle record field is validated before use."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    record = cast(dict[str, Any], cast(list[object], registry["reports"])[0])
    target = record
    for key in field_path[:-1]:
        target = cast(dict[str, Any], target[key])
    target[field_path[-1]] = value
    _persist_registry(registry_path, registry)

    with pytest.raises(LifecycleRegistryError, match=message):
        _build_fixture(reports_root, registry_path)


def test_lifecycle_registry_rejects_duplicate_extra_missing_and_count_drift(tmp_path: Path) -> None:
    """Registry coverage is bijective and audited bucket counts are immutable."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    reports = cast(list[object], registry["reports"])
    reports.append(deepcopy(reports[0]))
    _persist_registry(registry_path, registry)
    with pytest.raises(LifecycleRegistryError, match="duplicate lifecycle report path"):
        _build_fixture(reports_root, registry_path)

    reports.pop()
    (reports_root / "extra.json").write_text("{}\n", encoding="utf-8")
    _persist_registry(registry_path, registry)
    with pytest.raises(LifecycleRegistryError, match="coverage drift"):
        _build_fixture(reports_root, registry_path)

    (reports_root / "extra.json").unlink()
    (reports_root / "gk_interface_artifacts.json").unlink()
    with pytest.raises(LifecycleRegistryError, match="registered tracked report does not exist"):
        _build_fixture(reports_root, registry_path)

    source = ROOT / "validation/reports/gk_interface_artifacts.json"
    (reports_root / "gk_interface_artifacts.json").write_bytes(source.read_bytes())
    counts = cast(dict[str, int], registry["expected_bucket_counts"])
    counts["external_artifact_blocked"] = 0
    counts["historical_only"] = 1
    _persist_registry(registry_path, registry)
    with pytest.raises(LifecycleRegistryError, match="bucket-count drift"):
        _build_fixture(reports_root, registry_path)


def _make_refreshed_local(record: dict[str, Any], tmp_path: Path) -> None:
    artifact_path = tmp_path / "validation" / "report_refreshes" / "test.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text('{"schema_version": "test"}\n', encoding="utf-8")
    record["lifecycle_bucket"] = "rerunnable_local"
    record["evidence_class"] = "local_proxy"
    record["evidence_time_utc"] = "2026-08-26T00:00:00Z"
    record["refresh"] = {
        "locally_rerunnable": True,
        "status": "refreshed",
        "commands": ["python benchmark.py --json-out report.json"],
        "artifact_path": "validation/report_refreshes/test.json",
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "evidence_time_utc": "2026-08-26T00:00:00Z",
    }
    record["claim_boundary"] = {
        "current_evidence": True,
        "scientific_admission": True,
        "production_admission": False,
        "public_claim_allowed": True,
        "rationale": "Fresh local evidence admitted for its declared bounded claim only.",
    }
    record["provenance"] = {
        "source_commit": "a" * 40,
        "dependency_lock_sha256": "b" * 64,
        "host_id": "ws-aud002",
        "host_class": "workstation-x86_64",
        "host_load": {"load1": 2.0, "captured_at_utc": "2026-08-26T00:00:00Z"},
        "samples": 100,
        "repeats": 3,
        "warmup": 10,
        "artifacts": [record["path"]],
        "failures": [],
    }


def _make_pending_local(record: dict[str, Any], commands: list[str]) -> None:
    record["lifecycle_bucket"] = "rerunnable_local"
    record["evidence_class"] = "local_proxy"
    record["refresh"] = {
        "locally_rerunnable": True,
        "status": "pending_refresh",
        "commands": commands,
        "artifact_path": None,
        "artifact_sha256": None,
        "evidence_time_utc": None,
    }
    record["claim_boundary"] = {
        "current_evidence": False,
        "scientific_admission": False,
        "production_admission": False,
        "public_claim_allowed": False,
        "rationale": "Pending local refresh fixture.",
    }


@pytest.mark.parametrize(
    ("commands", "expected_rationale"),
    [
        (["python benchmark.py ..."], "abbreviated or partial"),
        ([], "No exact command metadata"),
    ],
)
def test_pending_local_refresh_plan_requires_reconstruction(
    tmp_path: Path,
    commands: list[str],
    expected_rationale: str,
) -> None:
    """Partial and absent legacy commands remain explicit reconstruction blockers."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    record = cast(dict[str, Any], cast(list[object], registry["reports"])[0])
    _make_pending_local(record, commands)
    registry["expected_bucket_counts"] = {
        "external_artifact_blocked": 0,
        "historical_only": 0,
        "rerunnable_local": 1,
    }
    _persist_registry(registry_path, registry)

    plan = (
        build_validation_report_freshness_matrix(
            reports_root,
            as_of=AUDIT_AS_OF,
            max_age_days=21,
            registry_path=registry_path,
        )
        .rerunnable_local_reports[0]
        .refresh_plan
    )

    assert plan.status == "manual_reconstruction_required"
    assert expected_rationale in plan.rationale


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["refresh"].__setitem__("artifact_path", "validation/report_refreshes/x.json"),
            "cannot bind a refresh artifact",
        ),
        (
            lambda report: report["refresh"].__setitem__("artifact_path", "validation/reports/x.json"),
            "must be below validation/report_refreshes",
        ),
        (
            lambda report: report["refresh"].__setitem__("artifact_path", "validation/report_refreshes/../escape.json"),
            "escapes the repository",
        ),
        (
            lambda report: report["refresh"].__setitem__("artifact_path", "validation/report_refreshes/missing.json"),
            "refresh artifact does not exist",
        ),
        (lambda report: report["refresh"].__setitem__("artifact_sha256", "bad"), "lowercase SHA-256"),
        (lambda report: report["refresh"].__setitem__("artifact_sha256", "0" * 64), "artifact digest drift"),
        (
            lambda report: report["refresh"].__setitem__("evidence_time_utc", "2027-01-01T00:00:00Z"),
            "future refresh evidence timestamp",
        ),
    ],
)
def test_refresh_artifact_binding_fails_closed(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Refresh state, location, digest, existence, and time are independently bound."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    record = cast(dict[str, Any], cast(list[object], registry["reports"])[0])
    if "cannot bind" not in message:
        _make_refreshed_local(record, tmp_path)
        registry["expected_bucket_counts"] = {
            "external_artifact_blocked": 0,
            "historical_only": 0,
            "rerunnable_local": 1,
        }
    mutate(record)
    _persist_registry(registry_path, registry)

    with pytest.raises(LifecycleRegistryError, match=message):
        _build_fixture(reports_root, registry_path)


def test_fresh_admitted_local_report_is_selectable(tmp_path: Path) -> None:
    """Only a fresh, fully provenanced, explicitly admitted report is selected."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    record = cast(dict[str, Any], cast(list[object], registry["reports"])[0])
    _make_refreshed_local(record, tmp_path)
    registry["expected_bucket_counts"] = {
        "external_artifact_blocked": 0,
        "historical_only": 0,
        "rerunnable_local": 1,
    }
    _persist_registry(registry_path, registry)

    matrix = build_validation_report_freshness_matrix(
        reports_root,
        as_of=AUDIT_AS_OF,
        max_age_days=21,
        registry_path=registry_path,
    )

    assert len(matrix.current_admitted_reports) == 1
    assert matrix.current_admitted_reports[0].lifecycle.production_admission is False
    rendered = matrix.to_markdown()
    assert "No stale validation report JSON artifacts were found." in rendered
    assert "validation/report_refreshes/test.json" in json.dumps(matrix.to_dict())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["refresh"].__setitem__("locally_rerunnable", False),
            "local-rerun permission drift",
        ),
        (
            lambda report: report["claim_boundary"].update({"current_evidence": False, "scientific_admission": True}),
            "admission requires current evidence",
        ),
        (
            lambda report: report["claim_boundary"].update(
                {"scientific_admission": False, "production_admission": True}
            ),
            "production admission requires scientific admission",
        ),
        (
            lambda report: report["claim_boundary"].update(
                {"scientific_admission": False, "public_claim_allowed": True}
            ),
            "public claim permission requires scientific admission",
        ),
        (
            lambda report: report["refresh"].__setitem__("evidence_time_utc", "2026-06-01T00:00:00Z"),
            "stale report marked as current evidence",
        ),
    ],
)
def test_lifecycle_registry_rejects_invalid_local_admission(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Local refresh cannot bypass freshness or hierarchical admissions."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    record = cast(dict[str, Any], cast(list[object], registry["reports"])[0])
    _make_refreshed_local(record, tmp_path)
    mutate(record)
    registry["expected_bucket_counts"] = {
        "external_artifact_blocked": 0,
        "historical_only": 0,
        "rerunnable_local": 1,
    }
    _persist_registry(registry_path, registry)

    with pytest.raises(LifecycleRegistryError, match=message):
        _build_fixture(reports_root, registry_path)


@pytest.mark.parametrize(
    ("missing_key", "message"),
    [
        ("source_commit", "source_commit"),
        ("dependency_lock_sha256", "dependency_lock_sha256"),
        ("samples", "samples"),
        ("repeats", "repeats"),
        ("warmup", "warmup"),
        ("host_load", "host_load"),
    ],
)
def test_refreshed_report_requires_complete_provenance(
    tmp_path: Path,
    missing_key: str,
    message: str,
) -> None:
    """Each mandatory refreshed-report provenance field fails independently."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    record = cast(dict[str, Any], cast(list[object], registry["reports"])[0])
    _make_refreshed_local(record, tmp_path)
    provenance = cast(dict[str, Any], record["provenance"])
    provenance[missing_key] = None
    registry["expected_bucket_counts"] = {
        "external_artifact_blocked": 0,
        "historical_only": 0,
        "rerunnable_local": 1,
    }
    _persist_registry(registry_path, registry)

    with pytest.raises(LifecycleRegistryError, match=message):
        _build_fixture(reports_root, registry_path)


@pytest.mark.parametrize("host_key", ["host_id", "host_class"])
@pytest.mark.parametrize("host_value", [None, "unknown"])
def test_refreshed_report_rejects_ambiguous_host(
    tmp_path: Path,
    host_key: str,
    host_value: object,
) -> None:
    """Refreshed evidence requires concrete host identity and class."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    record = cast(dict[str, Any], cast(list[object], registry["reports"])[0])
    _make_refreshed_local(record, tmp_path)
    cast(dict[str, Any], record["provenance"])[host_key] = host_value
    registry["expected_bucket_counts"] = {
        "external_artifact_blocked": 0,
        "historical_only": 0,
        "rerunnable_local": 1,
    }
    _persist_registry(registry_path, registry)

    with pytest.raises(LifecycleRegistryError, match=f"ambiguous {host_key}"):
        _build_fixture(reports_root, registry_path)


def test_owner_local_report_can_be_absent_but_not_claim_a_commit(tmp_path: Path) -> None:
    """An owner-local frozen artifact remains indexed without clone-time fabrication."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    record = cast(dict[str, Any], cast(list[object], registry["reports"])[0])
    record["storage_class"] = "owner_local_untracked"
    record["report_commit"] = None
    (reports_root / "gk_interface_artifacts.json").unlink()
    _persist_registry(registry_path, registry)
    matrix = build_validation_report_freshness_matrix(
        reports_root,
        as_of=AUDIT_AS_OF,
        max_age_days=21,
        registry_path=registry_path,
    )
    assert len(matrix.reports) == 1

    record["report_commit"] = "a" * 40
    _persist_registry(registry_path, registry)
    with pytest.raises(LifecycleRegistryError, match="report_commit must be null"):
        _build_fixture(reports_root, registry_path)


def test_source_claim_boundary_drift_is_rejected(tmp_path: Path) -> None:
    """Embedded source-boundary metadata cannot drift behind a stable digest."""
    reports_root, registry_path, registry = _write_single_report_registry(tmp_path)
    record = cast(dict[str, Any], cast(list[object], registry["reports"])[0])
    record["source_claim_boundary_present"] = not record["source_claim_boundary_present"]
    _persist_registry(registry_path, registry)

    with pytest.raises(LifecycleRegistryError, match="source claim-boundary drift"):
        _build_fixture(reports_root, registry_path)


def test_datetime_and_build_input_validation(tmp_path: Path) -> None:
    """Timestamp and report-root inputs reject malformed and ambiguous values."""
    assert parse_datetime("20260604T121529Z") == datetime(2026, 6, 4, 12, 15, 29, tzinfo=timezone.utc)
    assert parse_datetime("2026-06-04T12:15:29") == datetime(2026, 6, 4, 12, 15, 29, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="non-empty"):
        parse_datetime(" ")
    with pytest.raises(ValueError, match="non-negative"):
        build_validation_report_freshness_matrix(tmp_path, as_of=AUDIT_AS_OF, max_age_days=-1)
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="does not exist"):
        build_validation_report_freshness_matrix(missing, as_of=AUDIT_AS_OF, max_age_days=21)
    regular_file = tmp_path / "file"
    regular_file.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        build_validation_report_freshness_matrix(regular_file, as_of=AUDIT_AS_OF, max_age_days=21)


def test_cli_stdout_modes_default_clock_and_error_path(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """JSON, Markdown, default-clock, and malformed-registry CLI paths are observable."""
    assert main(["--as-of", "2026-08-30T05:31:18Z", "--json-out"]) == 0
    assert '"schema_version": "scpn-control.validation-report-freshness.v2"' in capsys.readouterr().out
    assert main(["--as-of", "2026-08-30T05:31:18Z", "--markdown-out"]) == 0
    assert "# SCPN Control Validation Report Freshness" in capsys.readouterr().out
    assert main(["--max-age-days", "10000"]) == 0
    assert "Validation report freshness:" in capsys.readouterr().out

    bad_registry = tmp_path / "bad.json"
    bad_registry.write_text("[]\n", encoding="utf-8")
    assert main(["--registry", str(bad_registry)]) == 1
    assert "must contain a JSON object" in capsys.readouterr().err
