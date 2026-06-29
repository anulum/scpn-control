# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Validation report freshness tests.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pytest import CaptureFixture

from tools.validation_report_freshness import ROOT, build_validation_report_freshness_matrix, main


def test_validation_report_freshness_inventory_finds_live_stale_reports() -> None:
    matrix = build_validation_report_freshness_matrix(
        ROOT / "validation" / "reports",
        as_of=datetime(2026, 6, 29, tzinfo=timezone.utc),
        max_age_days=21,
    )

    stale_paths = {report.path.relative_to(ROOT).as_posix() for report in matrix.stale_reports}
    assert len(matrix.reports) >= 118
    assert "validation/reports/pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json" in stale_paths
    assert matrix.source_counts["generated_at_utc"] >= 1
    assert matrix.bucket_counts["rerunnable_local"] > 0
    assert matrix.bucket_counts["external_artifact_blocked"] > 0
    assert matrix.bucket_counts["historical_only"] > 0


def test_validation_report_freshness_can_render_markdown_summary() -> None:
    matrix = build_validation_report_freshness_matrix(
        ROOT / "validation" / "reports",
        as_of=datetime(2026, 6, 29, tzinfo=timezone.utc),
        max_age_days=21,
    )
    rendered = matrix.to_markdown()

    assert "# SCPN Control Validation Report Freshness" in rendered
    assert "validation/reports/pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json" in rendered
    assert "Reports missing claim-boundary signal" in rendered
    assert "## Classification Buckets" in rendered
    assert "rerunnable_local" in rendered


def test_validation_report_freshness_cli_writes_json_and_markdown(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    output_json = tmp_path / "freshness.json"
    output_md = tmp_path / "freshness.md"

    assert (
        main(
            [
                "--as-of",
                "2026-06-29T00:00:00Z",
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
    assert payload["schema_version"] == "scpn-control.validation-report-freshness.v1"
    assert payload["summary"]["stale_report_count"] > 0
    assert payload["summary"]["bucket_counts"]["rerunnable_local"] > 0
    assert payload["stale_reports"][0]["classification"]["bucket"] in {
        "rerunnable_local",
        "external_artifact_blocked",
        "historical_only",
    }
    assert "Stale Reports" in output_md.read_text(encoding="utf-8")


def test_validation_report_freshness_classifies_known_live_reports() -> None:
    matrix = build_validation_report_freshness_matrix(
        ROOT / "validation" / "reports",
        as_of=datetime(2026, 6, 29, tzinfo=timezone.utc),
        max_age_days=21,
    )
    reports = {report.path.relative_to(ROOT).as_posix(): report for report in matrix.stale_reports}

    assert (
        reports["validation/reports/pulsed_scenario_scheduler_v2_soft_isolated_20260604T113618Z.json"]
        .classification.bucket
        == "rerunnable_local"
    )
    assert (
        reports["validation/reports/gk_interface_artifacts.json"].classification.bucket
        == "external_artifact_blocked"
    )
    assert (
        reports["validation/reports/mast_efm_neural_equilibrium_dataset.json"].classification.bucket
        == "historical_only"
    )


def test_validation_report_freshness_cli_can_fail_on_stale_reports(capsys: CaptureFixture[str]) -> None:
    assert main(["--as-of", "2026-06-29T00:00:00Z", "--max-age-days", "21", "--fail-on-stale"]) == 1
    assert "Stale validation reports detected:" in capsys.readouterr().err


def test_validation_report_freshness_cli_accepts_current_window(capsys: CaptureFixture[str]) -> None:
    assert main(["--as-of", "2026-06-06T00:00:00Z", "--max-age-days", "10000", "--fail-on-stale"]) == 0
    assert "Validation report freshness:" in capsys.readouterr().out


def test_validation_report_freshness_docs_include_entrypoint() -> None:
    validation_docs = (ROOT / "docs" / "validation.md").read_text(encoding="utf-8")

    assert "scpn-validation-report-freshness --output-json artifacts/validation_report_freshness.json" in validation_docs
    assert "scpn-control.validation-report-freshness.v1" in validation_docs
