# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Generated traceability freshness check tests

from __future__ import annotations

from pathlib import Path

from tools.check_generated_traceability import ROOT, generated_traceability_is_current, main


def test_generated_traceability_check_passes_for_repository_state(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "Generated traceability documentation is current:" in output
    assert "validation/physics_traceability.json" in output
    assert "docs/physics_traceability.md" in output


def test_generated_traceability_check_detects_stale_report(tmp_path: Path) -> None:
    stale_report = tmp_path / "physics_traceability.md"
    stale_report.write_text("stale report\n", encoding="utf-8")

    assert not generated_traceability_is_current(
        ROOT / "validation" / "physics_traceability.json",
        stale_report,
    )


def test_generated_traceability_cli_accepts_explicit_paths(tmp_path: Path, capsys) -> None:
    registry = ROOT / "validation" / "physics_traceability.json"
    report = ROOT / "docs" / "physics_traceability.md"

    assert main(["--registry", str(registry), "--report", str(report)]) == 0
    output = capsys.readouterr().out
    assert "Generated traceability documentation is current:" in output
    assert str(registry) in output
    assert str(report) in output


def test_generated_traceability_cli_reports_stale_explicit_path(tmp_path: Path, capsys) -> None:
    registry = ROOT / "validation" / "physics_traceability.json"
    stale_report = tmp_path / "physics_traceability.md"
    stale_report.write_text("stale report\n", encoding="utf-8")

    assert main(["--registry", str(registry), "--report", str(stale_report)]) == 1
    assert f"{stale_report} is stale" in capsys.readouterr().err


def test_generated_traceability_cli_reports_missing_explicit_path(tmp_path: Path, capsys) -> None:
    registry = ROOT / "validation" / "physics_traceability.json"
    missing_report = tmp_path / "missing_physics_traceability.md"

    assert main(["--registry", str(registry), "--report", str(missing_report)]) == 1
    assert f"{missing_report} is missing" in capsys.readouterr().err


def test_generated_traceability_cli_reports_missing_registry_path(tmp_path: Path, capsys) -> None:
    missing_registry = tmp_path / "missing_physics_traceability.json"
    report = ROOT / "docs" / "physics_traceability.md"

    assert main(["--registry", str(missing_registry), "--report", str(report)]) == 1
    assert f"{missing_registry} is missing" in capsys.readouterr().err
