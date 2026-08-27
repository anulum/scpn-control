# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — CI RMSE gate tests.
"""Regressions for bounded reference-evidence consumption in the RMSE gate."""

from __future__ import annotations

import json
from pathlib import Path

from tools.ci_rmse_gate import main


def _write_reports(
    root: Path,
    *,
    disruption_fpr: float,
    rmse: dict[str, object] | None = None,
) -> None:
    """Write minimal RMSE and reference-evidence reports."""
    artifacts = root / "artifacts"
    artifacts.mkdir()
    (artifacts / "rmse_dashboard_ci.json").write_text(json.dumps(rmse or {}) + "\n", encoding="utf-8")
    reference = {
        "schema": "scpn-control.reference-evidence-validation.v1",
        "lanes": {"disruption_synthetic": {"false_positive_rate": disruption_fpr}},
    }
    (artifacts / "reference_evidence_validation.json").write_text(json.dumps(reference), encoding="utf-8")


def test_gate_reads_bounded_disruption_lane(tmp_path: Path, monkeypatch) -> None:
    """The current bounded report supplies the disruption regression metric."""
    _write_reports(tmp_path, disruption_fpr=0.0)
    monkeypatch.chdir(tmp_path)

    assert main() == 0


def test_gate_rejects_excessive_bounded_disruption_fpr(tmp_path: Path, monkeypatch) -> None:
    """A synthetic lane still fails its computational FPR regression guard."""
    _write_reports(tmp_path, disruption_fpr=0.5)
    monkeypatch.chdir(tmp_path)

    assert main() == 1


def test_gate_requires_rmse_dashboard(tmp_path: Path, monkeypatch) -> None:
    """The gate fails explicitly when its primary dashboard is absent."""
    monkeypatch.chdir(tmp_path)

    assert main() == 1


def test_gate_accepts_all_bounded_rmse_metrics_without_optional_reference(tmp_path: Path, monkeypatch) -> None:
    """Every dashboard metric passes below its declared regression threshold."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    report = {
        "confinement_itpa": {"tau_rmse_s": 0.1},
        "sparc_axis": {"axis_rmse_m": 1.0},
        "beta_iter_sparc": {"beta_n_rmse": 0.05},
    }
    (artifacts / "rmse_dashboard_ci.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main() == 0


def test_gate_reports_every_bounded_rmse_regression(tmp_path: Path, monkeypatch) -> None:
    """Independent dashboard regressions accumulate instead of short-circuiting."""
    _write_reports(
        tmp_path,
        disruption_fpr=0.5,
        rmse={
            "confinement_itpa": {"tau_rmse_s": 0.3},
            "sparc_axis": {"axis_rmse_m": 3.0},
            "beta_iter_sparc": {"beta_n_rmse": 0.2},
        },
    )
    monkeypatch.chdir(tmp_path)

    assert main() == 1
