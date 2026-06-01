# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear GK Cyclone validation tests

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from validation import gk_nonlinear_cyclone as cyclone
from validation.gk_nonlinear_cyclone import assess_cbc_saturation_evidence


def _result(*, chi_i_gb: float, q_i_t: list[float], converged: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        chi_i_gB=chi_i_gb,
        chi_e=1.0,
        Q_i_t=np.asarray(q_i_t, dtype=np.float64),
        converged=converged,
    )


def test_repo_src_bootstrap_supports_direct_script_execution(monkeypatch) -> None:
    repo_src = str(Path(cyclone.__file__).resolve().parents[1] / "src")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != repo_src])

    cyclone.ensure_repo_src_on_path()

    assert sys.path[0] == repo_src


def test_cbc_saturation_evidence_accepts_long_flat_reference_range_trace() -> None:
    cfg = SimpleNamespace(n_steps=2500, save_interval=100)

    report = assess_cbc_saturation_evidence(
        _result(chi_i_gb=2.4, q_i_t=[2.2, 2.35, 2.42, 2.39, 2.41]),
        cfg,
    )

    assert report["passed"] is True
    assert report["chi_i_gB"] == 2.4
    assert report["tail_relative_drift"] < 0.1
    assert report["reasons"] == []


def test_cbc_saturation_evidence_rejects_short_trace_even_when_finite() -> None:
    cfg = SimpleNamespace(n_steps=200, save_interval=20)

    report = assess_cbc_saturation_evidence(
        _result(chi_i_gb=2.4, q_i_t=[2.3, 2.4, 2.5, 2.45]),
        cfg,
    )

    assert report["passed"] is False
    assert "n_steps below saturation evidence minimum" in report["reasons"]


def test_cbc_saturation_evidence_rejects_out_of_reference_band() -> None:
    cfg = SimpleNamespace(n_steps=2500, save_interval=100)

    report = assess_cbc_saturation_evidence(
        _result(chi_i_gb=0.2, q_i_t=[0.18, 0.19, 0.2, 0.21, 0.2]),
        cfg,
    )

    assert report["passed"] is False
    assert "chi_i_gB outside CBC reference band" in report["reasons"]


def test_cbc_saturation_evidence_rejects_drifting_tail() -> None:
    cfg = SimpleNamespace(n_steps=2500, save_interval=100)

    report = assess_cbc_saturation_evidence(
        _result(chi_i_gb=2.5, q_i_t=[1.0, 1.4, 1.8, 2.4, 3.2]),
        cfg,
    )

    assert report["passed"] is False
    assert "tail heat-flux drift exceeds saturation threshold" in report["reasons"]


def test_cbc_report_keeps_diagnostics_separate_from_saturation_admission() -> None:
    results = [
        {"test": "V1_linear_recovery", "passed": True},
        {"test": "V2_energy_conservation", "passed": True},
        {"test": "V3_zonal_flow_generation", "passed": True},
        {
            "test": "V4_cbc_saturated",
            "passed": False,
            "saturation_evidence": {
                "passed": False,
                "reasons": ["n_steps below saturation evidence minimum"],
                "chi_i_gB": 0.0,
                "n_steps": 200,
            },
        },
    ]

    report = cyclone.build_cbc_report(results)

    assert report["schema_version"] == cyclone.REPORT_SCHEMA_VERSION
    assert report["diagnostic_checks_passed"] is True
    assert report["saturation_claim_admitted"] is False
    assert report["claim_status"] == "diagnostic evidence only; saturated nonlinear CBC chi_i claim remains blocked"
    assert report["blocked_reasons"] == ["n_steps below saturation evidence minimum"]
    assert cyclone.verify_payload_digest(report) is True


def test_cbc_report_admits_saturation_only_when_v4_evidence_passes() -> None:
    results = [
        {"test": "V1_linear_recovery", "passed": True},
        {"test": "V2_energy_conservation", "passed": True},
        {"test": "V3_zonal_flow_generation", "passed": True},
        {
            "test": "V4_cbc_saturated",
            "passed": True,
            "saturation_evidence": {
                "passed": True,
                "reasons": [],
                "chi_i_gB": 2.2,
                "n_steps": 2500,
            },
        },
    ]

    report = cyclone.build_cbc_report(results)

    assert report["diagnostic_checks_passed"] is True
    assert report["saturation_claim_admitted"] is True
    assert report["blocked_reasons"] == []
    assert report["claim_status"] == "nonlinear CBC saturation evidence admitted"


def test_cbc_report_digest_rejects_tampering() -> None:
    report = cyclone.build_cbc_report(
        [
            {"test": "V1_linear_recovery", "passed": True},
            {
                "test": "V4_cbc_saturated",
                "passed": False,
                "saturation_evidence": {"passed": False, "reasons": ["short"], "chi_i_gB": 0.0},
            },
        ]
    )
    report["results"][0]["passed"] = False

    assert cyclone.verify_payload_digest(report) is False
