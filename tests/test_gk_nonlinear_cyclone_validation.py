# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear GK Cyclone validation tests

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from validation.gk_nonlinear_cyclone import assess_cbc_saturation_evidence


def _result(*, chi_i_gb: float, q_i_t: list[float], converged: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        chi_i_gB=chi_i_gb,
        chi_e=1.0,
        Q_i_t=np.asarray(q_i_t, dtype=np.float64),
        converged=converged,
    )


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
