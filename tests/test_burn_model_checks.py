# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real burn diagnostics and analytical pressure units.
"""Exercise the public physical conversion, refusal and actual provider surfaces."""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from validation.burn_model_checks import BurnPoint, check_history, evidence_failures
from validation.burn_model_provider import SOURCE_HASHES, evaluate_burn_model

ROOT = Path(__file__).resolve().parents[1]


def point() -> BurnPoint:
    """Return a declared uniform input, with no fitted design beta target."""
    return BurnPoint(6.2, 2.0, 5.3, 15.0, 1.7, 1.01, 50.0)


def test_total_pressure_matches_independent_ideal_gas_definition() -> None:
    """Total electron+ion energy yields twice the electron-only pressure conversion."""
    p = point()
    temperature_j = 8e3 * 1.602e-19
    density = p.ne20 * 1e20
    stored_mj = 3 * density * temperature_j * p.volume_m3 / 1e6
    expected_pressure = 2 * density * temperature_j
    expected = 100 * 2 * (4 * math.pi * 1e-7) * expected_pressure * p.a_m / (p.b_t * p.ip_ma)
    assert p.beta_n(stored_mj) == pytest.approx(expected, rel=1e-14)
    assert replace(p, ip_ma=2 * p.ip_ma).beta_n(stored_mj) == pytest.approx(expected / 2)
    assert p.beta_n(0.0) == 0.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_invalid_physical_input_and_stored_energy_refuse(value: float) -> None:
    """The public physical conversion does not coerce invalid inputs into pressure."""
    with pytest.raises(ValueError):
        replace(point(), b_t=value)
    with pytest.raises(ValueError):
        point().beta_n(value)


def test_negative_impurity_radiation_is_not_a_supported_operating_point() -> None:
    """A subunit effective charge cannot silently produce negative line radiation."""
    with pytest.raises(ValueError, match="at least one"):
        replace(point(), z_eff=0.5)


@pytest.mark.parametrize("history", [[], [float("nan")] * 10, [1.0] * 9, [[1.0] * 10]])
def test_incomplete_or_nonfinite_history_cannot_claim_numerical_evidence(history: object) -> None:
    """Actual public history admission refuses malformed first-channel observations."""
    with pytest.raises(ValueError, match="finite samples"):
        check_history({"time_s": history}, point(), 1.0, 0.1)


@pytest.mark.parametrize("reference_name", ["iter_reference.json", "sparc_reference.json"])
def test_real_source_bound_provider_or_explicit_unavailability(reference_name: str) -> None:
    """Execute actual reference inputs; the hosted provider job requires real success.

    Dependency-light platform jobs exercise the explicit optional-provider
    refusal. The dedicated pinned-provider job sets the requirement environment
    variable and exercises original coarse/refined numerical histories.
    """
    reference = json.loads((ROOT / "validation/reference_data" / reference_name).read_text())
    try:
        beta, evidence = evaluate_burn_model(reference)
    except RuntimeError as exc:
        if os.environ.get("SCPN_REQUIRE_BURN_PROVIDER") == "1":
            raise
        assert "reference values cannot replace model predictions" in str(exc)
        return
    assert beta > 0 and math.isfinite(beta)
    assert evidence["model_provenance"]["source_files_sha256"] == SOURCE_HASHES
    assert evidence["physical_validation_claim"] is False
    assert evidence_failures(evidence["checks"]) == []
    coarse, refined = evidence["native_histories"]
    assert len(coarse["W_MJ"]) == 20_000 and len(refined["W_MJ"]) == 40_000
    assert evidence["W_MJ"] == refined["W_MJ"][-1]
    assert evidence["run_checks"][1]["beta_energy_time_s"] == 200.0
    assert evidence["run_checks"][1]["power_sample_time_s"] == pytest.approx(199.995)
    assert evidence["Q_unclipped"] == pytest.approx(refined["P_fus_MW"][-1] / reference["P_aux_MW"])


def test_reference_beta_is_never_a_model_input() -> None:
    """Invalid physical fields refuse before a reference beta could supply a prediction."""
    reference = json.loads((ROOT / "validation/reference_data/iter_reference.json").read_text())
    reference.update(beta_N=0.0, B_t_T=0.0)
    with pytest.raises(ValueError, match="B_t_T"):
        evaluate_burn_model(reference)
