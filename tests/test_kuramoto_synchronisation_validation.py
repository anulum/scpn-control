# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kuramoto synchronisation validation tests
"""Tests for the published-benchmark Kuramoto-Sakaguchi synchronisation validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from validation.validate_kuramoto_synchronisation import (
    KURAMOTO_SYNCHRONISATION_SCHEMA_VERSION,
    SynchronisationCase,
    SynchronisationValidationResult,
    build_evidence,
    critical_coupling,
    lorentzian_frequencies,
    simulate_steady_order_parameter,
    synchronised_order_parameter,
    validate_evidence_payload,
    validate_synchronisation,
)


@pytest.fixture(scope="module")
def result() -> SynchronisationValidationResult:
    """Run the default published-benchmark validation once for the module."""
    return validate_synchronisation()


def test_critical_coupling_matches_kuramoto_lorentzian() -> None:
    assert critical_coupling(1.0, 0.0) == pytest.approx(2.0)
    assert critical_coupling(2.5, 0.0) == pytest.approx(5.0)


def test_critical_coupling_sakaguchi_phase_lag_shift() -> None:
    expected = 2.0 / math.cos(math.pi / 4.0)
    assert critical_coupling(1.0, math.pi / 4.0) == pytest.approx(expected)
    assert critical_coupling(1.0, math.pi / 4.0) > critical_coupling(1.0, 0.0)


def test_critical_coupling_rejects_phase_lag_at_half_pi() -> None:
    with pytest.raises(ValueError, match="finite critical coupling"):
        critical_coupling(1.0, math.pi / 2.0)


def test_critical_coupling_rejects_nonpositive_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        critical_coupling(0.0, 0.0)


def test_synchronised_order_parameter_exact_branch() -> None:
    assert synchronised_order_parameter(4.0, 1.0, 0.0) == pytest.approx(math.sqrt(0.5))
    assert synchronised_order_parameter(8.0, 1.0, 0.0) == pytest.approx(math.sqrt(1.0 - 2.0 / 8.0))
    assert synchronised_order_parameter(2.0, 1.0, 0.0) == 0.0
    assert synchronised_order_parameter(1.0, 1.0, 0.0) == 0.0


def test_order_parameter_branch_matches_reference(result: SynchronisationValidationResult) -> None:
    assert result.branch_passed is True
    assert result.branch_max_abs_error <= result.branch_tolerance
    assert len(result.branch_records) == 3
    for record in result.branch_records:
        assert record.coupling_over_critical > 1.0
        assert record.within_tolerance is True
        assert abs(record.order_parameter_measured - record.order_parameter_reference) <= result.branch_tolerance


def test_subthreshold_population_is_incoherent(result: SynchronisationValidationResult) -> None:
    assert result.subthreshold_passed is True
    assert result.subthreshold_order_parameter < result.incoherent_ceiling


def test_sakaguchi_onset_shift_is_observed(result: SynchronisationValidationResult) -> None:
    assert result.onset_shift_passed is True
    assert result.onset_above_order_parameter > result.onset_below_order_parameter


def test_overall_validation_passes(result: SynchronisationValidationResult) -> None:
    assert result.passed is True
    assert result.critical_coupling == pytest.approx(2.0)


def test_simulation_is_deterministic_for_fixed_seed() -> None:
    case = SynchronisationCase(
        n_oscillators=800,
        gamma=1.0,
        coupling=6.0,
        alpha=0.0,
        dt_s=0.05,
        n_steps=400,
        n_average=100,
        seed=11,
    )
    assert simulate_steady_order_parameter(case) == simulate_steady_order_parameter(case)


def test_lorentzian_frequencies_are_finite_and_shaped() -> None:
    rng = np.random.default_rng(3)
    frequencies = lorentzian_frequencies(500, 1.5, rng)
    assert frequencies.shape == (500,)
    assert frequencies.dtype == np.float64
    assert np.all(np.isfinite(frequencies))


def test_case_rejects_nonpositive_oscillator_count() -> None:
    with pytest.raises(ValueError, match="n_oscillators"):
        SynchronisationCase(0, 1.0, 5.0, 0.0, 0.05, 100, 50, 1)


def test_case_rejects_average_window_exceeding_steps() -> None:
    with pytest.raises(ValueError, match="n_average"):
        SynchronisationCase(100, 1.0, 5.0, 0.0, 0.05, 100, 200, 1)


def test_case_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        SynchronisationCase(100, 1.0, 5.0, 0.0, 0.05, 100, 50, -1)


def test_evidence_roundtrip_is_sealed_and_passing(result: SynchronisationValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    assert evidence["schema_version"] == KURAMOTO_SYNCHRONISATION_SCHEMA_VERSION
    assert validate_evidence_payload(evidence) is True


def test_evidence_tamper_is_rejected(result: SynchronisationValidationResult) -> None:
    evidence = build_evidence(result, target_id="test-target")
    evidence["branch_max_abs_error"] = 999.0
    with pytest.raises(ValueError, match="payload_sha256 does not match"):
        validate_evidence_payload(evidence)


def test_evidence_rejects_empty_target_id(result: SynchronisationValidationResult) -> None:
    with pytest.raises(ValueError, match="target_id"):
        build_evidence(result, target_id="   ")
