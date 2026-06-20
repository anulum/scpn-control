# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Orbit-following tests
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from scpn_control.core.orbit_following import (
    EnsembleResult,
    GuidingCenterOrbit,
    MonteCarloEnsemble,
    OrbitClassifier,
    assert_orbit_following_external_claim_admissible,
    SlowingDown,
    _finite_scalar,
    _non_empty_text,
    _profile_array,
    banana_orbit_width,
    first_orbit_loss,
    orbit_following_claim_evidence,
    save_orbit_following_claim_evidence,
)


def mock_b_field(R, Z):
    # Pure toroidal field (no poloidal -> no drifts in passing)
    B0 = 5.0
    R0 = 6.0
    B_phi = B0 * R0 / R
    # Add small poloidal field for trapping
    B_R = -0.1 * Z
    B_Z = 0.1 * (R - R0)
    return B_R, B_Z, B_phi


def test_passing_orbit():
    # Pitch angle 0 -> purely parallel -> passing
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, 0.0, 6.2, 0.0)

    # Evolve a bit
    for _ in range(10):
        orbit.step(mock_b_field, 1e-6)

    assert orbit.v_par > 0.0  # Never reversed
    assert orbit.R > 0.0


def test_trapped_orbit():
    # Pitch angle pi/2 -> purely perpendicular -> trapped instantly
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, math.pi / 2 - 0.1, 6.2, 0.0)

    # We must have computed mu
    orbit.step(mock_b_field, 1e-6)
    assert orbit.mu > 0.0

    # In a full simulation it would reverse v_par


def test_orbit_classifier():
    R = np.ones(10) * 6.0
    Z = np.zeros(10)
    v_par_pass = np.ones(10)
    v_par_trap = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 1.0, 1.0, 1.0, 1.0])

    assert OrbitClassifier.classify(R, Z, v_par_pass, 10.0, 5.0) == "passing"
    assert OrbitClassifier.classify(R, Z, v_par_trap, 10.0, 5.0) == "trapped"

    # Lost
    R_lost = np.array([6.0, 7.0, 11.0, 12.0])
    v_lost = np.ones(4)
    assert OrbitClassifier.classify(R_lost, Z[:4], v_lost, 10.0, 5.0) == "lost"


def test_first_orbit_loss():
    # ITER: large Ip -> low loss
    iter_loss = first_orbit_loss(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0)
    assert iter_loss < 0.05

    # NSTX: small Ip, small a -> higher loss
    nstx_loss = first_orbit_loss(R0=0.9, a=0.6, B0=1.0, Ip_MA=1.0)
    assert nstx_loss > iter_loss
    assert nstx_loss > 0.5


def test_slowing_down():
    tau = SlowingDown.tau_sd(Te_keV=20.0, ne_20=1.0, Z_eff=1.5)
    # Stix 1972, Eq. 7; Wesson Table 7.1: ~0.2 s for Te=20 keV, ne=1e20, Zeff=1.5
    assert 0.1 < tau < 1.0

    vc = SlowingDown.critical_velocity(20.0, 1.0)

    # 3.5 MeV alpha v ~ 1.3e7 m/s
    v_alpha = 1.3e7
    f_i, f_e = SlowingDown.heating_partition(v_alpha, vc)

    # Fast alpha heats electrons primarily
    assert f_e > f_i


def test_mc_ensemble():
    ens = MonteCarloEnsemble(10, 3500.0, 6.2, 2.0, 5.3)
    ens.initialize(np.ones(10), np.ones(10), np.linspace(0, 1, 10))

    assert len(ens.particles) == 10

    res = ens.follow(mock_b_field)
    assert res.n_passing + res.n_trapped + res.n_lost == 10


# ── New physics tests ────────────────────────────────────────────────────────


def test_stix_slowing_down():
    """dE/dt < 0 at all energies — Stix 1972, Eq. 12."""
    Te_keV = 20.0
    ne_20 = 1.0
    Z_eff = 1.5
    tau_s = SlowingDown.tau_sd(Te_keV, ne_20, Z_eff)
    E_crit = SlowingDown.critical_energy(Te_keV, A_fast=4.0, A_bg=2.5, Z_bg=1, ne_20=ne_20)

    for E_keV in [50.0, 500.0, 3520.0]:
        dE = SlowingDown.dE_dt(E_keV, E_crit, tau_s)
        assert dE < 0.0, f"dE/dt must be negative at E={E_keV} keV (Stix 1972, Eq. 12)"


def test_critical_energy_partition():
    """
    Below E_crit ions absorb >50% of fast-ion power.
    Stix 1972, Eq. 16: f_ion = v_c³ / (v³ + v_c³).
    At v < v_c → f_ion > 0.5.
    """
    Te_keV = 20.0
    ne_20 = 1.0
    v_c = SlowingDown.critical_velocity(Te_keV, ne_20)

    # Speed well below v_c → dominant ion heating
    v_slow = 0.3 * v_c
    f_ion, f_elec = SlowingDown.heating_partition(v_slow, v_c)
    assert f_ion > 0.5, f"Below v_c ion fraction must exceed 0.5 (got {f_ion:.3f})"

    # Speed well above v_c → dominant electron heating
    v_fast = 3.0 * v_c
    f_ion_hi, f_elec_hi = SlowingDown.heating_partition(v_fast, v_c)
    assert f_elec_hi > 0.5, f"Above v_c electron fraction must exceed 0.5 (got {f_elec_hi:.3f})"

    # At v = v_c: equal partition within 5%
    f_ion_eq, f_elec_eq = SlowingDown.heating_partition(v_c, v_c)
    assert abs(f_ion_eq - 0.5) < 0.05


def test_banana_width_scaling():
    """
    Δ_b = q ρ_L / √ε — Wesson 2011, Eq. 5.4.14.
    Verify linear scaling with q and inverse-sqrt scaling with ε.
    """
    rho_L = 0.05  # m, typical alpha Larmor radius

    # Linear in q
    w1 = banana_orbit_width(q=1.5, rho_L=rho_L, epsilon=0.3)
    w2 = banana_orbit_width(q=3.0, rho_L=rho_L, epsilon=0.3)
    assert abs(w2 / w1 - 2.0) < 1e-10, "Δ_b must scale linearly with q (Wesson Eq. 5.4.14)"

    # Inverse sqrt in ε
    w3 = banana_orbit_width(q=2.0, rho_L=rho_L, epsilon=0.1)
    w4 = banana_orbit_width(q=2.0, rho_L=rho_L, epsilon=0.4)
    expected_ratio = math.sqrt(0.4 / 0.1)  # √(ε2/ε1)
    assert abs(w3 / w4 - expected_ratio) < 1e-10, "Δ_b must scale as 1/√ε (Wesson Eq. 5.4.14)"


def test_first_orbit_loss_current_dependence():
    """
    Higher I_p → smaller prompt loss fraction.
    Goldston 1981, J. Comput. Phys. 43, 61, Eq. 15: f_lost ∝ 1/I_p.
    """
    base = dict(R0=6.2, a=2.0, B0=5.3, E_alpha_keV=3520.0)

    f_low_Ip = first_orbit_loss(Ip_MA=5.0, **base)
    f_high_Ip = first_orbit_loss(Ip_MA=15.0, **base)

    assert f_high_Ip < f_low_Ip, "Higher I_p must reduce prompt losses (Goldston 1981, Eq. 15)"

    # Ratio should match 1/I_p scaling within 5% (both in unsaturated regime)
    expected_ratio = 5.0 / 15.0
    actual_ratio = f_high_Ip / f_low_Ip
    assert abs(actual_ratio - expected_ratio) < 0.05, (
        f"f_lost ratio {actual_ratio:.3f} deviates from 1/I_p = {expected_ratio:.3f}"
    )


def test_guiding_center_orbit_rejects_nonphysical_particle_and_step_inputs():
    with pytest.raises(ValueError, match="m_amu"):
        GuidingCenterOrbit(0.0, 2, 3500.0, 0.0, 6.2, 0.0)

    with pytest.raises(ValueError, match="Z"):
        GuidingCenterOrbit(4.0, 0, 3500.0, 0.0, 6.2, 0.0)

    with pytest.raises(ValueError, match="Z"):
        GuidingCenterOrbit(4.0, True, 3500.0, 0.0, 6.2, 0.0)

    with pytest.raises(ValueError, match="E_keV"):
        GuidingCenterOrbit(4.0, 2, 0.0, 0.0, 6.2, 0.0)

    with pytest.raises(ValueError, match="pitch_angle"):
        GuidingCenterOrbit(4.0, 2, 3500.0, -0.1, 6.2, 0.0)

    with pytest.raises(ValueError, match="pitch_angle"):
        GuidingCenterOrbit(4.0, 2, 3500.0, math.pi + 0.1, 6.2, 0.0)

    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, 0.0, 6.2, 0.0)

    with pytest.raises(ValueError, match="dt"):
        orbit.step(mock_b_field, 0.0)


def test_guiding_center_orbit_rejects_invalid_field():
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, 0.0, 6.2, 0.0)

    def zero_b_field(R, Z):
        return 0.0, 0.0, 0.0

    with pytest.raises(ValueError, match="B_field magnitude"):
        orbit.step(zero_b_field, 1e-7)

    def nan_b_field(R, Z):
        return np.nan, 0.0, 5.0

    with pytest.raises(ValueError, match="B_field"):
        orbit.step(nan_b_field, 1e-7)


def test_orbit_classifier_rejects_malformed_traces():
    R = np.ones(4)
    Z = np.zeros(4)
    v = np.ones(4)

    with pytest.raises(ValueError, match="orbit_Z"):
        OrbitClassifier.classify(R, Z[:-1], v, R_wall=10.0, Z_wall_upper=5.0)

    with pytest.raises(ValueError, match="R_wall"):
        OrbitClassifier.classify(R, Z, v, R_wall=0.0, Z_wall_upper=5.0)

    with pytest.raises(ValueError, match="v_par"):
        OrbitClassifier.classify(R, Z, np.array([1.0, np.nan, 1.0, 1.0]), R_wall=10.0, Z_wall_upper=5.0)


def test_monte_carlo_ensemble_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="n_particles"):
        MonteCarloEnsemble(0, 3500.0, 6.2, 2.0, 5.3)

    with pytest.raises(ValueError, match="n_particles"):
        MonteCarloEnsemble(True, 3500.0, 6.2, 2.0, 5.3)

    with pytest.raises(ValueError, match="a must be smaller"):
        MonteCarloEnsemble(10, 3500.0, 2.0, 2.0, 5.3)

    ens = MonteCarloEnsemble(2, 3500.0, 6.2, 2.0, 5.3)

    with pytest.raises(ValueError, match="rho"):
        ens.initialize(np.ones(3), np.ones(3), np.array([0.0, 0.8, 0.7]))

    with pytest.raises(ValueError, match="rho"):
        ens.initialize(np.ones(3), np.ones(3), np.array([0.0, 0.5, 0.5]))

    with pytest.raises(ValueError, match="ne_profile"):
        ens.initialize(np.array([1.0, 0.0, 1.0]), np.ones(3), np.array([0.0, 0.5, 1.0]))

    with pytest.raises(ValueError, match="particles"):
        ens.follow(mock_b_field)

    ens.initialize(np.ones(3), np.ones(3), np.array([0.0, 0.5, 1.0]))
    with pytest.raises(ValueError, match="n_bounces"):
        ens.follow(mock_b_field, n_bounces=0)

    with pytest.raises(ValueError, match="n_bounces"):
        ens.follow(mock_b_field, n_bounces=True)


def test_first_orbit_loss_and_banana_width_reject_invalid_physics():
    with pytest.raises(ValueError, match="Ip_MA"):
        first_orbit_loss(R0=6.2, a=2.0, B0=5.3, Ip_MA=0.0)

    with pytest.raises(ValueError, match="a must be smaller"):
        first_orbit_loss(R0=2.0, a=2.0, B0=5.3, Ip_MA=10.0)

    with pytest.raises(ValueError, match="q"):
        banana_orbit_width(q=0.0, rho_L=0.05, epsilon=0.3)

    with pytest.raises(ValueError, match="rho_L"):
        banana_orbit_width(q=2.0, rho_L=0.0, epsilon=0.3)


def test_slowing_down_rejects_invalid_domains():
    with pytest.raises(ValueError, match="Te_keV"):
        SlowingDown.critical_velocity(0.0, 1.0)

    with pytest.raises(ValueError, match="Z_bg"):
        SlowingDown.critical_energy(20.0, A_fast=4.0, A_bg=2.5, Z_bg=0, ne_20=1.0)

    with pytest.raises(ValueError, match="Z_bg"):
        SlowingDown.critical_energy(20.0, A_fast=4.0, A_bg=2.5, Z_bg=True, ne_20=1.0)

    with pytest.raises(ValueError, match="ne_20"):
        SlowingDown.tau_sd(20.0, ne_20=0.0, Z_eff=1.5)

    with pytest.raises(ValueError, match="E_keV"):
        SlowingDown.dE_dt(0.0, E_crit_keV=100.0, tau_s=0.1)

    with pytest.raises(ValueError, match="v_c"):
        SlowingDown.heating_partition(1.0, v_c=0.0)


def test_orbit_following_claim_evidence_records_bounded_provenance(tmp_path):
    loss = first_orbit_loss(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0)
    ensemble = EnsembleResult(
        loss_fraction=0.2,
        heating_profile=np.zeros(50),
        current_drive=0.0,
        n_passing=6,
        n_trapped=2,
        n_lost=2,
    )

    evidence = orbit_following_claim_evidence(
        source="synthetic_regression_reference",
        source_id="orbit-following-bounded-regression-v1",
        geometry_source="repository large-aspect-ratio tokamak fixture",
        particle_source="repository alpha-particle birth fixture",
        collision_model="Stix slowing-down fixture",
        loss_boundary_source="repository first-orbit wall boundary fixture",
        q=2.0,
        rho_L_m=0.05,
        epsilon=0.25,
        first_orbit_loss_fraction=loss,
        ensemble_result=ensemble,
    )
    report_path = tmp_path / "orbit_following_claim.json"
    save_orbit_following_claim_evidence(evidence, report_path)

    assert evidence.external_orbit_claim_allowed is False
    assert evidence.claim_status == "bounded_orbit_following_evidence"
    assert evidence.banana_width_m == pytest.approx(0.2)
    assert evidence.first_orbit_loss_fraction == pytest.approx(loss)
    assert evidence.ensemble_particles == 10
    assert evidence.ensemble_lost == 2
    assert '"external_orbit_claim_allowed": false' in report_path.read_text(encoding="utf-8")


def test_orbit_following_external_admission_requires_matched_references():
    loss = first_orbit_loss(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0)
    width = banana_orbit_width(q=2.0, rho_L=0.05, epsilon=0.25)

    matched = orbit_following_claim_evidence(
        source="external_orbit_code",
        source_id="matched-orbit-reference",
        geometry_source="documented equilibrium geometry",
        particle_source="documented alpha-particle birth distribution",
        collision_model="documented slowing-down model",
        loss_boundary_source="documented first-wall boundary",
        q=2.0,
        rho_L_m=0.05,
        epsilon=0.25,
        first_orbit_loss_fraction=loss,
        reference_banana_width_m=width,
        reference_loss_fraction=loss,
    )
    assert_orbit_following_external_claim_admissible(matched)
    assert matched.external_orbit_claim_allowed is True

    mismatched_loss = orbit_following_claim_evidence(
        source="external_orbit_code",
        source_id="mismatched-loss-reference",
        geometry_source="documented equilibrium geometry",
        particle_source="documented alpha-particle birth distribution",
        collision_model="documented slowing-down model",
        loss_boundary_source="documented first-wall boundary",
        q=2.0,
        rho_L_m=0.05,
        epsilon=0.25,
        first_orbit_loss_fraction=loss,
        reference_banana_width_m=width,
        reference_loss_fraction=min(loss + 0.2, 1.0),
        loss_fraction_abs_tolerance=0.01,
    )
    with pytest.raises(ValueError, match="external orbit claim requires matched"):
        assert_orbit_following_external_claim_admissible(mismatched_loss)
    assert mismatched_loss.external_orbit_claim_allowed is False


def test_orbit_following_claim_evidence_rejects_invalid_claim_inputs():
    loss = first_orbit_loss(R0=6.2, a=2.0, B0=5.3, Ip_MA=15.0)

    with pytest.raises(ValueError, match="source must be one of"):
        orbit_following_claim_evidence(
            source="untracked_reference",
            source_id="bad-source",
            geometry_source="documented geometry",
            particle_source="documented particles",
            collision_model="documented collisions",
            loss_boundary_source="documented wall",
            q=2.0,
            rho_L_m=0.05,
            epsilon=0.25,
            first_orbit_loss_fraction=loss,
        )

    with pytest.raises(ValueError, match="loss_fraction"):
        orbit_following_claim_evidence(
            source="external_orbit_code",
            source_id="bad-loss",
            geometry_source="documented geometry",
            particle_source="documented particles",
            collision_model="documented collisions",
            loss_boundary_source="documented wall",
            q=2.0,
            rho_L_m=0.05,
            epsilon=0.25,
            first_orbit_loss_fraction=1.2,
        )

    with pytest.raises(ValueError, match="loss fraction must match"):
        orbit_following_claim_evidence(
            source="external_orbit_code",
            source_id="bad-ensemble",
            geometry_source="documented geometry",
            particle_source="documented particles",
            collision_model="documented collisions",
            loss_boundary_source="documented wall",
            q=2.0,
            rho_L_m=0.05,
            epsilon=0.25,
            first_orbit_loss_fraction=loss,
            ensemble_result=EnsembleResult(0.1, np.zeros(50), 0.0, 1, 1, 1),
        )


def test_finite_scalar_rejects_nonfinite_and_negative() -> None:
    with pytest.raises(ValueError, match="x must be finite"):
        _finite_scalar("x", math.inf)
    with pytest.raises(ValueError, match="x must be non-negative"):
        _finite_scalar("x", -1.0, nonnegative=True)


def test_profile_array_rejects_empty_profile() -> None:
    with pytest.raises(ValueError, match="must be a one-dimensional non-empty profile"):
        _profile_array("rho", np.array([]))


def test_non_empty_text_rejects_blank() -> None:
    assert _non_empty_text("source", "  orbit-x  ") == "orbit-x"
    with pytest.raises(ValueError, match="source must be a non-empty string"):
        _non_empty_text("source", "   ")


def _evidence_kwargs() -> dict[str, Any]:
    return dict(
        source="synthetic_regression_reference",
        source_id="orbit-following-bounded-regression-v1",
        geometry_source="repository large-aspect-ratio tokamak fixture",
        particle_source="repository alpha-particle birth fixture",
        collision_model="Stix slowing-down fixture",
        loss_boundary_source="repository first-orbit wall boundary fixture",
        q=2.0,
        rho_L_m=0.05,
        epsilon=0.25,
        first_orbit_loss_fraction=0.1,
    )


def test_claim_evidence_rejects_out_of_range_reference_loss() -> None:
    with pytest.raises(ValueError, match=r"reference_loss_fraction must stay within \[0, 1\]"):
        orbit_following_claim_evidence(reference_loss_fraction=1.5, **_evidence_kwargs())


def test_claim_evidence_rejects_unclassified_ensemble() -> None:
    empty = EnsembleResult(
        loss_fraction=0.0,
        heating_profile=np.zeros(10),
        current_drive=0.0,
        n_passing=0,
        n_trapped=0,
        n_lost=0,
    )
    with pytest.raises(ValueError, match="must contain at least one classified particle"):
        orbit_following_claim_evidence(ensemble_result=empty, **_evidence_kwargs())


def test_eom_rejects_nonphysical_state() -> None:
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, 0.0, 6.2, 0.0)
    with pytest.raises(ValueError, match="orbit state must be finite with positive major radius"):
        orbit._eom(np.array([-1.0, 0.0, 0.0, 0.0]), lambda R, Z: (1.0, 1.0, 1.0))


def test_initialize_rejects_nonpositive_temperature() -> None:
    ens = MonteCarloEnsemble(10, 3500.0, 6.2, 2.0, 5.3)
    te = np.ones(10)
    te[3] = 0.0
    with pytest.raises(ValueError, match="Te_profile must be positive everywhere"):
        ens.initialize(np.ones(10), te, np.linspace(0.0, 1.0, 10))


def test_follow_counts_each_classified_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    ens = MonteCarloEnsemble(4, 3500.0, 6.2, 2.0, 5.3)
    ens.initialize(np.ones(4), np.ones(4), np.linspace(0.0, 1.0, 4))
    outcomes = iter(["lost", "trapped", "passing", "lost"])
    monkeypatch.setattr(OrbitClassifier, "classify", staticmethod(lambda *args, **kwargs: next(outcomes)))

    def steady_b_field(R: float, Z: float) -> tuple[float, float, float]:
        return (0.0, 1.0, 5.3 * 6.2 / R)

    result = ens.follow(steady_b_field, n_bounces=1, dt=1.0e-7)
    assert result.n_lost == 2
    assert result.n_trapped == 1
    assert result.n_passing == 1
    assert result.loss_fraction == pytest.approx(0.5)
