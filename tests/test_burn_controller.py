# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Burn Control Tests
from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_control.control.burn_controller import (
    AlphaHeating,
    BurnController,
    assert_burn_control_reactor_claim_admissible,
    burn_control_claim_evidence,
    BurnStabilityAnalysis,
    SubignitedBurnPoint,
    save_burn_control_claim_evidence,
    _extract_burn_reference_artifact,
    _non_empty_text,
    _nonnegative_reference_scalar,
    _positive_reference_scalar,
    _require_nonnegative_profile,
    _sha256_text,
    _weighted_average,
)


def _valid_burn_reference_artifact() -> dict:
    return {
        "source": "integrated_transport_benchmark",
        "reference_dataset_id": "burn-integrated-transport-fixture-v1",
        "reference_artifact_sha256": "b" * 64,
        "reference_case_count": 3,
        "units": {
            "density": "m^-3",
            "temperature": "keV",
            "power": "MW",
            "time": "s",
            "reactivity": "m^3/s",
            "triple_product": "m^-3 s keV",
            "dimensionless": "1",
        },
        "metrics": {
            "P_alpha_relative_error": 0.01,
            "Q_abs_error": 0.05,
            "lawson_margin_abs_error": 0.02,
            "burn_fraction_relative_error": 0.03,
            "reactivity_exponent_abs_error": 0.04,
        },
        "tolerances": {
            "P_alpha_relative_error": 0.05,
            "Q_abs_error": 0.2,
            "lawson_margin_abs_error": 0.1,
            "burn_fraction_relative_error": 0.1,
            "reactivity_exponent_abs_error": 0.2,
        },
    }


def test_zero_temperature():
    alpha = AlphaHeating(R0=6.2, a=2.0)
    P = alpha.power(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.5]))
    assert P == 0.0


def test_alpha_power_density_rejects_nonphysical_inputs():
    alpha = AlphaHeating(R0=6.2, a=2.0)

    with np.testing.assert_raises(ValueError):
        alpha.power_density(np.array([-1.0]), np.array([10.0]), np.array([10.0]))

    with np.testing.assert_raises(ValueError):
        alpha.power_density(np.array([1.0]), np.array([np.inf]), np.array([10.0]))

    with np.testing.assert_raises(ValueError):
        alpha.power_density(np.array([1.0]), np.array([10.0]), np.array([-0.1]))


def test_iter_alpha_power():
    alpha = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)

    # 20 keV, 1e20 m-3 flat
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 1.0
    Te = np.ones(50) * 20.0
    Ti = np.ones(50) * 20.0

    P_alpha = alpha.power(ne, Te, Ti, rho)

    # A completely flat 20 keV profile yields much higher power than a peaked profile.
    # ~ 500 MW is physically correct for a uniform 800 m^3 plasma at 20 keV.
    assert 400.0 < P_alpha < 600.0


def test_Q_definition():
    alpha = AlphaHeating(R0=6.2, a=2.0)
    Q = alpha.Q(P_alpha_MW=20.0, P_aux_MW=10.0)
    assert Q == 10.0

    # Ignition
    Q_ign = alpha.Q(P_alpha_MW=20.0, P_aux_MW=0.0)
    assert Q_ign == float("inf")


def test_stability_boundary():
    alpha = AlphaHeating(R0=6.2, a=2.0)
    analysis = BurnStabilityAnalysis(alpha)

    # Exponent < 2 for T > 15
    assert analysis.reactivity_exponent(20.0) < 2.0
    assert analysis.is_thermally_stable(20.0)

    # Exponent > 2 for T < 10
    assert analysis.reactivity_exponent(8.0) > 2.0
    assert not analysis.is_thermally_stable(8.0)

    T_bound = analysis.stability_boundary_keV()
    assert 12.0 < T_bound < 16.0


def test_burn_controller():
    ctrl = BurnController(Q_target=10.0, T_target_keV=20.0, P_aux_max_MW=50.0)

    # Test emergency cooling
    u_emerg = ctrl.step(Q_meas=15.0, T_meas_keV=35.0, P_alpha_MW=100.0, dt=0.1)
    assert u_emerg == 0.0

    # Test normal response (T too low -> increase power)
    ctrl.integral_T = 0.0
    u_norm = ctrl.step(Q_meas=5.0, T_meas_keV=10.0, P_alpha_MW=20.0, dt=0.1)

    # K_T_p = -5, e_T = -10 => +50 MW. Base is 25. Total > 50 -> clipped to 50
    assert u_norm == 50.0

    # T too high -> decrease power
    ctrl.integral_T = 0.0
    u_high = ctrl.step(Q_meas=10.0, T_meas_keV=25.0, P_alpha_MW=80.0, dt=0.1)

    # K_T_p = -5, e_T = 5 => -25 MW. Base is 25. Total 0.
    assert u_high == 0.0


def test_subignited_burn_point():
    alpha = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)
    sbp = SubignitedBurnPoint(alpha)

    tau_E = 3.0
    ne_20 = 1.0
    P_aux = 10.0

    pts = sbp.find_operating_point(ne_20, P_aux, tau_E)

    assert len(pts) > 0
    pts.sort(key=lambda p: p.Te_keV)
    assert pts[-1].P_alpha_MW > 0.0


# ── Physics-grounded tests (ITER Physics Basis 1999; Lawson 1957) ──

from scpn_control.control.burn_controller import (
    LAWSON_TRIPLE_PRODUCT,
    burn_fraction,
    lawson_triple_product,
)
from scpn_control.core.uncertainty import bosch_hale_reactivity


def test_burn_alpha_power_positive() -> None:
    """P_alpha > 0 at ignition-relevant conditions (T_i = 20 keV, n = 1e20 m^-3).

    P_α = n_D n_T <σv> E_α × V.
    ITER Physics Basis 1999, Nucl. Fusion 39, 2137.
    """
    alpha = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)
    rho = np.linspace(0.0, 1.0, 50)
    ne = np.ones(50)
    T = np.ones(50) * 20.0
    P = alpha.power(ne, T, T, rho)
    assert P > 0.0


def test_lawson_criterion_iter() -> None:
    """ITER Q=10 operating point satisfies n τ_E T > 3e21 m^-3 s keV.

    Lawson 1957, Proc. Phys. Soc. B 70, 6.
    ITER: n ≈ 1e20 m^-3, τ_E ≈ 3.7 s, T ≈ 20 keV  →  n τ_E T ≈ 7.4e21.
    """
    ntp = lawson_triple_product(ne_m3=1.0e20, tau_E_s=3.7, T_keV=20.0)
    assert ntp > LAWSON_TRIPLE_PRODUCT


def test_burn_fraction_positive() -> None:
    """f_b > 0 for finite n, <σv>, v_th.

    Wesson 2011, Eq. 1.7.3: f_b ≈ a² n_DT <σv> / (4 v_th).
    """
    sigv = float(bosch_hale_reactivity(20.0))
    fb = burn_fraction(n_dt_m3=1.0e20, sigv=sigv, v_th_ms=1.0e6, a_m=2.0)
    assert fb > 0.0


def test_reactivity_exponent_edge_cases():
    """Near-cold ions return the conservative unstable reactivity exponent."""
    alpha = AlphaHeating(R0=6.2, a=2.0)
    analysis = BurnStabilityAnalysis(alpha)

    # Ti_keV <= 0.1 returns 10.0 (line 122)
    assert analysis.reactivity_exponent(0.05) == 10.0

    # Ti_keV exactly at threshold
    assert analysis.reactivity_exponent(0.1) == 10.0


def test_alpha_heating_rejects_nonphysical_geometry_and_grids() -> None:
    with pytest.raises(ValueError, match="R0 must be finite and > 0"):
        AlphaHeating(R0=0.0, a=2.0)
    with pytest.raises(ValueError, match="a must be smaller than R0"):
        AlphaHeating(R0=2.0, a=2.0)
    with pytest.raises(ValueError, match="kappa must be finite and > 0"):
        AlphaHeating(R0=6.2, a=2.0, kappa=float("nan"))

    alpha = AlphaHeating(R0=6.2, a=2.0)
    with pytest.raises(ValueError, match="ne_20 must be one-dimensional"):
        alpha.power_density(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))

    with pytest.raises(ValueError, match="ne_20 must have shape"):
        alpha.power(np.ones(3), np.ones(3), np.ones(3), np.array([0.0, 1.0]))

    bad_rho = np.array([0.0, 0.5, 0.4])
    with pytest.raises(ValueError, match="rho must be strictly increasing"):
        alpha.power(np.ones(3), np.ones(3), np.ones(3), bad_rho)

    with pytest.raises(ValueError, match="normalised interval"):
        alpha.power(np.ones(3), np.ones(3), np.ones(3), np.array([0.0, 0.5, 1.1]))

    with pytest.raises(ValueError, match="rho must be strictly increasing"):
        alpha.power(np.ones(3), np.ones(3), np.ones(3), np.array([0.0, 0.5, 0.5]))


def test_burn_scalar_contracts_reject_nonphysical_inputs() -> None:
    with pytest.raises(ValueError, match="ne_m3 must be finite and >= 0"):
        lawson_triple_product(ne_m3=-1.0, tau_E_s=3.0, T_keV=20.0)
    with pytest.raises(ValueError, match="tau_E_s must be finite and > 0"):
        lawson_triple_product(ne_m3=1e20, tau_E_s=0.0, T_keV=20.0)

    with pytest.raises(ValueError, match="v_th_ms must be finite and > 0"):
        burn_fraction(n_dt_m3=1e20, sigv=1e-22, v_th_ms=0.0, a_m=2.0)


def test_burn_controller_rejects_invalid_control_domains() -> None:
    with pytest.raises(ValueError, match="Q_target must be finite and > 0"):
        BurnController(Q_target=0.0)

    ctrl = BurnController(Q_target=10.0, T_target_keV=20.0, P_aux_max_MW=50.0)
    with pytest.raises(ValueError, match="dt must be finite and > 0"):
        ctrl.step(Q_meas=10.0, T_meas_keV=20.0, P_alpha_MW=10.0, dt=0.0)
    with pytest.raises(ValueError, match="T_meas_keV must be finite and >= 0"):
        ctrl.step(Q_meas=10.0, T_meas_keV=-1.0, P_alpha_MW=10.0, dt=0.1)
    with pytest.raises(ValueError, match="P_alpha_MW must be finite and >= 0"):
        ctrl.step(Q_meas=10.0, T_meas_keV=20.0, P_alpha_MW=-1.0, dt=0.1)


def test_subignited_burn_point_rejects_nonphysical_scan_inputs() -> None:
    sbp = SubignitedBurnPoint(AlphaHeating(R0=6.2, a=2.0))
    with pytest.raises(ValueError, match="ne_20 must be finite and > 0"):
        sbp.find_operating_point(ne_20=0.0, P_aux_MW=10.0, tau_E_s=3.0)
    with pytest.raises(ValueError, match="P_aux_MW must be finite and >= 0"):
        sbp.find_operating_point(ne_20=1.0, P_aux_MW=-1.0, tau_E_s=3.0)
    with pytest.raises(ValueError, match="tau_E_s must be finite and > 0"):
        sbp.find_operating_point(ne_20=1.0, P_aux_MW=10.0, tau_E_s=0.0)


def test_burn_claim_evidence_records_bounded_operating_point(tmp_path) -> None:
    alpha = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)
    controller = BurnController(Q_target=10.0, T_target_keV=20.0, P_aux_max_MW=73.0)
    rho = np.linspace(0.0, 1.0, 24)
    ne = np.full(rho.shape, 1.0)
    temp = np.full(rho.shape, 20.0)

    evidence = burn_control_claim_evidence(
        alpha,
        controller,
        rho=rho,
        ne_20=ne,
        Te_keV=temp,
        Ti_keV=temp,
        tau_E_s=3.7,
        P_aux_MW=50.0,
        source="repository_burn_regression",
        source_id="burn-control-regression-v1",
    )

    assert evidence.claim_status == "bounded_burn_control_evidence"
    assert evidence.reactor_claim_allowed is False
    assert evidence.P_alpha_MW > 0.0
    assert evidence.lawson_margin > 1.0
    assert evidence.thermally_stable is True
    with pytest.raises(ValueError, match="reactor burn-control claim requires matched"):
        assert_burn_control_reactor_claim_admissible(evidence)

    output = tmp_path / "burn_claim.json"
    save_burn_control_claim_evidence(evidence, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1
    assert persisted["claim_status"] == "bounded_burn_control_evidence"


def test_burn_reactor_claim_requires_reference_artifact() -> None:
    alpha = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)
    controller = BurnController(Q_target=10.0, T_target_keV=20.0, P_aux_max_MW=73.0)
    rho = np.linspace(0.0, 1.0, 16)
    temp = np.full(rho.shape, 18.0)
    artifact = {
        "source": "integrated_transport_benchmark",
        "reference_dataset_id": "burn-integrated-transport-fixture-v1",
        "reference_artifact_sha256": "b" * 64,
        "reference_case_count": 3,
        "units": {
            "density": "m^-3",
            "temperature": "keV",
            "power": "MW",
            "time": "s",
            "reactivity": "m^3/s",
            "triple_product": "m^-3 s keV",
            "dimensionless": "1",
        },
        "metrics": {
            "P_alpha_relative_error": 0.01,
            "Q_abs_error": 0.05,
            "lawson_margin_abs_error": 0.02,
            "burn_fraction_relative_error": 0.03,
            "reactivity_exponent_abs_error": 0.04,
        },
        "tolerances": {
            "P_alpha_relative_error": 0.05,
            "Q_abs_error": 0.2,
            "lawson_margin_abs_error": 0.1,
            "burn_fraction_relative_error": 0.1,
            "reactivity_exponent_abs_error": 0.2,
        },
    }
    evidence = burn_control_claim_evidence(
        alpha,
        controller,
        rho=rho,
        ne_20=np.full(rho.shape, 0.95),
        Te_keV=temp,
        Ti_keV=temp,
        tau_E_s=3.4,
        P_aux_MW=45.0,
        source="integrated_transport_benchmark",
        source_id="burn-integrated-transport-fixture-v1",
        reference_artifact=artifact,
    )
    assert evidence.reactor_claim_allowed is True
    assert evidence.reference_dataset_id == "burn-integrated-transport-fixture-v1"
    assert assert_burn_control_reactor_claim_admissible(evidence) is evidence

    bad_artifact = dict(artifact)
    bad_artifact["metrics"] = dict(artifact["metrics"])
    bad_artifact["metrics"]["Q_abs_error"] = 1.0
    with pytest.raises(ValueError, match="Q_abs_error exceeds declared tolerance"):
        burn_control_claim_evidence(
            alpha,
            controller,
            rho=rho,
            ne_20=np.full(rho.shape, 0.95),
            Te_keV=temp,
            Ti_keV=temp,
            tau_E_s=3.4,
            P_aux_MW=45.0,
            source="integrated_transport_benchmark",
            source_id="burn-integrated-transport-fixture-v1",
            reference_artifact=bad_artifact,
        )


# ── Profile / scalar / text validation helpers ───────────────────────


def test_require_nonnegative_profile_rejects_empty():
    with pytest.raises(ValueError, match="must be non-empty"):
        _require_nonnegative_profile("p", np.array([]))


def test_non_empty_text_rejects_blank_and_non_string():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        _non_empty_text("field", "   ")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        _non_empty_text("field", 7)


def test_weighted_average_returns_first_value_for_zero_weights():
    # A single-point normalised rho [0.0] passes validation and zeroes the weights.
    assert _weighted_average(np.array([5.0]), np.array([0.0])) == 5.0


def test_sha256_text_rejects_non_digest():
    with pytest.raises(ValueError, match="must be a SHA-256 hex digest"):
        _sha256_text("digest", "abc")


@pytest.mark.parametrize("value", [True, float("inf"), "x"])
def test_positive_reference_scalar_rejects_non_numeric_or_non_finite(value):
    with pytest.raises(ValueError, match="finite and positive"):
        _positive_reference_scalar("metric", value)


def test_positive_reference_scalar_rejects_non_positive():
    with pytest.raises(ValueError, match="finite and positive"):
        _positive_reference_scalar("metric", 0.0)


@pytest.mark.parametrize("value", [True, float("nan"), "x"])
def test_nonnegative_reference_scalar_rejects_non_numeric_or_non_finite(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _nonnegative_reference_scalar("metric", value)


def test_nonnegative_reference_scalar_rejects_negative():
    with pytest.raises(ValueError, match="finite and non-negative"):
        _nonnegative_reference_scalar("metric", -1.0)


# ── Reference-artifact extraction rejection matrix ───────────────────


def test_extract_burn_reference_artifact_none_returns_inactive():
    assert _extract_burn_reference_artifact(None) == (None, False)


def test_extract_burn_reference_artifact_rejects_non_dict():
    with pytest.raises(ValueError, match="must be a dictionary"):
        _extract_burn_reference_artifact(["not", "a", "dict"])


def test_extract_burn_reference_artifact_rejects_inadmissible_source():
    artifact = _valid_burn_reference_artifact()
    artifact["source"] = "repository_burn_regression"  # bounded but not facility
    with pytest.raises(ValueError, match="source must be one of"):
        _extract_burn_reference_artifact(artifact)


def test_extract_burn_reference_artifact_rejects_bad_units():
    artifact = _valid_burn_reference_artifact()
    artifact["units"] = dict(artifact["units"])
    artifact["units"]["power"] = "W"
    with pytest.raises(ValueError, match="units must declare burn-control unit contracts"):
        _extract_burn_reference_artifact(artifact)


@pytest.mark.parametrize("count", [0, -2, True])
def test_extract_burn_reference_artifact_rejects_bad_case_count(count):
    artifact = _valid_burn_reference_artifact()
    artifact["reference_case_count"] = count
    with pytest.raises(ValueError, match="reference_case_count must be a positive integer"):
        _extract_burn_reference_artifact(artifact)


def test_extract_burn_reference_artifact_rejects_non_dict_metric_blocks():
    artifact = _valid_burn_reference_artifact()
    artifact["tolerances"] = "not a dict"
    with pytest.raises(ValueError, match="metrics and tolerances must be dictionaries"):
        _extract_burn_reference_artifact(artifact)


# ── Claim-evidence and admission guards ──────────────────────────────


def test_claim_evidence_rejects_inadmissible_source():
    alpha = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)
    controller = BurnController(Q_target=10.0, T_target_keV=20.0, P_aux_max_MW=73.0)
    rho = np.linspace(0.0, 1.0, 16)
    temp = np.full(rho.shape, 18.0)
    with pytest.raises(ValueError, match="source must be one of"):
        burn_control_claim_evidence(
            alpha,
            controller,
            rho=rho,
            ne_20=np.full(rho.shape, 0.95),
            Te_keV=temp,
            Ti_keV=temp,
            tau_E_s=3.4,
            P_aux_MW=45.0,
            source="not_a_declared_source",
            source_id="case",
        )


def test_assert_reactor_admissible_rejects_non_evidence_object():
    with pytest.raises(ValueError, match="must be BurnControlClaimEvidence"):
        assert_burn_control_reactor_claim_admissible({"not": "evidence"})


def test_save_claim_evidence_rejects_non_evidence_object(tmp_path):
    with pytest.raises(ValueError, match="must be BurnControlClaimEvidence"):
        save_burn_control_claim_evidence({"not": "evidence"}, tmp_path / "x.json")


def test_reactivity_exponent_returns_conservative_value_for_non_positive_reactivity(monkeypatch):
    # A reactivity model returning non-positive <σv> must fall back to the
    # conservative unstable exponent rather than take a logarithm of zero.
    import scpn_control.control.burn_controller as burn_mod

    monkeypatch.setattr(burn_mod, "bosch_hale_reactivity", lambda arr: np.zeros_like(np.asarray(arr, dtype=float)))
    analysis = BurnStabilityAnalysis(AlphaHeating(R0=6.2, a=2.0))
    assert analysis.reactivity_exponent(0.5) == 10.0
