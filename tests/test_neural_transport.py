# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neural transport tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

"""Tests for scpn_control.core.neural_transport."""

import numpy as np

import pytest

from scpn_control.core.neural_transport import (
    NeuralTransportModel,
    TransportInputs,
    critical_gradient_model,
)


class TestTransportInputs:
    def test_defaults(self):
        inp = TransportInputs()
        assert inp.rho == 0.5
        assert inp.te_kev == 5.0
        assert inp.q == 1.5

    def test_custom(self):
        inp = TransportInputs(rho=0.3, te_kev=10.0, grad_ti=8.0)
        assert inp.rho == 0.3
        assert inp.grad_ti == 8.0


class TestCriticalGradientModel:
    def test_below_threshold_stable(self):
        inp = TransportInputs(grad_ti=2.0, grad_te=2.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_i == 0.0
        assert fluxes.chi_e == 0.0
        assert fluxes.channel == "stable"

    def test_itg_dominant(self):
        inp = TransportInputs(grad_ti=8.0, grad_te=2.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_i > 0.0
        assert fluxes.chi_e == 0.0
        assert fluxes.channel == "ITG"

    def test_tem_dominant(self):
        inp = TransportInputs(grad_ti=2.0, grad_te=9.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_e > 0.0
        assert fluxes.chi_i == 0.0
        assert fluxes.channel == "TEM"

    def test_particle_diffusivity_uses_density_gradient_and_shear(self):
        inp = TransportInputs(grad_ti=2.0, grad_te=9.0)
        fluxes = critical_gradient_model(inp)
        steeper_density = critical_gradient_model(TransportInputs(grad_ti=2.0, grad_te=9.0, grad_ne=5.0))
        higher_shear = critical_gradient_model(TransportInputs(grad_ti=2.0, grad_te=9.0, s_hat=2.0))

        fixed_ratio_alias = 1.0 / 3.0
        assert fluxes.d_e != pytest.approx(fluxes.chi_e * fixed_ratio_alias)
        assert steeper_density.d_e > fluxes.d_e
        assert higher_shear.d_e > fluxes.d_e

    def test_stiffness_exponent(self):
        inp = TransportInputs(grad_ti=5.0, grad_te=2.0)
        fluxes = critical_gradient_model(inp)
        # chi_i = 1.0 * (5.0 - 4.0)^2.0 = 1.0
        assert fluxes.chi_i == pytest.approx(1.0)

    def test_both_channels_itg_wins(self):
        inp = TransportInputs(grad_ti=8.0, grad_te=7.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_i > fluxes.chi_e
        assert fluxes.channel == "ITG"

    def test_profile_fallback_particle_diffusivity_matches_local_model(self):
        rho = np.linspace(0.2, 0.8, 7)
        te = np.linspace(7.0, 4.0, 7)
        ti = np.linspace(5.0, 4.5, 7)
        ne = np.linspace(8.0, 3.0, 7)
        q_profile = 1.0 + rho
        s_hat_profile = np.linspace(0.2, 1.4, 7)

        model = NeuralTransportModel(auto_discover=False)
        chi_e, _, d_e = model.predict_profile(rho, te, ti, ne, q_profile, s_hat_profile, r_major=6.2)

        assert np.any(chi_e > 0.0)
        assert not np.allclose(d_e[chi_e > 0.0], chi_e[chi_e > 0.0] / 3.0)


def test_neural_transport_explicit_weights_fail_closed_without_legacy_fallback(tmp_path):
    from scpn_control.core.neural_transport import NeuralTransportModel

    missing = tmp_path / "missing_weights.npz"

    with pytest.raises(FileNotFoundError, match="weights not found"):
        NeuralTransportModel(weights_path=missing)

    model = NeuralTransportModel(
        weights_path=missing,
        allow_weight_load_fallback=True,
        allow_legacy_weight_load_fallback=True,
    )
    fluxes = model.predict(TransportInputs(grad_ti=8.0, grad_te=2.0))
    assert model.is_neural is False
    assert fluxes.channel == "ITG"


def test_neural_transport_closure_profiles_validate_radial_contracts():
    from scpn_control.core.neural_transport import neural_transport_closure_profiles

    rho = np.array([0.0, 0.5, 0.25])
    profile = np.array([6.0, 5.0, 4.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        neural_transport_closure_profiles(rho, profile, profile, profile, profile, profile)

    with pytest.raises(ValueError, match="strictly positive"):
        neural_transport_closure_profiles(
            np.array([0.0, 0.5, 1.0]),
            np.array([6.0, 0.0, 4.0]),
            profile,
            profile,
            profile,
            profile,
        )


def test_neural_transport_claim_evidence_blocks_quantitative_claim_without_reference():
    from scpn_control.core.neural_transport import (
        assert_neural_transport_quantitative_claim_admissible,
        neural_transport_claim_evidence,
    )

    validation_result = {
        "mode": "analytic_fallback",
        "n_cases": 2,
        "max_abs_error": 0.0,
        "channel_agreement": 1.0,
        "per_channel_relative_rmse": (0.0, 0.0, 0.0),
        "profile_per_channel_relative_rmse": (0.0, 0.0, 0.0),
    }

    evidence = neural_transport_claim_evidence(
        validation_result,
        source="local_regression_reference",
        source_id="tests/test_neural_transport.py::analytic_fallback",
    )

    assert evidence.quantitative_claim_allowed is False
    assert evidence.feature_schema[0] == "R_LTi"
    with pytest.raises(ValueError, match="blocked without matched reference"):
        assert_neural_transport_quantitative_claim_admissible(evidence)
