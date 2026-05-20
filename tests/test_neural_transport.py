# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Neural Transport
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
