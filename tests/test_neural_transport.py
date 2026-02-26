"""Tests for scpn_control.core.neural_transport."""

import numpy as np
import pytest

from scpn_control.core.neural_transport import (
    TransportInputs,
    TransportFluxes,
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

    def test_particle_diffusivity_one_third(self):
        inp = TransportInputs(grad_ti=2.0, grad_te=9.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.d_e == pytest.approx(fluxes.chi_e / 3.0)

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
