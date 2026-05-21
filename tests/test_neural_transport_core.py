# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Neural Transport Core
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Neural Transport (pure-function coverage)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for MLP internals, weight loading, and profile prediction."""

import tempfile

import numpy as np
import pytest

from scpn_control.core.neural_transport import (
    MLPWeights,
    NeuralTransportModel,
    TransportFluxes,
    TransportInputs,
    _mlp_forward,
    _relu,
    _softplus,
    critical_gradient_model,
)


# ── Activation functions ────────────────────────────────────────────


class TestRelu:
    def test_positive_passthrough(self):
        assert np.allclose(_relu(np.array([1.0, 2.0])), [1.0, 2.0])

    def test_negative_zeroed(self):
        assert np.allclose(_relu(np.array([-1.0, -0.5])), [0.0, 0.0])


class TestSoftplus:
    def test_positive_input(self):
        out = _softplus(np.array([5.0]))
        assert out[0] > 5.0  # softplus(x) > x for x > 0

    def test_nonnegative(self):
        out = _softplus(np.array([-10.0, 0.0, 10.0]))
        assert np.all(out >= 0.0)

    def test_large_negative_clipped(self):
        out = _softplus(np.array([-100.0]))
        assert np.isfinite(out[0])


# ── MLP forward pass ───────────────────────────────────────────────


def _make_weights(h1=16, h2=8):
    """Build a small random MLP weight set for testing."""
    rng = np.random.default_rng(0)
    return MLPWeights(
        w1=rng.standard_normal((10, h1)),
        b1=np.zeros(h1),
        w2=rng.standard_normal((h1, h2)),
        b2=np.zeros(h2),
        w3=rng.standard_normal((h2, 3)),
        b3=np.zeros(3),
        input_mean=np.zeros(10),
        input_std=np.ones(10),
        output_scale=np.ones(3),
    )


class TestMLPForward:
    def test_single_input_shape(self):
        w = _make_weights()
        out = _mlp_forward(np.zeros(10), w)
        assert out.shape == (3,)

    def test_batch_input_shape(self):
        w = _make_weights()
        out = _mlp_forward(np.zeros((5, 10)), w)
        assert out.shape == (5, 3)

    def test_output_nonnegative(self):
        w = _make_weights()
        out = _mlp_forward(np.random.default_rng(1).standard_normal(10), w)
        assert np.all(out >= 0.0)  # softplus ensures this

    def test_output_scale_applied(self):
        w = _make_weights()
        w_scaled = _make_weights()
        w_scaled.output_scale = np.array([2.0, 3.0, 4.0])
        out_base = _mlp_forward(np.zeros(10), w)
        out_scaled = _mlp_forward(np.zeros(10), w_scaled)
        assert np.allclose(out_scaled, out_base * [2.0, 3.0, 4.0])

    def test_input_normalisation(self):
        w = _make_weights()
        w.input_mean = np.ones(10) * 5.0
        w.input_std = np.ones(10) * 2.0
        out = _mlp_forward(np.ones(10) * 5.0, w)
        out_zero = _mlp_forward(np.zeros(10), w)
        # Feeding the mean should produce the same as feeding zero (after norm)
        # ... when mean is subtracted, it becomes zero in both cases only if input == mean
        # Actually: (5 - 5)/2 = 0 vs (0 - 5)/2 = -2.5, so they differ
        assert not np.allclose(out, out_zero)


# ── Weight loading via .npz ────────────────────────────────────────


class TestNeuralTransportModel:
    def test_fallback_mode(self):
        model = NeuralTransportModel(auto_discover=False)
        assert not model.is_neural

    def test_missing_path_falls_back(self):
        with pytest.raises(FileNotFoundError, match="allow_weight_load_fallback=True"):
            NeuralTransportModel("/nonexistent/path.npz")

    def test_missing_path_legacy_fallback_opt_in(self):
        model = NeuralTransportModel(
            "/nonexistent/path.npz",
            allow_weight_load_fallback=True,
            allow_legacy_weight_load_fallback=True,
        )
        assert not model.is_neural

    def test_load_valid_weights(self):
        w = _make_weights()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                w1=w.w1,
                b1=w.b1,
                w2=w.w2,
                b2=w.b2,
                w3=w.w3,
                b3=w.b3,
                input_mean=w.input_mean,
                input_std=w.input_std,
                output_scale=w.output_scale,
                version=np.array(1),
            )
            model = NeuralTransportModel(f.name)
        assert model.is_neural
        assert model.weights_checksum is not None

    def test_predict_neural(self):
        w = _make_weights()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                w1=w.w1,
                b1=w.b1,
                w2=w.w2,
                b2=w.b2,
                w3=w.w3,
                b3=w.b3,
                input_mean=w.input_mean,
                input_std=w.input_std,
                output_scale=w.output_scale,
                version=np.array(1),
            )
            model = NeuralTransportModel(f.name)
        fluxes = model.predict(TransportInputs(grad_ti=8.0))
        assert isinstance(fluxes, TransportFluxes)
        assert fluxes.chi_e >= 0
        assert fluxes.chi_i >= 0

    def test_predict_fallback(self):
        model = NeuralTransportModel(auto_discover=False)
        fluxes = model.predict(TransportInputs(grad_ti=8.0))
        expected = critical_gradient_model(TransportInputs(grad_ti=8.0))
        assert fluxes.chi_i == expected.chi_i

    def test_wrong_version_falls_back(self):
        w = _make_weights()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                w1=w.w1,
                b1=w.b1,
                w2=w.w2,
                b2=w.b2,
                w3=w.w3,
                b3=w.b3,
                input_mean=w.input_mean,
                input_std=w.input_std,
                output_scale=w.output_scale,
                version=np.array(99),
            )
            with pytest.raises(
                RuntimeError,
                match="Failed to load explicit neural transport weights.*allow_weight_load_fallback=True",
            ):
                NeuralTransportModel(f.name)

    def test_wrong_version_legacy_fallback_opt_in(self):
        w = _make_weights()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                w1=w.w1,
                b1=w.b1,
                w2=w.w2,
                b2=w.b2,
                w3=w.w3,
                b3=w.b3,
                input_mean=w.input_mean,
                input_std=w.input_std,
                output_scale=w.output_scale,
                version=np.array(99),
            )
            model = NeuralTransportModel(
                f.name,
                allow_weight_load_fallback=True,
                allow_legacy_weight_load_fallback=True,
            )
        assert not model.is_neural

    def test_missing_key_falls_back(self):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, w1=np.zeros((10, 16)))
            with pytest.raises(
                RuntimeError,
                match="Failed to load explicit neural transport weights.*allow_weight_load_fallback=True",
            ):
                NeuralTransportModel(f.name)

    def test_missing_key_legacy_fallback_opt_in(self):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, w1=np.zeros((10, 16)))
            model = NeuralTransportModel(
                f.name,
                allow_weight_load_fallback=True,
                allow_legacy_weight_load_fallback=True,
            )
        assert not model.is_neural

    def test_weight_load_legacy_fallback_requires_explicit_opt_in(self):
        with pytest.raises(ValueError, match="allow_legacy_weight_load_fallback=True"):
            NeuralTransportModel(
                "/nonexistent/path.npz",
                allow_weight_load_fallback=True,
                allow_legacy_weight_load_fallback=False,
            )


# ── Profile prediction ─────────────────────────────────────────────


class TestPredictProfile:
    def _make_profiles(self, n=32):
        rho = np.linspace(0.01, 0.99, n)
        te = 10.0 * (1 - rho**2)
        ti = 9.0 * (1 - rho**2)
        ne = 8.0 * (1 - 0.5 * rho**2)
        q = 1.0 + 2.0 * rho**2
        s_hat = 0.5 + 1.5 * rho
        return rho, te, ti, ne, q, s_hat

    def test_fallback_profile_shape(self):
        model = NeuralTransportModel(auto_discover=False)
        rho, te, ti, ne, q, s_hat = self._make_profiles()
        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        assert chi_e.shape == rho.shape
        assert chi_i.shape == rho.shape
        assert d_e.shape == rho.shape

    def test_fallback_profile_nonneg(self):
        model = NeuralTransportModel(auto_discover=False)
        rho, te, ti, ne, q, s_hat = self._make_profiles()
        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        assert np.all(chi_e >= 0)
        assert np.all(chi_i >= 0)
        assert np.all(d_e >= 0)

    def test_neural_profile_shape(self):
        w = _make_weights()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                w1=w.w1,
                b1=w.b1,
                w2=w.w2,
                b2=w.b2,
                w3=w.w3,
                b3=w.b3,
                input_mean=w.input_mean,
                input_std=w.input_std,
                output_scale=w.output_scale,
                version=np.array(1),
            )
            model = NeuralTransportModel(f.name)
        rho, te, ti, ne, q, s_hat = self._make_profiles()
        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        assert chi_e.shape == rho.shape
        assert chi_i.shape == rho.shape

    def test_d_e_uses_profile_density_gradient_and_shear(self):
        model = NeuralTransportModel(auto_discover=False)
        rho, te, ti, ne, q, s_hat = self._make_profiles()
        chi_e, _, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        mask = chi_e > 0
        if mask.any():
            ratio = d_e[mask] / chi_e[mask]
            assert not np.allclose(ratio, 1.0 / 3.0, atol=1e-10)
            assert np.all((0.05 <= ratio) & (ratio <= 0.65))
