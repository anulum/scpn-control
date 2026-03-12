# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Tests for QLKNN-10D trained transport model
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Tests: QLKNN weight loading, inference, training script, profile prediction."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core.neural_transport import (
    NeuralTransportModel,
    TransportFluxes,
    TransportInputs,
    _DEFAULT_WEIGHTS_PATH,
    cross_validate_neural_transport,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPO_ROOT / "weights" / "neural_transport_qlknn.npz"
METRICS_PATH = REPO_ROOT / "weights" / "neural_transport_qlknn.metrics.json"


# ── Weight file existence ────────────────────────────────────────────


class TestWeightFiles:
    def test_weights_exist(self) -> None:
        assert WEIGHTS_PATH.exists(), f"Missing {WEIGHTS_PATH}"

    def test_metrics_exist(self) -> None:
        assert METRICS_PATH.exists(), f"Missing {METRICS_PATH}"

    def test_weights_contain_required_keys(self) -> None:
        data = np.load(WEIGHTS_PATH)
        for key in ["w1", "b1", "w2", "b2", "w3", "b3", "input_mean", "input_std", "output_scale"]:
            assert key in data, f"Missing key: {key}"

    def test_weight_shapes(self) -> None:
        data = np.load(WEIGHTS_PATH)
        assert data["w1"].shape == (10, 128)
        assert data["b1"].shape == (128,)
        assert data["w2"].shape == (128, 64)
        assert data["b2"].shape == (64,)
        assert data["w3"].shape == (64, 3)
        assert data["b3"].shape == (3,)
        assert data["input_mean"].shape == (10,)
        assert data["input_std"].shape == (10,)
        assert data["output_scale"].shape == (3,)

    def test_metrics_valid(self) -> None:
        with open(METRICS_PATH) as f:
            m = json.load(f)
        assert m["n_total"] == 5000
        assert m["synthetic"] is True
        assert len(m["per_channel_rmse"]) == 3
        assert m["test_rmse"] > 0


# ── Model loading and inference ──────────────────────────────────────


class TestQLKNNModel:
    @pytest.fixture()
    def model(self) -> NeuralTransportModel:
        return NeuralTransportModel(WEIGHTS_PATH)

    def test_loads_as_neural(self, model: NeuralTransportModel) -> None:
        assert model.is_neural

    def test_auto_discover(self) -> None:
        m = NeuralTransportModel()
        assert m.is_neural
        assert m.weights_path == _DEFAULT_WEIGHTS_PATH

    def test_auto_discover_disabled(self) -> None:
        m = NeuralTransportModel(auto_discover=False)
        assert not m.is_neural

    def test_checksum_stable(self, model: NeuralTransportModel) -> None:
        m2 = NeuralTransportModel(WEIGHTS_PATH)
        assert model.weights_checksum == m2.weights_checksum

    def test_predict_returns_fluxes(self, model: NeuralTransportModel) -> None:
        fluxes = model.predict(TransportInputs(grad_ti=8.0, grad_te=10.0))
        assert isinstance(fluxes, TransportFluxes)

    def test_output_nonnegative(self, model: NeuralTransportModel) -> None:
        fluxes = model.predict(TransportInputs(grad_ti=8.0, grad_te=10.0))
        assert fluxes.chi_e >= 0
        assert fluxes.chi_i >= 0
        assert fluxes.d_e >= 0

    def test_high_gradient_produces_transport(self, model: NeuralTransportModel) -> None:
        fluxes = model.predict(TransportInputs(grad_ti=20.0, grad_te=20.0))
        assert fluxes.chi_i > 0
        assert fluxes.chi_e > 0

    def test_gradient_monotonicity(self, model: NeuralTransportModel) -> None:
        low = model.predict(TransportInputs(grad_ti=0.0, grad_te=0.0))
        high = model.predict(TransportInputs(grad_ti=20.0, grad_te=20.0))
        assert high.chi_i > low.chi_i
        assert high.chi_e > low.chi_e

    def test_deterministic(self, model: NeuralTransportModel) -> None:
        inp = TransportInputs(grad_ti=12.0, grad_te=8.0)
        f1 = model.predict(inp)
        f2 = model.predict(inp)
        assert f1.chi_e == f2.chi_e
        assert f1.chi_i == f2.chi_i

    def test_channel_assignment(self, model: NeuralTransportModel) -> None:
        fluxes = model.predict(TransportInputs(grad_ti=20.0, grad_te=2.0))
        assert fluxes.channel in {"ITG", "TEM", "stable"}


# ── Profile prediction with QLKNN weights ────────────────────────────


class TestQLKNNProfile:
    @pytest.fixture()
    def model(self) -> NeuralTransportModel:
        return NeuralTransportModel(WEIGHTS_PATH)

    def _profiles(self, n: int = 32) -> tuple:
        rho = np.linspace(0.01, 0.99, n)
        te = 10.0 * (1 - rho**2)
        ti = 9.0 * (1 - rho**2)
        ne = 8.0 * (1 - 0.5 * rho**2)
        q = 1.0 + 2.0 * rho**2
        s_hat = 0.5 + 1.5 * rho
        return rho, te, ti, ne, q, s_hat

    def test_profile_shape(self, model: NeuralTransportModel) -> None:
        rho, te, ti, ne, q, s_hat = self._profiles()
        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        assert chi_e.shape == rho.shape
        assert chi_i.shape == rho.shape
        assert d_e.shape == rho.shape

    def test_profile_nonneg(self, model: NeuralTransportModel) -> None:
        rho, te, ti, ne, q, s_hat = self._profiles()
        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        assert np.all(chi_e >= 0)
        assert np.all(chi_i >= 0)
        assert np.all(d_e >= 0)

    def test_profile_finite(self, model: NeuralTransportModel) -> None:
        rho, te, ti, ne, q, s_hat = self._profiles()
        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)
        assert np.all(np.isfinite(chi_e))
        assert np.all(np.isfinite(chi_i))
        assert np.all(np.isfinite(d_e))


class TestReferenceCrossValidation:
    def test_fallback_matches_reference_exactly(self) -> None:
        model = NeuralTransportModel(auto_discover=False)
        metrics = cross_validate_neural_transport(model)
        assert metrics["mode"] == "analytic_fallback"
        assert metrics["channel_agreement"] == pytest.approx(1.0)
        assert metrics["max_abs_error"] == pytest.approx(0.0)
        assert np.allclose(metrics["per_channel_rmse"], 0.0)
        assert np.allclose(metrics["per_channel_mae"], 0.0)
        assert np.allclose(metrics["profile_per_channel_rmse"], 0.0)

    def test_neural_model_stays_close_to_reference_benchmark(self) -> None:
        model = NeuralTransportModel(WEIGHTS_PATH)
        metrics = cross_validate_neural_transport(model)
        assert metrics["mode"] == "neural"
        assert metrics["n_cases"] == 7
        assert len(metrics["per_channel_relative_rmse"]) == 3
        assert len(metrics["profile_per_channel_relative_rmse"]) == 3
        assert metrics["channel_agreement"] >= 5.0 / 7.0
        assert max(metrics["per_channel_relative_rmse"]) < 0.65
        assert max(metrics["profile_per_channel_relative_rmse"]) < 0.35
        assert metrics["max_abs_error"] < 12.0


# ── Training script ──────────────────────────────────────────────────


class TestTrainingScript:
    def test_script_runs_synthetic(self, tmp_path: Path) -> None:
        out = tmp_path / "test_weights.npz"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "train_neural_transport_qlknn.py"),
                "--synthetic",
                "--epochs",
                "5",
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        assert out.with_suffix(".metrics.json").exists()

    def test_trained_weights_loadable(self, tmp_path: Path) -> None:
        out = tmp_path / "test_weights.npz"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "train_neural_transport_qlknn.py"),
                "--synthetic",
                "--epochs",
                "5",
                "--output",
                str(out),
            ],
            capture_output=True,
            timeout=60,
        )
        model = NeuralTransportModel(out)
        assert model.is_neural
