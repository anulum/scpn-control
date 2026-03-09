"""Tests for Python fallback implementations when Rust backend is unavailable.

These tests exercise _python_svd_optimal_correction, _python_multigrid_vcycle,
and the Python path of RustSPIMitigation directly (bypassing Rust try/except).
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.core._rust_compat import (
    RustSPIMitigation,
    _python_multigrid_vcycle,
    _python_svd_optimal_correction,
)


# ─── SVD optimal correction fallback ──────────────────────────────────


class TestSVDFallback:
    def test_identity_response(self):
        J = np.eye(2)
        e = np.array([0.1, -0.2])
        delta = _python_svd_optimal_correction(J, e, gain=0.8)
        np.testing.assert_allclose(delta, 0.8 * e, atol=1e-12)

    def test_rectangular_response(self):
        J = np.array([[1.0, 0.5, 0.0], [0.0, 0.3, 1.0]])
        e = np.array([0.1, -0.2])
        delta = _python_svd_optimal_correction(J, e, gain=1.0)
        assert delta.shape == (3,)
        assert np.all(np.isfinite(delta))
        residual = np.linalg.norm(J @ delta - e)
        assert residual < np.linalg.norm(e)

    def test_gain_scaling(self):
        J = np.eye(3)
        e = np.array([1.0, 2.0, 3.0])
        d1 = _python_svd_optimal_correction(J, e, gain=0.5)
        d2 = _python_svd_optimal_correction(J, e, gain=1.0)
        np.testing.assert_allclose(d1, 0.5 * d2, atol=1e-12)

    def test_singular_value_cutoff(self):
        J = np.array([[1.0, 0.0], [0.0, 1e-10]])
        e = np.array([1.0, 1.0])
        delta = _python_svd_optimal_correction(J, e, gain=1.0)
        assert np.isfinite(delta[0])
        assert np.isfinite(delta[1])

    def test_rejects_dimension_mismatch(self):
        J = np.eye(2)
        with pytest.raises(ValueError, match="error length"):
            _python_svd_optimal_correction(J, np.array([1.0, 2.0, 3.0]), gain=1.0)

    def test_rejects_1d_matrix(self):
        with pytest.raises(ValueError, match="2D"):
            _python_svd_optimal_correction(np.array([1.0, 2.0]), np.array([1.0]), gain=1.0)


# ─── SPI mitigation fallback ─────────────────────────────────────────


class TestSPIFallback:
    def _make_spi_python(self, **kwargs):
        """Force Python path by patching."""
        spi = object.__new__(RustSPIMitigation)
        defaults = {"w_th_mj": 300.0, "ip_ma": 15.0, "te_kev": 20.0}
        defaults.update(kwargs)
        spi._use_rust = False
        spi._w_th = defaults["w_th_mj"] * 1e6
        spi._ip = defaults["ip_ma"] * 1e6
        spi._te = defaults["te_kev"]
        return spi

    def test_returns_dict_list(self):
        spi = self._make_spi_python()
        history = spi.run()
        assert isinstance(history, list)
        assert len(history) > 0
        snap = history[0]
        assert set(snap.keys()) == {"time", "w_th_mj", "ip_ma", "te_kev", "phase"}

    def test_energy_decreases(self):
        spi = self._make_spi_python()
        history = spi.run()
        energies = [s["w_th_mj"] for s in history]
        assert energies[-1] < energies[0]

    def test_current_decreases(self):
        spi = self._make_spi_python()
        history = spi.run()
        currents = [s["ip_ma"] for s in history]
        assert currents[-1] < currents[0]

    def test_phase_progression(self):
        spi = self._make_spi_python()
        history = spi.run()
        phases = [s["phase"] for s in history]
        assert phases[0] == "Assimilation"
        assert "ThermalQuench" in phases
        assert "CurrentQuench" in phases

    def test_correct_step_count(self):
        spi = self._make_spi_python()
        history = spi.run()
        expected = int(0.05 / 1e-5)
        assert len(history) == expected

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError, match="w_th_mj"):
            RustSPIMitigation(w_th_mj=0.0)
        with pytest.raises(ValueError, match="ip_ma"):
            RustSPIMitigation(ip_ma=-1.0)
        with pytest.raises(ValueError, match="te_kev"):
            RustSPIMitigation(te_kev=float("nan"))


# ─── Multigrid V-cycle fallback ───────────────────────────────────────


class TestMultigridFallback:
    def test_residual_decreases(self):
        nr, nz = 17, 17
        source = np.ones((nz, nr))
        psi_init = np.zeros((nz, nr))

        _, r1, _, _ = _python_multigrid_vcycle(
            source,
            psi_init,
            1.0,
            9.0,
            -5.0,
            5.0,
            nr,
            nz,
            tol=1e-12,
            max_cycles=1,
        )
        _, r5, _, _ = _python_multigrid_vcycle(
            source,
            psi_init,
            1.0,
            9.0,
            -5.0,
            5.0,
            nr,
            nz,
            tol=1e-12,
            max_cycles=5,
        )
        assert r5 < r1, f"residual should decrease: {r1} → {r5}"

    def test_preserves_boundaries(self):
        nr, nz = 17, 17
        source = np.ones((nz, nr))
        psi_init = np.zeros((nz, nr))
        psi, _, _, _ = _python_multigrid_vcycle(
            source,
            psi_init,
            1.0,
            9.0,
            -5.0,
            5.0,
            nr,
            nz,
            tol=1e-4,
            max_cycles=30,
        )
        assert np.max(np.abs(psi[0, :])) < 1e-14
        assert np.max(np.abs(psi[-1, :])) < 1e-14
        assert np.max(np.abs(psi[:, 0])) < 1e-14
        assert np.max(np.abs(psi[:, -1])) < 1e-14

    def test_zero_source_gives_zero_solution(self):
        nr, nz = 9, 9
        source = np.zeros((nz, nr))
        psi_init = np.zeros((nz, nr))
        psi, residual, _, _ = _python_multigrid_vcycle(
            source,
            psi_init,
            1.0,
            5.0,
            -2.0,
            2.0,
            nr,
            nz,
            tol=1e-8,
            max_cycles=10,
        )
        assert np.max(np.abs(psi)) < 1e-12
        assert residual < 1e-8

    def test_returns_tuple_format(self):
        nr, nz = 9, 9
        result = _python_multigrid_vcycle(
            np.ones((nz, nr)),
            np.zeros((nz, nr)),
            1.0,
            5.0,
            -2.0,
            2.0,
            nr,
            nz,
            tol=1e-4,
            max_cycles=5,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4
        psi, residual, n_cycles, converged = result
        assert isinstance(psi, np.ndarray)
        assert isinstance(residual, float)
        assert isinstance(n_cycles, int)
        assert isinstance(converged, bool)
