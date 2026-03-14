# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Coverage tests for _rust_compat uncovered lines
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core import _rust_compat


# ── Module-level flags ──────────────────────────────────────────────


def test_rust_available_returns_false():
    assert _rust_compat._rust_available() is False


def test_rust_backend_is_false():
    assert _rust_compat.RUST_BACKEND is False


# ── Rust-only helpers raise ImportError ─────────────────────────────


def test_rust_shafranov_bv_raises():
    with pytest.raises(ImportError, match="scpn_control_rs not installed"):
        _rust_compat.rust_shafranov_bv(6.2, 2.0, 15.0)


def test_rust_solve_coil_currents_raises():
    with pytest.raises(ImportError, match="scpn_control_rs not installed"):
        _rust_compat.rust_solve_coil_currents()


def test_rust_bosch_hale_dt_raises():
    with pytest.raises(ImportError, match="scpn_control_rs not installed"):
        _rust_compat.rust_bosch_hale_dt(10.0)


# ── rust_simulate_tearing_mode Python fallback ──────────────────────


def test_tearing_mode_fallback_no_seed():
    signal, label, ttd = _rust_compat.rust_simulate_tearing_mode(steps=30)
    assert len(signal) == 30
    assert label in (0, 1)


def test_tearing_mode_fallback_with_seed():
    a = _rust_compat.rust_simulate_tearing_mode(steps=20, seed=99)
    b = _rust_compat.rust_simulate_tearing_mode(steps=20, seed=99)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]


# ── Rust controller classes raise ImportError ───────────────────────


def test_rust_snn_pool_init_raises():
    with pytest.raises(ImportError):
        _rust_compat.RustSnnPool()


def test_rust_snn_controller_init_raises():
    with pytest.raises(ImportError):
        _rust_compat.RustSnnController()


def test_rust_pid_controller_init_raises():
    with pytest.raises(ImportError):
        _rust_compat.RustPIDController(1.0, 0.1, 0.01)


def test_rust_pid_controller_radial_raises():
    with pytest.raises(ImportError):
        _rust_compat.RustPIDController.radial()


def test_rust_pid_controller_vertical_raises():
    with pytest.raises(ImportError):
        _rust_compat.RustPIDController.vertical()


def test_rust_isoflux_controller_init_raises():
    with pytest.raises(ImportError):
        _rust_compat.RustIsoFluxController(6.2, 0.0)


def test_rust_hinf_controller_init_raises():
    with pytest.raises(ImportError):
        _rust_compat.RustHInfController()


# ── RustAcceleratedKernel (requires mock PyFusionKernel) ────────────


class _NoJPhiKernel:
    """Mock Rust kernel where get_j_phi raises AttributeError."""

    def __init__(self, _path: str):
        self._psi = np.zeros((5, 5))
        self._r = np.linspace(1.0, 3.0, 5)
        self._z = np.linspace(-1.0, 1.0, 5)

    def get_r(self):
        return self._r.tolist()

    def get_z(self):
        return self._z.tolist()

    def get_psi(self):
        return self._psi.copy()

    def get_j_phi(self):
        raise AttributeError("no j_phi")

    def solve_equilibrium(self):
        return object()

    def compute_b_field(self):
        raise AttributeError("no compute_b_field")


class _NoProbesKernel(_NoJPhiKernel):
    """Mock missing sample_psi_at_probes (triggers per-point fallback)."""

    def get_j_phi(self):
        return np.zeros((5, 5))

    def sample_psi_at(self, r, z):
        return r + z

    def compute_b_field(self):
        return np.zeros((5, 5)), np.zeros((5, 5))


def _cfg_path(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"dimensions": {"Z_min": -1.0}}), encoding="utf-8")
    return p


def test_init_j_phi_attribute_error(monkeypatch, tmp_path):
    """Line 70-71: get_j_phi raises → J_phi = zeros."""
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _NoJPhiKernel, raising=False)
    w = _rust_compat.RustAcceleratedKernel(str(_cfg_path(tmp_path)))
    np.testing.assert_array_equal(w.J_phi, np.zeros((5, 5)))


def test_solve_equilibrium_j_phi_attribute_error(monkeypatch, tmp_path):
    """Line 84-85: get_j_phi raises in solve_equilibrium → pass."""
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _NoJPhiKernel, raising=False)
    w = _rust_compat.RustAcceleratedKernel(str(_cfg_path(tmp_path)))
    w.solve_equilibrium()
    # J_phi stays zeros since get_j_phi raises
    np.testing.assert_array_equal(w.J_phi, np.zeros((5, 5)))


def test_compute_b_field_fallback(monkeypatch, tmp_path):
    """Lines 118-122: compute_b_field raises → Python gradient fallback."""
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _NoJPhiKernel, raising=False)
    w = _rust_compat.RustAcceleratedKernel(str(_cfg_path(tmp_path)))
    w.Psi = np.random.default_rng(0).standard_normal((5, 5))
    w.compute_b_field()
    assert w.B_R.shape == (5, 5)
    assert w.B_Z.shape == (5, 5)
    assert np.all(np.isfinite(w.B_R))


def test_compute_b_field_fallback_matches_analytic_field(monkeypatch, tmp_path):
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _NoJPhiKernel, raising=False)
    w = _rust_compat.RustAcceleratedKernel(str(_cfg_path(tmp_path)))
    w.Psi = w.RR**2 + 2.0 * w.ZZ**2
    w.compute_b_field()

    r_safe = np.maximum(w.RR, 1e-6)
    expected_b_r = -(4.0 * w.ZZ) / r_safe
    expected_b_z = np.full_like(w.RR, 2.0)

    np.testing.assert_allclose(w.B_R[1:-1, 1:-1], expected_b_r[1:-1, 1:-1], atol=1e-10)
    np.testing.assert_allclose(w.B_Z[1:-1, 1:-1], expected_b_z[1:-1, 1:-1], atol=1e-10)


def test_sample_psi_at_probes_no_rust_method(monkeypatch, tmp_path):
    """Line 110: sample_psi_at_probes absent → per-point loop."""
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _NoProbesKernel, raising=False)
    w = _rust_compat.RustAcceleratedKernel(str(_cfg_path(tmp_path)))
    # Remove sample_psi_at_probes from the inner object
    if hasattr(w._rust, "sample_psi_at_probes"):
        delattr(w._rust, "sample_psi_at_probes")
    vals = w.sample_psi_at_probes([(1.5, 0.0), (2.0, 0.5)])
    np.testing.assert_allclose(vals, [1.5, 2.5])


def test_find_x_point_no_divertor_mask(monkeypatch, tmp_path):
    """Line 140: no divertor region → return (0,0), min(Psi)."""
    # Z_min=-100 → threshold=-50 → ZZ (in [-1,1]) never below → empty mask
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"dimensions": {"Z_min": -100.0}}), encoding="utf-8")
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _NoProbesKernel, raising=False)
    w = _rust_compat.RustAcceleratedKernel(str(p))
    psi = np.random.default_rng(7).standard_normal((5, 5))
    (rx, zx), psi_x = w.find_x_point(psi)
    assert rx == 0
    assert zx == 0
    assert psi_x == np.min(psi)


def test_find_x_point_prefers_saddle_field(monkeypatch, tmp_path):
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _NoProbesKernel, raising=False)
    w = _rust_compat.RustAcceleratedKernel(str(_cfg_path(tmp_path)))
    psi = (w.RR - 2.0) ** 2 - (w.ZZ + 0.75) ** 2

    (rx, zx), psi_x = w.find_x_point(psi)

    assert abs(rx - 2.0) <= w.dR
    assert abs(zx + 0.75) <= w.dZ
    assert abs(float(psi_x)) <= 4.0 * max(w.dR, w.dZ)


def test_calculate_vacuum_field(monkeypatch, tmp_path):
    """Lines 144-147: delegates to Python FusionKernel."""
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _NoProbesKernel, raising=False)
    w = _rust_compat.RustAcceleratedKernel(str(_cfg_path(tmp_path)))

    # Mock the Python FusionKernel so we don't need a real config
    class _MockPyFK:
        def __init__(self, path):
            pass

        def calculate_vacuum_field(self):
            return np.ones((3, 3))

    monkeypatch.setattr("scpn_control.core.fusion_kernel.FusionKernel", _MockPyFK, raising=True)
    result = w.calculate_vacuum_field()
    np.testing.assert_array_equal(result, np.ones((3, 3)))


# ── Public fallback functions (hit except ImportError → Python) ─────


def test_rust_multigrid_vcycle_fallback():
    """Lines 467-473: Rust import fails → _python_multigrid_vcycle."""
    nr, nz = 9, 9
    source = np.ones((nz, nr))
    psi_bc = np.zeros((nz, nr))
    psi, residual, n_cycles, converged = _rust_compat.rust_multigrid_vcycle(
        source, psi_bc, 1.0, 5.0, -2.0, 2.0, nr, nz, tol=1e-4, max_cycles=30
    )
    assert psi.shape == (nz, nr)
    assert isinstance(residual, float)
    assert isinstance(n_cycles, int)
    assert isinstance(converged, bool)


def test_rust_svd_optimal_correction_fallback():
    """Lines 619-630: Rust import fails → _python_svd_optimal_correction."""
    J = np.eye(2)
    e = np.array([0.1, -0.2])
    delta = _rust_compat.rust_svd_optimal_correction(J, e, gain=0.8)
    np.testing.assert_allclose(delta, 0.8 * e, atol=1e-12)


# ── RustSPIMitigation Python path via __init__ ─────────────────────


def test_spi_init_falls_back_to_python():
    """Lines 542-551: Rust import fails → Python fallback."""
    spi = _rust_compat.RustSPIMitigation(w_th_mj=100.0, ip_ma=10.0, te_kev=15.0)
    assert spi._use_rust is False
    assert spi._w_th == 100.0e6
    assert spi._ip == 10.0e6
    assert spi._te == 15.0


def test_spi_run_python_path():
    """Lines 558-594: full Python SPI simulation."""
    spi = _rust_compat.RustSPIMitigation(w_th_mj=300.0, ip_ma=15.0, te_kev=20.0)
    history = spi.run()
    assert len(history) == 5000
    assert history[0]["phase"] == "Assimilation"
    phases = {s["phase"] for s in history}
    assert "ThermalQuench" in phases
    assert "CurrentQuench" in phases
    assert history[-1]["w_th_mj"] < history[0]["w_th_mj"]


# ── _python_svd_optimal_correction edge cases ──────────────────────


def test_python_svd_rejects_3d_matrix():
    with pytest.raises(ValueError, match="2D"):
        _rust_compat._python_svd_optimal_correction(np.zeros((2, 2, 2)), np.zeros(2), 1.0)


def test_python_svd_rejects_length_mismatch():
    with pytest.raises(ValueError, match="error length"):
        _rust_compat._python_svd_optimal_correction(np.eye(2), np.zeros(3), 1.0)


# ── Rust wrapper method bodies (bypassing __init__ via object.__new__) ──


def _make_stub(cls):
    """Construct cls without calling __init__, injecting a mock _inner."""
    return object.__new__(cls)


class _MockInner:
    """Generic mock for Rust inner objects."""

    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)

    def step(self, *args):
        if len(args) == 1:
            return 0.5
        return 0.1, -0.1

    def reset(self):
        pass


class TestRustSnnPoolMethods:
    def _make(self):
        obj = _make_stub(_rust_compat.RustSnnPool)
        obj._inner = _MockInner(n_neurons=50, gain=10.0)
        return obj

    def test_step(self):
        assert self._make().step(0.1) == 0.5

    def test_n_neurons(self):
        assert self._make().n_neurons == 50

    def test_gain(self):
        assert self._make().gain == 10.0

    def test_repr(self):
        assert "RustSnnPool" in repr(self._make())


class TestRustSnnControllerMethods:
    def _make(self):
        obj = _make_stub(_rust_compat.RustSnnController)
        obj._inner = _MockInner(target_r=6.2, target_z=0.0)
        return obj

    def test_step(self):
        r, z = self._make().step(6.3, 0.1)
        assert isinstance(r, float) and isinstance(z, float)

    def test_target_r(self):
        assert self._make().target_r == 6.2

    def test_target_z(self):
        assert self._make().target_z == 0.0

    def test_repr(self):
        assert "RustSnnController" in repr(self._make())


class TestRustPIDControllerMethods:
    def _make(self):
        obj = _make_stub(_rust_compat.RustPIDController)
        obj._inner = _MockInner(kp=1.0, ki=0.1, kd=0.01)
        return obj

    def test_step(self):
        assert self._make().step(0.5) == 0.5

    def test_reset(self):
        self._make().reset()

    def test_kp(self):
        assert self._make().kp == 1.0

    def test_ki(self):
        assert self._make().ki == 0.1

    def test_kd(self):
        assert self._make().kd == 0.01

    def test_repr(self):
        assert "RustPIDController" in repr(self._make())


class TestRustIsoFluxControllerMethods:
    def _make(self):
        obj = _make_stub(_rust_compat.RustIsoFluxController)
        obj._inner = _MockInner(target_r=6.2, target_z=0.0)
        return obj

    def test_step(self):
        r, z = self._make().step(6.3, 0.1)
        assert isinstance(r, float) and isinstance(z, float)

    def test_target_r(self):
        assert self._make().target_r == 6.2

    def test_target_z(self):
        assert self._make().target_z == 0.0

    def test_repr(self):
        assert "RustIsoFluxController" in repr(self._make())


class _HInfMockInner:
    gamma = 1.0

    def step(self, y, dt):
        return 0.5

    def reset(self):
        pass


class TestRustHInfControllerMethods:
    def _make(self):
        obj = _make_stub(_rust_compat.RustHInfController)
        obj._inner = _HInfMockInner()
        obj._u_max = 10.0
        return obj

    def test_step(self):
        u = self._make().step(0.5, 1e-3)
        assert isinstance(u, float)
        assert abs(u) <= 10.0

    def test_step_saturation(self):
        obj = self._make()
        obj._inner.step = lambda y, dt: 999.0
        assert obj.step(0.5, 1e-3) == 10.0

    def test_reset(self):
        self._make().reset()

    def test_gamma(self):
        assert self._make().gamma == 1.0

    def test_u_max(self):
        assert self._make().u_max == 10.0

    def test_repr(self):
        assert "RustHInfController" in repr(self._make())


class TestRustSPIMitigationRustPath:
    """Exercise line 556: self._inner.run() when _use_rust is True."""

    def test_run_delegates_to_inner(self):
        obj = _make_stub(_rust_compat.RustSPIMitigation)
        obj._use_rust = True
        obj._inner = type("FakeInner", (), {"run": lambda self: [{"t": 0}]})()
        assert obj.run() == [{"t": 0}]
