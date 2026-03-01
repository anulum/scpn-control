# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Rust Compat Wrapper Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.core import _rust_compat

_HAS_RUST = _rust_compat._rust_available()


class _DummyRustKernel:
    def __init__(self, _config_path: str) -> None:
        self._method = "sor"
        self._psi = np.zeros((5, 5), dtype=float)
        self._j_phi = np.zeros((5, 5), dtype=float)
        self._r = np.linspace(1.0, 3.0, 5, dtype=float)
        self._z = np.linspace(-1.0, 1.0, 5, dtype=float)

    def grid_shape(self) -> tuple[int, int]:
        return (5, 5)

    def get_r(self) -> list[float]:
        return self._r.tolist()

    def get_z(self) -> list[float]:
        return self._z.tolist()

    def get_psi(self) -> np.ndarray:
        return self._psi.copy()

    def get_j_phi(self) -> np.ndarray:
        return self._j_phi.copy()

    def solve_equilibrium(self) -> object:
        return object()

    def calculate_thermodynamics(self, _p_aux_mw: float) -> object:
        return object()

    def set_solver_method(self, method: str) -> None:
        aliases = {
            "sor": "sor",
            "picard_sor": "sor",
            "multigrid": "multigrid",
            "picard_multigrid": "multigrid",
            "mg": "multigrid",
        }
        if method not in aliases:
            raise ValueError("Unknown solver method")
        self._method = aliases[method]

    def solver_method(self) -> str:
        return self._method

    def sample_psi_at(self, r: float, z: float) -> float:
        return 0.0

    def sample_psi_at_probes(self, probes: list) -> list[float]:
        return [0.0] * len(probes)

    def compute_b_field(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((5, 5)), np.zeros((5, 5))

    def find_x_point(self) -> tuple:
        return (2.0, -0.5), 0.0

    def calculate_vacuum_field(self) -> np.ndarray:
        return np.zeros((5, 5))


def _write_config(path: Path) -> None:
    cfg = {"dimensions": {"Z_min": -1.0}}
    path.write_text(json.dumps(cfg), encoding="utf-8")


# ── Solver method tests (existing) ─────────────────────────────────


def test_rust_wrapper_solver_method_forwarding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))
    assert wrapper.solver_method() == "sor"

    wrapper.set_solver_method("mg")
    assert wrapper.solver_method() == "multigrid"

    wrapper.set_solver_method("picard_sor")
    assert wrapper.solver_method() == "sor"


def test_rust_wrapper_solver_method_propagates_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    wrapper = _rust_compat.RustAcceleratedKernel(str(cfg))
    with pytest.raises(ValueError, match="Unknown solver method"):
        wrapper.set_solver_method("invalid")


# ── RustAcceleratedKernel method coverage ───────────────────────────


def test_wrapper_grid_attributes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    w = _rust_compat.RustAcceleratedKernel(str(cfg))
    assert w.NR == 5
    assert w.NZ == 5
    assert len(w.R) == 5
    assert len(w.Z) == 5
    assert w.RR.shape == (5, 5)
    assert w.ZZ.shape == (5, 5)


def test_wrapper_solve_equilibrium(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    w = _rust_compat.RustAcceleratedKernel(str(cfg))
    result = w.solve_equilibrium()
    assert result is not None
    assert w.Psi.shape == (5, 5)
    assert w.B_R.shape == (5, 5)


def test_wrapper_sample_psi_at(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    w = _rust_compat.RustAcceleratedKernel(str(cfg))
    val = w.sample_psi_at(2.0, 0.0)
    assert isinstance(val, float)


def test_wrapper_sample_psi_at_probes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    w = _rust_compat.RustAcceleratedKernel(str(cfg))
    vals = w.sample_psi_at_probes([(2.0, 0.0), (2.5, 0.5)])
    assert len(vals) == 2


def test_wrapper_compute_b_field(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    w = _rust_compat.RustAcceleratedKernel(str(cfg))
    w.compute_b_field()
    assert w.B_R.shape == (5, 5)
    assert w.B_Z.shape == (5, 5)


def test_wrapper_find_x_point(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    w = _rust_compat.RustAcceleratedKernel(str(cfg))
    (r_x, z_x), psi_x = w.find_x_point(w.Psi)
    assert np.isfinite(r_x)
    assert np.isfinite(z_x)


def test_wrapper_save_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)
    monkeypatch.setattr(_rust_compat, "PyFusionKernel", _DummyRustKernel, raising=False)

    w = _rust_compat.RustAcceleratedKernel(str(cfg))
    out = tmp_path / "eq.npz"
    w.save_results(str(out))
    assert out.exists()


# ── rust_simulate_tearing_mode (Python fallback) ───────────────────


def test_tearing_mode_python_fallback() -> None:
    result = _rust_compat.rust_simulate_tearing_mode(steps=50)
    assert len(result) == 3
    signal, label, ttd = result
    assert len(signal) == 50
    assert label in (0, 1)


def test_tearing_mode_seeded_deterministic() -> None:
    sig_a, lab_a, ttd_a = _rust_compat.rust_simulate_tearing_mode(steps=20, seed=42)
    sig_b, lab_b, ttd_b = _rust_compat.rust_simulate_tearing_mode(steps=20, seed=42)
    np.testing.assert_array_equal(sig_a, sig_b)
    assert lab_a == lab_b


# ── Rust-gated controller wrappers ─────────────────────────────────


@pytest.mark.skipif(not _HAS_RUST, reason="Rust backend not available")
def test_rust_snn_pool_step() -> None:
    pool = _rust_compat.RustSnnPool(n_neurons=10, gain=5.0, window_size=10)
    u = pool.step(0.1)
    assert np.isfinite(u)
    assert repr(pool).startswith("RustSnnPool")


@pytest.mark.skipif(not _HAS_RUST, reason="Rust backend not available")
def test_rust_snn_controller_step() -> None:
    ctrl = _rust_compat.RustSnnController(target_r=6.2, target_z=0.0)
    u_r, u_z = ctrl.step(6.3, 0.1)
    assert np.isfinite(u_r) and np.isfinite(u_z)


@pytest.mark.skipif(not _HAS_RUST, reason="Rust backend not available")
def test_rust_pid_radial() -> None:
    ctrl = _rust_compat.RustPIDController.radial()
    u = ctrl.step(0.5)
    assert np.isfinite(u)
    ctrl.reset()
    assert np.isfinite(ctrl.kp)


@pytest.mark.skipif(not _HAS_RUST, reason="Rust backend not available")
def test_rust_pid_vertical() -> None:
    ctrl = _rust_compat.RustPIDController.vertical()
    u = ctrl.step(-0.3)
    assert np.isfinite(u)


@pytest.mark.skipif(not _HAS_RUST, reason="Rust backend not available")
def test_rust_isoflux_step() -> None:
    ctrl = _rust_compat.RustIsoFluxController(target_r=6.2, target_z=0.0)
    u_r, u_z = ctrl.step(6.3, 0.1)
    assert np.isfinite(u_r) and np.isfinite(u_z)


@pytest.mark.skipif(not _HAS_RUST, reason="Rust backend not available")
def test_rust_hinf_step() -> None:
    ctrl = _rust_compat.RustHInfController(gamma_growth=100.0, damping=10.0, gamma=1.0, u_max=10.0, dt=1e-3)
    u = ctrl.step(0.5, 1e-3)
    assert np.isfinite(u) and abs(u) <= 10.0
    ctrl.reset()


@pytest.mark.skipif(not _HAS_RUST, reason="Rust backend not available")
def test_rust_spi_mitigation_run() -> None:
    spi = _rust_compat.RustSPIMitigation(w_th_mj=300.0, ip_ma=15.0, te_kev=20.0)
    result = spi.run()
    assert len(result) > 0


@pytest.mark.skipif(not _HAS_RUST, reason="Rust backend not available")
def test_rust_multigrid_vcycle() -> None:
    nr, nz = 17, 17
    source = np.ones((nz, nr))
    psi_bc = np.zeros((nz, nr))
    psi, residual, n_cycles, converged = _rust_compat.rust_multigrid_vcycle(
        source, psi_bc, 1.0, 9.0, -5.0, 5.0, nr, nz, tol=1e-4, max_cycles=100
    )
    assert psi.shape == (nz, nr)
    assert np.isfinite(residual)


@pytest.mark.skipif(not _HAS_RUST, reason="Rust backend not available")
def test_rust_svd_optimal_correction() -> None:
    response = np.array([[1.0, 0.5], [0.0, 1.0]])
    error = np.array([0.1, -0.2])
    delta = _rust_compat.rust_svd_optimal_correction(response, error, gain=0.8)
    assert delta.shape == (2,)
    assert np.all(np.isfinite(delta))
