# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Rust PyRealtimeMonitor parity test
# ──────────────────────────────────────────────────────────────────────
"""Verify Rust PyRealtimeMonitor tick matches Python RealtimeMonitor."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import scpn_control_rs  # noqa: F401
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust bindings not compiled")


@pytest.fixture
def shared_state():
    """Build matching Python and Rust monitors from identical initial state."""
    from scpn_control.phase.knm import build_knm_paper27, OMEGA_N_16
    from scpn_control.phase.realtime_monitor import RealtimeMonitor

    L, N_per = 4, 20
    seed = 42
    dt = 1e-3
    zeta_val = 0.5
    psi = 0.3

    spec = build_knm_paper27(L=L, zeta_uniform=zeta_val)
    rng = np.random.default_rng(seed)
    theta_layers = [rng.uniform(-np.pi, np.pi, N_per) for _ in range(L)]
    omega_layers = [OMEGA_N_16[m % 16] + rng.normal(0, 0.2, N_per) for m in range(L)]

    # Python monitor
    py_mon = RealtimeMonitor.from_paper27(
        L=L, N_per=N_per, dt=dt, zeta_uniform=zeta_val, psi_driver=psi, seed=seed,
    )

    # Rust monitor — needs flat arrays
    knm_flat = spec.K.ravel().astype(np.float64)
    zeta_flat = np.full(L, zeta_val, dtype=np.float64)
    theta_flat = np.concatenate([t.ravel() for t in py_mon.theta_layers]).astype(np.float64)
    omega_flat = np.concatenate([o.ravel() for o in py_mon.omega_layers]).astype(np.float64)

    rs_mon = scpn_control_rs.PyRealtimeMonitor(
        knm_flat, zeta_flat, theta_flat, omega_flat,
        L, N_per, dt=dt, psi_driver=psi,
    )

    return py_mon, rs_mon, L, N_per


def test_single_tick_parity(shared_state):
    py_mon, rs_mon, L, N_per = shared_state

    py_snap = py_mon.tick()
    rs_snap = rs_mon.tick()

    assert rs_snap["tick"] == 1
    np.testing.assert_allclose(rs_snap["R_global"], py_snap["R_global"], atol=0.05)
    np.testing.assert_allclose(
        np.array(rs_snap["R_layer"]), np.array(py_snap["R_layer"]), atol=0.05,
    )


def test_multi_tick_convergence(shared_state):
    py_mon, rs_mon, L, N_per = shared_state

    for _ in range(200):
        py_snap = py_mon.tick()
        rs_snap = rs_mon.tick()

    # Both should converge — R_global should be positive
    assert py_snap["R_global"] > 0.1
    assert rs_snap["R_global"] > 0.1
    # V_global should be finite
    assert np.isfinite(rs_snap["V_global"])


def test_rust_monitor_reset(shared_state):
    _, rs_mon, L, N_per = shared_state

    rs_mon.tick()
    rs_mon.tick()
    assert rs_mon.tick_count == 2

    rng = np.random.default_rng(99)
    new_theta = rng.uniform(-np.pi, np.pi, L * N_per).astype(np.float64)
    rs_mon.reset(new_theta)
    assert rs_mon.tick_count == 0
