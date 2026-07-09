# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Test Rust Realtime Parity
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Rust PyRealtimeMonitor parity test
# ──────────────────────────────────────────────────────────────────────
"""Verify Rust PyRealtimeMonitor tick matches Python RealtimeMonitor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scpn_control.phase.realtime_monitor import RealtimeMonitor


class _RustMonitor(Protocol):
    @property
    def tick_count(self) -> int:
        """Return the number of native monitor ticks since construction/reset."""

    def tick(self) -> dict[str, object]:
        """Advance the native monitor and return the public snapshot mapping."""

    def reset(self, theta_flat: NDArray[np.float64]) -> None:
        """Reset native monitor state from a contiguous flattened phase vector."""


class _RustBindings(Protocol):
    def PyRealtimeMonitor(
        self,
        knm_flat: NDArray[np.float64],
        zeta_flat: NDArray[np.float64],
        theta_flat: NDArray[np.float64],
        omega_flat: NDArray[np.float64],
        n_layers: int,
        n_per: int,
        *,
        dt: float,
        psi_driver: float,
    ) -> _RustMonitor:
        """Construct the native real-time monitor binding."""


try:
    import scpn_control_rs as _scpn_control_rs

    HAS_RUST = True
except ImportError:
    _scpn_control_rs = None
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust bindings not compiled")

SharedState = tuple["RealtimeMonitor", _RustMonitor, int, int]


def _as_float(value: object) -> float:
    """Return a Python float from a scalar snapshot value."""
    if isinstance(value, int | float | np.floating):
        return float(value)
    raise AssertionError(f"expected numeric scalar, got {type(value).__name__}")


@pytest.fixture
def shared_state() -> SharedState:
    """Build matching Python and Rust monitors from identical initial state."""
    from scpn_control.phase.knm import OMEGA_N_16, build_knm_paper27
    from scpn_control.phase.realtime_monitor import RealtimeMonitor

    rust_bindings = cast(_RustBindings, _scpn_control_rs)

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
        L=L,
        N_per=N_per,
        dt=dt,
        zeta_uniform=zeta_val,
        psi_driver=psi,
        seed=seed,
    )

    # Rust monitor — needs flat arrays
    knm_flat = spec.K.ravel().astype(np.float64)
    zeta_flat = np.full(L, zeta_val, dtype=np.float64)
    theta_flat = np.concatenate([t.ravel() for t in py_mon.theta_layers]).astype(np.float64)
    omega_flat = np.concatenate([o.ravel() for o in py_mon.omega_layers]).astype(np.float64)

    rs_mon = rust_bindings.PyRealtimeMonitor(
        knm_flat,
        zeta_flat,
        theta_flat,
        omega_flat,
        L,
        N_per,
        dt=dt,
        psi_driver=psi,
    )

    return py_mon, rs_mon, L, N_per


def test_single_tick_parity(shared_state: SharedState) -> None:
    py_mon, rs_mon, L, N_per = shared_state

    py_snap = py_mon.tick()
    rs_snap = rs_mon.tick()

    assert rs_snap["tick"] == 1
    assert set(rs_snap) >= {
        "dtheta_flat",
        "Psi_layer",
        "R_global",
        "Psi_global",
        "V_layer",
        "V_global",
    }
    assert np.asarray(rs_snap["dtheta_flat"]).shape == (L * N_per,)
    assert np.asarray(rs_snap["Psi_layer"]).shape == (L,)
    np.testing.assert_allclose(_as_float(rs_snap["R_global"]), _as_float(py_snap["R_global"]), atol=0.05)
    np.testing.assert_allclose(_as_float(rs_snap["Psi_global"]), _as_float(py_snap["Psi_global"]), atol=0.05)
    np.testing.assert_allclose(
        np.array(rs_snap["R_layer"], dtype=np.float64),
        np.array(py_snap["R_layer"], dtype=np.float64),
        atol=0.05,
    )


def test_multi_tick_convergence(shared_state: SharedState) -> None:
    py_mon, rs_mon, _layers, _n_per = shared_state

    py_snap: dict[str, object] = {}
    rs_snap: dict[str, object] = {}
    for _ in range(200):
        py_snap = py_mon.tick()
        rs_snap = rs_mon.tick()

    # Both should converge — R_global should be positive
    assert _as_float(py_snap["R_global"]) > 0.1
    assert _as_float(rs_snap["R_global"]) > 0.1
    # V_global should be finite
    assert np.isfinite(_as_float(rs_snap["V_global"]))


def test_rust_monitor_reset(shared_state: SharedState) -> None:
    _, rs_mon, L, N_per = shared_state

    rs_mon.tick()
    rs_mon.tick()
    assert rs_mon.tick_count == 2

    rng = np.random.default_rng(99)
    new_theta = rng.uniform(-np.pi, np.pi, L * N_per).astype(np.float64)
    rs_mon.reset(new_theta)
    assert rs_mon.tick_count == 0
