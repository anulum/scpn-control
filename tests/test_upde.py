# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — UPDE non-uniform fallback and step-input validation tests.
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_control.phase.upde as upde_module
from scpn_control.phase.knm import KnmSpec
from scpn_control.phase.kuramoto import order_parameter
from scpn_control.phase.upde import UPDESystem

_UpdeTick = Callable[
    [
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
        int,
        float,
        float,
        float,
    ],
    object,
]


class _NativeUpdeModule(Protocol):
    upde_tick: _UpdeTick


class _UpdeProbeModule(Protocol):
    HAS_RUST_UPDE: bool
    _rust_upde_tick: _UpdeTick


@dataclass(frozen=True)
class _FakeRustUpdeResult:
    theta_flat: NDArray[np.float64]
    dtheta_flat: NDArray[np.float64]
    r_layer: NDArray[np.float64]
    psi_layer: NDArray[np.float64]
    r_global: float
    psi_global: float
    v_layer: NDArray[np.float64]
    v_global: float


def _two_layer_spec() -> KnmSpec:
    return KnmSpec(
        K=np.array([[0.5, 0.2], [0.3, 0.6]], dtype=np.float64),
        zeta=np.array([0.1, 0.1], dtype=np.float64),
    )


def test_import_enables_rust_upde_when_symbol_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise import-time wiring for the optional Rust UPDE tick."""

    def fake_upde_tick(
        theta_flat: NDArray[np.float64],
        _omega_flat: NDArray[np.float64],
        _k_flat: NDArray[np.float64],
        _alpha_flat: NDArray[np.float64],
        _zeta: NDArray[np.float64],
        _layers: int,
        _n_per_layer: int,
        _dt: float,
        _psi_global: float,
        _pac_gamma: float,
    ) -> _FakeRustUpdeResult:
        return _FakeRustUpdeResult(
            theta_flat=theta_flat,
            dtheta_flat=np.zeros_like(theta_flat),
            r_layer=np.array([1.0], dtype=np.float64),
            psi_layer=np.array([0.0], dtype=np.float64),
            r_global=1.0,
            psi_global=0.0,
            v_layer=np.array([0.0], dtype=np.float64),
            v_global=0.0,
        )

    native = types.ModuleType("scpn_control_rs")
    cast(_NativeUpdeModule, native).upde_tick = fake_upde_tick

    module_path = Path(upde_module.__file__).resolve()
    spec = importlib.util.spec_from_file_location("_scpn_control_upde_cov1_probe", module_path)
    assert spec is not None
    assert spec.loader is not None
    probe = importlib.util.module_from_spec(spec)

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "scpn_control_rs", native)
        patch.setitem(sys.modules, spec.name, probe)
        spec.loader.exec_module(probe)

        probe_view = cast(_UpdeProbeModule, probe)
        assert probe_view.HAS_RUST_UPDE is True
        assert probe_view._rust_upde_tick is fake_upde_tick


def test_step_uses_rust_fast_path_for_uniform_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the optional Rust UPDE path with a typed fake native result."""

    calls: list[tuple[int, int, float, float, float]] = []

    def fake_upde_tick(
        theta_flat: NDArray[np.float64],
        omega_flat: NDArray[np.float64],
        _k_flat: NDArray[np.float64],
        _alpha_flat: NDArray[np.float64],
        _zeta: NDArray[np.float64],
        layers: int,
        n_per_layer: int,
        dt: float,
        psi_global: float,
        pac_gamma: float,
    ) -> _FakeRustUpdeResult:
        calls.append((layers, n_per_layer, dt, psi_global, pac_gamma))
        return _FakeRustUpdeResult(
            theta_flat=theta_flat + dt * omega_flat,
            dtheta_flat=omega_flat,
            r_layer=np.array([0.8, 0.6], dtype=np.float64),
            psi_layer=np.array([0.1, -0.2], dtype=np.float64),
            r_global=0.7,
            psi_global=0.9,
            v_layer=np.array([0.03, 0.04], dtype=np.float64),
            v_global=0.05,
        )

    monkeypatch.setattr(upde_module, "HAS_RUST_UPDE", True)
    monkeypatch.setattr(upde_module, "_rust_upde_tick", fake_upde_tick, raising=False)

    system = upde_module.UPDESystem(spec=_two_layer_spec(), dt=0.25, psi_mode="external", wrap=True)
    theta = [
        np.array([0.0, 0.5], dtype=np.float64),
        np.array([1.0, -0.5], dtype=np.float64),
    ]
    omega = [
        np.array([0.2, -0.1], dtype=np.float64),
        np.array([0.3, 0.4], dtype=np.float64),
    ]

    out = system.step(theta, omega, psi_driver=0.4, pac_gamma=0.2)

    assert calls == [(2, 2, 0.25, 0.4, 0.2)]
    assert len(out["theta1"]) == 2
    np.testing.assert_allclose(out["theta1"][0], np.array([0.05, 0.475]))
    np.testing.assert_allclose(out["theta1"][1], np.array([1.075, -0.4]))
    np.testing.assert_allclose(out["dtheta"][0], omega[0])
    np.testing.assert_allclose(out["dtheta"][1], omega[1])
    np.testing.assert_allclose(out["R_layer"], np.array([0.8, 0.6]))
    np.testing.assert_allclose(out["Psi_layer"], np.array([0.1, -0.2]))
    assert float(out["R_global"]) == 0.7
    assert float(out["Psi_global"]) == 0.9
    np.testing.assert_allclose(out["V_layer"], np.array([0.03, 0.04]))
    assert float(out["V_global"]) == 0.05


def test_step_falls_back_when_rust_result_shape_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject native UPDE snapshots that cannot satisfy the public contract."""

    calls = 0

    def malformed_upde_tick(
        theta_flat: NDArray[np.float64],
        omega_flat: NDArray[np.float64],
        _k_flat: NDArray[np.float64],
        _alpha_flat: NDArray[np.float64],
        _zeta: NDArray[np.float64],
        _layers: int,
        _n_per_layer: int,
        dt: float,
        _psi_global: float,
        _pac_gamma: float,
    ) -> _FakeRustUpdeResult:
        nonlocal calls
        calls += 1
        return _FakeRustUpdeResult(
            theta_flat=theta_flat + dt * omega_flat,
            dtheta_flat=np.array([0.0], dtype=np.float64),
            r_layer=np.array([0.1, 0.2], dtype=np.float64),
            psi_layer=np.array([0.3, 0.4], dtype=np.float64),
            r_global=0.123,
            psi_global=0.456,
            v_layer=np.array([0.5, 0.6], dtype=np.float64),
            v_global=0.789,
        )

    monkeypatch.setattr(upde_module, "HAS_RUST_UPDE", True)
    monkeypatch.setattr(upde_module, "_rust_upde_tick", malformed_upde_tick, raising=False)

    system = upde_module.UPDESystem(spec=_two_layer_spec(), dt=0.25, psi_mode="external", wrap=True)
    theta = [
        np.array([0.0, 0.5], dtype=np.float64),
        np.array([1.0, -0.5], dtype=np.float64),
    ]
    omega = [
        np.array([0.2, -0.1], dtype=np.float64),
        np.array([0.3, 0.4], dtype=np.float64),
    ]

    out = system.step(theta, omega, psi_driver=0.4, pac_gamma=0.2)

    assert calls == 1
    assert len(out["dtheta"]) == 2
    assert out["dtheta"][0].shape == (2,)
    assert float(out["R_global"]) != pytest.approx(0.123)
    _expected_r, expected_psi = order_parameter(np.concatenate(out["theta1"]))
    assert float(out["Psi_global"]) == pytest.approx(expected_psi)


def test_step_uses_python_fallback_for_non_uniform_layers() -> None:
    # Layers of unequal length disqualify the uniform-N Rust fast-path, so the
    # Python fallback (with the global-field driver active via non-zero zeta)
    # runs and must return per-layer phases of the original sizes.
    system = UPDESystem(spec=_two_layer_spec(), dt=1e-3, psi_mode="external", wrap=True)
    theta_layers = [
        np.array([0.0, 0.5, 1.0], dtype=np.float64),
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
    ]
    omega_layers = [
        np.array([0.2, 0.1, 0.0], dtype=np.float64),
        np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float64),
    ]

    out = system.step(theta_layers, omega_layers, psi_driver=0.0, pac_gamma=0.2)

    assert len(out["theta1"]) == 2
    assert out["theta1"][0].shape == (3,)
    assert out["theta1"][1].shape == (4,)
    assert out["R_layer"].shape == (2,)
    assert np.isfinite(out["R_global"])
    assert np.isfinite(out["V_global"])
    assert np.all(np.abs(out["theta1"][0]) <= np.pi + 1e-9)
    expected_global_r, expected_global_psi = order_parameter(np.concatenate(out["theta1"]))
    assert float(out["R_global"]) == pytest.approx(expected_global_r)
    assert float(out["Psi_global"]) == pytest.approx(expected_global_psi)
    for layer_index, theta_next in enumerate(out["theta1"]):
        expected_r, expected_psi = order_parameter(theta_next)
        assert float(out["R_layer"][layer_index]) == pytest.approx(expected_r)
        assert float(out["Psi_layer"][layer_index]) == pytest.approx(expected_psi)


def test_step_requires_external_psi_driver() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    omega = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]

    with pytest.raises(ValueError, match="psi_driver required"):
        system.step(theta, omega)


def test_step_supports_global_mean_field_without_wrapping() -> None:
    system = UPDESystem(spec=_two_layer_spec(), dt=1e-3, psi_mode="global_mean_field", wrap=False)
    theta = [
        np.array([0.0, 0.1, 0.2], dtype=np.float64),
        np.array([0.3, 0.4, 0.5], dtype=np.float64),
    ]
    omega = [
        np.full(3, 0.01, dtype=np.float64),
        np.full(3, 0.02, dtype=np.float64),
    ]

    out = system.step(theta, omega, psi_driver=None)

    assert len(out["theta1"]) == 2
    assert np.isfinite(out["Psi_global"])
    _expected_r, expected_psi = order_parameter(np.concatenate(out["theta1"]))
    assert float(out["Psi_global"]) == pytest.approx(expected_psi)
    assert np.asarray(out["V_layer"]).shape == (2,)


def test_step_rejects_unknown_psi_mode() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="unsupported")
    theta = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    omega = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]

    with pytest.raises(ValueError, match="Unknown psi_mode"):
        system.step(theta, omega, psi_driver=0.0)


def test_step_rejects_nonpositive_dt() -> None:
    system = UPDESystem(spec=_two_layer_spec(), dt=0.0, psi_mode="external")
    theta = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    omega = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]

    with pytest.raises(ValueError, match="dt must be positive and finite"):
        system.step(theta, omega, psi_driver=0.0)


def test_step_rejects_nonfinite_actuation_gain() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3), np.zeros(3)]
    omega = [np.zeros(3), np.zeros(3)]
    with pytest.raises(ValueError, match="actuation_gain must be finite"):
        system.step(theta, omega, psi_driver=0.0, actuation_gain=float("nan"))


def test_step_rejects_nonfinite_pac_gamma() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3), np.zeros(3)]
    omega = [np.zeros(3), np.zeros(3)]
    with pytest.raises(ValueError, match="pac_gamma must be finite"):
        system.step(theta, omega, psi_driver=0.0, pac_gamma=float("inf"))


def test_step_rejects_invalid_k_override_shape() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    omega = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]

    with pytest.raises(ValueError, match="K_override shape"):
        system.step(theta, omega, psi_driver=0.0, K_override=np.ones((3, 3), dtype=np.float64))


def test_step_rejects_nonfinite_k_override_values() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    omega = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    bad_k = np.array([[0.5, np.nan], [0.2, 0.6]], dtype=np.float64)

    with pytest.raises(ValueError, match="K_override must contain only finite values"):
        system.step(theta, omega, psi_driver=0.0, K_override=bad_k)


def test_step_rejects_negative_k_override_values() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    omega = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    bad_k = np.array([[0.5, -0.1], [0.2, 0.6]], dtype=np.float64)

    with pytest.raises(ValueError, match="K_override must be non-negative"):
        system.step(theta, omega, psi_driver=0.0, K_override=bad_k)


def test_step_rejects_layer_count_mismatch() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3, dtype=np.float64)]
    omega = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]

    with pytest.raises(ValueError, match="Expected 2 layers"):
        system.step(theta, omega, psi_driver=0.0)


def test_step_rejects_non_1d_theta_layer() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros((2, 2)), np.zeros(3)]
    omega = [np.zeros(4), np.zeros(3)]
    with pytest.raises(ValueError, match=r"theta_layers\[0\] must be a 1D phase vector"):
        system.step(theta, omega, psi_driver=0.0)


def test_step_rejects_non_1d_omega_layer() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3), np.zeros(3)]
    omega = [np.zeros((3, 1)), np.zeros(3)]
    with pytest.raises(ValueError, match=r"omega_layers\[0\] must be a 1D frequency vector"):
        system.step(theta, omega, psi_driver=0.0)


def test_step_rejects_omega_shape_mismatch() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    omega = [np.zeros(2, dtype=np.float64), np.zeros(3, dtype=np.float64)]

    with pytest.raises(ValueError, match=r"omega_layers\[0\] shape"):
        system.step(theta, omega, psi_driver=0.0)


def test_step_rejects_nonfinite_theta_values() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.array([0.0, np.inf, 0.0], dtype=np.float64), np.zeros(3, dtype=np.float64)]
    omega = [np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]

    with pytest.raises(ValueError, match=r"theta_layers\[0\] must contain only finite values"):
        system.step(theta, omega, psi_driver=0.0)


def test_step_rejects_nonfinite_omega_values() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3), np.zeros(3)]
    omega = [np.array([0.0, np.nan, 0.0]), np.zeros(3)]
    with pytest.raises(ValueError, match=r"omega_layers\[0\] must contain only finite values"):
        system.step(theta, omega, psi_driver=0.0)


def test_run_returns_layer_and_global_histories() -> None:
    system = UPDESystem(spec=_two_layer_spec(), dt=1e-3, psi_mode="external")
    theta = [
        np.array([0.0, 0.1, 0.2], dtype=np.float64),
        np.array([0.3, 0.4, 0.5], dtype=np.float64),
    ]
    omega = [
        np.full(3, 0.01, dtype=np.float64),
        np.full(3, 0.02, dtype=np.float64),
    ]

    out = system.run(3, theta, omega, psi_driver=0.0, pac_gamma=0.1)

    assert len(out["theta_final"]) == 2
    assert out["R_layer_hist"].shape == (3, 2)
    assert out["R_global_hist"].shape == (3,)
    assert np.isfinite(out["R_global_hist"]).all()


def test_run_lyapunov_returns_exponents_and_histories() -> None:
    system = UPDESystem(spec=_two_layer_spec(), dt=1e-3, psi_mode="external")
    theta = [
        np.array([0.0, 0.1, 0.2], dtype=np.float64),
        np.array([0.3, 0.4, 0.5], dtype=np.float64),
    ]
    omega = [
        np.full(3, 0.01, dtype=np.float64),
        np.full(3, 0.02, dtype=np.float64),
    ]

    out = system.run_lyapunov(3, theta, omega, psi_driver=0.0, pac_gamma=0.1)

    assert len(out["theta_final"]) == 2
    assert out["R_layer_hist"].shape == (3, 2)
    assert out["V_layer_hist"].shape == (3, 2)
    assert out["lambda_layer"].shape == (2,)
    assert np.isfinite(out["lambda_global"])
