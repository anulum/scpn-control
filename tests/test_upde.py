# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — UPDE non-uniform fallback and step-input validation tests.
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.phase.knm import KnmSpec
from scpn_control.phase.upde import UPDESystem


def _two_layer_spec() -> KnmSpec:
    return KnmSpec(
        K=np.array([[0.5, 0.2], [0.3, 0.6]], dtype=np.float64),
        zeta=np.array([0.1, 0.1], dtype=np.float64),
    )


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


def test_step_rejects_nonfinite_omega_values() -> None:
    system = UPDESystem(spec=_two_layer_spec(), psi_mode="external")
    theta = [np.zeros(3), np.zeros(3)]
    omega = [np.array([0.0, np.nan, 0.0]), np.zeros(3)]
    with pytest.raises(ValueError, match=r"omega_layers\[0\] must contain only finite values"):
        system.step(theta, omega, psi_driver=0.0)
