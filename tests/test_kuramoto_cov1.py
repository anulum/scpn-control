# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Kuramoto COV-1 fallback coverage tests.
"""Focused COV-1 tests for reachable Kuramoto Python fallback behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Callable, Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_control.phase.kuramoto as kuramoto_module
from scpn_control.phase.kuramoto import kuramoto_sakaguchi_step

_RustStep = Callable[
    [NDArray[np.float64], NDArray[np.float64], float, float, float, float, float],
    dict[str, object],
]


class _NativeKuramotoModule(Protocol):
    kuramoto_step: _RustStep


class _KuramotoProbeModule(Protocol):
    RUST_KURAMOTO: bool
    _rust_step: _RustStep


def test_kuramoto_import_enables_rust_backend_when_symbol_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise import-time wiring for the optional Rust Kuramoto step."""

    def fake_kuramoto_step(
        theta: NDArray[np.float64],
        omega: NDArray[np.float64],
        dt: float,
        _k: float,
        _alpha: float,
        _zeta: float,
        psi: float,
    ) -> dict[str, object]:
        theta_next = theta + dt * omega
        return {
            "theta": theta_next,
            "r": 0.75,
            "psi_r": -0.25,
            "psi_global": psi,
        }

    native = types.ModuleType("scpn_control_rs")
    cast(_NativeKuramotoModule, native).kuramoto_step = fake_kuramoto_step

    module_path = Path(kuramoto_module.__file__).resolve()
    spec = importlib.util.spec_from_file_location("_scpn_control_kuramoto_cov1_probe", module_path)
    assert spec is not None
    assert spec.loader is not None
    probe = importlib.util.module_from_spec(spec)

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "scpn_control_rs", native)
        patch.setitem(sys.modules, spec.name, probe)
        spec.loader.exec_module(probe)

        probe_view = cast(_KuramotoProbeModule, probe)
        assert probe_view.RUST_KURAMOTO is True
        assert probe_view._rust_step is fake_kuramoto_step


def test_kuramoto_step_uses_python_fallback_when_wrap_disabled() -> None:
    """Exercise the NumPy fallback return even when the optional Rust backend exists."""

    theta = np.array([3.2, -3.0, 0.25], dtype=np.float64)
    omega = np.array([0.5, -0.25, 0.125], dtype=np.float64)

    out = kuramoto_sakaguchi_step(
        theta,
        omega,
        dt=0.1,
        K=0.0,
        zeta=0.0,
        psi_driver=0.0,
        psi_mode="external",
        wrap=False,
    )

    expected = theta + 0.1 * omega
    np.testing.assert_allclose(np.asarray(out["theta1"], dtype=np.float64), expected)
    np.testing.assert_allclose(np.asarray(out["dtheta"], dtype=np.float64), omega)
    assert float(out["Psi"]) == 0.0
