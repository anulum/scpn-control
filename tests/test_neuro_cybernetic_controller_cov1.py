# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Neuro-cybernetic controller coverage hardening tests
"""COV-1 regression tests for neuro-cybernetic optional-path coverage."""

from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import scpn_control.control.neuro_cybernetic_controller as controller_mod

FloatArray = npt.NDArray[np.float64]


class _FakeStochasticLIFNeuron:
    """Minimal sc-neurocore LIF stand-in for import-success coverage."""

    def __init__(self, *, seed: int, entropy_source: object | None = None) -> None:
        self.seed = int(seed)
        self.entropy_source = entropy_source

    def step(self, input_current: float) -> bool:
        """Return a deterministic spike decision for the optional backend."""
        return float(input_current) > 0.5


class _FakeQuantumEntropySource:
    """Minimal entropy-source stand-in for sc-neurocore import-success coverage."""

    def __init__(self, *, n_qubits: int) -> None:
        self.n_qubits = int(n_qubits)


class _FakeFusionKernel:
    """Small FusionKernel stand-in for the optional Rust compatibility import."""

    def __init__(self, _config_file: str) -> None:
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R: FloatArray = np.asarray(np.linspace(5.9, 6.5, 9), dtype=np.float64)
        self.Z: FloatArray = np.asarray(np.linspace(-0.2, 0.2, 9), dtype=np.float64)
        self.RR: FloatArray
        self.ZZ: FloatArray
        rr, zz = np.meshgrid(self.R, self.Z)
        self.RR = np.asarray(rr, dtype=np.float64)
        self.ZZ = np.asarray(zz, dtype=np.float64)
        self.Psi: FloatArray = np.zeros((9, 9), dtype=np.float64)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        """Build a stable centered flux surface for deterministic shots."""
        self.Psi = 1.0 - ((self.RR - 6.2) ** 2 + (self.ZZ / 1.4) ** 2)


def _module(name: str, **attrs: object) -> ModuleType:
    """Create a module object populated with dynamic test attributes."""
    module = ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def test_optional_sc_neurocore_import_success_path_is_exercised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reload the controller with fake sc-neurocore modules to cover the success guard."""
    try:
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "sc_neurocore", _module("sc_neurocore"))
            patch.setitem(sys.modules, "sc_neurocore.neurons", _module("sc_neurocore.neurons"))
            patch.setitem(
                sys.modules,
                "sc_neurocore.neurons.stochastic_lif",
                _module("sc_neurocore.neurons.stochastic_lif", StochasticLIFNeuron=_FakeStochasticLIFNeuron),
            )
            patch.setitem(sys.modules, "sc_neurocore.sources", _module("sc_neurocore.sources"))
            patch.setitem(
                sys.modules,
                "sc_neurocore.sources.quantum_entropy",
                _module("sc_neurocore.sources.quantum_entropy", QuantumEntropySource=_FakeQuantumEntropySource),
            )

            importlib.reload(controller_mod)
            assert controller_mod.SC_NEUROCORE_AVAILABLE is True
            pool = controller_mod.SpikingControllerPool(n_neurons=2, use_quantum=True)

            assert pool.backend == "sc_neurocore"
            assert pool.step(0.25) > 0.0
    finally:
        importlib.reload(controller_mod)


def test_optional_rust_kernel_import_path_runs_public_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the public control entry point through the optional Rust-kernel resolver."""
    try:
        with monkeypatch.context() as patch:
            patch.setitem(
                sys.modules,
                "scpn_control.core._rust_compat",
                _module("scpn_control.core._rust_compat", FusionKernel=_FakeFusionKernel),
            )
            importlib.reload(controller_mod)
            summary = controller_mod.run_neuro_cybernetic_control(
                config_file="fake.json",
                shot_duration=3,
                seed=7,
                save_plot=False,
                verbose=False,
                allow_numpy_fallback=True,
                allow_legacy_numpy_fallback=True,
            )

            assert summary["steps"] == 3
            assert summary["plot_saved"] is False
            # Environment may have sc_neurocore installed; both backends exercise the public path.
            assert summary["backend_r"] in {"numpy_lif", "sc_neurocore"}
    finally:
        importlib.reload(controller_mod)


def test_plot_export_failure_is_reported_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A plot-export I/O failure must become summary metadata, not a shot failure."""

    def _raise_plot_error(
        self: controller_mod.NeuroCyberneticController,
        title: str,
        *,
        output_path: str | None = None,
        verbose: bool = True,
    ) -> str:
        del self, title, output_path, verbose
        raise OSError("plot directory unavailable")

    monkeypatch.setattr(controller_mod.NeuroCyberneticController, "visualize", _raise_plot_error)
    with caplog.at_level(logging.INFO, logger="scpn_control.control.neuro_cybernetic_controller"):
        summary = controller_mod.run_neuro_cybernetic_control(
            config_file="fake.json",
            shot_duration=3,
            seed=11,
            save_plot=True,
            verbose=True,
            allow_numpy_fallback=True,
            allow_legacy_numpy_fallback=True,
            kernel_factory=_FakeFusionKernel,
        )

    assert summary["plot_saved"] is False
    assert summary["plot_error"] == "plot directory unavailable"
    assert "Plot export skipped due to error: plot directory unavailable" in caplog.text
