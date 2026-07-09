# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fusion control room import path tests

"""Import-path and renderer setup tests for the fusion control room."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType
from typing import Any

import pytest

MODULE_NAME = "scpn_control.control.fusion_control_room"


def _reload_control_room(monkeypatch: pytest.MonkeyPatch, *, blocked_imports: set[str]) -> ModuleType:
    """Reload the control-room module with selected absolute imports blocked."""
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level == 0 and name in blocked_imports:
            raise ImportError(f"blocked test import: {name}")
        return original_import(name, globals_, locals_, fromlist, level)

    original_module = sys.modules.pop(MODULE_NAME, None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        return importlib.import_module(MODULE_NAME)
    finally:
        sys.modules.pop(MODULE_NAME, None)
        if original_module is not None:
            sys.modules[MODULE_NAME] = original_module


def test_fusion_kernel_loader_falls_back_to_python_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader uses the Python kernel when the Rust compatibility wrapper is absent."""
    module = _reload_control_room(
        monkeypatch,
        blocked_imports={"scpn_control.core._rust_compat"},
    )

    assert callable(module.FusionKernel)


def test_fusion_kernel_loader_returns_none_without_available_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader disables kernel auto-construction when both kernel imports are absent."""
    module = _reload_control_room(
        monkeypatch,
        blocked_imports={
            "scpn_control.core._rust_compat",
            "scpn_control.core.fusion_kernel",
        },
    )

    assert module.FusionKernel is None
    summary = module.run_control_room(
        sim_duration=1,
        config_file="missing.json",
        save_animation=False,
        save_report=False,
        verbose=False,
    )
    assert summary["psi_source"] == "analytic"


def test_render_outputs_reports_backend_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Renderer setup errors are reported through the normal fail-closed contract."""
    import scpn_control.control.fusion_control_room as fcr

    class _BrokenPyplot:
        """Pyplot stand-in that fails at figure construction."""

        def figure(self, **_kwargs: object) -> object:
            raise TypeError("backend setup failed")

        def close(self, _fig: object) -> None:
            return None

    monkeypatch.setattr(fcr, "HAS_MPL", True)
    monkeypatch.setattr(fcr, "plt", _BrokenPyplot())

    animation_saved, animation_error, report_saved, report_error = fcr._render_outputs(
        [
            {
                "density": [[0.0]],
                "r_min": 1.0,
                "r_max": 2.0,
                "z_min": -1.0,
                "z_max": 1.0,
            }
        ],
        [0.0],
        [0.0],
        [0.0],
        save_animation=True,
        save_report=True,
        output_gif="unused.gif",
        output_report="unused.png",
    )

    assert animation_saved is False
    assert animation_error == "backend setup failed"
    assert report_saved is False
    assert report_error == "backend setup failed"


def test_render_outputs_closes_partial_figure_after_backend_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Renderer setup errors after figure creation still close the partial figure."""
    import scpn_control.control.fusion_control_room as fcr

    class _BrokenFigure:
        """Figure stand-in that fails after construction."""

        def add_gridspec(self, *_args: object) -> object:
            raise ValueError("grid setup failed")

    class _ClosingPyplot:
        """Pyplot stand-in that records cleanup for partially-created figures."""

        def __init__(self) -> None:
            self.figure_instance = _BrokenFigure()
            self.closed: object | None = None

        def figure(self, **_kwargs: object) -> _BrokenFigure:
            return self.figure_instance

        def close(self, fig: object) -> None:
            self.closed = fig

    fake_pyplot = _ClosingPyplot()
    monkeypatch.setattr(fcr, "HAS_MPL", True)
    monkeypatch.setattr(fcr, "plt", fake_pyplot)

    animation_saved, animation_error, report_saved, report_error = fcr._render_outputs(
        [{"density": [[0.0]], "r_min": 1.0, "r_max": 2.0, "z_min": -1.0, "z_max": 1.0}],
        [0.0],
        [0.0],
        [0.0],
        save_animation=True,
        save_report=True,
        output_gif="unused.gif",
        output_report="unused.png",
    )

    assert fake_pyplot.closed is fake_pyplot.figure_instance
    assert animation_saved is False
    assert animation_error == "grid setup failed"
    assert report_saved is False
    assert report_error == "grid setup failed"


def test_render_outputs_reports_partial_backend_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Renderer setup errors are mapped only to requested output channels."""
    import scpn_control.control.fusion_control_room as fcr

    class _BrokenPyplot:
        """Pyplot stand-in that fails at figure construction."""

        def figure(self, **_kwargs: object) -> object:
            raise RuntimeError("plot grid failed")

        def close(self, _fig: object) -> None:
            return None

    monkeypatch.setattr(fcr, "HAS_MPL", True)
    monkeypatch.setattr(fcr, "plt", _BrokenPyplot())

    animation_saved, animation_error, report_saved, report_error = fcr._render_outputs(
        [{"density": [[0.0]], "r_min": 1.0, "r_max": 2.0, "z_min": -1.0, "z_max": 1.0}],
        [0.0],
        [0.0],
        [0.0],
        save_animation=False,
        save_report=True,
        output_gif="unused.gif",
        output_report="unused.png",
    )

    assert animation_saved is False
    assert animation_error is None
    assert report_saved is False
    assert report_error == "plot grid failed"
