# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Director interface COV-1 tests.
"""Focused COV-1 tests for Director interface guard and plotting paths."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pytest

import scpn_control.control.director_interface as director_module
from scpn_control.control.director_interface import DirectorInterface


class _FallbackKernel:
    """Sentinel kernel class returned by the import fallback fixture."""


class _ApprovingDirector:
    """Director stand-in that approves every proposed mission action."""

    def review_action(self, _prompt: str, _proposed_action: str) -> tuple[bool, float]:
        """Approve the proposed action with a zero security score."""
        return True, 0.0


class _Brain:
    """Minimal neural controller channel for closed-loop mission tests."""

    def step(self, error: float) -> float:
        """Return a deterministic bounded correction for the supplied error."""
        return -0.1 * error


class _Kernel:
    """Minimal kernel exposing the runtime contract expected by the mission loop."""

    def __init__(self) -> None:
        self.cfg: dict[str, Any] = {
            "physics": {"plasma_current_target": 5.0},
            "coils": [{"current": 0.0} for _ in range(5)],
        }
        self.R: NDArray[np.float64] = np.linspace(5.8, 6.6, 9, dtype=np.float64)
        self.Z: NDArray[np.float64] = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
        rr, zz = np.meshgrid(self.R, self.Z)
        self.Psi: NDArray[np.float64] = -((rr - 6.2) ** 2 + zz**2)

    def solve_equilibrium(self) -> None:
        """Satisfy the legacy kernel solve contract used by solve_kernel."""


class _Controller:
    """Controller factory product used by DirectorInterface tests."""

    def __init__(self, _config_path: str) -> None:
        self.kernel = _Kernel()
        self.brain_R = _Brain()
        self.brain_Z = _Brain()

    def initialize_brains(self, *, use_quantum: bool = True) -> None:
        """Accept the public initialization call without mutating the fixture."""


class _FakeAxes:
    """Small plotting-axis test double for the public visualize method."""

    def __init__(self) -> None:
        self.plots: list[tuple[list[float], list[float], str, str]] = []
        self.ylabel: str | None = None
        self.title: str | None = None

    def set_title(self, title: str) -> None:
        """Record the configured plot title."""
        self.title = title

    def plot(self, x: list[float], y: list[float], style: str, *, label: str) -> None:
        """Record plotted series without invoking a graphical backend."""
        self.plots.append((x, y, style, label))

    def set_ylabel(self, label: str, *, color: str) -> None:
        """Record the y-axis label; color is part of the production contract."""
        self.ylabel = f"{label}:{color}"

    def tick_params(self, *, axis: str, labelcolor: str) -> None:
        """Accept y-axis color configuration from the production renderer."""
        assert axis == "y"
        assert labelcolor in {"b", "r"}

    def twinx(self) -> _FakeAxes:
        """Return a second independent axis for radial-error telemetry."""
        return _FakeAxes()


class _FakeFigure:
    """Small figure test double for legend placement calls."""

    def __init__(self) -> None:
        self.legend_args: tuple[str, tuple[float, float]] | None = None

    def legend(self, *, loc: str, bbox_to_anchor: tuple[float, float]) -> None:
        """Record legend placement options."""
        self.legend_args = (loc, bbox_to_anchor)


class _FakePyplot:
    """Module-shaped pyplot test double used by DirectorInterface.visualize."""

    def __init__(self) -> None:
        self.figure = _FakeFigure()
        self.primary_axis = _FakeAxes()
        self.saved_path: str | None = None
        self.tight_layout_called = False
        self.vline: tuple[int, str, str, str] | None = None

    def subplots(self, *, figsize: tuple[int, int]) -> tuple[_FakeFigure, _FakeAxes]:
        """Return deterministic fake figure and axis objects."""
        assert figsize == (10, 6)
        return self.figure, self.primary_axis

    def axvline(self, x: int, *, color: str, linestyle: str, label: str) -> None:
        """Record the disturbance marker requested by the production plot."""
        self.vline = (x, color, linestyle, label)

    def tight_layout(self) -> None:
        """Record that layout compaction was requested."""
        self.tight_layout_called = True

    def savefig(self, output_path: str) -> None:
        """Record the requested output path instead of touching a GUI backend."""
        self.saved_path = output_path


def _interface(*, director: object | None = None) -> DirectorInterface:
    """Build a DirectorInterface wired to deterministic production-compatible fakes."""
    if director is not None:
        return DirectorInterface("mock.json", controller_factory=_Controller, director=director)
    return DirectorInterface(
        "mock.json",
        controller_factory=_Controller,
        allow_fallback=True,
        allow_legacy_fallback=True,
    )


def test_load_fusion_kernel_falls_back_to_python_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader falls back to the Python FusionKernel when native Rust is absent."""
    fallback_module = ModuleType("scpn_control.core.fusion_kernel")
    cast(Any, fallback_module).FusionKernel = _FallbackKernel

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        assert package is None
        if name == "scpn_control.core._rust_compat":
            raise ImportError("native backend absent")
        if name == "scpn_control.core.fusion_kernel":
            return fallback_module
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(director_module, "import_module", fake_import)

    kernel_type, rust_backend = director_module._load_fusion_kernel()

    assert kernel_type is _FallbackKernel
    assert rust_backend is False


def test_load_fusion_kernel_falls_back_when_rust_compat_lacks_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader accepts a partial rust-compat module and uses the Python kernel."""
    rust_compat_module = ModuleType("scpn_control.core._rust_compat")
    cast(Any, rust_compat_module).RUST_BACKEND = False
    fallback_module = ModuleType("scpn_control.core.fusion_kernel")
    cast(Any, fallback_module).FusionKernel = _FallbackKernel

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        assert package is None
        if name == "scpn_control.core._rust_compat":
            return rust_compat_module
        if name == "scpn_control.core.fusion_kernel":
            return fallback_module
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(director_module, "import_module", fake_import)

    kernel_type, rust_backend = director_module._load_fusion_kernel()

    assert kernel_type is _FallbackKernel
    assert rust_backend is False


def test_load_fusion_kernel_reports_missing_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader raises the operator-facing import guidance when both backends fail."""

    def fake_import(name: str, package: str | None = None) -> ModuleType:
        assert package is None
        raise ImportError(name)

    monkeypatch.setattr(director_module, "import_module", fake_import)

    with pytest.raises(ImportError, match="Unable to import FusionKernel"):
        director_module._load_fusion_kernel()


def test_approved_director_increments_target_current() -> None:
    """An approving injected director advances the mission target on review steps."""
    summary = _interface(director=_ApprovingDirector()).run_directed_mission(
        duration=6,
        glitch_start_step=99,
        save_plot=False,
        verbose=False,
    )

    assert summary["final_target_ip"] == 7.0
    assert summary["intervention_count"] == 0


def test_plot_export_success_marks_plot_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful public plot export marks the mission summary as saved."""
    interface = _interface()

    def visualize_success(*, output_path: str = "Director_Interface_Result.png") -> str:
        return output_path

    monkeypatch.setattr(interface, "visualize", visualize_success)

    summary = interface.run_directed_mission(duration=1, save_plot=True, verbose=False)

    assert summary["plot_saved"] is True
    assert summary["plot_error"] is None


def test_plot_export_failure_is_reported_under_explicit_legacy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plot failures fail open only when both legacy fallback switches are explicit."""
    interface = _interface()

    def visualize_failure(*, output_path: str = "Director_Interface_Result.png") -> str:
        raise ValueError(f"cannot write {output_path}")

    monkeypatch.setattr(interface, "visualize", visualize_failure)

    summary = interface.run_directed_mission(
        duration=1,
        save_plot=True,
        verbose=False,
        allow_plot_fallback=True,
        allow_legacy_plot_fallback=True,
    )

    assert summary["plot_saved"] is False
    assert summary["plot_error"] == "ValueError: cannot write Director_Interface_Result.png"


def test_plot_export_failure_raises_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plot failures stay fail-closed unless the legacy fallback is explicitly enabled."""
    interface = _interface()

    def visualize_failure(*, output_path: str = "Director_Interface_Result.png") -> str:
        raise OSError(f"disk denied {output_path}")

    monkeypatch.setattr(interface, "visualize", visualize_failure)

    with pytest.raises(RuntimeError, match="legacy plot fallback is disabled"):
        interface.run_directed_mission(duration=1, save_plot=True, verbose=False)


def test_visualize_writes_telemetry_plot_with_configured_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The public visualize method sends mission telemetry to the configured backend."""
    fake_pyplot = _FakePyplot()
    monkeypatch.setattr(director_module, "plt", fake_pyplot, raising=False)
    output_path = str(tmp_path / "telemetry.png")
    interface = _interface()
    interface.log = [
        {"t": 0.0, "Ip": 5.0, "Err_R": 0.01, "Director_Intervention": 0.0},
        {"t": 1.0, "Ip": 6.0, "Err_R": -0.02, "Director_Intervention": 0.0},
    ]

    rendered_path = interface.visualize(output_path=output_path)

    assert rendered_path == output_path
    assert fake_pyplot.saved_path == output_path
    assert fake_pyplot.tight_layout_called is True
    assert fake_pyplot.vline == (50, "k", ":", "External Disturbance")
    assert fake_pyplot.figure.legend_args == ("upper left", (0.15, 0.85))
    assert fake_pyplot.primary_axis.title == "Director-Mediated Fusion Control"
