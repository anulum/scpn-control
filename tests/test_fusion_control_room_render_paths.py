# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fusion control room render path tests

"""Renderer path tests for fusion-control-room output handling."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest


class _PlotArtist:
    """Minimal artist object used by the renderer line-coverage tests."""

    def set_data(self, *_args: object) -> None:
        return None

    def set_alpha(self, _alpha: float) -> None:
        return None

    def set_height(self, _height: float) -> None:
        return None


class _PlotAxis:
    """Minimal axis facade for the renderer line-coverage tests."""

    def set_facecolor(self, _color: str) -> None:
        return None

    def set_title(self, _title: str, *, color: str) -> None:
        return None

    def imshow(self, *_args: object, **_kwargs: object) -> _PlotArtist:
        return _PlotArtist()

    def set_ylim(self, _lower: float, _upper: float) -> None:
        return None

    def grid(self, _enabled: bool, *, color: str) -> None:
        return None

    def plot(self, *_args: object, **_kwargs: object) -> tuple[_PlotArtist]:
        return (_PlotArtist(),)

    def set_xlim(self, _lower: int, _upper: int) -> None:
        return None

    def bar(self, *_args: object, **_kwargs: object) -> list[_PlotArtist]:
        return [_PlotArtist()]

    def set_xticks(self, _ticks: list[int]) -> None:
        return None

    def set_xticklabels(self, _labels: list[str], *, color: str) -> None:
        return None

    def legend(self) -> None:
        return None

    def add_patch(self, _patch: object) -> None:
        return None


class _PlotGrid:
    """Grid facade that supports Matplotlib-style indexing."""

    def __getitem__(self, _key: object) -> object:
        return object()


class _PlotFigure:
    """Figure facade for successful and failing save paths."""

    def __init__(self, *, fail_save: bool = False) -> None:
        self._fail_save = fail_save

    def add_gridspec(self, _rows: int, _cols: int) -> _PlotGrid:
        return _PlotGrid()

    def add_subplot(self, _spec: object) -> _PlotAxis:
        return _PlotAxis()

    def savefig(self, _path: str) -> None:
        if self._fail_save:
            raise OSError("report save failed")


class _FakeAnimation:
    """Animation facade that executes the renderer update function."""

    def __init__(self, _fig: object, update: object, *, frames: int, interval: int, blit: bool) -> None:
        update_fn = cast("Callable[[int], object]", update)
        for frame_idx in range(frames):
            update_fn(frame_idx)

    def save(self, _path: str, *, writer: object) -> None:
        return None


class _FailingAnimation(_FakeAnimation):
    """Animation facade that raises during save."""

    def save(self, _path: str, *, writer: object) -> None:
        raise RuntimeError("animation save failed")


class _FakePyplot:
    """Pyplot facade for deterministic renderer line coverage."""

    def __init__(self, *, fail_save: bool = False) -> None:
        self._fail_save = fail_save

    def figure(self, **_kwargs: object) -> _PlotFigure:
        return _PlotFigure(fail_save=self._fail_save)

    def tight_layout(self) -> None:
        return None

    def close(self, _fig: object) -> None:
        return None


class _FakeWriter:
    """PillowWriter facade used by animation save tests."""

    def __init__(self, *, fps: int) -> None:
        self.fps = fps


def _frames() -> list[dict[str, object]]:
    """Return two renderer frames with density and coordinate extents."""
    return [
        {"density": [[0.1, 0.2], [0.3, 0.4]], "r_min": 1.0, "r_max": 2.0, "z_min": -1.0, "z_max": 1.0},
        {"density": [[0.2, 0.1], [0.4, 0.3]], "r_min": 1.0, "r_max": 2.0, "z_min": -1.0, "z_max": 1.0},
    ]


def _patch_plotting(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pyplot: _FakePyplot,
    animation: type[_FakeAnimation],
) -> None:
    """Patch the control-room module to use deterministic plotting facades."""
    import scpn_control.control.fusion_control_room as fcr

    monkeypatch.setattr(fcr, "HAS_MPL", True)
    monkeypatch.setattr(fcr, "plt", pyplot)
    monkeypatch.setattr(fcr, "Rectangle", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(fcr, "FuncAnimation", animation)
    monkeypatch.setattr(fcr, "PillowWriter", _FakeWriter)


def test_render_outputs_saves_animation_and_report_with_plotting_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The renderer drives animation and report export through the plotting API."""
    import scpn_control.control.fusion_control_room as fcr

    _patch_plotting(monkeypatch, pyplot=_FakePyplot(), animation=_FakeAnimation)
    animation_saved, animation_error, report_saved, report_error = fcr._render_outputs(
        _frames(),
        [0.0, 0.1],
        [0.0, 0.2],
        [0.0, 0.3],
        save_animation=True,
        save_report=True,
        output_gif="out.gif",
        output_report="out.png",
    )

    assert animation_saved is True
    assert animation_error is None
    assert report_saved is True
    assert report_error is None


def test_render_outputs_reports_animation_save_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Animation save errors are returned without suppressing report export."""
    import scpn_control.control.fusion_control_room as fcr

    _patch_plotting(monkeypatch, pyplot=_FakePyplot(), animation=_FailingAnimation)
    animation_saved, animation_error, report_saved, report_error = fcr._render_outputs(
        _frames(),
        [0.0, 0.1],
        [0.0, 0.2],
        [0.0, 0.3],
        save_animation=True,
        save_report=True,
        output_gif="out.gif",
        output_report="out.png",
    )

    assert animation_saved is False
    assert animation_error == "animation save failed"
    assert report_saved is True
    assert report_error is None


def test_render_outputs_reports_report_save_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report save errors are returned without suppressing animation export."""
    import scpn_control.control.fusion_control_room as fcr

    _patch_plotting(monkeypatch, pyplot=_FakePyplot(fail_save=True), animation=_FakeAnimation)
    animation_saved, animation_error, report_saved, report_error = fcr._render_outputs(
        _frames(),
        [0.0, 0.1],
        [0.0, 0.2],
        [0.0, 0.3],
        save_animation=True,
        save_report=True,
        output_gif="out.gif",
        output_report="out.png",
    )

    assert animation_saved is True
    assert animation_error is None
    assert report_saved is False
    assert report_error == "report save failed"
