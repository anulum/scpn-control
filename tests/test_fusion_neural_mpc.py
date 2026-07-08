# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fusion neural MPC tests.
"""Deterministic tests for fusion_neural_mpc runtime/controller paths."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_control.control.fusion_neural_mpc import (
    HAS_MPL,
    ModelPredictiveController,
    NeuralSurrogate,
    run_neural_mpc_simulation,
)

FloatArray = NDArray[np.float64]


class _CoilConfig(TypedDict):
    """Coil state used by the deterministic dummy kernel."""

    name: str
    current: float


class _PhysicsConfig(TypedDict):
    """Physics block used by the deterministic dummy kernel."""

    plasma_current_target: float


class _KernelConfig(TypedDict):
    """Configuration shape consumed by ``solve_equilibrium``."""

    physics: _PhysicsConfig
    coils: list[_CoilConfig]


class _RunKwargs(TypedDict, total=False):
    """Typed keyword arguments for ``run_neural_mpc_simulation``."""

    config_file: str | None
    shot_length: int
    prediction_horizon: int
    target_vector: FloatArray
    disturbance_start_step: int
    disturbance_per_step_ma: float
    current_target_bounds: tuple[float, float]
    action_limit: float
    coil_current_limits: tuple[float, float]
    save_plot: bool
    output_path: str
    verbose: bool
    kernel_factory: Callable[[str], "_DummyKernel"]


class _ControllerKwargs(TypedDict, total=False):
    """Typed keyword arguments for ``ModelPredictiveController``."""

    prediction_horizon: int
    learning_rate: float
    iterations: int
    action_limit: float
    action_regularization: float


class _DummyKernel:
    """Deterministic stand-in for FusionKernel used by neural MPC tests."""

    def __init__(self, _config_file: str) -> None:
        self.cfg: _KernelConfig = {
            "physics": {"plasma_current_target": 7.0},
            "coils": [
                {"name": "PF1", "current": 0.0},
                {"name": "PF2", "current": 0.0},
                {"name": "PF3", "current": 0.0},
                {"name": "PF4", "current": 0.0},
            ],
        }
        self.R: FloatArray = np.asarray(np.linspace(5.8, 6.3, 25), dtype=np.float64)
        self.Z: FloatArray = np.asarray(np.linspace(-0.4, 0.4, 25), dtype=np.float64)
        rr, zz = np.meshgrid(self.R, self.Z)
        self.RR: FloatArray = np.asarray(rr, dtype=np.float64)
        self.ZZ: FloatArray = np.asarray(zz, dtype=np.float64)
        self.Psi: FloatArray = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._xp: tuple[float, float] = (5.0, -3.5)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        i = [float(c["current"]) for c in self.cfg["coils"]]
        radial_drive = 0.8 * i[2] - 0.4 * i[1] + 0.1 * i[3]
        vertical_drive = 0.7 * i[3] - 0.6 * i[0] + 0.1 * i[2]

        center_r = 6.0 + 0.08 * np.tanh(radial_drive / 7.0)
        center_z = 0.0 + 0.06 * np.tanh(vertical_drive / 7.0)

        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0 + 0.001 * float(self.cfg["physics"]["plasma_current_target"])

        xr = 5.0 + 0.03 * np.tanh((i[1] - i[0]) / 6.0)
        xz = -3.5 + 0.03 * np.tanh((i[3] - i[2]) / 6.0)
        self._xp = (float(xr), float(xz))

    def find_x_point(self, _psi: FloatArray) -> tuple[tuple[float, float], float]:
        return self._xp, 0.0


def _default_run_kwargs() -> _RunKwargs:
    """Return deterministic simulation arguments used by repeated-call tests."""

    return {
        "config_file": "dummy.json",
        "shot_length": 18,
        "prediction_horizon": 5,
        "disturbance_start_step": 4,
        "disturbance_per_step_ma": 0.5,
        "current_target_bounds": (6.5, 8.0),
        "action_limit": 0.35,
        "coil_current_limits": (-1.0, 1.0),
        "save_plot": False,
        "verbose": False,
        "kernel_factory": _DummyKernel,
    }


def _runtime_kwargs_with(field: str, value: int) -> _RunKwargs:
    """Return runtime arguments with one invalid integer field overridden."""

    kwargs = _default_run_kwargs()
    if field == "shot_length":
        kwargs["shot_length"] = value
    elif field == "disturbance_start_step":
        kwargs["disturbance_start_step"] = value
    else:
        raise AssertionError(f"unhandled runtime field {field!r}")
    return kwargs


def _controller_kwargs_with(field: str, value: float | int) -> _ControllerKwargs:
    """Return controller constructor arguments with one invalid field."""

    if field == "prediction_horizon":
        return {"prediction_horizon": int(value)}
    if field == "learning_rate":
        return {"learning_rate": float(value)}
    if field == "iterations":
        return {"iterations": int(value)}
    if field == "action_limit":
        return {"action_limit": float(value)}
    if field == "action_regularization":
        return {"action_regularization": float(value)}
    raise AssertionError(f"unhandled controller field {field!r}")


def test_run_neural_mpc_simulation_returns_finite_bounded_summary() -> None:
    summary = run_neural_mpc_simulation(
        config_file="dummy.json",
        shot_length=22,
        prediction_horizon=6,
        disturbance_start_step=3,
        disturbance_per_step_ma=0.7,
        current_target_bounds=(7.0, 9.0),
        action_limit=0.4,
        coil_current_limits=(-1.25, 1.25),
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    for key in (
        "config_path",
        "steps",
        "prediction_horizon",
        "runtime_seconds",
        "final_target_ip_ma",
        "final_r_axis",
        "final_z_axis",
        "final_xpoint_r",
        "final_xpoint_z",
        "mean_tracking_error",
        "max_abs_action",
        "max_abs_coil_current",
        "plot_saved",
    ):
        assert key in summary
    assert summary["config_path"] == "dummy.json"
    assert summary["steps"] == 22
    assert summary["prediction_horizon"] == 6
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert np.isfinite(summary["runtime_seconds"])
    assert np.isfinite(summary["mean_tracking_error"])
    assert summary["max_abs_action"] <= 0.4 + 1e-9
    assert summary["max_abs_coil_current"] <= 1.25 + 1e-9
    assert 7.0 <= summary["final_target_ip_ma"] <= 9.0


def test_run_neural_mpc_simulation_is_deterministic_for_fixed_inputs() -> None:
    kwargs = _default_run_kwargs()
    a = run_neural_mpc_simulation(**kwargs)
    b = run_neural_mpc_simulation(**kwargs)
    for key in (
        "final_target_ip_ma",
        "final_r_axis",
        "final_z_axis",
        "final_xpoint_r",
        "final_xpoint_z",
        "mean_tracking_error",
        "max_abs_action",
        "max_abs_coil_current",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_mpc_plan_is_clipped_to_action_limit() -> None:
    surrogate = NeuralSurrogate(n_coils=3, n_state=4, verbose=False)
    surrogate.B[:] = 10.0

    mpc = ModelPredictiveController(
        surrogate=surrogate,
        target_state=np.zeros(4, dtype=np.float64),
        prediction_horizon=4,
        learning_rate=1.0,
        iterations=5,
        action_limit=0.3,
        action_regularization=0.0,
    )
    action = mpc.plan_trajectory(np.full(4, 5.0, dtype=np.float64))
    assert float(np.max(np.abs(action))) <= 0.3 + 1e-12


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shot_length": 0}, "shot_length"),
        ({"disturbance_start_step": -1}, "disturbance_start_step"),
    ],
)
def test_run_neural_mpc_simulation_rejects_invalid_runtime_inputs(kwargs: dict[str, int], match: str) -> None:
    [(field, value)] = kwargs.items()
    with pytest.raises(ValueError, match=match):
        run_neural_mpc_simulation(**_runtime_kwargs_with(field, value))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prediction_horizon": 0}, "prediction_horizon"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"iterations": 0}, "iterations"),
        ({"action_limit": 0.0}, "action_limit"),
        ({"action_regularization": -1.0}, "action_regularization"),
    ],
)
def test_mpc_controller_rejects_invalid_constructor_inputs(kwargs: dict[str, float | int], match: str) -> None:
    surrogate = NeuralSurrogate(n_coils=3, n_state=4, verbose=False)
    [(field, value)] = kwargs.items()
    with pytest.raises(ValueError, match=match):
        ModelPredictiveController(
            surrogate=surrogate,
            target_state=np.zeros(4, dtype=np.float64),
            **_controller_kwargs_with(field, value),
        )


def test_surrogate_rejects_invalid_perturbation() -> None:
    surrogate = NeuralSurrogate(n_coils=4, n_state=4, verbose=False)
    kernel = _DummyKernel("dummy.json")
    with pytest.raises(ValueError, match="perturbation"):
        surrogate.train_on_kernel(kernel, perturbation=0.0)


def test_run_neural_mpc_simulation_default_config_path() -> None:
    summary = run_neural_mpc_simulation(
        config_file=None,
        shot_length=8,
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    assert "iter_config.json" in summary["config_path"]


def test_run_neural_mpc_simulation_explicit_target_vector() -> None:
    summary = run_neural_mpc_simulation(
        config_file="dummy.json",
        shot_length=8,
        target_vector=np.array([6.1, 0.1, 5.2, -3.3]),
        save_plot=False,
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    assert summary["steps"] == 8
    assert np.isfinite(summary["mean_tracking_error"])


@pytest.mark.skipif(not HAS_MPL, reason="requires matplotlib (optional viz dependency)")
def test_run_neural_mpc_simulation_saves_plot(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    out = tmp_path / "results.png"
    with caplog.at_level(logging.INFO, logger="scpn_control.control.fusion_neural_mpc"):
        summary = run_neural_mpc_simulation(
            config_file="dummy.json",
            shot_length=6,
            save_plot=True,
            output_path=str(out),
            verbose=True,
            kernel_factory=_DummyKernel,
        )
    assert summary["plot_saved"] is True
    assert summary["plot_error"] is None
    assert out.exists()
    assert "Neural-MPC analysis saved" in caplog.text


@pytest.mark.skipif(not HAS_MPL, reason="requires matplotlib (optional viz dependency)")
def test_run_neural_mpc_simulation_reports_plot_failure(tmp_path: Path) -> None:
    # The parent directory does not exist, so savefig raises an OSError the
    # plotting helper must catch and surface as plot_error.
    bad = tmp_path / "missing_dir" / "results.png"
    summary = run_neural_mpc_simulation(
        config_file="dummy.json",
        shot_length=6,
        save_plot=True,
        output_path=str(bad),
        verbose=False,
        kernel_factory=_DummyKernel,
    )
    assert summary["plot_saved"] is False
    assert summary["plot_error"]
    assert not bad.exists()


def test_run_neural_mpc_simulation_verbose_prints_output(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="scpn_control.control.fusion_neural_mpc"):
        run_neural_mpc_simulation(
            config_file="dummy.json",
            shot_length=12,
            save_plot=False,
            verbose=True,
            kernel_factory=_DummyKernel,
        )
    assert "Neural-MPC Hybrid Control" in caplog.text
    assert "Simulation finished" in caplog.text
