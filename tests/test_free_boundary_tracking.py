# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Free-Boundary Tracking Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for free-boundary target tracking control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_control.control.free_boundary_tracking import (
    FreeBoundaryTrackingController,
    run_free_boundary_tracking,
)
from scpn_control.core.fusion_kernel import CoilSet, FusionKernel


class _DummyFreeBoundaryKernel:
    """Linear free-boundary plant with explicit boundary/X-point/divertor observables."""

    def __init__(self, _config_file: str) -> None:
        self._boundary_points = np.array(
            [
                [3.4, -0.1],
                [4.0, 0.3],
                [4.6, -0.4],
            ],
            dtype=np.float64,
        )
        self._divertor_points = np.array(
            [
                [3.1, -2.6],
                [4.9, -2.6],
            ],
            dtype=np.float64,
        )
        self._x_target = np.array([4.2, -1.4], dtype=np.float64)
        self._target_vector = np.array(
            [0.12, 0.18, 0.10, 4.2, -1.4, 0.15, 0.15, 0.15],
            dtype=np.float64,
        )
        self._response_matrix = np.array(
            [
                [0.60, -0.20, 0.15, 0.05],
                [0.10, 0.55, -0.15, 0.05],
                [-0.20, 0.10, 0.50, 0.10],
                [0.12, -0.08, 0.03, 0.01],
                [-0.04, 0.02, 0.11, -0.09],
                [0.22, 0.10, -0.05, 0.04],
                [0.14, -0.12, 0.18, 0.05],
                [0.11, 0.09, -0.04, 0.16],
            ],
            dtype=np.float64,
        )
        self._bias = self._response_matrix @ np.array([0.45, -0.35, 0.30, -0.20], dtype=np.float64)
        self._drift_vector = self._response_matrix @ np.array([0.10, -0.06, 0.08, -0.04], dtype=np.float64)
        self.cfg = {
            "physics": {"drift_scale": 0.0},
            "coils": [
                {"name": "PF1", "current": 0.0},
                {"name": "PF2", "current": 0.0},
                {"name": "PF3", "current": 0.0},
                {"name": "PF4", "current": 0.0},
            ],
            "free_boundary": {
                "objective_tolerances": {
                    "shape_rms": 0.025,
                    "x_point_position": 0.08,
                    "x_point_flux": 0.03,
                    "divertor_rms": 0.025,
                }
            },
        }
        self.R = np.linspace(3.0, 5.2, 8)
        self.Z = np.linspace(-3.0, 1.0, 8)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((len(self.Z), len(self.R)), dtype=np.float64)
        self._state = self._target_vector + self._bias
        self.solve()

    def build_coilset_from_config(self) -> CoilSet:
        return CoilSet(
            positions=[(3.0, 2.2), (3.6, -2.1), (4.4, 2.0), (5.0, -2.2)],
            currents=np.zeros(4, dtype=np.float64),
            turns=[12, 12, 12, 12],
            current_limits=np.full(4, 3.0, dtype=np.float64),
            target_flux_points=self._boundary_points.copy(),
            target_flux_values=self._target_vector[:3].copy(),
            x_point_target=self._x_target.copy(),
            x_point_flux_target=float(self._target_vector[5]),
            divertor_strike_points=self._divertor_points.copy(),
            divertor_flux_values=self._target_vector[6:].copy(),
        )

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, float | bool | str]:
        active_coils = coils if coils is not None else self.build_coilset_from_config()
        currents = np.asarray(active_coils.currents, dtype=np.float64).reshape(-1)
        drift = float(self.cfg.get("physics", {}).get("drift_scale", 0.0))
        disturbance = drift * self._drift_vector
        self._state = self._target_vector + self._bias + disturbance + self._response_matrix @ currents
        for idx, current in enumerate(currents):
            self.cfg["coils"][idx]["current"] = float(current)
        self.Psi.fill(0.0)
        return {
            "boundary_variant": "free_boundary" if boundary_variant is None else str(boundary_variant),
            "converged": True,
            "outer_iterations": 1,
            "final_diff": float(np.linalg.norm(self._response_matrix @ currents)),
        }

    def _sample_flux_at_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == self._boundary_points.shape and np.allclose(pts, self._boundary_points):
            return self._state[:3].copy()
        if pts.shape == self._divertor_points.shape and np.allclose(pts, self._divertor_points):
            return self._state[6:].copy()
        raise ValueError("Unexpected probe points for dummy free-boundary kernel.")

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return (float(self._state[3]), float(self._state[4])), float(self._state[5])

    def _interp_psi(self, r_pt: float, z_pt: float) -> float:
        if np.allclose([r_pt, z_pt], self._x_target):
            return float(self._state[5])
        return float(np.mean(self._state[:3]))


class _NoTargetKernel(_DummyFreeBoundaryKernel):
    def build_coilset_from_config(self) -> CoilSet:
        coils = super().build_coilset_from_config()
        coils.target_flux_values = None
        coils.x_point_target = None
        coils.x_point_flux_target = None
        coils.divertor_flux_values = None
        return coils


def _write_real_kernel_tracking_config(path: Path) -> Path:
    cfg = {
        "reactor_name": "Real-Free-Boundary-Tracking-Test",
        "grid_resolution": [12, 12],
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0},
        ],
        "solver": {
            "max_iterations": 10,
            "convergence_threshold": 1e-3,
            "relaxation_factor": 0.15,
            "solver_method": "sor",
            "boundary_variant": "free_boundary",
        },
        "free_boundary": {
            "current_limits": [5.0e4, 5.0e4],
            "target_flux_points": [[3.5, 0.0], [4.0, 0.5]],
            "objective_tolerances": {"shape_rms": 0.25, "shape_max_abs": 0.35},
        },
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")

    kernel = FusionKernel(path)
    coils = kernel.build_coilset_from_config()
    kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=1e-2,
        optimize_shape=False,
    )
    flux_targets = kernel._sample_flux_at_points(coils.target_flux_points)
    cfg["free_boundary"]["target_flux_values"] = [float(v) for v in flux_targets]
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_run_free_boundary_tracking_returns_bounded_converged_summary() -> None:
    summary = run_free_boundary_tracking(
        config_file="dummy.json",
        shot_steps=8,
        gain=0.9,
        verbose=False,
        kernel_factory=_DummyFreeBoundaryKernel,
        stop_on_convergence=False,
    )
    for key in (
        "config_path",
        "steps",
        "runtime_seconds",
        "boundary_variant",
        "final_tracking_error_norm",
        "mean_tracking_error_norm",
        "max_abs_delta_i",
        "max_abs_coil_current",
        "objective_convergence_active",
        "objective_converged",
        "objective_checks",
    ):
        assert key in summary
    assert summary["config_path"] == "dummy.json"
    assert summary["boundary_variant"] == "free_boundary"
    assert summary["steps"] == 8
    assert np.isfinite(summary["runtime_seconds"])
    assert summary["max_abs_coil_current"] <= 3.0 + 1e-9
    assert summary["objective_convergence_active"] is True
    assert summary["objective_converged"] is True
    assert summary["final_tracking_error_norm"] < 0.15


def test_free_boundary_tracking_is_deterministic_for_fixed_inputs() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_steps=6,
        gain=0.85,
        verbose=False,
        kernel_factory=_DummyFreeBoundaryKernel,
        stop_on_convergence=False,
    )
    a = run_free_boundary_tracking(**kwargs)
    b = run_free_boundary_tracking(**kwargs)
    for key in (
        "final_tracking_error_norm",
        "mean_tracking_error_norm",
        "max_abs_delta_i",
        "max_abs_coil_current",
        "shape_rms",
        "x_point_position_error",
        "x_point_flux_error",
        "divertor_rms",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_controller_reduces_error_under_disturbance_callback() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_DummyFreeBoundaryKernel,
        verbose=False,
        response_refresh_steps=1,
    )
    controller._solve_free_boundary_state()
    initial_metrics = controller.evaluate_objectives(controller._observe_objectives())

    def disturbance(kernel: _DummyFreeBoundaryKernel, _coils: CoilSet, step: int) -> None:
        kernel.cfg.setdefault("physics", {})["drift_scale"] = 0.8 if step < 2 else 0.2

    summary = controller.run_tracking_shot(
        shot_steps=7,
        gain=0.9,
        disturbance_callback=disturbance,
        stop_on_convergence=False,
    )

    assert summary["final_tracking_error_norm"] < initial_metrics["tracking_error_norm"]
    assert summary["objective_converged"] is True
    assert summary["max_abs_coil_current"] <= 3.0 + 1e-9


def test_controller_rejects_invalid_runtime_inputs() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_DummyFreeBoundaryKernel,
        verbose=False,
    )
    with pytest.raises(ValueError, match="shot_steps"):
        controller.run_tracking_shot(shot_steps=0)
    with pytest.raises(ValueError, match="gain"):
        controller.run_tracking_shot(shot_steps=2, gain=0.0)
    with pytest.raises(ValueError, match="perturbation"):
        controller.identify_response_matrix(perturbation=0.0)


def test_controller_rejects_missing_explicit_targets() -> None:
    with pytest.raises(ValueError, match="explicit target values"):
        FreeBoundaryTrackingController(
            "dummy.json",
            kernel_factory=_NoTargetKernel,
            verbose=False,
        )


def test_controller_backtracks_aggressive_gain() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_DummyFreeBoundaryKernel,
        verbose=False,
    )

    summary = controller.run_tracking_shot(
        shot_steps=3,
        gain=12.0,
        stop_on_convergence=False,
    )

    assert summary["mean_accepted_gain"] > 0.0
    assert summary["min_accepted_gain"] > 0.0
    assert summary["mean_accepted_gain"] < 12.0
    assert max(controller.history["accepted_gain"]) < 12.0
    assert summary["final_tracking_error_norm"] < 0.25


def test_controller_enforces_coil_slew_limits() -> None:
    summary = run_free_boundary_tracking(
        config_file="dummy.json",
        shot_steps=1,
        gain=8.0,
        verbose=False,
        kernel_factory=_DummyFreeBoundaryKernel,
        control_dt_s=0.1,
        coil_actuator_tau_s=0.05,
        coil_slew_limits=0.5,
        stop_on_convergence=False,
    )

    assert summary["max_abs_coil_current"] <= 0.05 + 1e-9
    assert summary["max_abs_actuator_lag"] > 0.01
    assert summary["mean_abs_actuator_lag"] > 0.01


def test_controller_supervisor_rejects_and_holds_unsafe_updates() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_DummyFreeBoundaryKernel,
        verbose=False,
        control_dt_s=0.1,
        coil_actuator_tau_s=0.05,
        coil_slew_limits=0.5,
        supervisor_limits={"max_abs_actuator_lag": 0.0},
        hold_steps_after_reject=2,
    )
    controller._solve_free_boundary_state()
    initial_metrics = controller.evaluate_objectives(controller._observe_objectives())

    summary = controller.run_tracking_shot(
        shot_steps=4,
        gain=8.0,
        stop_on_convergence=False,
    )

    assert summary["supervisor_active"] is True
    assert summary["supervisor_intervention_count"] >= 3
    assert summary["hold_steps_after_reject"] == 2
    assert controller.history["supervisor_intervened"][0] is True
    assert controller.history["supervisor_hold_steps_remaining"][0] == 2
    assert controller.history["supervisor_hold_steps_remaining"][1] == 1
    assert controller.history["supervisor_hold_steps_remaining"][2] == 0
    assert summary["final_tracking_error_norm"] == pytest.approx(initial_metrics["tracking_error_norm"])
    assert summary["max_abs_coil_current"] == pytest.approx(0.0)


def test_run_free_boundary_tracking_with_real_kernel_smoke(tmp_path: Path) -> None:
    cfg_path = _write_real_kernel_tracking_config(tmp_path / "real_tracking.json")

    summary = run_free_boundary_tracking(
        config_file=str(cfg_path),
        shot_steps=2,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        stop_on_convergence=False,
    )

    assert summary["boundary_variant"] == "free_boundary"
    assert summary["steps"] == 2
    assert summary["objective_convergence_active"] is True
    assert summary["shape_rms"] is not None
    assert np.isfinite(summary["final_tracking_error_norm"])
    assert np.isfinite(summary["mean_tracking_error_norm"])
    assert summary["max_abs_coil_current"] <= 5.0e4 + 1e-9
