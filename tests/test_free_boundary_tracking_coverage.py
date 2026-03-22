# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Free-Boundary Tracking Coverage Gap Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Coverage for EKF integration, disturbance observer, latency compensation,
supervisor ramp toward fallback, and sensor bias+drift paths."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.free_boundary_tracking import (
    FreeBoundaryTrackingController,
    run_free_boundary_tracking,
)
from scpn_control.core.fusion_kernel import CoilSet


class _DummyFreeBoundaryKernel:
    """Minimal linear free-boundary plant reused from the main test module."""

    def __init__(self, _config_file: str) -> None:
        self._boundary_points = np.array(
            [[3.4, -0.1], [4.0, 0.3], [4.6, -0.4]],
            dtype=np.float64,
        )
        self._divertor_points = np.array(
            [[3.1, -2.6], [4.9, -2.6]],
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
        self._bias = self._response_matrix @ np.array([0.45, -0.35, 0.30, -0.20])
        self._drift_vector = self._response_matrix @ np.array([0.10, -0.06, 0.08, -0.04])
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

    def solve(self, *, boundary_variant=None, coils=None, max_outer_iter=20,
              tol=1e-4, optimize_shape=False, tikhonov_alpha=1e-4):
        active_coils = coils if coils is not None else self.build_coilset_from_config()
        currents = np.asarray(active_coils.currents, dtype=np.float64).reshape(-1)
        drift = float(self.cfg.get("physics", {}).get("drift_scale", 0.0))
        self._state = (self._target_vector + self._bias
                       + drift * self._drift_vector
                       + self._response_matrix @ currents)
        for idx, current in enumerate(currents):
            self.cfg["coils"][idx]["current"] = float(current)
        self.Psi.fill(0.0)
        return {"boundary_variant": "free_boundary", "converged": True,
                "outer_iterations": 1,
                "final_diff": float(np.linalg.norm(self._response_matrix @ currents))}

    def _sample_flux_at_points(self, points):
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape == self._boundary_points.shape and np.allclose(pts, self._boundary_points):
            return self._state[:3].copy()
        if pts.shape == self._divertor_points.shape and np.allclose(pts, self._divertor_points):
            return self._state[6:].copy()
        raise ValueError("Unexpected probe points")

    def find_x_point(self, _psi):
        return (float(self._state[3]), float(self._state[4])), float(self._state[5])

    def _interp_psi(self, r_pt, z_pt):
        if np.allclose([r_pt, z_pt], self._x_target):
            return float(self._state[5])
        return float(np.mean(self._state[:3]))


# ── Kernel variants for specific coverage paths ─────────────────────


class _EKFKernel(_DummyFreeBoundaryKernel):
    pass


class _DisturbanceObserverKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "observer_gain": 0.6,
            "observer_forgetting": 0.1,
            "observer_max_abs": 0.5,
        }


class _LatencyWithCompensationKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "measurement_latency_steps": 3,
            "latency_compensation_gain": 0.8,
            "latency_rate_max_abs": 1.0,
        }


class _SensorBiasDriftKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "measurement_bias": {
                "shape_flux": [0.02, -0.01, 0.015],
                "x_point_position": [0.03, -0.02],
                "x_point_flux": 0.01,
                "divertor_flux": [0.01, -0.008],
            },
            "measurement_drift_per_step": {
                "shape_flux": [0.003, -0.002, 0.002],
                "x_point_position": [0.005, -0.004],
                "x_point_flux": 0.002,
                "divertor_flux": [0.002, -0.001],
            },
        }


class _SupervisorFallbackRampKernel(_DummyFreeBoundaryKernel):
    """Degenerate response forces supervisor intervention; fallback currents configured."""

    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "fallback_currents": [0.15, -0.15, 0.10, -0.10],
        }
        self._response_matrix = np.ones((8, 4), dtype=np.float64) * 1e-14


# ── Tests ────────────────────────────────────────────────────────────


class TestEKFIntegration:
    """use_ekf=True path: EKF predict/update cycle in _observe_snapshot."""

    def test_ekf_refines_observation(self):
        from scpn_control.control.state_estimator import ExtendedKalmanFilter

        x0 = np.array([4.2, -1.4, 0.0, 0.0, 15.0, 5.0])
        P0 = np.eye(6) * 0.1
        Q = np.eye(6) * 0.01
        R_cov = np.eye(4) * 0.01

        ekf = ExtendedKalmanFilter(x0=x0, P0=P0, Q=Q, R_cov=R_cov)
        controller = FreeBoundaryTrackingController(
            "dummy.json",
            kernel_factory=_EKFKernel,
            verbose=False,
            state_estimator=ekf,
        )
        summary = controller.run_tracking_shot(shot_steps=3, gain=0.5)
        assert summary["steps"] == 3
        est = ekf.estimate()
        assert np.all(np.isfinite(est))


class TestDisturbanceObserverAccumulation:
    """do_enabled=True: observer_gain > 0 accumulates objective bias."""

    def test_observer_gain_accumulates_residual(self):
        controller = FreeBoundaryTrackingController(
            "dummy.json",
            kernel_factory=_DisturbanceObserverKernel,
            verbose=False,
        )
        summary = controller.run_tracking_shot(shot_steps=5, gain=0.3)
        assert summary["observer_enabled"] is True
        assert summary["observer_gain"] == pytest.approx(0.6)
        assert summary["max_abs_objective_bias_estimate"] > 0.0

    def test_observer_with_forgetting_caps_bias(self):
        controller = FreeBoundaryTrackingController(
            "dummy.json",
            kernel_factory=_DisturbanceObserverKernel,
            verbose=False,
        )
        summary = controller.run_tracking_shot(shot_steps=8, gain=0.3)
        assert summary["max_abs_objective_bias_estimate"] <= 0.5 + 1e-6


class TestLatencyWithCompensation:
    """latency_steps > 0 with compensate_latency=True."""

    def test_latency_compensation_path(self):
        def disturbance(kernel, _coils, step):
            schedule = (0.0, 0.5, 0.3, 0.1, 0.0)
            kernel.cfg.setdefault("physics", {})["drift_scale"] = schedule[min(step, len(schedule) - 1)]

        summary = run_free_boundary_tracking(
            config_file="dummy.json",
            kernel_factory=_LatencyWithCompensationKernel,
            shot_steps=5,
            gain=0.2,
            verbose=False,
            disturbance_callback=disturbance,
        )
        assert summary["measurement_latency_enabled"] is True
        assert summary["measurement_latency_steps"] == 3
        assert summary["latency_compensation_enabled"] is True
        assert summary["latency_compensation_gain"] == pytest.approx(0.8)


class TestSupervisorFallbackRamp:
    """When response is degenerate, supervisor ramps toward safe_fallback."""

    def test_supervisor_ramp_to_fallback(self):
        summary = run_free_boundary_tracking(
            config_file="dummy.json",
            kernel_factory=_SupervisorFallbackRampKernel,
            shot_steps=3,
            gain=0.5,
            verbose=False,
        )
        assert summary["response_degenerate_count"] > 0
        assert summary["fallback_active_steps"] > 0
        assert summary["supervisor_intervention_count"] > 0
        assert summary["fallback_configured"] is True


class TestSensorBiasWithDrift:
    """sensor_bias nonzero with drift accumulates measurement_offset."""

    def test_bias_drift_grows_measurement_error(self):
        summary = run_free_boundary_tracking(
            config_file="dummy.json",
            kernel_factory=_SensorBiasDriftKernel,
            shot_steps=5,
            gain=0.2,
            verbose=False,
        )
        assert summary["measurement_distortion_enabled"] is True
        assert summary["max_measurement_error_norm"] > 0.0
        assert summary["max_abs_measurement_offset"] > 0.0
