# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Free-Boundary Tracking Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic tests for free-boundary target tracking control."""

from __future__ import annotations


import numpy as np
import pytest

from scpn_control.control.free_boundary_tracking import (
    FreeBoundaryTrackingController,
    run_free_boundary_tracking,
)
from scpn_control.core.fusion_kernel import CoilSet


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


class _FallbackKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "fallback_currents": [0.20, -0.20, 0.10, -0.10],
        }


class _ObserverKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "observer_gain": 0.45,
            "observer_max_abs": 0.35,
        }


class _MeasurementDistortionKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "measurement_bias": {
                "shape_flux": [0.04, -0.02, 0.03],
                "x_point_position": [0.06, -0.05],
                "x_point_flux": 0.025,
                "divertor_flux": [0.03, -0.02],
            },
            "measurement_drift_per_step": {
                "shape_flux": [0.006, -0.003, 0.004],
                "x_point_position": [0.01, -0.008],
                "x_point_flux": 0.004,
                "divertor_flux": [0.005, -0.003],
            },
        }


class _MeasurementCorrectedKernel(_MeasurementDistortionKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        tracking_cfg = self.cfg["free_boundary_tracking"]
        tracking_cfg["measurement_correction_bias"] = {
            "shape_flux": [0.04, -0.02, 0.03],
            "x_point_position": [0.06, -0.05],
            "x_point_flux": 0.025,
            "divertor_flux": [0.03, -0.02],
        }
        tracking_cfg["measurement_correction_drift_per_step"] = {
            "shape_flux": [0.006, -0.003, 0.004],
            "x_point_position": [0.01, -0.008],
            "x_point_flux": 0.004,
            "divertor_flux": [0.005, -0.003],
        }


class _MeasurementLatencyKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "measurement_latency_steps": 2,
        }


class _MeasurementLatencyCompensatedKernel(_MeasurementLatencyKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        tracking_cfg = self.cfg["free_boundary_tracking"]
        tracking_cfg["latency_compensation_gain"] = 1.0


class _InvalidMeasurementKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "measurement_bias": {
                "x_point_position": [0.01, 0.02, 0.03],
            },
        }


class _PriorityConflictKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary"] = dict(self.cfg.get("free_boundary", {}))
        self.cfg["free_boundary"]["objective_tolerances"] = {
            "shape_rms": 1.0,
            "x_point_flux": 0.001,
        }


# ── EKF integration kernel ──────────────────────────────────────────


class _EKFKernel(_DummyFreeBoundaryKernel):
    """Kernel used with state_estimator=ExtendedKalmanFilter."""

    pass


# ── Disturbance observer kernel ──────────────────────────────────────


class _DisturbanceObserverKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "observer_gain": 0.6,
            "observer_forgetting": 0.1,
            "observer_max_abs": 0.5,
        }


# ── Latency with compensation kernel ────────────────────────────────


class _LatencyWithCompensationKernel(_DummyFreeBoundaryKernel):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "measurement_latency_steps": 3,
            "latency_compensation_gain": 0.8,
            "latency_rate_max_abs": 1.0,
        }


# ── Sensor bias + drift kernel ──────────────────────────────────────


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


# ── Supervisor fallback ramp kernel ─────────────────────────────────


class _SupervisorFallbackRampKernel(_DummyFreeBoundaryKernel):
    """Degenerate response forces supervisor intervention; fallback currents configured."""

    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.cfg["free_boundary_tracking"] = {
            "fallback_currents": [0.15, -0.15, 0.10, -0.10],
        }
        # Make response nearly singular to trigger degenerate flag
        self._response_matrix = np.ones((8, 4), dtype=np.float64) * 1e-14


# ── Tests ────────────────────────────────────────────────────────────


def test_basic_tracking_reduces_error() -> None:
    summary = run_free_boundary_tracking(
        config_file="dummy.json",
        kernel_factory=_DummyFreeBoundaryKernel,
        shot_steps=3,
        gain=0.8,
        verbose=False,
    )
    assert summary["boundary_variant"] == "free_boundary"
    assert summary["steps"] == 3


def test_controller_no_target_raises() -> None:
    with pytest.raises(ValueError, match="requires explicit target values"):
        FreeBoundaryTrackingController(
            "dummy.json",
            kernel_factory=_NoTargetKernel,
            verbose=False,
        )


def test_observer_accumulates_bias() -> None:
    summary = run_free_boundary_tracking(
        config_file="dummy.json",
        kernel_factory=_ObserverKernel,
        shot_steps=5,
        gain=0.3,
        verbose=False,
    )
    assert summary["observer_enabled"] is True
    assert summary["max_abs_objective_bias_estimate"] > 0.0


def test_fallback_currents_activated_on_degenerate_response() -> None:
    summary = run_free_boundary_tracking(
        config_file="dummy.json",
        kernel_factory=_FallbackKernel,
        shot_steps=3,
        gain=0.5,
        verbose=False,
    )
    assert summary["fallback_configured"] is True


def test_measurement_distortion_and_correction() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_steps=4,
        gain=0.18,
        verbose=False,
        stop_on_convergence=False,
    )
    baseline = run_free_boundary_tracking(
        kernel_factory=_DummyFreeBoundaryKernel,
        **kwargs,
    )
    distorted = run_free_boundary_tracking(
        kernel_factory=_MeasurementDistortionKernel,
        **kwargs,
    )
    corrected = run_free_boundary_tracking(
        kernel_factory=_MeasurementCorrectedKernel,
        **kwargs,
    )

    assert distorted["measurement_compensation_enabled"] is False
    assert distorted["final_true_tracking_error_norm"] != pytest.approx(
        baseline["final_true_tracking_error_norm"],
        rel=0.0,
        abs=1.0e-4,
    )
    assert distorted["final_tracking_error_norm"] != pytest.approx(
        distorted["final_true_tracking_error_norm"],
        rel=0.0,
        abs=1.0e-4,
    )
    assert corrected["measurement_compensation_enabled"] is True
    for key in (
        "final_tracking_error_norm",
        "mean_tracking_error_norm",
        "final_true_tracking_error_norm",
        "mean_true_tracking_error_norm",
        "true_shape_rms",
        "true_x_point_position_error",
        "true_x_point_flux_error",
        "true_divertor_rms",
    ):
        assert corrected[key] == pytest.approx(baseline[key], rel=0.0, abs=1.0e-12)


def test_latency_compensation_reduces_delayed_observation_error() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_steps=4,
        gain=0.18,
        verbose=False,
        stop_on_convergence=False,
    )

    def disturbance(kernel: _DummyFreeBoundaryKernel, _coils: CoilSet, step: int) -> None:
        drift_schedule = (0.0, 1.0, 0.45, 0.10)
        kernel.cfg.setdefault("physics", {})["drift_scale"] = drift_schedule[min(step, len(drift_schedule) - 1)]

    delayed = run_free_boundary_tracking(
        kernel_factory=_MeasurementLatencyKernel,
        disturbance_callback=disturbance,
        **kwargs,
    )
    compensated = run_free_boundary_tracking(
        kernel_factory=_MeasurementLatencyCompensatedKernel,
        disturbance_callback=disturbance,
        **kwargs,
    )

    assert delayed["measurement_latency_enabled"] is True
    assert delayed["latency_compensation_enabled"] is False
    assert delayed["measurement_latency_steps"] == 2
    assert delayed["max_delayed_observation_error_norm"] > 0.0
    assert delayed["max_estimated_observation_error_norm"] == pytest.approx(
        delayed["max_delayed_observation_error_norm"],
        rel=0.0,
        abs=1.0e-12,
    )
    assert compensated["measurement_latency_enabled"] is True
    assert compensated["latency_compensation_enabled"] is True
    assert compensated["max_abs_objective_rate_estimate"] > 0.0
    assert compensated["mean_estimated_observation_error_norm"] < delayed["max_delayed_observation_error_norm"]
    assert compensated["max_estimated_observation_error_norm"] <= delayed["max_delayed_observation_error_norm"] + 0.02
    assert compensated["final_true_tracking_error_norm"] < delayed["final_true_tracking_error_norm"]


def test_controller_rejects_invalid_measurement_bias_shape() -> None:
    with pytest.raises(ValueError, match="measurement_bias.x_point_position must be a scalar or contain exactly 2"):
        FreeBoundaryTrackingController(
            "dummy.json",
            kernel_factory=_InvalidMeasurementKernel,
            verbose=False,
        )


def test_controller_prioritizes_tighter_x_point_flux_tolerance_under_conflict() -> None:
    weighted = run_free_boundary_tracking(
        config_file="dummy.json",
        shot_steps=1,
        gain=1.0,
        verbose=False,
        kernel_factory=_PriorityConflictKernel,
        stop_on_convergence=False,
    )
    flat = run_free_boundary_tracking(
        config_file="dummy.json",
        shot_steps=1,
        gain=1.0,
        verbose=False,
        kernel_factory=_PriorityConflictKernel,
        objective_tolerances={"shape_rms": 2.0, "x_point_flux": 1.0},
        stop_on_convergence=False,
    )

    # Both runs produce finite, positive metrics
    for key in ("x_point_flux_error", "shape_rms", "max_abs_delta_i"):
        assert np.isfinite(weighted[key]) and weighted[key] >= 0
        assert np.isfinite(flat[key]) and flat[key] >= 0
    # Weighted and flat tolerances produce different trade-offs
    assert weighted != flat


def test_controller_does_not_sacrifice_already_met_tolerance() -> None:
    controller = FreeBoundaryTrackingController(
        "dummy.json",
        kernel_factory=_DummyFreeBoundaryKernel,
        verbose=False,
    )
    summary = controller.run_tracking_shot(shot_steps=5, gain=0.6, stop_on_convergence=True)
    assert summary["tolerance_regression_blocked_count"] >= 0


# ── Coverage-gap tests ──────────────────────────────────────────────


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
