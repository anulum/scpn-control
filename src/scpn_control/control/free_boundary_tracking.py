# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Free-Boundary Tracking Control
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: MIT OR Apache-2.0
# ──────────────────────────────────────────────────────────────────────
"""Closed-loop free-boundary tracking without a surrogate plant.

The controller re-identifies the local coil-to-objective response directly
from repeated :class:`scpn_control.core.fusion_kernel.FusionKernel` solves and
applies bounded least-squares coil corrections.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.fusion_kernel import CoilSet, FusionKernel

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _ObjectiveBlock:
    name: str
    start: int
    stop: int


class FreeBoundaryTrackingController:
    """Direct free-boundary controller using local coil-response identification.

    This path keeps the full Grad-Shafranov kernel in the loop instead of
    replacing it with a reduced-order plant.

    Examples
    --------
    >>> from scpn_control.control.free_boundary_tracking import run_free_boundary_tracking
    >>> summary = run_free_boundary_tracking(
    ...     "iter_config.json",
    ...     shot_steps=3,
    ...     gain=0.8,
    ...     verbose=False,
    ... )
    >>> summary["boundary_variant"]
    'free_boundary'
    """

    def __init__(
        self,
        config_file: str,
        *,
        kernel_factory: Callable[[str], Any] = FusionKernel,
        verbose: bool = True,
        identification_perturbation: float = 0.25,
        correction_limit: float = 2.0,
        response_regularization: float = 1e-3,
        response_refresh_steps: int = 1,
        solve_max_outer_iter: int = 10,
        solve_tol: float = 1e-4,
        objective_tolerances: dict[str, float] | None = None,
    ) -> None:
        self.kernel = kernel_factory(config_file)
        self.verbose = bool(verbose)
        self.identification_perturbation = float(identification_perturbation)
        if not np.isfinite(self.identification_perturbation) or self.identification_perturbation <= 0.0:
            raise ValueError("identification_perturbation must be finite and > 0.")
        self.correction_limit = float(correction_limit)
        if not np.isfinite(self.correction_limit) or self.correction_limit <= 0.0:
            raise ValueError("correction_limit must be finite and > 0.")
        self.response_regularization = float(response_regularization)
        if not np.isfinite(self.response_regularization) or self.response_regularization < 0.0:
            raise ValueError("response_regularization must be finite and >= 0.")
        self.response_refresh_steps = int(response_refresh_steps)
        if self.response_refresh_steps < 1:
            raise ValueError("response_refresh_steps must be >= 1.")
        self.solve_max_outer_iter = int(solve_max_outer_iter)
        if self.solve_max_outer_iter < 1:
            raise ValueError("solve_max_outer_iter must be >= 1.")
        self.solve_tol = float(solve_tol)
        if not np.isfinite(self.solve_tol) or self.solve_tol <= 0.0:
            raise ValueError("solve_tol must be finite and > 0.")

        build_coilset = getattr(self.kernel, "build_coilset_from_config", None)
        if not callable(build_coilset):
            raise TypeError("kernel must define build_coilset_from_config() for free-boundary tracking.")
        self.coils = build_coilset()
        self.n_coils = int(len(self.coils.positions))
        if self.n_coils < 1:
            raise ValueError("free-boundary tracking requires at least one external coil.")

        if self.coils.current_limits is None:
            self.coil_current_limits = np.full(self.n_coils, np.inf, dtype=np.float64)
        else:
            self.coil_current_limits = np.asarray(self.coils.current_limits, dtype=np.float64).reshape(-1)
            if self.coil_current_limits.shape != (self.n_coils,):
                raise ValueError("CoilSet.current_limits must match the number of coils.")
            if np.any(~np.isfinite(self.coil_current_limits)) or np.any(self.coil_current_limits <= 0.0):
                raise ValueError("CoilSet.current_limits must be finite and > 0.")

        self.objective_tolerances = self._resolve_objective_tolerances(
            self.kernel.cfg.get("free_boundary", {}).get("objective_tolerances"),
            objective_tolerances,
        )
        self.target_vector, self.objective_blocks = self._build_target_vector()
        if self.target_vector.size < 1:
            raise ValueError("free-boundary tracking requires explicit target values.")

        self.response_matrix = np.zeros((self.target_vector.size, self.n_coils), dtype=np.float64)
        self.history: dict[str, list[Any]] = {
            "t": [],
            "tracking_error_norm": [],
            "shape_rms": [],
            "shape_max_abs": [],
            "x_point_position_error": [],
            "x_point_flux_error": [],
            "divertor_rms": [],
            "divertor_max_abs": [],
            "max_abs_delta_i": [],
            "max_abs_coil_current": [],
            "accepted_gain": [],
            "objective_converged": [],
        }

    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    @staticmethod
    def _resolve_objective_tolerances(
        cfg_tolerances: Any,
        override_tolerances: dict[str, float] | None,
    ) -> dict[str, float]:
        allowed = {
            "shape_rms",
            "shape_max_abs",
            "x_point_position",
            "x_point_flux",
            "divertor_rms",
            "divertor_max_abs",
        }
        merged: dict[str, float] = {}
        for raw, name in (
            (cfg_tolerances, "free_boundary.objective_tolerances"),
            (override_tolerances, "objective_tolerances"),
        ):
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"{name} must be a mapping of tolerance names to non-negative floats.")
            for key, value in raw.items():
                if key not in allowed:
                    allowed_keys = ", ".join(sorted(allowed))
                    raise ValueError(f"Unknown {name} key {key!r}. Allowed keys: {allowed_keys}.")
                tol_value = float(value)
                if not np.isfinite(tol_value) or tol_value < 0.0:
                    raise ValueError(f"{name}.{key} must be finite and >= 0.")
                merged[key] = tol_value
        return merged

    def _build_target_vector(self) -> tuple[FloatArray, tuple[_ObjectiveBlock, ...]]:
        values: list[float] = []
        blocks: list[_ObjectiveBlock] = []
        start = 0

        if self.coils.target_flux_points is not None and self.coils.target_flux_values is not None:
            target_flux = np.asarray(self.coils.target_flux_values, dtype=np.float64).reshape(-1)
            values.extend(float(v) for v in target_flux)
            stop = start + target_flux.size
            blocks.append(_ObjectiveBlock("shape_flux", start, stop))
            start = stop

        if self.coils.x_point_target is not None:
            x_target = np.asarray(self.coils.x_point_target, dtype=np.float64).reshape(2)
            values.extend((float(x_target[0]), float(x_target[1])))
            stop = start + 2
            blocks.append(_ObjectiveBlock("x_point_position", start, stop))
            start = stop
            if self.coils.x_point_flux_target is not None:
                values.append(float(self.coils.x_point_flux_target))
                stop = start + 1
                blocks.append(_ObjectiveBlock("x_point_flux", start, stop))
                start = stop
        elif self.coils.x_point_flux_target is not None:
            raise ValueError("x_point_flux_target requires x_point_target for free-boundary tracking.")

        if self.coils.divertor_strike_points is not None and self.coils.divertor_flux_values is not None:
            divertor_flux = np.asarray(self.coils.divertor_flux_values, dtype=np.float64).reshape(-1)
            values.extend(float(v) for v in divertor_flux)
            stop = start + divertor_flux.size
            blocks.append(_ObjectiveBlock("divertor_flux", start, stop))
            start = stop

        return np.asarray(values, dtype=np.float64), tuple(blocks)

    def _sync_config_currents(self) -> None:
        coils_cfg = self.kernel.cfg.setdefault("coils", [])
        while len(coils_cfg) < self.n_coils:
            coils_cfg.append({})
        for idx in range(self.n_coils):
            coils_cfg[idx]["current"] = float(self.coils.currents[idx])

    def _solve_free_boundary_state(self) -> dict[str, Any]:
        self._sync_config_currents()
        solve = getattr(self.kernel, "solve", None)
        if callable(solve):
            return cast(
                dict[str, Any],
                solve(
                    boundary_variant="free_boundary",
                    coils=self.coils,
                    max_outer_iter=self.solve_max_outer_iter,
                    tol=self.solve_tol,
                    optimize_shape=False,
                ),
            )

        solve_free_boundary = getattr(self.kernel, "solve_free_boundary", None)
        if callable(solve_free_boundary):
            return cast(
                dict[str, Any],
                solve_free_boundary(
                    self.coils,
                    max_outer_iter=self.solve_max_outer_iter,
                    tol=self.solve_tol,
                    optimize_shape=False,
                ),
            )

        raise AttributeError("kernel must define solve() or solve_free_boundary() for free-boundary tracking.")

    def _observe_objectives(self) -> FloatArray:
        observed: list[float] = []
        for block in self.objective_blocks:
            if block.name == "shape_flux":
                if self.coils.target_flux_points is None:
                    raise ValueError("shape-flux observation requires target_flux_points.")
                flux = np.asarray(
                    self.kernel._sample_flux_at_points(self.coils.target_flux_points), dtype=np.float64
                ).reshape(-1)
                observed.extend(float(v) for v in flux)
            elif block.name == "x_point_position":
                x_pos, _ = self.kernel.find_x_point(self.kernel.Psi)
                observed.extend((float(x_pos[0]), float(x_pos[1])))
            elif block.name == "x_point_flux":
                if self.coils.x_point_target is None:
                    raise ValueError("x_point_flux observation requires x_point_target.")
                x_target = np.asarray(self.coils.x_point_target, dtype=np.float64).reshape(2)
                observed.append(float(self.kernel._interp_psi(float(x_target[0]), float(x_target[1]))))
            elif block.name == "divertor_flux":
                if self.coils.divertor_strike_points is None:
                    raise ValueError("divertor-flux observation requires divertor_strike_points.")
                flux = np.asarray(
                    self.kernel._sample_flux_at_points(self.coils.divertor_strike_points),
                    dtype=np.float64,
                ).reshape(-1)
                observed.extend(float(v) for v in flux)
            else:
                raise ValueError(f"Unknown objective block {block.name!r}.")
        return np.asarray(observed, dtype=np.float64)

    def identify_response_matrix(self, perturbation: float | None = None) -> FloatArray:
        p = self.identification_perturbation if perturbation is None else float(perturbation)
        if not np.isfinite(p) or p <= 0.0:
            raise ValueError("perturbation must be finite and > 0.")

        self._solve_free_boundary_state()
        original_currents = self.coils.currents.copy()
        for idx in range(self.n_coils):
            hi = float(self.coil_current_limits[idx])
            plus = original_currents.copy()
            minus = original_currents.copy()
            plus[idx] = float(np.clip(original_currents[idx] + p, -hi, hi))
            minus[idx] = float(np.clip(original_currents[idx] - p, -hi, hi))
            denom = plus[idx] - minus[idx]
            if abs(denom) < 1e-12:
                self.response_matrix[:, idx] = 0.0
                continue

            self.coils.currents = plus
            self._solve_free_boundary_state()
            obs_plus = self._observe_objectives()

            self.coils.currents = minus
            self._solve_free_boundary_state()
            obs_minus = self._observe_objectives()

            self.response_matrix[:, idx] = (obs_plus - obs_minus) / denom

        self.coils.currents = original_currents
        self._solve_free_boundary_state()
        return self.response_matrix.copy()

    def compute_correction(self, observation: FloatArray) -> FloatArray:
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        if obs.shape != self.target_vector.shape:
            raise ValueError("observation must match the free-boundary target vector shape.")
        error = self.target_vector - obs
        aug_matrix = np.vstack(
            [
                self.response_matrix,
                np.sqrt(self.response_regularization) * np.eye(self.n_coils, dtype=np.float64),
            ]
        )
        aug_rhs = np.concatenate([error, np.zeros(self.n_coils, dtype=np.float64)])
        delta, *_ = np.linalg.lstsq(aug_matrix, aug_rhs, rcond=None)
        clipped = np.clip(np.asarray(delta, dtype=np.float64), -self.correction_limit, self.correction_limit)
        return cast(FloatArray, np.asarray(clipped, dtype=np.float64))

    def _apply_correction(self, delta_currents: FloatArray, gain: float) -> FloatArray:
        g = float(gain)
        if not np.isfinite(g) or g <= 0.0:
            raise ValueError("gain must be finite and > 0.")
        updated = np.asarray(self.coils.currents.copy(), dtype=np.float64)
        for idx in range(self.n_coils):
            limit = float(self.coil_current_limits[idx])
            updated[idx] = float(np.clip(updated[idx] + g * float(delta_currents[idx]), -limit, limit))
        self.coils.currents = updated
        return cast(FloatArray, np.asarray(updated.copy(), dtype=np.float64))

    def evaluate_objectives(self, observation: np.ndarray) -> dict[str, Any]:
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        error = self.target_vector - obs
        metrics: dict[str, Any] = {
            "tracking_error_norm": float(np.linalg.norm(error)),
            "shape_rms": None,
            "shape_max_abs": None,
            "x_point_position_error": None,
            "x_point_flux_error": None,
            "divertor_rms": None,
            "divertor_max_abs": None,
        }

        for block in self.objective_blocks:
            block_error = error[block.start : block.stop]
            if block.name == "shape_flux":
                metrics["shape_rms"] = float(np.sqrt(np.mean(block_error**2)))
                metrics["shape_max_abs"] = float(np.max(np.abs(block_error)))
            elif block.name == "x_point_position":
                metrics["x_point_position_error"] = float(np.linalg.norm(block_error))
            elif block.name == "x_point_flux":
                metrics["x_point_flux_error"] = float(abs(block_error[0]))
            elif block.name == "divertor_flux":
                metrics["divertor_rms"] = float(np.sqrt(np.mean(block_error**2)))
                metrics["divertor_max_abs"] = float(np.max(np.abs(block_error)))

        checks: dict[str, bool] = {}
        if "shape_rms" in self.objective_tolerances and metrics["shape_rms"] is not None:
            checks["shape_rms"] = bool(metrics["shape_rms"] <= self.objective_tolerances["shape_rms"])
        if "shape_max_abs" in self.objective_tolerances and metrics["shape_max_abs"] is not None:
            checks["shape_max_abs"] = bool(metrics["shape_max_abs"] <= self.objective_tolerances["shape_max_abs"])
        if "x_point_position" in self.objective_tolerances and metrics["x_point_position_error"] is not None:
            checks["x_point_position"] = bool(
                metrics["x_point_position_error"] <= self.objective_tolerances["x_point_position"]
            )
        if "x_point_flux" in self.objective_tolerances and metrics["x_point_flux_error"] is not None:
            checks["x_point_flux"] = bool(metrics["x_point_flux_error"] <= self.objective_tolerances["x_point_flux"])
        if "divertor_rms" in self.objective_tolerances and metrics["divertor_rms"] is not None:
            checks["divertor_rms"] = bool(metrics["divertor_rms"] <= self.objective_tolerances["divertor_rms"])
        if "divertor_max_abs" in self.objective_tolerances and metrics["divertor_max_abs"] is not None:
            checks["divertor_max_abs"] = bool(
                metrics["divertor_max_abs"] <= self.objective_tolerances["divertor_max_abs"]
            )
        metrics["objective_checks"] = checks
        metrics["objective_convergence_active"] = bool(checks)
        metrics["objective_converged"] = all(checks.values()) if checks else True
        metrics["objective_tolerances"] = self.objective_tolerances.copy()
        return metrics

    def run_tracking_shot(
        self,
        *,
        shot_steps: int = 10,
        gain: float = 1.0,
        disturbance_callback: Callable[[Any, CoilSet, int], None] | None = None,
        stop_on_convergence: bool = False,
    ) -> dict[str, Any]:
        steps = int(shot_steps)
        if steps < 1:
            raise ValueError("shot_steps must be >= 1.")
        gain_value = float(gain)
        if not np.isfinite(gain_value) or gain_value <= 0.0:
            raise ValueError("gain must be finite and > 0.")
        start_time = time.time()
        self.history = {key: [] for key in self.history}
        self._solve_free_boundary_state()
        last_metrics = self.evaluate_objectives(self._observe_objectives())

        for step in range(steps):
            if disturbance_callback is not None:
                disturbance_callback(self.kernel, self.coils, step)

            self._solve_free_boundary_state()
            if step % self.response_refresh_steps == 0:
                self.identify_response_matrix()

            observation_before = self._observe_objectives()
            metrics_before = self.evaluate_objectives(observation_before)
            delta_currents = self.compute_correction(observation_before)
            baseline_currents = self.coils.currents.copy()
            accepted_gain = 0.0
            last_metrics = metrics_before

            trial_gain = gain_value
            for _ in range(6):
                self.coils.currents = baseline_currents.copy()
                self._apply_correction(delta_currents, gain=trial_gain)
                self._solve_free_boundary_state()
                observation_after = self._observe_objectives()
                metrics_after = self.evaluate_objectives(observation_after)
                improved = metrics_after["tracking_error_norm"] <= metrics_before["tracking_error_norm"] + 1e-12
                newly_converged = (not metrics_before["objective_converged"]) and metrics_after["objective_converged"]
                if improved or newly_converged:
                    accepted_gain = trial_gain
                    last_metrics = metrics_after
                    break
                trial_gain *= 0.5

            if accepted_gain == 0.0:
                self.coils.currents = baseline_currents.copy()
                self._solve_free_boundary_state()
                last_metrics = metrics_before

            max_abs_delta = float(np.max(np.abs(delta_currents))) if delta_currents.size > 0 else 0.0
            max_abs_coil = float(np.max(np.abs(self.coils.currents))) if self.coils.currents.size > 0 else 0.0
            self.history["t"].append(int(step))
            self.history["tracking_error_norm"].append(last_metrics["tracking_error_norm"])
            self.history["shape_rms"].append(last_metrics["shape_rms"])
            self.history["shape_max_abs"].append(last_metrics["shape_max_abs"])
            self.history["x_point_position_error"].append(last_metrics["x_point_position_error"])
            self.history["x_point_flux_error"].append(last_metrics["x_point_flux_error"])
            self.history["divertor_rms"].append(last_metrics["divertor_rms"])
            self.history["divertor_max_abs"].append(last_metrics["divertor_max_abs"])
            self.history["max_abs_delta_i"].append(max_abs_delta)
            self.history["max_abs_coil_current"].append(max_abs_coil)
            self.history["accepted_gain"].append(float(accepted_gain))
            self.history["objective_converged"].append(bool(last_metrics["objective_converged"]))

            self._log(
                f"Step {step}: err={last_metrics['tracking_error_norm']:.4e} | "
                f"max dI={max_abs_delta:.3e} | gain={accepted_gain:.3e} | "
                f"converged={last_metrics['objective_converged']}"
            )
            if stop_on_convergence and last_metrics["objective_converged"]:
                break

        runtime_seconds = time.time() - start_time
        error_arr = np.asarray(self.history["tracking_error_norm"], dtype=np.float64)
        delta_arr = np.asarray(self.history["max_abs_delta_i"], dtype=np.float64)
        coil_arr = np.asarray(self.history["max_abs_coil_current"], dtype=np.float64)
        gain_arr = np.asarray(self.history["accepted_gain"], dtype=np.float64)
        return {
            "steps": int(len(self.history["t"])),
            "runtime_seconds": float(runtime_seconds),
            "boundary_variant": "free_boundary",
            "final_tracking_error_norm": float(error_arr[-1]) if error_arr.size else 0.0,
            "mean_tracking_error_norm": float(np.mean(error_arr)) if error_arr.size else 0.0,
            "max_abs_delta_i": float(np.max(delta_arr)) if delta_arr.size else 0.0,
            "max_abs_coil_current": float(np.max(coil_arr)) if coil_arr.size else 0.0,
            "mean_accepted_gain": float(np.mean(gain_arr)) if gain_arr.size else 0.0,
            "min_accepted_gain": float(np.min(gain_arr)) if gain_arr.size else 0.0,
            "objective_tolerances": last_metrics["objective_tolerances"],
            "objective_checks": last_metrics["objective_checks"],
            "objective_convergence_active": last_metrics["objective_convergence_active"],
            "objective_converged": last_metrics["objective_converged"],
            "shape_rms": last_metrics["shape_rms"],
            "shape_max_abs": last_metrics["shape_max_abs"],
            "x_point_position_error": last_metrics["x_point_position_error"],
            "x_point_flux_error": last_metrics["x_point_flux_error"],
            "divertor_rms": last_metrics["divertor_rms"],
            "divertor_max_abs": last_metrics["divertor_max_abs"],
        }


def run_free_boundary_tracking(
    config_file: str | None = None,
    *,
    shot_steps: int = 10,
    gain: float = 1.0,
    verbose: bool = True,
    kernel_factory: Callable[[str], Any] = FusionKernel,
    objective_tolerances: dict[str, float] | None = None,
    disturbance_callback: Callable[[Any, CoilSet, int], None] | None = None,
    stop_on_convergence: bool = False,
) -> dict[str, Any]:
    """Run deterministic free-boundary tracking over the configured objectives.

    Examples
    --------
    >>> summary = run_free_boundary_tracking("iter_config.json", shot_steps=2, gain=0.7, verbose=False)
    >>> bool(summary["objective_convergence_active"])
    True
    """
    if config_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        config_file = str(repo_root / "iter_config.json")

    controller = FreeBoundaryTrackingController(
        str(config_file),
        kernel_factory=kernel_factory,
        verbose=verbose,
        objective_tolerances=objective_tolerances,
    )
    summary = controller.run_tracking_shot(
        shot_steps=shot_steps,
        gain=gain,
        disturbance_callback=disturbance_callback,
        stop_on_convergence=stop_on_convergence,
    )
    summary["config_path"] = str(config_file)
    return summary


__all__ = ["FreeBoundaryTrackingController", "run_free_boundary_tracking"]
