# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary solve orchestration

"""Outer free-boundary solve loop composing equilibrium and coil control.

This leaf owns the experimental external-coil outer loop around the fixed-
boundary Grad-Shafranov solve. The CONTROL
:class:`~scpn_control.core.fusion_kernel.FusionKernel` product surface remains
first-class under dual-home C and keeps a thin wrapper that supplies mesh,
equilibrium, Green/vacuum, and already-extracted free-boundary control helpers.
Phase-sync steps and the Rust multigrid bridge stay on the owner (R0-S8/S9).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.core.gs_free_boundary_control import CoilSet

logger = logging.getLogger(__name__)


@runtime_checkable
class FreeBoundarySolveKernel(Protocol):
    """Minimal owner surface required by free-boundary solve orchestration."""

    Psi: FloatArray
    cfg: dict[str, Any]

    def _compute_external_flux(self, coils: CoilSet) -> FloatArray:
        raise NotImplementedError

    def solve_equilibrium(
        self,
        preserve_initial_state: bool = False,
        boundary_flux: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Run the inner fixed-boundary Grad-Shafranov solve."""
        raise NotImplementedError

    def _sample_flux_at_points(self, obs_points: FloatArray) -> FloatArray:
        raise NotImplementedError

    def _resolve_shape_target_flux(self, coils: CoilSet, current_flux: FloatArray) -> tuple[FloatArray, str]:
        raise NotImplementedError

    def _shape_error_metrics(self, current_flux: FloatArray, target_flux: FloatArray) -> dict[str, float]:
        raise NotImplementedError

    def _resolve_separatrix_flux_target(self, coils: CoilSet, shape_target_flux: FloatArray | None) -> float | None:
        raise NotImplementedError

    def _resolve_x_point_flux_target(
        self, coils: CoilSet, separatrix_flux_target: float | None
    ) -> tuple[float | None, str]:
        raise NotImplementedError

    def _resolve_divertor_flux_targets(
        self, coils: CoilSet, separatrix_flux_target: float | None
    ) -> tuple[FloatArray | None, str]:
        raise NotImplementedError

    def _interp_psi_gradient(self, R_pt: float, Z_pt: float) -> tuple[float, float]:
        raise NotImplementedError

    def find_x_point(self, Psi: FloatArray) -> tuple[tuple[float, float], float]:
        """Locate an X-point candidate on the current flux map."""
        raise NotImplementedError

    def _interp_psi(self, R_pt: float, Z_pt: float) -> float:
        raise NotImplementedError

    def _evaluate_free_boundary_objective_status(
        self,
        tolerances: dict[str, float],
        *,
        shape_error_rms: float | None,
        shape_error_max_abs: float | None,
        x_point_detected_error: float | None,
        x_point_gradient_norm: float | None,
        x_point_flux_error: float | None,
        divertor_error_rms: float | None,
        divertor_error_max_abs: float | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _resolve_free_boundary_objective_tolerances(
        self,
        cfg_objective_tolerances: Any,
        override_objective_tolerances: dict[str, float] | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError

    def optimize_coil_currents(
        self,
        coils: CoilSet,
        target_flux: FloatArray,
        tikhonov_alpha: float = 1e-4,
        *,
        x_point_flux_target: float | None = None,
        divertor_flux_targets: FloatArray | None = None,
    ) -> FloatArray:
        """Optimise coil currents against free-boundary objective targets."""
        raise NotImplementedError

    def _divertor_configuration_label(self, strike_points: FloatArray | None) -> str:
        raise NotImplementedError


def solve_free_boundary(
    kernel: FreeBoundarySolveKernel,
    coils: CoilSet,
    max_outer_iter: int = 20,
    tol: float = 1e-4,
    optimize_shape: bool = False,
    tikhonov_alpha: float = 1e-4,
    objective_tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Experimental external-coil outer loop around the fixed-boundary GS solve.

    Iterates between updating boundary flux from coils and solving the
    internal GS equation.  When ``optimize_shape=True`` and the coil set
    has ``target_flux_points``, an additional outer loop optimises the
    coil currents to match the desired plasma boundary shape.

    This helper should not be interpreted as a complete production-grade
    free-boundary solver.  The current project standard for closing the
    free-boundary roadmap item is higher: shape control, X-point geometry,
    and divertor-configuration support must all be demonstrated.

    Parameters
    ----------
    coils : CoilSet
        External coil set.
    max_outer_iter : int
        Maximum outer-loop iterations for the experimental coil-coupled path.
    tol : float
        Convergence tolerance on max |delta psi|.
    optimize_shape : bool
        When True, run coil-current optimisation at each outer step.
    tikhonov_alpha : float
        Tikhonov regularisation for coil optimisation.
    objective_tolerances : dict or None
        Optional convergence gates for free-boundary target objectives.
        Supported keys are ``shape_rms``, ``shape_max_abs``,
        ``x_point_position``, ``x_point_gradient``, ``x_point_flux``,
        ``divertor_rms``, and ``divertor_max_abs``. When omitted, the
        method falls back to ``free_boundary.objective_tolerances`` in the
        config, if present.

    Returns
    -------
    dict
        ``{"outer_iterations": int, "final_diff": float,
        "coil_currents": AnyFloatArray}``
    """
    psi_ext = kernel._compute_external_flux(coils)
    shape_error_history: list[float] = []
    shape_error_max_history: list[float] = []
    x_point_detected_error_history: list[float] = []
    x_point_gradient_norm_history: list[float] = []
    divertor_error_history: list[float] = []
    divertor_error_max_history: list[float] = []
    shape_objective_mode = "disabled"
    x_point_objective_mode = "disabled"
    divertor_objective_mode = "disabled"
    target_flux_used: FloatArray | None = None
    x_point_flux_target_used: float | None = None
    divertor_flux_target_used: FloatArray | None = None
    objective_tolerances_resolved = kernel._resolve_free_boundary_objective_tolerances(
        kernel.cfg.get("free_boundary", {}).get("objective_tolerances"),
        objective_tolerances,
    )
    objective_status = kernel._evaluate_free_boundary_objective_status(
        objective_tolerances_resolved,
        shape_error_rms=None,
        shape_error_max_abs=None,
        x_point_detected_error=None,
        x_point_gradient_norm=None,
        x_point_flux_error=None,
        divertor_error_rms=None,
        divertor_error_max_abs=None,
    )
    currents_updated = False
    objective_enabled = optimize_shape and (
        coils.target_flux_points is not None
        or (coils.x_point_target is not None and (coils.x_point_weight > 0.0 or coils.x_point_null_weight > 0.0))
        or (coils.divertor_strike_points is not None and coils.divertor_weight > 0.0)
    )
    converged = False
    diff = float("inf")

    for outer in range(max_outer_iter):
        # Apply external flux as boundary condition
        kernel.Psi[0, :] = psi_ext[0, :]
        kernel.Psi[-1, :] = psi_ext[-1, :]
        kernel.Psi[:, 0] = psi_ext[:, 0]
        kernel.Psi[:, -1] = psi_ext[:, -1]

        # Inner GS solve (use existing Picard iteration)
        psi_old = kernel.Psi.copy()
        kernel.solve_equilibrium(
            preserve_initial_state=True,
            boundary_flux=psi_ext,
        )

        # Check equilibrium change on the actual solved state.
        diff = float(np.max(np.abs(kernel.Psi - psi_old)))

        # Optional: optimise coil currents to match target shape / X-point / divertor constraints.
        if objective_enabled:
            target_psi: FloatArray | None = None
            shape_error_rms_current: float | None = None
            shape_error_max_abs_current: float | None = None
            if coils.target_flux_points is not None:
                current_flux = kernel._sample_flux_at_points(coils.target_flux_points)
                target_psi, shape_objective_mode = kernel._resolve_shape_target_flux(coils, current_flux)
                target_flux_used = target_psi.copy()
                metrics = kernel._shape_error_metrics(current_flux, target_psi)
                shape_error_rms_current = metrics["shape_error_rms"]
                shape_error_max_abs_current = metrics["shape_error_max_abs"]
                shape_error_history.append(shape_error_rms_current)
                shape_error_max_history.append(shape_error_max_abs_current)

            separatrix_flux_target = kernel._resolve_separatrix_flux_target(coils, target_psi)
            x_point_flux_target, x_point_objective_mode = kernel._resolve_x_point_flux_target(
                coils,
                separatrix_flux_target,
            )
            x_point_detected_error_current: float | None = None
            x_point_gradient_norm_current: float | None = None
            x_point_flux_error_current: float | None = None
            if x_point_flux_target is not None:
                x_point_flux_target_used = float(x_point_flux_target)

            divertor_flux_targets, divertor_objective_mode = kernel._resolve_divertor_flux_targets(
                coils,
                separatrix_flux_target,
            )
            divertor_error_rms_current: float | None = None
            divertor_error_max_abs_current: float | None = None
            if divertor_flux_targets is not None:
                divertor_flux_target_used = np.asarray(divertor_flux_targets, dtype=np.float64).copy()

            if coils.x_point_target is not None:
                x_target = np.asarray(coils.x_point_target, dtype=np.float64).reshape(2)
                dpsi_dR_target, dpsi_dZ_target = kernel._interp_psi_gradient(float(x_target[0]), float(x_target[1]))
                x_point_gradient_norm_current = float(np.hypot(dpsi_dR_target, dpsi_dZ_target))
                x_point_gradient_norm_history.append(x_point_gradient_norm_current)
                x_detected, _ = kernel.find_x_point(kernel.Psi)
                x_point_detected_error_current = float(
                    np.hypot(float(x_detected[0]) - float(x_target[0]), float(x_detected[1]) - float(x_target[1]))
                )
                x_point_detected_error_history.append(x_point_detected_error_current)
                if x_point_flux_target is not None:  # pragma: no branch - x_point set => target non-None; #129
                    x_point_flux_actual_current = float(kernel._interp_psi(float(x_target[0]), float(x_target[1])))
                    x_point_flux_error_current = float(x_point_flux_actual_current - x_point_flux_target)

            if coils.divertor_strike_points is not None and divertor_flux_targets is not None:
                current_divertor_flux = kernel._sample_flux_at_points(coils.divertor_strike_points)
                divertor_metrics = kernel._shape_error_metrics(current_divertor_flux, divertor_flux_targets)
                divertor_error_rms_current = divertor_metrics["shape_error_rms"]
                divertor_error_max_abs_current = divertor_metrics["shape_error_max_abs"]
                divertor_error_history.append(divertor_error_rms_current)
                divertor_error_max_history.append(divertor_error_max_abs_current)

            objective_status = kernel._evaluate_free_boundary_objective_status(
                objective_tolerances_resolved,
                shape_error_rms=shape_error_rms_current,
                shape_error_max_abs=shape_error_max_abs_current,
                x_point_detected_error=x_point_detected_error_current,
                x_point_gradient_norm=x_point_gradient_norm_current,
                x_point_flux_error=x_point_flux_error_current,
                divertor_error_rms=divertor_error_rms_current,
                divertor_error_max_abs=divertor_error_max_abs_current,
            )
        else:
            objective_status = kernel._evaluate_free_boundary_objective_status(
                objective_tolerances_resolved,
                shape_error_rms=None,
                shape_error_max_abs=None,
                x_point_detected_error=None,
                x_point_gradient_norm=None,
                x_point_flux_error=None,
                divertor_error_rms=None,
                divertor_error_max_abs=None,
            )

        if diff < tol and objective_status["objective_converged"]:
            logger.info("Free-boundary converged at outer iter %d (diff=%.2e)", outer, diff)
            converged = True
            break

        if objective_enabled:
            new_currents = kernel.optimize_coil_currents(
                coils,
                np.array([], dtype=np.float64) if target_psi is None else target_psi,
                tikhonov_alpha=tikhonov_alpha,
                x_point_flux_target=x_point_flux_target,
                divertor_flux_targets=divertor_flux_targets,
            )
            coils.currents = new_currents
            psi_ext = kernel._compute_external_flux(coils)
            currents_updated = True

    # Keep the returned Psi consistent with the final coil currents.
    if currents_updated:
        psi_before_final = kernel.Psi.copy()
        kernel.solve_equilibrium(
            preserve_initial_state=True,
            boundary_flux=psi_ext,
        )
        diff = float(np.max(np.abs(kernel.Psi - psi_before_final)))

    shape_error_final_rms: float | None = None
    shape_error_final_max_abs: float | None = None
    current_flux_final: FloatArray | None = None
    if objective_enabled and coils.target_flux_points is not None:
        current_flux_final = kernel._sample_flux_at_points(coils.target_flux_points)
        target_final, shape_objective_mode = kernel._resolve_shape_target_flux(coils, current_flux_final)
        target_flux_used = target_final.copy()
        metrics_final = kernel._shape_error_metrics(current_flux_final, target_final)
        shape_error_final_rms = metrics_final["shape_error_rms"]
        shape_error_final_max_abs = metrics_final["shape_error_max_abs"]
        if currents_updated or not shape_error_history:
            shape_error_history.append(shape_error_final_rms)
            shape_error_max_history.append(shape_error_final_max_abs)

    x_point_detected = None
    x_point_detected_error: float | None = None
    x_point_target_gradient_norm: float | None = None
    x_point_flux_actual: float | None = None
    x_point_flux_error: float | None = None
    if objective_enabled and coils.x_point_target is not None:
        x_target = np.asarray(coils.x_point_target, dtype=np.float64).reshape(2)
        x_detected, _ = kernel.find_x_point(kernel.Psi)
        x_point_detected = (float(x_detected[0]), float(x_detected[1]))
        x_point_detected_error = float(
            np.hypot(float(x_detected[0]) - float(x_target[0]), float(x_detected[1]) - float(x_target[1]))
        )
        dpsi_dR_target, dpsi_dZ_target = kernel._interp_psi_gradient(float(x_target[0]), float(x_target[1]))
        x_point_target_gradient_norm = float(np.hypot(dpsi_dR_target, dpsi_dZ_target))
        x_point_flux_actual = float(kernel._interp_psi(float(x_target[0]), float(x_target[1])))
        if x_point_flux_target_used is None:  # pragma: no cover - defensive free-boundary fallback path
            x_point_flux_target_used, x_point_objective_mode = kernel._resolve_x_point_flux_target(
                coils,
                kernel._resolve_separatrix_flux_target(coils, target_flux_used),
            )
        if x_point_flux_target_used is not None:  # pragma: no branch - always set when x_point set; #129
            x_point_flux_error = float(x_point_flux_actual - x_point_flux_target_used)
        if (currents_updated or not x_point_detected_error_history) and x_point_detected_error is not None:
            x_point_detected_error_history.append(x_point_detected_error)
        if (currents_updated or not x_point_gradient_norm_history) and x_point_target_gradient_norm is not None:
            x_point_gradient_norm_history.append(x_point_target_gradient_norm)

    divertor_flux_actual: FloatArray | None = None
    divertor_error_final_rms: float | None = None
    divertor_error_final_max_abs: float | None = None
    if objective_enabled and coils.divertor_strike_points is not None:
        divertor_flux_actual = kernel._sample_flux_at_points(coils.divertor_strike_points)
        if divertor_flux_target_used is None:  # pragma: no cover - defensive free-boundary fallback path
            divertor_flux_target_used, divertor_objective_mode = kernel._resolve_divertor_flux_targets(
                coils,
                kernel._resolve_separatrix_flux_target(coils, target_flux_used),
            )
        if divertor_flux_target_used is not None:  # pragma: no branch - always set when divertor set; #129
            divertor_metrics_final = kernel._shape_error_metrics(divertor_flux_actual, divertor_flux_target_used)
            divertor_error_final_rms = divertor_metrics_final["shape_error_rms"]
            divertor_error_final_max_abs = divertor_metrics_final["shape_error_max_abs"]
            if currents_updated or not divertor_error_history:
                divertor_error_history.append(divertor_error_final_rms)
                divertor_error_max_history.append(divertor_error_final_max_abs)

    objective_status = kernel._evaluate_free_boundary_objective_status(
        objective_tolerances_resolved,
        shape_error_rms=shape_error_final_rms,
        shape_error_max_abs=shape_error_final_max_abs,
        x_point_detected_error=x_point_detected_error,
        x_point_gradient_norm=x_point_target_gradient_norm,
        x_point_flux_error=x_point_flux_error,
        divertor_error_rms=divertor_error_final_rms,
        divertor_error_max_abs=divertor_error_final_max_abs,
    )
    equilibrium_converged = diff < tol
    converged = equilibrium_converged and objective_status["objective_converged"]

    return {
        "outer_iterations": outer + 1,
        "final_diff": diff,
        "converged": converged,
        "equilibrium_converged": equilibrium_converged,
        "coil_currents": coils.currents.copy(),
        "boundary_variant": "free_boundary",
        "shape_objective_mode": shape_objective_mode,
        "shape_error_history": shape_error_history,
        "shape_error_max_history": shape_error_max_history,
        "shape_error_final_rms": shape_error_final_rms,
        "shape_error_final_max_abs": shape_error_final_max_abs,
        "shape_target_flux": None if target_flux_used is None else target_flux_used.copy(),
        "shape_current_flux": None if current_flux_final is None else current_flux_final.copy(),
        "x_point_objective_mode": x_point_objective_mode,
        "x_point_target": None
        if coils.x_point_target is None
        else np.asarray(coils.x_point_target, dtype=np.float64).copy(),
        "x_point_detected": x_point_detected,
        "x_point_detected_error_history": x_point_detected_error_history,
        "x_point_detected_error": x_point_detected_error,
        "x_point_gradient_norm_history": x_point_gradient_norm_history,
        "x_point_target_gradient_norm": x_point_target_gradient_norm,
        "x_point_flux_target": x_point_flux_target_used,
        "x_point_flux_actual": x_point_flux_actual,
        "x_point_flux_error": x_point_flux_error,
        "divertor_objective_mode": divertor_objective_mode,
        "divertor_configuration": kernel._divertor_configuration_label(coils.divertor_strike_points),
        "divertor_strike_points": (
            None
            if coils.divertor_strike_points is None
            else np.asarray(coils.divertor_strike_points, dtype=np.float64).copy()
        ),
        "divertor_flux_target": None if divertor_flux_target_used is None else divertor_flux_target_used.copy(),
        "divertor_flux_actual": None if divertor_flux_actual is None else divertor_flux_actual.copy(),
        "divertor_error_history": divertor_error_history,
        "divertor_error_max_history": divertor_error_max_history,
        "divertor_error_final_rms": divertor_error_final_rms,
        "divertor_error_final_max_abs": divertor_error_final_max_abs,
        "objective_tolerances": objective_status["objective_tolerances"],
        "objective_checks": objective_status["objective_checks"],
        "objective_convergence_active": objective_status["objective_convergence_active"],
        "objective_converged": objective_status["objective_converged"],
    }


# Historical alias used by thin FusionKernel wrapper.
_solve_free_boundary = solve_free_boundary
