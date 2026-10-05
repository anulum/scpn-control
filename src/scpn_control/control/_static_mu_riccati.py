# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static Mu Riccati

"""Static mu riccati for bounded structured-uncertainty analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, solve_continuous_are

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.control._static_mu_bounds import _compute_static_mu_upper_bound_and_scalings
from scpn_control.control._static_mu_structure import StructuredUncertainty, _finite_scalar


def _validate_state_space(
    plant_ss: tuple[AnyFloatArray, AnyFloatArray, AnyFloatArray, AnyFloatArray],
    uncertainty: StructuredUncertainty,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    A, B, C, D_mat = (np.atleast_2d(np.asarray(mat, dtype=float)) for mat in plant_ss)

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    n = A.shape[0]
    if B.shape[0] != n:
        raise ValueError("B row count must match A.")
    if C.shape[1] != n:
        raise ValueError("C column count must match A.")
    if D_mat.shape != (C.shape[0], B.shape[1]):
        raise ValueError("D must have shape (C rows, B columns).")
    for name, mat in [("A", A), ("B", B), ("C", C), ("D", D_mat)]:
        if not np.all(np.isfinite(mat)):
            raise ValueError(f"{name} must contain only finite values.")

    uncertainty_size = uncertainty.total_size()
    if uncertainty_size != B.shape[1] or uncertainty_size != C.shape[0]:
        raise ValueError("static mu analysis requires uncertainty size to match B columns and C rows.")

    return A, B, C, D_mat


def _stable_open_loop_state_feedback(A: AnyFloatArray, B: AnyFloatArray, exc: Exception) -> FloatArray:
    """Return zero feedback for already-stable plants when SciPy CARE is unavailable."""
    try:
        spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(A))))
    except np.linalg.LinAlgError as eig_exc:
        raise RuntimeError("Riccati design failed; plant stability could not be checked.") from eig_exc
    if spectral_abscissa < 0.0:
        return np.zeros((B.shape[1], A.shape[0]), dtype=float)
    raise RuntimeError("Riccati design failed; plant is not stabilisable in this bounded domain.") from exc


def _riccati_state_feedback(A: AnyFloatArray, B: AnyFloatArray, C: AnyFloatArray) -> FloatArray:
    """Return the continuous-time CARE state-feedback gain."""
    q_weight = C.T @ C + np.eye(A.shape[0]) * 1e-9
    r_weight = np.eye(B.shape[1])
    try:
        P = solve_continuous_are(A, B, q_weight, r_weight)
    except (LinAlgError, TypeError, ValueError) as exc:
        return _stable_open_loop_state_feedback(A, B, exc)
    K = B.T @ P
    if not np.all(np.isfinite(K)):
        raise RuntimeError("Riccati design produced a non-finite controller gain.")
    return np.asarray(K, dtype=float)


def _closed_loop_dc_uncertainty_map(
    A: AnyFloatArray,
    B: AnyFloatArray,
    C: AnyFloatArray,
    D_mat: AnyFloatArray,
    K: AnyFloatArray,
) -> FloatArray:
    """Return C (0I - A_cl)^-1 B + D for the static robust-performance channel."""
    A_cl = A - B @ K
    try:
        state_response = np.linalg.solve(-A_cl, B)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Closed-loop DC map is singular; robust mu evidence is unavailable.") from exc
    M = C @ state_response + D_mat
    return np.asarray(M, dtype=complex)


@dataclass(frozen=True)
class StaticMuAnalysisResult:
    """Result of one Riccati feedback design and static DC mu analysis.

    ``mu_upper_bound`` is evaluated only at ``analysis_frequency_rad_s`` for
    the bound-scaled closed-loop map. It is not a frequency peak and does not
    constitute D-K synthesis or a robust-stability certificate over frequency.
    """

    controller_gain: FloatArray
    mu_upper_bound: float
    d_scalings: FloatArray
    analysis_frequency_rad_s: float
    closed_loop_spectral_abscissa: float


def design_riccati_state_feedback_with_static_mu_analysis(
    plant_ss: tuple[AnyFloatArray, AnyFloatArray, AnyFloatArray, AnyFloatArray],
    uncertainty: StructuredUncertainty,
) -> StaticMuAnalysisResult:
    """Design CARE state feedback and evaluate its static DC mu upper bound.

    The reduced plant uses ``B`` both as the state-feedback input channel and as
    the disturbance channel of the static transfer map; ``C`` is both the CARE
    state weighting source and the performance output. This is not the
    partitioned generalized plant required by H-infinity or D-K synthesis.

    Parameters
    ----------
    plant_ss : (A, B, C, D)
        Reduced state-space matrices with matched uncertainty input/output
        dimensions.
    uncertainty : StructuredUncertainty
        Block structure of Δ.

    Returns
    -------
    StaticMuAnalysisResult
        Controller gain, single-frequency upper bound, fitted static scalings,
        zero analysis frequency, and closed-loop spectral abscissa.
    """
    A, B, C, D_mat = _validate_state_space(plant_ss, uncertainty)

    controller_gain = _riccati_state_feedback(A, B, C)
    closed_loop = A - B @ controller_gain
    spectral_abscissa = float(np.max(np.real(np.linalg.eigvals(closed_loop))))
    if not np.isfinite(spectral_abscissa) or spectral_abscissa >= 0.0:
        raise RuntimeError("CARE state feedback did not produce a finite stable closed loop.")
    closed_loop_map = _closed_loop_dc_uncertainty_map(A, B, C, D_mat, controller_gain) @ uncertainty.bound_matrix()
    mu_upper_bound, d_scalings = _compute_static_mu_upper_bound_and_scalings(
        closed_loop_map,
        uncertainty.build_delta_structure(),
    )
    return StaticMuAnalysisResult(
        controller_gain=np.asarray(controller_gain, dtype=float).copy(),
        mu_upper_bound=float(mu_upper_bound),
        d_scalings=np.asarray(d_scalings, dtype=float).copy(),
        analysis_frequency_rad_s=0.0,
        closed_loop_spectral_abscissa=spectral_abscissa,
    )


class RiccatiStateFeedbackController:
    """CARE state-feedback controller with bounded static DC mu analysis.

    The controller gain is designed from ``A``, ``B``, and ``C`` by a continuous
    algebraic Riccati equation. Structured uncertainty affects only the attached
    single-frequency analysis; it does not participate in controller synthesis.

    Physical uncertainty model for tokamak control follows Ariola & Pironti
    2008, Ch. 7:
        - plasma_position  real_scalar  ±2 cm
        - plasma_current   real_scalar  ±3 %
        - plasma_shape     full         ±5 %
    """

    def __init__(
        self,
        plant_ss: tuple[AnyFloatArray, AnyFloatArray, AnyFloatArray, AnyFloatArray],
        uncertainty: StructuredUncertainty,
    ) -> None:
        self.plant_ss = plant_ss
        self.uncertainty = uncertainty
        self.analysis_result: StaticMuAnalysisResult | None = None

    def design(self) -> StaticMuAnalysisResult:
        """Design state feedback, run static analysis, and store the result."""
        result = design_riccati_state_feedback_with_static_mu_analysis(
            self.plant_ss,
            self.uncertainty,
        )
        self.analysis_result = result
        return result

    def step(self, x: AnyFloatArray, dt: float) -> FloatArray:
        """Apply the designed state-feedback law ``u = -K x``."""
        if self.analysis_result is None:
            raise RuntimeError("Controller not designed yet")
        _finite_scalar("dt", dt, positive=True)
        gain = self.analysis_result.controller_gain
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape != (gain.shape[1],):
            raise ValueError(f"x must have shape ({gain.shape[1]},)")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("x must contain only finite values")
        return np.asarray(-gain @ x_arr)

    def inverse_static_mu_upper_bound(self) -> float:
        """Return the reciprocal of the single-frequency upper bound.

        This diagnostic is relative to the declared bound-scaled uncertainty map
        at zero frequency. It is not a robust-stability margin over frequency.
        """
        if self.analysis_result is None:
            raise RuntimeError("Controller not designed yet")
        if self.analysis_result.mu_upper_bound <= 0.0:
            return float("inf")
        return 1.0 / self.analysis_result.mu_upper_bound
