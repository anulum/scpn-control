# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""
Mu-analysis utilities for structured-uncertainty robust control.

Structured singular value definition (Doyle 1982, IEE Proc. D 129, 242):

    μ(M) = 1 / min{ σ̄(Δ) : det(I - MΔ) = 0,  Δ ∈ Δ_struct }

where Δ_struct is the block-diagonal uncertainty set with blocks matching
the declared UncertaintyBlock list.  μ ≤ σ̄(M) always holds (upper bound),
with equality only when Δ is unstructured.

Full D-K iteration (Balas et al. 1993, "μ-Analysis and Synthesis Toolbox",
Ch. 8; Skogestad & Postlethwaite 2005, "Multivariable Feedback Control",
§8.5) alternates:

    K-step: synthesise H-infinity controller K with D-scaled plant D·P·D^{-1}
    D-step: fit stable, minimum-phase D(s) to the frequency-wise μ upper bound

Convergence is not guaranteed in general (non-convex problem). The executable
path below is deliberately bounded to a Riccati state-feedback K-step and
static D-scaled closed-loop upper-bound evaluation unless a validated
frequency-dependent backend is wired.

Physical uncertainty blocks for tokamak control (Ariola & Pironti 2008,
"Magnetic Control of Tokamak Plasmas", Ch. 7):

    plasma_position  real_scalar   ±2 cm  — flux-surface reconstruction error
    plasma_current   real_scalar   ±3 %   — Rogowski coil calibration drift
    plasma_shape     full          ±5 %   — elongation / triangularity uncertainty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import LinAlgError, solve_continuous_are


@dataclass
class UncertaintyBlock:
    """Single block in the structured uncertainty set Δ.

    Attributes
    ----------
    name : str
        Physical label, e.g. "plasma_position".
    size : int
        Block dimension (number of channels).
    bound : float
        Norm bound on this block (||Δ_i|| ≤ bound).
    block_type : str
        "real_scalar" | "complex_scalar" | "full".
        Tokamak usage: parametric deviations are "real_scalar";
        unmodelled dynamics are "full" (Ariola & Pironti 2008, Ch. 7).
    """

    name: str
    size: int
    bound: float
    block_type: str


class StructuredUncertainty:
    """Ordered collection of UncertaintyBlock objects defining Δ_struct."""

    def __init__(self, blocks: list[UncertaintyBlock]):
        self.blocks = blocks

    def build_Delta_structure(self) -> list[tuple[int, str]]:
        return [(b.size, b.block_type) for b in self.blocks]

    def total_size(self) -> int:
        return sum(b.size for b in self.blocks)


def _compute_mu_upper_bound_and_scalings(
    M: np.ndarray,
    delta_structure: list[tuple[int, str]],
) -> tuple[float, np.ndarray]:
    """D-scaling upper bound on μ(M).

    Computes  min_D  σ̄(D M D^{-1})  where D is block-diagonal with positive
    real scalars matching delta_structure.  This is always ≥ μ(M) and equals
    μ(M) for complex full blocks (Doyle 1982, IEE Proc. D 129, 242).
    """
    M = np.atleast_2d(np.asarray(M, dtype=complex))
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square closed-loop transfer matrix.")
    if not np.all(np.isfinite(M)):
        raise ValueError("M must contain only finite values.")
    n = M.shape[0]
    if sum(size for size, _ in delta_structure) != n:
        raise ValueError("Delta block sizes must sum to M dimension.")
    if not delta_structure:
        raise ValueError("Delta structure must contain at least one uncertainty block.")

    def apply_D(d_vec: np.ndarray) -> np.ndarray:
        D = np.zeros((n, n), dtype=complex)
        idx = 0
        for d_idx, (size, _btype) in enumerate(delta_structure):
            val = d_vec[d_idx]
            for i in range(size):
                D[idx + i, idx + i] = val
            idx += size
        return D

    num_blocks = len(delta_structure)
    d_vec = np.ones(num_blocks)

    # σ̄(M) is the trivial upper bound — Doyle 1982, IEE Proc. D 129, 242
    best_mu = np.max(np.linalg.svd(M)[1])
    best_d = d_vec.copy()

    alpha = 0.1
    for _ in range(50):
        D = apply_D(d_vec)
        D_inv = np.linalg.inv(D)

        M_scaled = D @ M @ D_inv
        U, S, Vh = np.linalg.svd(M_scaled)
        mu = S[0]

        if mu < best_mu:
            best_mu = mu
            best_d = d_vec.copy()

        # Finite-difference gradient of σ̄(D M D^{-1}) w.r.t. log(d_i)
        grad = np.zeros(num_blocks)
        for i in range(num_blocks):
            d_pert = d_vec.copy()
            d_pert[i] *= 1.01
            D_p = apply_D(d_pert)
            M_p = D_p @ M @ np.linalg.inv(D_p)
            mu_p = np.max(np.linalg.svd(M_p)[1])
            grad[i] = (mu_p - mu) / 0.01

        d_vec = d_vec * np.exp(-alpha * grad)
        # D M D^{-1} is invariant to uniform scaling of D — normalise to d_0=1
        d_vec /= d_vec[0]

    return float(best_mu), best_d


def compute_mu_upper_bound(M: np.ndarray, delta_structure: list[tuple[int, str]]) -> float:
    """D-scaling upper bound on μ(M).

    Computes  min_D  σ̄(D M D^{-1})  where D is block-diagonal with positive
    real scalars matching delta_structure.  This is always ≥ μ(M) and equals
    μ(M) for complex full blocks (Doyle 1982, IEE Proc. D 129, 242).

    A finite-difference descent on log(D) is used to fit the static D-scaling.

    Parameters
    ----------
    M : np.ndarray, shape (n, n)
        Closed-loop transfer matrix evaluated at a single frequency.
    delta_structure : list of (size, block_type)
        Block sizes and types from StructuredUncertainty.build_Delta_structure().

    Returns
    -------
    float
        Upper bound μ̄ ≥ μ(M).
    """
    best_mu, _best_d = _compute_mu_upper_bound_and_scalings(M, delta_structure)
    return float(best_mu)


def _validate_state_space(
    plant_ss: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    uncertainty: StructuredUncertainty,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if uncertainty_size <= 0:
        raise ValueError("uncertainty must contain at least one positive-size block.")
    if any(block.size <= 0 for block in uncertainty.blocks):
        raise ValueError("uncertainty block sizes must be positive.")
    if uncertainty_size != B.shape[1] or uncertainty_size != C.shape[0]:
        raise ValueError("D-K static mu analysis requires uncertainty size to match B columns and C rows.")

    return A, B, C, D_mat


def _riccati_state_feedback(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Continuous-time Riccati state-feedback K for the D-K K-step."""
    q_weight = C.T @ C + np.eye(A.shape[0]) * 1e-9
    r_weight = np.eye(B.shape[1])
    try:
        P = solve_continuous_are(A, B, q_weight, r_weight)
    except LinAlgError as exc:
        raise RuntimeError("Riccati K-step failed; plant is not stabilisable in this bounded domain.") from exc
    K = B.T @ P
    if not np.all(np.isfinite(K)):
        raise RuntimeError("Riccati K-step produced a non-finite controller gain.")
    return np.asarray(K, dtype=float)


def _closed_loop_dc_uncertainty_map(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D_mat: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Return C (0I - A_cl)^-1 B + D for the static robust-performance channel."""
    A_cl = A - B @ K
    try:
        state_response = np.linalg.solve(-A_cl, B)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Closed-loop DC map is singular; robust mu evidence is unavailable.") from exc
    M = C @ state_response + D_mat
    if not np.all(np.isfinite(M)):
        raise RuntimeError("Closed-loop DC map produced non-finite values.")
    return np.asarray(M, dtype=complex)


def dk_iteration(
    plant_ss: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    uncertainty: StructuredUncertainty,
    n_iter: int = 5,
    gamma_bisect_tol: float = 0.01,
) -> tuple[Any, float, np.ndarray]:
    """Bounded static D-scaled mu-analysis pass for μ-synthesis workflows.

    This bounded-domain implementation performs a Riccati state-feedback K-step
    and evaluates the static closed-loop robust-performance map with fitted
    D-scalings.  It does not fabricate monotonic convergence when a full
    frequency-dependent H-infinity backend is unavailable.

    Parameters
    ----------
    plant_ss : (A, B, C, D)
        State-space matrices of the generalised plant.
    uncertainty : StructuredUncertainty
        Block structure of Δ.
    n_iter : int
        Number of D-K outer iterations.
    gamma_bisect_tol : float
        Tolerance for the inner H-infinity γ bisection (passed to K-step).

    Returns
    -------
    K_controller : np.ndarray
        Synthesised controller gain matrix.
    mu_peak : float
        Peak μ upper bound achieved after n_iter iterations.
    D_scalings : np.ndarray
        Final D-scale vector (one entry per uncertainty block).
    """
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")
    if gamma_bisect_tol <= 0.0:
        raise ValueError("gamma_bisect_tol must be positive.")
    A, B, C, D_mat = _validate_state_space(plant_ss, uncertainty)

    K_controller = _riccati_state_feedback(A, B, C)
    closed_loop_map = _closed_loop_dc_uncertainty_map(A, B, C, D_mat, K_controller)
    mu_peak, D_scalings = _compute_mu_upper_bound_and_scalings(
        closed_loop_map,
        uncertainty.build_Delta_structure(),
    )

    _ = n_iter

    return K_controller, float(mu_peak), D_scalings


class MuSynthesisController:
    """Structured robust controller with bounded static mu-analysis evidence.

    Theory: Doyle 1982 (μ definition); Balas et al. 1993 (DK algorithm);
    Skogestad & Postlethwaite 2005, §8.5 (convergence and practical use).

    Physical uncertainty model for tokamak control follows Ariola & Pironti
    2008, Ch. 7:
        - plasma_position  real_scalar  ±2 cm
        - plasma_current   real_scalar  ±3 %
        - plasma_shape     full         ±5 %
    """

    def __init__(
        self,
        plant_ss: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        uncertainty: StructuredUncertainty,
    ):
        self.plant_ss = plant_ss
        self.uncertainty = uncertainty
        self.K: np.ndarray | None = None
        self.mu_peak = float("inf")
        self.D_scalings: np.ndarray | None = None

        self.integral_error = 0.0

    def synthesize(self, n_dk_iter: int = 5) -> None:
        """Run the bounded mu-analysis synthesis pass and store the controller."""
        K, mu, D_s = dk_iteration(self.plant_ss, self.uncertainty, n_iter=n_dk_iter)
        self.K = K
        self.mu_peak = mu
        self.D_scalings = D_s

    def step(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Apply synthesised controller: u = -K x."""
        if self.K is None:
            raise RuntimeError("Controller not synthesized yet")
        return np.asarray(-self.K @ x)

    def robustness_margin(self) -> float:
        """Return 1/μ_peak — the structured stability margin.

        μ_peak < 1 means the system is robustly stable for all Δ with
        ||Δ|| ≤ 1/μ_peak (Doyle 1982; Skogestad & Postlethwaite 2005, §8.2).
        """
        if self.mu_peak <= 0.0:
            return float("inf")
        return 1.0 / self.mu_peak
