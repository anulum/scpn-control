# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static Mu Bounds

"""Static mu bounds for bounded structured-uncertainty analysis."""

from __future__ import annotations

import numpy as np

from scpn_control._typing import AnyComplexArray, AnyFloatArray, FloatArray
from scpn_control.control._static_mu_structure import _VALID_BLOCK_TYPES, _positive_int


def _compute_static_mu_upper_bound_and_scalings(
    M: AnyFloatArray | AnyComplexArray,
    delta_structure: list[tuple[int, str]],
) -> tuple[float, FloatArray]:
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
    for size, block_type in delta_structure:
        _positive_int("Delta block size", size)
        if block_type not in _VALID_BLOCK_TYPES:
            raise ValueError(f"Delta block_type must be one of {sorted(_VALID_BLOCK_TYPES)}")

    def apply_D(d_vec: AnyFloatArray) -> FloatArray:
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


def compute_static_mu_upper_bound(M: AnyFloatArray | AnyComplexArray, delta_structure: list[tuple[int, str]]) -> float:
    """D-scaling upper bound on μ(M).

    Computes  min_D  σ̄(D M D^{-1})  where D is block-diagonal with positive
    real scalars matching delta_structure.  This is always ≥ μ(M) and equals
    μ(M) for complex full blocks (Doyle 1982, IEE Proc. D 129, 242).

    A finite-difference descent on log(D) is used to fit the static D-scaling.

    Parameters
    ----------
    M : AnyFloatArray | AnyComplexArray, shape (n, n)
        Closed-loop transfer matrix evaluated at a single frequency (real or complex).
    delta_structure : list of (size, block_type)
        Block sizes and types from StructuredUncertainty.build_delta_structure().

    Returns
    -------
    float
        Upper bound μ̄ ≥ μ(M).
    """
    best_mu, _best_d = _compute_static_mu_upper_bound_and_scalings(M, delta_structure)
    return float(best_mu)
