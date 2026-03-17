# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Mu-Synthesis Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_control.control.mu_synthesis import (
    MuSynthesisController,
    StructuredUncertainty,
    UncertaintyBlock,
    compute_mu_upper_bound,
    dk_iteration,
)


def test_uncertainty_structure():
    b1 = UncertaintyBlock("tau_E", 1, 0.2, "real_scalar")
    b2 = UncertaintyBlock("noise", 2, 0.05, "complex_scalar")
    unc = StructuredUncertainty([b1, b2])

    assert unc.total_size() == 3
    struct = unc.build_Delta_structure()
    assert struct == [(1, "real_scalar"), (2, "complex_scalar")]


def test_compute_mu_upper_bound():
    # Construct a matrix where standard singular value > mu
    M = np.array([[2.0, 10.0], [0.0, 2.0]], dtype=complex)
    # sigma_max(M) ~ 10.2
    # If Delta is diagonal, D M D^-1 can reduce the off-diagonal

    struct = [(1, "complex_scalar"), (1, "complex_scalar")]

    sigma_max = np.max(np.linalg.svd(M)[1])
    mu_bound = compute_mu_upper_bound(M, struct)

    # mu should be tighter (smaller) than sigma_max
    assert mu_bound < sigma_max


def test_dk_iteration_convergence():
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2))
    plant = (A, B, C, D)

    unc = StructuredUncertainty([UncertaintyBlock("test", 2, 0.1, "full")])

    K, mu, D_s = dk_iteration(plant, unc, n_iter=5)

    assert mu < 1.0  # Simulated convergence
    assert K is not None


def test_mu_controller_robustness():
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2))
    plant = (A, B, C, D)

    unc = StructuredUncertainty([UncertaintyBlock("test", 2, 0.1, "full")])
    ctrl = MuSynthesisController(plant, unc)

    ctrl.synthesize(n_dk_iter=3)

    margin = ctrl.robustness_margin()
    assert margin > 1.0

    x = np.array([1.0, -1.0])
    u = ctrl.step(x, 0.1)

    assert u.shape == (2,)


# ── Physics-motivated citation tests ────────────────────────────────


def test_mu_upper_bound() -> None:
    """μ(M) ≤ σ̄(M) for any M and any structured Δ.

    Doyle 1982, IEE Proc. D 129, 242: the unstructured singular value is
    always an upper bound on the structured singular value.  The D-scaling
    approach can only tighten, never exceed, σ̄(M).
    """
    M = np.array([[2.0, 10.0], [0.0, 2.0]], dtype=complex)
    struct = [(1, "complex_scalar"), (1, "complex_scalar")]

    sigma_max = float(np.max(np.linalg.svd(M)[1]))
    mu_bound = compute_mu_upper_bound(M, struct)

    assert mu_bound <= sigma_max + 1e-10, f"μ upper bound {mu_bound} exceeds σ̄(M) = {sigma_max}"


def test_dk_iteration_converges() -> None:
    """μ_peak must decrease monotonically with D-K iterations.

    Skogestad & Postlethwaite 2005, §8.5: each outer D-K iterate either
    reduces or maintains the peak μ upper bound.
    """
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2))
    plant = (A, B, C, D)
    unc = StructuredUncertainty([UncertaintyBlock("plasma_position", 2, 0.1, "full")])

    _, mu_1, _ = dk_iteration(plant, unc, n_iter=1)
    _, mu_5, _ = dk_iteration(plant, unc, n_iter=5)

    assert mu_5 <= mu_1, f"μ did not decrease: {mu_5} > {mu_1} after 5 vs 1 iteration"
