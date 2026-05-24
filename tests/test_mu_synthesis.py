# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Mu-Synthesis Tests
from __future__ import annotations

import numpy as np
import pytest

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

    assert mu < 1.0
    assert K is not None


def test_dk_iteration_mu_depends_on_closed_loop_plant_dynamics() -> None:
    """D-K evidence must come from the closed-loop plant, not iteration count."""
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2))
    unc = StructuredUncertainty([UncertaintyBlock("plasma_position", 2, 0.1, "full")])

    _, mu_marginal, _ = dk_iteration((np.zeros((2, 2)), B, C, D), unc, n_iter=4)
    _, mu_unstable, _ = dk_iteration((np.eye(2) * 2.0, B, C, D), unc, n_iter=4)

    assert not np.isclose(mu_marginal, mu_unstable)
    assert mu_marginal > mu_unstable


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
    """Repeated bounded static passes must not fabricate worse mu evidence."""
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2))
    plant = (A, B, C, D)
    unc = StructuredUncertainty([UncertaintyBlock("plasma_position", 2, 0.1, "full")])

    _, mu_1, _ = dk_iteration(plant, unc, n_iter=1)
    _, mu_5, _ = dk_iteration(plant, unc, n_iter=5)

    assert mu_5 <= mu_1, f"μ did not decrease: {mu_5} > {mu_1} after 5 vs 1 iteration"


def test_unsynthesized_step_raises():
    """Exercise mu_synthesis.py line 239: step without synthesize raises."""
    import pytest

    unc = StructuredUncertainty([UncertaintyBlock("t", 2, 0.1, "full")])
    ctrl = MuSynthesisController((np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2))), unc)
    with pytest.raises(RuntimeError, match="not synthesized"):
        ctrl.step(np.array([1.0, -1.0]), 0.1)


def test_robustness_margin_zero_mu():
    """Exercise mu_synthesis.py line 249: mu_peak <= 0 returns inf."""
    unc = StructuredUncertainty([UncertaintyBlock("t", 2, 0.1, "full")])
    ctrl = MuSynthesisController((np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2))), unc)
    ctrl.K = np.ones((2, 2)) * 0.1
    ctrl.mu_peak = 0.0
    assert ctrl.robustness_margin() == float("inf")


def test_uncertainty_blocks_reject_nonphysical_contracts() -> None:
    """Structured uncertainty blocks reject invalid labels, sizes, bounds, and block types."""
    with pytest.raises(ValueError, match="name"):
        UncertaintyBlock("", 1, 0.1, "full")
    with pytest.raises(ValueError, match="size"):
        UncertaintyBlock("plasma_position", 0, 0.1, "full")
    with pytest.raises(ValueError, match="bound"):
        UncertaintyBlock("plasma_position", 1, 0.0, "full")
    with pytest.raises(ValueError, match="block_type"):
        UncertaintyBlock("plasma_position", 1, 0.1, "diagonal")
    with pytest.raises(ValueError, match="blocks"):
        StructuredUncertainty([])


def test_dk_iteration_uses_uncertainty_bounds_in_mu_evidence() -> None:
    """Declared uncertainty bounds must scale the robust-performance channel."""
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2))
    plant = (A, B, C, D)

    small = StructuredUncertainty([UncertaintyBlock("plasma_position", 2, 0.05, "full")])
    large = StructuredUncertainty([UncertaintyBlock("plasma_position", 2, 0.5, "full")])

    _, mu_small, _ = dk_iteration(plant, small, n_iter=3)
    _, mu_large, _ = dk_iteration(plant, large, n_iter=3)

    assert mu_large > mu_small


def test_mu_controller_step_rejects_invalid_state_and_timestep() -> None:
    """Synthesised controller rejects non-finite, wrong-shaped state vectors and invalid timesteps."""
    unc = StructuredUncertainty([UncertaintyBlock("t", 2, 0.1, "full")])
    ctrl = MuSynthesisController((np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2))), unc)
    ctrl.synthesize()

    with pytest.raises(ValueError, match="x"):
        ctrl.step(np.array([1.0, np.nan]), 0.1)
    with pytest.raises(ValueError, match="x"):
        ctrl.step(np.array([1.0, -1.0, 0.0]), 0.1)
    with pytest.raises(ValueError, match="dt"):
        ctrl.step(np.array([1.0, -1.0]), 0.0)
