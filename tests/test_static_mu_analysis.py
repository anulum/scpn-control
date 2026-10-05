# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static structured-mu analysis tests
"""Module-specific tests for bounded static mu-analysis contracts."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.control._static_mu_riccati as mu_riccati
import scpn_control.control.static_mu_analysis as mu
from scpn_control._typing import FloatArray
from scpn_control.control.static_mu_analysis import (
    RiccatiStateFeedbackController,
    StaticMuAnalysisResult,
    StructuredUncertainty,
    UncertaintyBlock,
    compute_static_mu_upper_bound,
    design_riccati_state_feedback_with_static_mu_analysis,
)


def _plant() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    A = np.array([[-1.4, 0.2], [-0.1, -0.9]], dtype=float)
    B = np.eye(2)
    C = np.eye(2)
    D = np.zeros((2, 2), dtype=float)
    return A, B, C, D


def _uncertainty() -> StructuredUncertainty:
    return StructuredUncertainty(
        [
            UncertaintyBlock("plasma_position", 1, 0.02, "real_scalar"),
            UncertaintyBlock("plasma_current", 1, 0.03, "real_scalar"),
        ]
    )


def test_static_mu_upper_bound_respects_block_scaling_and_domains() -> None:
    """Exercise static mu upper bound respects block scaling and domains."""
    M = np.array([[0.2, 0.05], [0.02, 0.15]], dtype=float)
    structure = [(1, "real_scalar"), (1, "real_scalar")]
    mu = compute_static_mu_upper_bound(M, structure)
    assert 0.0 < mu <= np.linalg.svd(M, compute_uv=False)[0] + 1.0e-12

    with pytest.raises(ValueError, match="M must be a square"):
        compute_static_mu_upper_bound(np.ones((2, 3)), structure)
    with pytest.raises(ValueError, match="Delta block sizes must sum"):
        compute_static_mu_upper_bound(M, [(1, "real_scalar")])
    with pytest.raises(ValueError, match="Delta structure"):
        compute_static_mu_upper_bound(np.zeros((0, 0)), [])
    with pytest.raises(ValueError, match="finite values"):
        compute_static_mu_upper_bound(np.array([[np.nan]]), [(1, "real_scalar")])
    with pytest.raises(ValueError, match="Delta block_type"):
        compute_static_mu_upper_bound(np.eye(1), [(1, "bad")])
    with pytest.raises(ValueError, match="Delta block size"):
        compute_static_mu_upper_bound(np.eye(1), [(0, "real_scalar")])


def test_static_mu_analysis_controller_designs_stable_static_controller() -> None:
    """Exercise static mu analysis controller designs stable static controller."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())
    result = controller.design()
    assert result.mu_upper_bound > 0.0
    assert controller.inverse_static_mu_upper_bound() > 0.0
    action = controller.step(np.array([0.1, -0.2]), dt=0.01)
    assert action.shape == (2,)

    with pytest.raises(ValueError, match="x must have shape"):
        controller.step(np.array([1.0]), dt=0.01)
    with pytest.raises(ValueError, match="dt must be positive"):
        controller.step(np.array([0.1, -0.2]), dt=0.0)
    with pytest.raises(ValueError, match="x must contain only finite"):
        controller.step(np.array([np.nan, -0.2]), dt=0.01)


def test_static_mu_analysis_controller_fails_closed_before_design() -> None:
    """Exercise static mu analysis controller fails closed before design."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())

    with pytest.raises(RuntimeError, match="not designed"):
        controller.step(np.array([0.1, -0.2]), dt=0.01)
    with pytest.raises(RuntimeError, match="not designed"):
        controller.inverse_static_mu_upper_bound()


def test_uncertainty_blocks_reject_invalid_contracts() -> None:
    """Exercise uncertainty blocks reject invalid contracts."""
    with pytest.raises(ValueError, match="name must be non-empty"):
        UncertaintyBlock(" ", 1, 0.1, "real_scalar")
    with pytest.raises(ValueError, match="size must be a positive integer"):
        UncertaintyBlock("bad", 0, 0.1, "real_scalar")
    with pytest.raises(ValueError, match="bound must be positive"):
        UncertaintyBlock("bad", 1, 0.0, "real_scalar")
    with pytest.raises(ValueError, match="block_type must be one"):
        UncertaintyBlock("bad", 1, 0.1, "unstructured")


def test_structured_uncertainty_exposes_physical_bound_matrix_contract() -> None:
    """Exercise structured uncertainty exposes physical bound matrix contract."""
    uncertainty = _uncertainty()

    assert uncertainty.build_delta_structure() == [(1, "real_scalar"), (1, "real_scalar")]
    assert uncertainty.total_size() == 2
    assert np.allclose(uncertainty.bound_matrix(), np.diag([0.02, 0.03]))
    with pytest.raises(ValueError, match="at least one"):
        StructuredUncertainty([])


@pytest.mark.parametrize(
    ("plant", "message"),
    [
        (
            (np.ones((2, 3)), np.eye(2), np.eye(2), np.zeros((2, 2))),
            "A must be square",
        ),
        (
            (np.eye(2), np.ones((3, 2)), np.eye(2), np.zeros((2, 2))),
            "B row count",
        ),
        (
            (np.eye(2), np.eye(2), np.ones((2, 3)), np.zeros((2, 2))),
            "C column count",
        ),
        (
            (np.eye(2), np.eye(2), np.eye(2), np.zeros((1, 2))),
            "D must have shape",
        ),
        (
            (np.array([[np.nan, 0.0], [0.0, 1.0]]), np.eye(2), np.eye(2), np.zeros((2, 2))),
            "A must contain only finite",
        ),
        (
            (np.eye(2), np.ones((2, 1)), np.eye(2), np.zeros((2, 1))),
            "uncertainty size",
        ),
    ],
)
def test_static_design_rejects_invalid_state_space_contracts(
    plant: tuple[FloatArray, FloatArray, FloatArray, FloatArray], message: str
) -> None:
    """Exercise static design rejects invalid state space contracts."""
    with pytest.raises(ValueError, match=message):
        design_riccati_state_feedback_with_static_mu_analysis(plant, _uncertainty())


def test_riccati_state_feedback_rejects_unstabilisable_plant() -> None:
    """Exercise riccati state feedback rejects unstabilisable plant."""
    A = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float)
    B = np.array([[1.0], [0.0]], dtype=float)  # second unstable mode uncontrollable
    C = np.eye(2)
    with pytest.raises(RuntimeError, match="not stabilisable"):
        mu._riccati_state_feedback(A, B, C)


def test_riccati_state_feedback_falls_back_for_stable_open_loop_plant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise riccati state feedback falls back for stable open loop plant."""
    A, B, C, _ = _plant()

    def broken_care(*args: object, **kwargs: object) -> FloatArray:
        raise TypeError("local SciPy CARE validation failed")

    monkeypatch.setattr(mu_riccati, "solve_continuous_are", broken_care)

    gain = mu._riccati_state_feedback(A, B, C)

    np.testing.assert_allclose(gain, np.zeros((B.shape[1], A.shape[0])))


def test_riccati_state_feedback_fails_closed_when_fallback_stability_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise riccati state feedback fails closed when fallback stability check fails."""
    A, B, C, _ = _plant()

    def broken_care(*args: object, **kwargs: object) -> FloatArray:
        raise TypeError("local SciPy CARE validation failed")

    def broken_eigvals(*args: object, **kwargs: object) -> FloatArray:
        raise np.linalg.LinAlgError("eigenvalue decomposition failed")

    monkeypatch.setattr(mu_riccati, "solve_continuous_are", broken_care)
    monkeypatch.setattr(np.linalg, "eigvals", broken_eigvals)

    with pytest.raises(RuntimeError, match="plant stability could not be checked"):
        mu._riccati_state_feedback(A, B, C)


def test_riccati_state_feedback_accepts_finite_solver_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise riccati state feedback accepts finite solver output."""
    A, B, C, _ = _plant()

    def finite_care(*args: object, **kwargs: object) -> FloatArray:
        return np.eye(A.shape[0])

    monkeypatch.setattr(mu_riccati, "solve_continuous_are", finite_care)

    gain = mu._riccati_state_feedback(A, B, C)

    np.testing.assert_allclose(gain, B.T)


def test_riccati_state_feedback_rejects_nonfinite_solver_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise riccati state feedback rejects nonfinite solver output."""
    A, B, C, _ = _plant()

    def nonfinite_care(*args: object, **kwargs: object) -> FloatArray:
        return np.full((A.shape[0], A.shape[0]), np.nan)

    monkeypatch.setattr(mu_riccati, "solve_continuous_are", nonfinite_care)

    with pytest.raises(RuntimeError, match="non-finite controller gain"):
        mu._riccati_state_feedback(A, B, C)


def test_closed_loop_dc_map_rejects_singular_system() -> None:
    """Exercise closed loop dc map rejects singular system."""
    identity = np.eye(2)
    with pytest.raises(RuntimeError, match="singular"):
        mu._closed_loop_dc_uncertainty_map(identity, identity, identity, np.zeros((2, 2)), identity)


def test_inverse_static_bound_is_infinite_for_nonpositive_bound() -> None:
    """Exercise inverse static bound is infinite for nonpositive bound."""
    controller = RiccatiStateFeedbackController(_plant(), _uncertainty())
    controller.analysis_result = StaticMuAnalysisResult(
        controller_gain=np.eye(2),
        mu_upper_bound=0.0,
        d_scalings=np.ones(2),
        analysis_frequency_rad_s=0.0,
        closed_loop_spectral_abscissa=-1.0,
    )
    assert controller.inverse_static_mu_upper_bound() == float("inf")
