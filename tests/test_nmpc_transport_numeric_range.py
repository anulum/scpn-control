# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — NMPC transport numeric range tests.
"""Exercise representable large tuning steps through the real JAX entry points."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_control._typing import FloatArray
from scpn_control.control.nmpc_transport_contracts import _bounded_tuning_update
from scpn_control.control.nmpc_transport_tuning import (
    tune_transport_coefficients_for_tracking,
    tune_transport_source_rollout_for_tracking,
    tune_transport_sources_for_tracking,
)
from scpn_control.core.differentiable_transport import differentiable_transport_rollout, has_jax


def _single_step_case() -> tuple[FloatArray, ...]:
    """Return a finite four-channel profile with nonzero real JAX gradients."""
    rho = np.linspace(0.05, 1.0, 24)
    profiles = np.stack(
        [
            8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2,
            6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2,
            4.0 + 0.8 * (1.0 - rho**2),
            0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02),
        ]
    )
    chi = np.stack([0.20 + 0.02 * rho, 0.16 + 0.02 * rho, 0.04 + 0.005 * rho, 0.012 + 0.001 * rho])
    sources = np.zeros_like(profiles)
    edge = np.array([0.2, 0.2, 4.0, 0.03])
    return profiles, chi, sources, rho, edge


def _rollout_case() -> tuple[FloatArray, ...]:
    """Return a real transport rollout with a distinct target history."""
    rho = np.linspace(0.0, 1.0, 7)
    profiles = np.vstack([6.0 - 4.0 * rho**2, 5.0 - 3.0 * rho**2, 8.0 - 2.0 * rho**2, 0.04 * (1.0 - rho**2)])
    chi = np.full_like(profiles, 0.03)
    sources = np.zeros((4, 4, rho.size), dtype=np.float64)
    sources[:, 0, 2:5] = 0.6
    sources[:, 1, 2:5] = 0.4
    sources[:, 2, 1:4] = 0.2
    edge = profiles[:, -1].copy()
    desired = sources.copy()
    desired[:, 0, 2:5] += 0.2
    desired[:, 2, 1:4] += 0.1
    history = np.asarray(
        differentiable_transport_rollout(profiles, chi, desired, rho, 0.01, edge, use_jax=False),
        dtype=np.float64,
    )
    return profiles, chi, sources, history, rho, edge


@pytest.mark.parametrize("surface", ["coefficients", "sources", "rollout"])
@pytest.mark.skipif(not has_jax(), reason="optional JAX backend unavailable")
def test_large_representable_tuning_step_has_finite_norm(surface: str) -> None:
    """The public optimizer must measure a large finite update without squaring overflow."""
    if surface == "rollout":
        profiles, chi, sources, history, rho, edge = _rollout_case()
        rollout_result = tune_transport_source_rollout_for_tracking(
            profiles,
            chi,
            sources,
            history,
            rho,
            0.01,
            edge,
            learning_rate=1.0e308,
            require_gradient_audit=False,
        )
        baseline = sources
        updated = rollout_result.updated_sources
        loss = rollout_result.loss
        step_norm = rollout_result.step_norm
    else:
        profiles, chi, sources, rho, edge = _single_step_case()
        target = profiles.copy()
        if surface == "coefficients":
            target[0, 8:16] *= 0.97
            target[1, 8:16] *= 0.98
            coefficient_result = tune_transport_coefficients_for_tracking(
                profiles,
                chi,
                sources,
                target,
                rho,
                1.0e-3,
                edge,
                learning_rate=1.0e308,
                max_fractional_update=None,
                require_gradient_audit=False,
            )
            baseline = chi
            updated = coefficient_result.updated_chi
            loss = coefficient_result.loss
            step_norm = coefficient_result.step_norm
        else:
            target[0, 8:16] += 0.02
            target[2, 5:12] += 0.01
            source_result = tune_transport_sources_for_tracking(
                profiles,
                chi,
                sources,
                target,
                rho,
                1.0e-3,
                edge,
                learning_rate=1.0e308,
                require_gradient_audit=False,
            )
            baseline = sources
            updated = source_result.updated_sources
            loss = source_result.loss
            step_norm = source_result.step_norm

    scale = float(np.max(np.abs(updated - baseline)))
    assert np.isfinite(loss)
    assert np.all(np.isfinite(updated))
    assert np.isfinite(step_norm)
    assert scale <= step_norm <= scale * np.sqrt(updated.size)


@pytest.mark.skipif(not has_jax(), reason="optional JAX backend unavailable")
def test_rollout_gradient_audit_rejects_fractional_sample_index() -> None:
    """The public audit must sample the requested coordinate without truncation."""
    profiles, chi, sources, history, rho, edge = _rollout_case()
    with pytest.raises(ValueError, match="integer"):
        tune_transport_source_rollout_for_tracking(
            profiles,
            chi,
            sources,
            history,
            rho,
            0.01,
            edge,
            learning_rate=0.2,
            gradient_audit_sample_indices=((0.5, 0, 1),),
        )


@pytest.mark.parametrize("surface", ["coefficients", "sources", "rollout"])
@pytest.mark.skipif(not has_jax(), reason="optional JAX backend unavailable")
def test_real_jax_overflowed_loss_is_not_returned_as_tuning_evidence(surface: str) -> None:
    """A finite target can overflow the real JAX loss while its gradient stays finite."""
    with pytest.raises(ValueError, match="transport loss must be finite"):
        if surface == "rollout":
            profiles, chi, sources, history, rho, edge = _rollout_case()
            history[0, 0, 2] = 1.0e155
            tune_transport_source_rollout_for_tracking(
                profiles,
                chi,
                sources,
                history,
                rho,
                0.01,
                edge,
                learning_rate=1.0e-160,
                require_gradient_audit=False,
            )
        else:
            profiles, chi, sources, rho, edge = _single_step_case()
            target = profiles.copy()
            target[0, 8:16] = 1.0e155
            if surface == "coefficients":
                tune_transport_coefficients_for_tracking(
                    profiles,
                    chi,
                    sources,
                    target,
                    rho,
                    1.0e-3,
                    edge,
                    learning_rate=1.0e-160,
                    require_gradient_audit=False,
                )
            else:
                tune_transport_sources_for_tracking(
                    profiles,
                    chi,
                    sources,
                    target,
                    rho,
                    1.0e-3,
                    edge,
                    learning_rate=1.0e-160,
                    require_gradient_audit=False,
                )


@pytest.mark.parametrize(
    ("baseline", "gradient", "rate", "fractional_cap", "upper", "match"),
    [
        (np.array([0.0]), np.array([1.0e308]), 2.0, None, None, "update must be finite"),
        (np.array([1.0e308]), np.array([0.0]), 1.0, 2.0, None, "update cap must be finite"),
        (np.array([1.0e308]), np.array([-1.0e308]), 1.0, None, None, "updated values must be finite"),
        (np.array([1.0e308]), np.array([0.0]), 1.0, None, -1.0e308, "difference must be finite"),
        (np.zeros(2), np.array([-1.4e308, -1.4e308]), 1.0, None, None, "step norm must be finite"),
    ],
)
def test_update_rejects_unrepresentable_numeric_result(
    baseline: FloatArray,
    gradient: FloatArray,
    rate: float,
    fractional_cap: float | None,
    upper: float | None,
    match: str,
) -> None:
    """Reject overflow at each distinct update stage before publishing a control candidate."""
    with pytest.raises(ValueError, match=match):
        _bounded_tuning_update(
            "sources",
            baseline,
            gradient,
            rate,
            fractional_cap=fractional_cap,
            upper=upper,
        )


def test_zero_update_has_zero_norm() -> None:
    """A stationary tuning step remains finite and does not fabricate progress."""
    updated, step_norm = _bounded_tuning_update(
        "sources",
        np.array([1.0, 2.0]),
        np.zeros(2),
        0.5,
    )
    np.testing.assert_array_equal(updated, np.array([1.0, 2.0]))
    assert step_norm == 0.0
