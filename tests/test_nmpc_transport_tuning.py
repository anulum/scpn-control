# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Nonlinear MPC transport-model tuning tests
"""Gradient-based NMPC transport-model tuning entry points.

Exercises the four tuning surfaces in
:mod:`scpn_control.control.nmpc_transport_tuning` — coefficient, single-step
source-schedule, multi-step source-rollout, and neural-closure tuning — covering
their fail-closed JAX-unavailable behaviour, campaign-metadata and closure
provenance, gradient-audit admission (including the explicit rollout ``warn``
mode), and, where the optional JAX backend is present, that a real gradient step
reduces the tracking loss. Fail-closed validator branches live in
``test_nmpc_transport_tuning_error_paths.py``.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from scpn_control.control import nmpc_transport_tuning as tuning_mod
from scpn_control.core import differentiable_transport as transport_mod
from scpn_control.core.neural_transport import NeuralTransportModel, neural_transport_closure_profiles


def _transport_tuning_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = np.linspace(0.05, 1.0, 24)
    profiles = np.stack(
        [
            8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2,
            6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2,
            4.0 + 0.8 * (1.0 - rho**2),
            0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02),
        ]
    )
    chi = np.stack(
        [
            0.20 + 0.02 * rho,
            0.16 + 0.02 * rho,
            0.04 + 0.005 * rho,
            0.012 + 0.001 * rho,
        ]
    )
    sources = np.zeros_like(profiles)
    target = profiles.copy()
    target[0, 8:16] *= 0.97
    target[1, 8:16] *= 0.98
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    return profiles, chi, sources, target, rho, edge_values


def test_nmpc_transport_tuning_fails_closed_without_jax(monkeypatch) -> None:
    """NMPC transport-coefficient tuning must not silently use finite differences."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: False)

    with pytest.raises(RuntimeError, match="requires JAX"):
        tuning_mod.tune_transport_coefficients_for_tracking(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            learning_rate=0.05,
        )


def test_nmpc_transport_tuning_result_carries_campaign_metadata(monkeypatch) -> None:
    """Every transport tuning update should carry its validated campaign contract."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    gradient = 0.01 * np.ones_like(chi)
    audit = transport_mod.TransportGradientAudit(
        loss=0.125,
        epsilon=1.0e-5,
        tolerance=5.0e-4,
        checked_indices=((0, 1),),
        chi_max_abs_error=1.0e-8,
        source_max_abs_error=1.0e-8,
        passed=True,
    )
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
    monkeypatch.setattr(tuning_mod, "transport_loss_gradient", lambda *args, **kwargs: (0.125, gradient))
    monkeypatch.setattr(tuning_mod, "assert_transport_parameter_gradients_consistent", lambda *args, **kwargs: audit)

    result = tuning_mod.tune_transport_coefficients_for_tracking(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        learning_rate=0.05,
        gradient_tolerance=1.0e-7,
    )

    assert isinstance(result.metadata, transport_mod.TransportCampaignMetadata)
    assert result.metadata.backend == "jax"
    assert result.metadata.dtype == "float64"
    assert result.metadata.channel_order == transport_mod.CHANNELS
    assert result.metadata.n_rho == rho.size
    assert result.metadata.gradient_tolerance == pytest.approx(1.0e-7)
    assert result.metadata.closure_source is None
    assert result.gradient_audit == audit
    assert asdict(result.metadata)["backend"] == "jax"


def test_nmpc_transport_tuning_fails_closed_on_gradient_audit(monkeypatch) -> None:
    """NMPC must not admit transport tuning when the gradient audit fails."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    gradient = 0.01 * np.ones_like(chi)
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
    monkeypatch.setattr(tuning_mod, "transport_loss_gradient", lambda *args, **kwargs: (0.125, gradient))

    def reject_audit(*args: object, **kwargs: object) -> transport_mod.TransportGradientAudit:
        raise ValueError("transport parameter gradient audit failed")

    monkeypatch.setattr(tuning_mod, "assert_transport_parameter_gradients_consistent", reject_audit)

    with pytest.raises(ValueError, match="gradient audit failed"):
        tuning_mod.tune_transport_coefficients_for_tracking(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            learning_rate=0.05,
        )


def test_nmpc_transport_tuning_explicitly_allows_audit_bypass_for_admission_tests(monkeypatch) -> None:
    """Bypass is explicit so tests can isolate update clipping without hiding production default."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    gradient = 0.01 * np.ones_like(chi)
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
    monkeypatch.setattr(tuning_mod, "transport_loss_gradient", lambda *args, **kwargs: (0.125, gradient))

    def reject_audit(*args: object, **kwargs: object) -> transport_mod.TransportGradientAudit:
        raise AssertionError("audit should not be called")

    monkeypatch.setattr(tuning_mod, "assert_transport_parameter_gradients_consistent", reject_audit)

    result = tuning_mod.tune_transport_coefficients_for_tracking(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        learning_rate=0.05,
        require_gradient_audit=False,
    )

    assert result.gradient_audit is None
    assert result.step_norm > 0.0


def test_nmpc_source_schedule_tuning_records_audited_update(monkeypatch) -> None:
    """NMPC source schedules should use audited source gradients, not coefficient gradients."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    source_gradient = np.zeros_like(sources)
    source_gradient[0, 6:10] = -0.25
    source_gradient[2, 4:8] = 0.10
    gradients = transport_mod.TransportParameterGradients(
        loss=0.5,
        chi_gradient=np.zeros_like(chi),
        source_gradient=source_gradient,
    )
    audit = transport_mod.TransportGradientAudit(
        loss=0.5,
        epsilon=1.0e-5,
        tolerance=5.0e-4,
        checked_indices=((0, 6), (2, 4)),
        chi_max_abs_error=1.0e-8,
        source_max_abs_error=1.0e-8,
        passed=True,
    )
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
    monkeypatch.setattr(tuning_mod, "transport_parameter_gradients", lambda *args, **kwargs: gradients)
    monkeypatch.setattr(tuning_mod, "assert_transport_parameter_gradients_consistent", lambda *args, **kwargs: audit)

    result = tuning_mod.tune_transport_sources_for_tracking(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        learning_rate=0.2,
        source_min=-0.02,
        source_max=0.03,
        max_absolute_update=0.015,
        gradient_tolerance=1.0e-7,
    )

    assert isinstance(result, tuning_mod.TransportSourceScheduleTuningResult)
    assert result.loss == pytest.approx(0.5)
    assert result.gradient_audit == audit
    assert result.metadata.gradient_tolerance == pytest.approx(1.0e-7)
    assert result.step_norm > 0.0
    assert np.all(result.updated_sources >= -0.02)
    assert np.all(result.updated_sources <= 0.03)
    assert np.max(np.abs(result.updated_sources - sources)) <= 0.015 + 1.0e-12


def test_nmpc_source_schedule_tuning_without_audit_or_update_clamp(monkeypatch) -> None:
    """Bypassing the gradient audit and omitting the update clamp still yields a bounded update."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    source_gradient = np.zeros_like(sources)
    source_gradient[0, 6:10] = -0.25
    gradients = transport_mod.TransportParameterGradients(
        loss=0.5,
        chi_gradient=np.zeros_like(chi),
        source_gradient=source_gradient,
    )
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
    monkeypatch.setattr(tuning_mod, "transport_parameter_gradients", lambda *args, **kwargs: gradients)

    result = tuning_mod.tune_transport_sources_for_tracking(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        learning_rate=0.2,
        source_min=-0.02,
        source_max=0.03,
        require_gradient_audit=False,
    )

    assert isinstance(result, tuning_mod.TransportSourceScheduleTuningResult)
    assert result.gradient_audit is None
    assert result.step_norm > 0.0


def test_nmpc_source_schedule_tuning_fails_closed_without_jax(monkeypatch) -> None:
    """Source tuning must not silently use finite differences when JAX is absent."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: False)

    with pytest.raises(RuntimeError, match="requires JAX"):
        tuning_mod.tune_transport_sources_for_tracking(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            learning_rate=0.05,
        )


def test_nmpc_source_schedule_tuning_fails_closed_on_gradient_audit(monkeypatch) -> None:
    """Source schedule updates must be blocked when the transport gradient audit fails."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    gradients = transport_mod.TransportParameterGradients(
        loss=0.5,
        chi_gradient=np.zeros_like(chi),
        source_gradient=np.ones_like(sources),
    )
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
    monkeypatch.setattr(tuning_mod, "transport_parameter_gradients", lambda *args, **kwargs: gradients)

    def reject_audit(*args: object, **kwargs: object) -> transport_mod.TransportGradientAudit:
        raise ValueError("transport parameter gradient audit failed")

    monkeypatch.setattr(tuning_mod, "assert_transport_parameter_gradients_consistent", reject_audit)

    with pytest.raises(ValueError, match="gradient audit failed"):
        tuning_mod.tune_transport_sources_for_tracking(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            learning_rate=0.05,
        )


def test_nmpc_source_schedule_tuning_rejects_invalid_bounds(monkeypatch) -> None:
    """Source bounds are explicit because physically valid schedules may include sinks."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)

    with pytest.raises(ValueError, match="source_min"):
        tuning_mod.tune_transport_sources_for_tracking(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            learning_rate=0.05,
            source_min=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="source_min entries"):
        tuning_mod.tune_transport_sources_for_tracking(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            learning_rate=0.05,
            source_min=1.0,
            source_max=0.0,
        )


def test_nmpc_neural_closure_tuning_fails_closed_without_jax(monkeypatch) -> None:
    """NMPC neural-closure tuning must use the differentiable coefficient path."""
    profiles, _, sources, target, rho, edge_values = _transport_tuning_case()
    closure = neural_transport_closure_profiles(
        rho,
        profiles[0],
        profiles[1],
        profiles[2],
        1.0 + 2.0 * rho**2,
        0.5 + 1.5 * rho,
        model=NeuralTransportModel(auto_discover=False),
    )
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: False)

    with pytest.raises(RuntimeError, match="requires JAX"):
        tuning_mod.tune_neural_transport_closure_for_tracking(
            profiles,
            closure,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            learning_rate=0.05,
        )


def test_nmpc_neural_closure_tuning_result_carries_closure_metadata(monkeypatch) -> None:
    """Neural-closure tuning should preserve closure provenance in the NMPC result."""
    profiles, _, sources, target, rho, edge_values = _transport_tuning_case()
    closure = neural_transport_closure_profiles(
        rho,
        profiles[0],
        profiles[1],
        profiles[2],
        1.0 + 2.0 * rho**2,
        0.5 + 1.5 * rho,
        model=NeuralTransportModel(auto_discover=False),
    )
    gradient = 0.01 * np.ones_like(profiles)
    audit = transport_mod.TransportGradientAudit(
        loss=0.25,
        epsilon=1.0e-5,
        tolerance=5.0e-4,
        checked_indices=((0, 1),),
        chi_max_abs_error=1.0e-8,
        source_max_abs_error=1.0e-8,
        passed=True,
    )
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
    monkeypatch.setattr(tuning_mod, "transport_loss_gradient", lambda *args, **kwargs: (0.25, gradient))
    monkeypatch.setattr(tuning_mod, "assert_transport_parameter_gradients_consistent", lambda *args, **kwargs: audit)

    result = tuning_mod.tune_neural_transport_closure_for_tracking(
        profiles,
        closure,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        learning_rate=0.05,
        impurity_diffusivity_fraction=0.5,
        gradient_tolerance=5.0e-8,
    )

    assert result.metadata.closure_source == "analytic_fallback"
    assert result.metadata.closure_weights_checksum is None
    assert result.metadata.gradient_tolerance == pytest.approx(5.0e-8)
    assert result.metadata.edge_boundary == "dirichlet"
    assert result.gradient_audit == audit


@pytest.mark.skipif(not transport_mod.has_jax(), reason="JAX optional dependency is not installed")
def test_nmpc_transport_tuning_reduces_tracking_loss() -> None:
    """A JAX gradient step should reduce the same transport loss seen by NMPC tuning."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    weights = np.array([1.0, 1.0, 0.25, 0.1])
    initial_loss = float(
        transport_mod.transport_tracking_loss(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            weights=weights,
        )
    )

    result = tuning_mod.tune_transport_coefficients_for_tracking(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        weights=weights,
        learning_rate=0.05,
        max_fractional_update=0.25,
        gradient_audit_tolerance=2.0e-3,
    )

    updated_loss = float(
        transport_mod.transport_tracking_loss(
            profiles,
            result.updated_chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            weights=weights,
        )
    )
    assert result.loss == pytest.approx(initial_loss)
    assert result.gradient.shape == chi.shape
    assert result.step_norm > 0.0
    assert result.gradient_audit is not None
    assert result.gradient_audit.passed
    assert np.all(result.updated_chi >= 0.0)
    assert updated_loss < initial_loss


@pytest.mark.skipif(not transport_mod.has_jax(), reason="JAX optional dependency is not installed")
def test_nmpc_source_schedule_tuning_reduces_tracking_loss() -> None:
    """A JAX source-gradient step should reduce the transport loss used by NMPC."""
    profiles, chi, sources, target, rho, edge_values = _transport_tuning_case()
    target = profiles.copy()
    target[0, 8:16] += 0.02
    target[2, 5:12] += 0.01
    weights = np.array([1.0, 0.5, 0.25, 0.1])
    initial_loss = float(
        transport_mod.transport_tracking_loss(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            weights=weights,
        )
    )

    result = tuning_mod.tune_transport_sources_for_tracking(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        weights=weights,
        learning_rate=0.1,
        max_absolute_update=0.02,
        gradient_audit_tolerance=2.0e-3,
    )

    updated_loss = float(
        transport_mod.transport_tracking_loss(
            profiles,
            chi,
            result.updated_sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            weights=weights,
        )
    )
    assert result.loss == pytest.approx(initial_loss)
    assert result.gradient.shape == sources.shape
    assert result.step_norm > 0.0
    assert result.gradient_audit is not None
    assert result.gradient_audit.passed
    assert updated_loss < initial_loss


def _rollout_transport_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray
]:
    rho = np.linspace(0.0, 1.0, 7)
    profiles = np.vstack(
        [
            6.0 - 4.0 * rho**2,
            5.0 - 3.0 * rho**2,
            8.0 - 2.0 * rho**2,
            0.04 * (1.0 - rho**2),
        ]
    )
    chi = np.full_like(profiles, 0.03)
    source_sequence = np.zeros((4, 4, rho.size), dtype=np.float64)
    source_sequence[:, 0, 2:5] = 0.6
    source_sequence[:, 1, 2:5] = 0.4
    source_sequence[:, 2, 1:4] = 0.2
    edge_values = profiles[:, -1].copy()
    return profiles, chi, source_sequence, rho, edge_values, 0.01, edge_values


def test_nmpc_transport_rollout_tuning_fails_closed_without_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    """NMPC must not admit rollout source-gradient tuning without JAX autodiff."""
    profiles, chi, source_sequence, rho, edge_values, dt, _ = _rollout_transport_fixture()
    target_history = np.repeat(profiles[None, :, :], source_sequence.shape[0], axis=0)
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: False)

    with pytest.raises(RuntimeError, match="requires JAX"):
        tuning_mod.tune_transport_source_rollout_for_tracking(
            profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            dt,
            edge_values,
            learning_rate=0.05,
        )


def test_nmpc_transport_rollout_tuning_rejects_malformed_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NMPC admission boundary must reject non-four-channel source schedules."""
    profiles, chi, source_sequence, rho, edge_values, dt, _ = _rollout_transport_fixture()
    target_history = np.repeat(profiles[None, :, :], source_sequence.shape[0], axis=0)
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)

    with pytest.raises(ValueError, match="source_sequence"):
        tuning_mod.tune_transport_source_rollout_for_tracking(
            profiles,
            chi,
            source_sequence[:, :3, :],
            target_history,
            rho,
            dt,
            edge_values,
            learning_rate=0.05,
        )


def test_nmpc_transport_rollout_tuning_rejects_unknown_audit_failure_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit failure handling must be explicit and bounded to known modes."""
    profiles, chi, source_sequence, rho, edge_values, dt, _ = _rollout_transport_fixture()
    target_history = np.repeat(profiles[None, :, :], source_sequence.shape[0], axis=0)
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)

    with pytest.raises(ValueError, match="gradient_audit_failure_mode"):
        tuning_mod.tune_transport_source_rollout_for_tracking(
            profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            dt,
            edge_values,
            learning_rate=0.05,
            gradient_audit_failure_mode="continue",
        )


def test_nmpc_transport_rollout_tuning_warn_mode_keeps_failed_audit_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warning mode is advisory-only and must preserve the failed audit result."""
    profiles, chi, source_sequence, rho, edge_values, dt, _ = _rollout_transport_fixture()
    target_history = np.repeat(profiles[None, :, :], source_sequence.shape[0], axis=0)
    gradient = np.full_like(source_sequence, 0.125)
    failed_audit = tuning_mod.TransportSourceRolloutGradientAudit(
        loss=1.0,
        epsilon=1.0e-5,
        tolerance=5.0e-4,
        checked_indices=((0, 0, 0),),
        source_max_abs_error=1.0,
        passed=False,
    )
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
    monkeypatch.setattr(
        tuning_mod,
        "transport_rollout_source_gradients",
        lambda *args, **kwargs: transport_mod.TransportRolloutSourceGradients(
            loss=2.0,
            source_gradient=gradient,
            final_profiles=profiles,
        ),
    )
    monkeypatch.setattr(tuning_mod, "_audit_transport_rollout_source_gradients", lambda *args, **kwargs: failed_audit)

    with pytest.warns(RuntimeWarning, match="advisory-only"):
        result = tuning_mod.tune_transport_source_rollout_for_tracking(
            profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            dt,
            edge_values,
            learning_rate=0.05,
            gradient_audit_failure_mode="warn",
        )

    assert result.gradient_audit == failed_audit
    assert result.gradient_audit.passed is False
    assert result.step_norm > 0.0
    np.testing.assert_allclose(result.updated_sources, source_sequence - 0.05 * gradient)


def test_rollout_audit_indices_deduplicates_repeated_entries() -> None:
    """Repeated sample indices collapse to a single audited entry."""
    indices = tuning_mod._rollout_audit_indices((3, 4, 8), [(0, 0, 1), (0, 0, 1), (1, 2, 4)])
    assert indices == ((0, 0, 1), (1, 2, 4))


def test_nmpc_transport_rollout_tuning_without_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypassing the rollout gradient audit records no audit evidence on the result."""
    profiles, chi, source_sequence, rho, edge_values, dt, _ = _rollout_transport_fixture()
    target_history = np.repeat(profiles[None, :, :], source_sequence.shape[0], axis=0)
    gradient = np.full_like(source_sequence, 0.05)
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
    monkeypatch.setattr(
        tuning_mod,
        "transport_rollout_source_gradients",
        lambda *args, **kwargs: transport_mod.TransportRolloutSourceGradients(
            loss=1.0,
            source_gradient=gradient,
            final_profiles=profiles,
        ),
    )

    result = tuning_mod.tune_transport_source_rollout_for_tracking(
        profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        dt,
        edge_values,
        learning_rate=0.05,
        require_gradient_audit=False,
    )

    assert result.gradient_audit is None
    assert result.step_norm > 0.0


def test_nmpc_transport_rollout_tuning_default_fails_closed_on_failed_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production default must still reject rollout updates with failed audit evidence."""
    profiles, chi, source_sequence, rho, edge_values, dt, _ = _rollout_transport_fixture()
    target_history = np.repeat(profiles[None, :, :], source_sequence.shape[0], axis=0)
    gradient = np.full_like(source_sequence, 0.125)
    failed_audit = tuning_mod.TransportSourceRolloutGradientAudit(
        loss=1.0,
        epsilon=1.0e-5,
        tolerance=5.0e-4,
        checked_indices=((0, 0, 0),),
        source_max_abs_error=1.0,
        passed=False,
    )
    monkeypatch.setattr(tuning_mod, "has_differentiable_transport_jax", lambda: True)
    monkeypatch.setattr(
        tuning_mod,
        "transport_rollout_source_gradients",
        lambda *args, **kwargs: transport_mod.TransportRolloutSourceGradients(
            loss=2.0,
            source_gradient=gradient,
            final_profiles=profiles,
        ),
    )
    monkeypatch.setattr(tuning_mod, "_audit_transport_rollout_source_gradients", lambda *args, **kwargs: failed_audit)

    with pytest.raises(ValueError, match="rollout source gradient audit failed"):
        tuning_mod.tune_transport_source_rollout_for_tracking(
            profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            dt,
            edge_values,
            learning_rate=0.05,
        )


@pytest.mark.skipif(not transport_mod.has_jax(), reason="JAX optional dependency not installed")
def test_nmpc_transport_rollout_tuning_updates_bounded_source_schedule() -> None:
    """Rollout tuning should produce finite audited source updates within actuator bounds."""
    profiles, chi, source_sequence, rho, edge_values, dt, _ = _rollout_transport_fixture()
    desired_sources = source_sequence.copy()
    desired_sources[:, 0, 2:5] += 0.2
    desired_sources[:, 2, 1:4] += 0.1
    target_history = np.asarray(
        transport_mod.differentiable_transport_rollout(
            profiles,
            chi,
            desired_sources,
            rho,
            dt,
            edge_values,
            use_jax=False,
        ),
        dtype=np.float64,
    )

    result = tuning_mod.tune_transport_source_rollout_for_tracking(
        profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        dt,
        edge_values,
        learning_rate=0.2,
        source_min=0.0,
        source_max=1.0,
        max_absolute_update=0.05,
        gradient_audit_tolerance=2.0e-3,
        gradient_audit_sample_indices=((0, 0, 2), (2, 2, 3), (3, 1, 4)),
    )

    assert np.isfinite(result.loss)
    assert result.gradient.shape == source_sequence.shape
    assert result.updated_sources.shape == source_sequence.shape
    assert result.final_profiles.shape == profiles.shape
    assert np.all(np.isfinite(result.gradient))
    assert np.all(result.updated_sources >= 0.0)
    assert np.all(result.updated_sources <= 1.0)
    assert 0.0 < result.step_norm <= np.sqrt(source_sequence.size) * 0.05
    assert result.metadata.backend == "jax"
    assert result.gradient_audit is not None
    assert result.gradient_audit.passed is True
    assert result.gradient_audit.checked_indices == ((0, 0, 2), (2, 2, 3), (3, 1, 4))
