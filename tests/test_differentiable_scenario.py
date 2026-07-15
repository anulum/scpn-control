# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — End-to-end differentiable scenario tests

from __future__ import annotations

import numpy as np
import pytest

import scpn_control.core.differentiable_scenario as ds
from scpn_control.core.differentiable_transport import differentiable_transport_rollout


def _profiles(rho: np.ndarray) -> np.ndarray:
    te = 8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2
    ti = 6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2
    ne = 4.0 + 0.8 * (1.0 - rho**2)
    nz = 0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02)
    return np.stack([te, ti, ne, nz])


def _scenario_setup(n_rho: int = 16, n_steps: int = 3):
    rho = np.linspace(0.05, 1.0, n_rho)
    profiles = _profiles(rho)
    chi = np.stack([0.20 + 0.02 * rho, 0.16 + 0.02 * rho, 0.04 + 0.005 * rho, 0.012 + 0.001 * rho])
    edge = np.array([0.2, 0.2, 4.0, 0.03])
    dt = 8.0e-4
    sources = np.zeros((n_steps, 4, n_rho))
    sources[:, 0, 4:8] = 0.01
    sources[:, 2, 3:7] = 0.004
    target = np.asarray(differentiable_transport_rollout(profiles, chi, sources, rho, dt, edge, use_jax=False))
    target[:, 0, 5:9] += 0.05
    target[:, 2, 4:8] += 0.02
    r_grid = np.linspace(1.2, 2.2, n_rho)
    z_grid = np.linspace(-0.8, 0.8, 13)
    params = np.array([1.3, 0.7])
    return params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge


def test_scenario_flux_matches_solovev_closed_form() -> None:
    r_grid = np.linspace(1.2, 2.2, 9)
    z_grid = np.linspace(-0.8, 0.8, 7)
    flux = ds.scenario_equilibrium_flux(np.array([1.3, 0.7]), r_grid, z_grid)
    rr, zz = np.meshgrid(r_grid, z_grid)
    np.testing.assert_allclose(flux, 1.3 * rr**4 / 8.0 + 0.7 * zz**2, atol=1e-12)
    assert flux.shape == (z_grid.size, r_grid.size)


def test_scenario_loss_numpy_matches_jax() -> None:
    pytest.importorskip("jax")
    params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    numpy_loss = float(
        ds.differentiable_scenario_loss(
            params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge, use_jax=False
        )
    )
    jax_loss = float(
        ds.differentiable_scenario_loss(
            params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge, use_jax=True
        )
    )
    assert jax_loss == pytest.approx(numpy_loss, rel=1e-7, abs=1e-14)


@pytest.mark.skipif(not ds.has_jax(), reason="JAX optional dependency is not installed")
def test_scenario_gradient_couples_equilibrium_and_sources() -> None:
    params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    result = ds.differentiable_scenario_gradient(params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge)
    assert result.equilibrium_param_gradient.shape == (2,)
    assert result.source_gradient.shape == sources.shape
    assert result.final_profiles.shape == profiles.shape
    assert np.all(np.isfinite(result.equilibrium_param_gradient))
    assert np.all(np.isfinite(result.source_gradient))
    # Both the equilibrium shape and the source schedule influence the loss.
    assert np.any(np.abs(result.equilibrium_param_gradient) > 0.0)
    assert np.any(np.abs(result.source_gradient) > 0.0)
    assert result.radial_weights.mean() == pytest.approx(1.0)


@pytest.mark.skipif(not ds.has_jax(), reason="JAX optional dependency is not installed")
def test_scenario_gradient_uses_explicit_channel_weights() -> None:
    """Explicit channel weights replace the uniform default (branch 249->252).

    When weights are supplied the uniform default is not constructed; the
    per-channel weighting reshapes the loss, so a non-uniform weight vector
    yields a different but still finite equilibrium-parameter gradient than the
    default uniform weighting.
    """
    setup = _scenario_setup()
    default = ds.differentiable_scenario_gradient(*setup)
    weights = np.arange(1.0, ds.CHANNEL_COUNT + 1.0)
    weighted = ds.differentiable_scenario_gradient(*setup, weights=weights)

    assert np.all(np.isfinite(weighted.equilibrium_param_gradient))
    assert not np.allclose(weighted.equilibrium_param_gradient, default.equilibrium_param_gradient)


@pytest.mark.skipif(not ds.has_jax(), reason="JAX optional dependency is not installed")
def test_scenario_gradient_audit_matches_finite_difference() -> None:
    params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    audit = ds.assert_differentiable_scenario_gradient_consistent(
        params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge
    )
    assert audit.passed is True
    assert audit.param_max_abs_error <= audit.tolerance
    assert audit.source_max_abs_error <= audit.tolerance
    assert audit.checked_param_indices == (0, 1)


@pytest.mark.skipif(not ds.has_jax(), reason="JAX optional dependency is not installed")
def test_scenario_assert_fails_closed_on_tight_tolerance() -> None:
    params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    with pytest.raises(ValueError, match="scenario gradient audit failed"):
        ds.assert_differentiable_scenario_gradient_consistent(
            params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge, tolerance=1e-30
        )


def test_scenario_gradient_fails_closed_without_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ds, "_HAS_JAX", False)
    params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    with pytest.raises(RuntimeError, match="requires JAX"):
        ds.differentiable_scenario_gradient(params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("params_size", "equilibrium_params must be a finite vector"),
        ("params_nan", "equilibrium_params must be a finite vector"),
        ("r_grid_short", "r_grid must be a finite one-dimensional axis"),
        ("z_grid_2d", "z_grid must be a finite one-dimensional axis"),
    ],
)
def test_scenario_flux_rejects_invalid_inputs(mutation: str, match: str) -> None:
    params = np.array([1.3, 0.7])
    r_grid = np.linspace(1.2, 2.2, 9)
    z_grid = np.linspace(-0.8, 0.8, 7)
    if mutation == "params_size":
        params = np.array([1.3, 0.7, 0.1])
    elif mutation == "params_nan":
        params = np.array([1.3, np.nan])
    elif mutation == "r_grid_short":
        r_grid = np.array([1.2, 2.2])
    elif mutation == "z_grid_2d":
        z_grid = z_grid.reshape(1, -1)
    with pytest.raises(ValueError, match=match):
        ds.scenario_equilibrium_flux(params, r_grid, z_grid)


@pytest.mark.skipif(not ds.has_jax(), reason="JAX optional dependency is not installed")
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"epsilon": 0.0}, "epsilon must be positive"),
        ({"tolerance": -1.0}, "tolerance must be positive"),
        ({"sample_indices": []}, "at least one rollout source index"),
        ({"sample_indices": [(0, 0, 99)]}, "out-of-range rollout source index"),
        ({"sample_indices": [(0, 0)]}, "three-part rollout source indices"),
    ],
)
def test_scenario_audit_rejects_invalid_arguments(kwargs: dict[str, object], match: str) -> None:
    params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    with pytest.raises(ValueError, match=match):
        ds.audit_differentiable_scenario_gradient(
            params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge, **kwargs
        )


@pytest.mark.skipif(not ds.has_jax(), reason="JAX optional dependency is not installed")
def test_scenario_audit_accepts_explicit_sample_indices() -> None:
    params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    audit = ds.audit_differentiable_scenario_gradient(
        params,
        profiles,
        chi,
        sources,
        target,
        rho,
        r_grid,
        z_grid,
        dt,
        edge,
        sample_indices=[(0, 0, 1), (1, 2, 3)],
    )
    assert audit.checked_source_indices == ((0, 0, 1), (1, 2, 3))


@pytest.mark.skipif(not ds.has_jax(), reason="JAX optional dependency is not installed")
def test_scenario_gradient_requires_target_history() -> None:
    params, profiles, chi, sources, _target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    with pytest.raises(ValueError, match="target_history is required"):
        ds.differentiable_scenario_gradient(params, profiles, chi, sources, None, rho, r_grid, z_grid, dt, edge)


def test_scenario_metadata_rejects_non_positive_tolerance() -> None:
    params, profiles, chi, sources, _target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    with pytest.raises(ValueError, match="gradient_tolerance must be positive and finite"):
        ds.scenario_campaign_metadata(
            params, profiles, chi, sources, rho, r_grid, z_grid, dt, edge, backend="jax", gradient_tolerance=-1.0
        )


@pytest.mark.skipif(not ds.has_jax(), reason="JAX optional dependency is not installed")
def test_scenario_readiness_rejects_audit_tolerance_mismatch() -> None:
    params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    audit = ds.audit_differentiable_scenario_gradient(
        params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge, tolerance=5.0e-4
    )
    metadata = ds.scenario_campaign_metadata(
        params, profiles, chi, sources, rho, r_grid, z_grid, dt, edge, backend="jax", gradient_tolerance=9.0e-4
    )
    with pytest.raises(ValueError, match="audit tolerance must match"):
        ds.differentiable_scenario_readiness_evidence(metadata, audit)


def test_scenario_metadata_records_grid_and_provenance() -> None:
    params, profiles, chi, sources, _, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    metadata = ds.scenario_campaign_metadata(
        params, profiles, chi, sources, rho, r_grid, z_grid, dt, edge, backend="jax", gradient_tolerance=5e-4
    )
    assert metadata.backend == "jax"
    assert metadata.n_rho == rho.size
    assert metadata.n_steps == sources.shape[0]
    assert metadata.flux_grid_shape == (z_grid.size, r_grid.size)
    assert metadata.equilibrium_param_count == 2
    assert len(metadata.inputs_sha256) == 64
    with pytest.raises(ValueError, match="backend must be"):
        ds.scenario_campaign_metadata(params, profiles, chi, sources, rho, r_grid, z_grid, dt, edge, backend="torch")


@pytest.mark.skipif(not ds.has_jax(), reason="JAX optional dependency is not installed")
def test_scenario_readiness_blocks_until_all_evidence_present() -> None:
    params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    audit = ds.assert_differentiable_scenario_gradient_consistent(
        params, profiles, chi, sources, target, rho, r_grid, z_grid, dt, edge
    )
    metadata = ds.scenario_campaign_metadata(
        params, profiles, chi, sources, rho, r_grid, z_grid, dt, edge, backend="jax", gradient_tolerance=5e-4
    )

    bounded = ds.differentiable_scenario_readiness_evidence(metadata, audit)
    assert bounded.claim_admissible is False
    assert "latency_evidence" in bounded.blocked_reasons
    assert "physics_traceability" in bounded.blocked_reasons
    with pytest.raises(ValueError, match="not ready"):
        ds.assert_scenario_claim_admissible(bounded)

    ready = ds.differentiable_scenario_readiness_evidence(metadata, audit, latency_p95_ms=2.5, traceability_passed=True)
    assert ready.claim_admissible is True
    assert ds.assert_scenario_claim_admissible(ready) is ready


def test_scenario_readiness_requires_tolerance_and_matching_audit() -> None:
    params, profiles, chi, sources, _, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    metadata_no_tol = ds.scenario_campaign_metadata(
        params, profiles, chi, sources, rho, r_grid, z_grid, dt, edge, backend="jax"
    )
    dummy_audit = ds.DifferentiableScenarioGradientAudit(
        loss=1e-6,
        epsilon=1e-5,
        tolerance=5e-4,
        checked_param_indices=(0, 1),
        checked_source_indices=((0, 0, 1),),
        param_max_abs_error=1e-7,
        source_max_abs_error=1e-7,
        passed=True,
    )
    with pytest.raises(ValueError, match="gradient_tolerance is required"):
        ds.differentiable_scenario_readiness_evidence(metadata_no_tol, dummy_audit)


def _passing_audit() -> ds.DifferentiableScenarioGradientAudit:
    return ds.DifferentiableScenarioGradientAudit(
        loss=1e-6,
        epsilon=1e-5,
        tolerance=5e-4,
        checked_param_indices=(0, 1),
        checked_source_indices=((0, 0, 1),),
        param_max_abs_error=1e-7,
        source_max_abs_error=1e-7,
        passed=True,
    )


def test_scenario_readiness_blocks_non_jax_backend_and_failed_audit() -> None:
    params, profiles, chi, sources, _, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    numpy_metadata = ds.scenario_campaign_metadata(
        params, profiles, chi, sources, rho, r_grid, z_grid, dt, edge, backend="numpy", gradient_tolerance=5e-4
    )
    failed_audit = ds.DifferentiableScenarioGradientAudit(
        loss=1e-6,
        epsilon=1e-5,
        tolerance=5e-4,
        checked_param_indices=(0, 1),
        checked_source_indices=((0, 0, 1),),
        param_max_abs_error=1.0,
        source_max_abs_error=1.0,
        passed=False,
    )
    evidence = ds.differentiable_scenario_readiness_evidence(numpy_metadata, failed_audit, latency_p95_ms=1.0)
    assert evidence.claim_admissible is False
    assert "jax_backend" in evidence.blocked_reasons
    assert "gradient_audit" in evidence.blocked_reasons


def test_scenario_readiness_rejects_negative_latency() -> None:
    params, profiles, chi, sources, _, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    metadata = ds.scenario_campaign_metadata(
        params, profiles, chi, sources, rho, r_grid, z_grid, dt, edge, backend="jax", gradient_tolerance=5e-4
    )
    with pytest.raises(ValueError, match="latency_p95_ms must be finite"):
        ds.differentiable_scenario_readiness_evidence(metadata, _passing_audit(), latency_p95_ms=-1.0)


def test_scenario_readiness_rejects_wrong_types() -> None:
    with pytest.raises(ValueError, match="metadata must be ScenarioCampaignMetadata"):
        ds.differentiable_scenario_readiness_evidence("x", _passing_audit())  # type: ignore[arg-type]
    params, profiles, chi, sources, _, rho, r_grid, z_grid, dt, edge = _scenario_setup()
    metadata = ds.scenario_campaign_metadata(
        params, profiles, chi, sources, rho, r_grid, z_grid, dt, edge, backend="jax", gradient_tolerance=5e-4
    )
    with pytest.raises(ValueError, match="audit must be DifferentiableScenarioGradientAudit"):
        ds.differentiable_scenario_readiness_evidence(metadata, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="evidence must be ScenarioReadinessEvidence"):
        ds.assert_scenario_claim_admissible(object())  # type: ignore[arg-type]
