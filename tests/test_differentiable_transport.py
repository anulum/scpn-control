# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport tests

from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pytest

from scpn_control.core import differentiable_transport as dt
from scpn_control.core.neural_transport import NeuralTransportModel, neural_transport_closure_profiles


def _profiles(rho: np.ndarray) -> np.ndarray:
    te = 8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2
    ti = 6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2
    ne = 4.0 + 0.8 * (1.0 - rho**2)
    nz = 0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02)
    return np.stack([te, ti, ne, nz])


def test_numpy_transport_step_diffuses_all_channels_and_applies_boundaries():
    rho = np.linspace(0.05, 1.0, 48)
    profiles = _profiles(rho)
    chi = np.stack(
        [
            0.22 + 0.03 * rho,
            0.18 + 0.02 * rho,
            0.05 + 0.01 * rho,
            0.015 + 0.002 * rho,
        ]
    )
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])

    stepped = dt.differentiable_transport_step(
        profiles,
        chi,
        sources,
        rho,
        2.0e-3,
        edge_values,
        use_jax=False,
    )

    assert stepped.shape == profiles.shape
    assert np.all(np.isfinite(stepped))
    assert np.max(stepped[0, 1:-1]) < np.max(profiles[0, 1:-1])
    assert np.max(stepped[1, 1:-1]) < np.max(profiles[1, 1:-1])
    np.testing.assert_allclose(stepped[:, -1], edge_values, atol=1e-12)
    np.testing.assert_allclose(stepped[:, 0], stepped[:, 1], atol=1e-12)


def test_transport_tracking_loss_is_zero_for_exact_next_step_target():
    rho = np.linspace(0.05, 1.0, 32)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    target = dt.differentiable_transport_step(
        profiles,
        chi,
        sources,
        rho,
        1.0e-3,
        edge_values,
        use_jax=False,
    )

    loss = dt.transport_tracking_loss(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        use_jax=False,
    )

    assert loss == pytest.approx(0.0, abs=1e-24)


def test_gradient_api_fails_closed_without_jax(monkeypatch):
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    target = profiles.copy()
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    monkeypatch.setattr(dt, "_HAS_JAX", False)

    with pytest.raises(RuntimeError, match="transport_loss_gradient requires JAX"):
        dt.transport_loss_gradient(profiles, chi, sources, target, rho, 1.0e-3, edge_values)


def test_transport_parameter_gradients_fail_closed_without_jax(monkeypatch):
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    target = profiles.copy()
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    monkeypatch.setattr(dt, "_HAS_JAX", False)

    with pytest.raises(RuntimeError, match="transport_parameter_gradients requires JAX"):
        dt.transport_parameter_gradients(profiles, chi, sources, target, rho, 1.0e-3, edge_values)


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_transport_loss_gradient_is_finite_with_jax():
    rho = np.linspace(0.05, 1.0, 24)
    profiles = _profiles(rho)
    chi = np.stack(
        [
            0.20 + 0.02 * rho,
            0.16 + 0.02 * rho,
            0.04 + 0.005 * rho,
            0.012 + 0.001 * rho,
        ]
    )
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    target = profiles.copy()
    target[0, 8:16] *= 0.97
    target[1, 8:16] *= 0.98
    weights = np.array([1.0, 1.0, 0.25, 0.1])

    loss, gradient = dt.transport_loss_gradient(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        weights=weights,
    )

    assert np.isfinite(loss)
    assert gradient.shape == chi.shape
    assert np.all(np.isfinite(gradient))
    assert np.any(np.abs(gradient) > 0.0)


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_transport_parameter_gradients_include_source_schedule_sensitivity():
    rho = np.linspace(0.05, 1.0, 24)
    profiles = _profiles(rho)
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
    target[0, 7:15] += 0.03
    target[2, 5:12] += 0.02
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])

    result = dt.transport_parameter_gradients(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        weights=np.array([1.0, 0.5, 0.25, 0.1]),
    )

    assert isinstance(result, dt.TransportParameterGradients)
    assert np.isfinite(result.loss)
    assert result.chi_gradient.shape == chi.shape
    assert result.source_gradient.shape == sources.shape
    assert np.all(np.isfinite(result.chi_gradient))
    assert np.all(np.isfinite(result.source_gradient))
    assert np.any(np.abs(result.chi_gradient) > 0.0)
    assert np.any(np.abs(result.source_gradient) > 0.0)


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_transport_parameter_gradient_audit_matches_finite_difference_contract():
    rho = np.linspace(0.05, 1.0, 21)
    profiles = _profiles(rho)
    chi = np.stack(
        [
            0.18 + 0.02 * rho,
            0.15 + 0.02 * rho,
            0.04 + 0.004 * rho,
            0.012 + 0.001 * rho,
        ]
    )
    sources = np.zeros_like(profiles)
    sources[0, 4:8] = 0.03
    sources[2, 7:11] = -0.01
    target = profiles.copy()
    target[0, 6:13] += 0.02
    target[1, 5:12] -= 0.015
    target[2, 4:10] += 0.01
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])

    audit = dt.assert_transport_parameter_gradients_consistent(
        profiles,
        chi,
        sources,
        target,
        rho,
        8.0e-4,
        edge_values,
        weights=np.array([1.0, 0.75, 0.25, 0.1]),
        epsilon=5.0e-5,
        tolerance=2.0e-3,
        sample_indices=((0, 5), (1, 10), (2, 7), (3, 12)),
    )

    assert isinstance(audit, dt.TransportGradientAudit)
    assert audit.passed
    assert audit.loss >= 0.0
    assert audit.checked_indices == ((0, 5), (1, 10), (2, 7), (3, 12))
    assert audit.chi_max_abs_error <= audit.tolerance
    assert audit.source_max_abs_error <= audit.tolerance


def test_transport_parameter_gradient_audit_rejects_invalid_admission_contract(monkeypatch):
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    target = profiles.copy()
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    monkeypatch.setattr(dt, "_HAS_JAX", False)

    with pytest.raises(ValueError, match="epsilon"):
        dt.audit_transport_parameter_gradients(profiles, chi, sources, target, rho, 1.0e-3, edge_values, epsilon=0.0)
    with pytest.raises(RuntimeError, match="transport_parameter_gradients requires JAX"):
        dt.audit_transport_parameter_gradients(profiles, chi, sources, target, rho, 1.0e-3, edge_values)


def test_equilibrium_weighted_transport_loss_uses_flux_radial_weight():
    rho = np.linspace(0.05, 1.0, 24)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    target = profiles.copy()
    target[:, 8:16] *= 0.98
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    psi = np.tile(np.linspace(0.2, 1.0, rho.size), (9, 1))

    uniform_loss = dt.transport_tracking_loss(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        use_jax=False,
    )
    weighted_loss = dt.equilibrium_weighted_transport_tracking_loss(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        psi,
        use_jax=False,
    )
    radial_weights = dt.equilibrium_radial_weights(psi, rho.size)

    assert weighted_loss != pytest.approx(uniform_loss)
    assert radial_weights.shape == (rho.size,)
    assert np.all(radial_weights > 0.0)
    assert np.mean(radial_weights) == pytest.approx(1.0)


def test_neural_transport_closure_maps_to_four_channel_coefficients():
    rho = np.linspace(0.05, 1.0, 24)
    profiles = _profiles(rho)
    closure = neural_transport_closure_profiles(
        rho,
        profiles[0],
        profiles[1],
        profiles[2],
        1.0 + 2.0 * rho**2,
        0.5 + 1.5 * rho,
        model=NeuralTransportModel(auto_discover=False),
    )

    chi = dt.transport_coefficients_from_neural_closure(
        closure,
        impurity_diffusivity_fraction=0.4,
        chi_floor=1.0e-6,
    )
    stepped = dt.differentiable_transport_step(
        profiles,
        chi,
        np.zeros_like(profiles),
        rho,
        1.0e-3,
        np.array([0.2, 0.2, 4.0, 0.03]),
        use_jax=False,
    )

    assert chi.shape == profiles.shape
    np.testing.assert_allclose(chi[0], np.maximum(closure.chi_e, 1.0e-6))
    np.testing.assert_allclose(chi[1], np.maximum(closure.chi_i, 1.0e-6))
    np.testing.assert_allclose(chi[2], np.maximum(closure.d_e, 1.0e-6))
    np.testing.assert_allclose(chi[3], np.maximum(0.4 * closure.d_e, 1.0e-6))
    assert np.all(np.isfinite(stepped))


def test_transport_campaign_metadata_records_numerical_contract_and_closure_provenance():
    rho = np.linspace(0.05, 1.0, 24)
    profiles = _profiles(rho)
    closure = neural_transport_closure_profiles(
        rho,
        profiles[0],
        profiles[1],
        profiles[2],
        1.0 + 2.0 * rho**2,
        0.5 + 1.5 * rho,
        model=NeuralTransportModel(auto_discover=False),
    )
    chi = dt.transport_coefficients_from_neural_closure(closure, impurity_diffusivity_fraction=0.25)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    equilibrium_psi = np.tile(np.linspace(0.2, 1.0, rho.size), (7, 1))

    metadata = dt.transport_campaign_metadata(
        profiles,
        chi,
        np.zeros_like(profiles),
        rho,
        1.0e-3,
        edge_values,
        backend="numpy",
        closure=closure,
        gradient_tolerance=1.0e-7,
        equilibrium_psi=equilibrium_psi,
    )

    assert isinstance(metadata, dt.TransportCampaignMetadata)
    assert metadata.backend == "numpy"
    assert metadata.dtype == "float64"
    assert metadata.channel_order == dt.CHANNELS
    assert metadata.n_rho == rho.size
    assert metadata.rho_min == pytest.approx(float(rho[0]))
    assert metadata.rho_max == pytest.approx(float(rho[-1]))
    assert metadata.rho_spacing == pytest.approx(float(rho[1] - rho[0]))
    assert metadata.dt == pytest.approx(1.0e-3)
    assert metadata.core_boundary == "zero_gradient"
    assert metadata.edge_boundary == "dirichlet"
    assert metadata.edge_values == tuple(float(x) for x in edge_values)
    assert metadata.closure_source == "analytic_fallback"
    assert metadata.closure_weights_checksum is None
    assert metadata.gradient_tolerance == pytest.approx(1.0e-7)
    assert metadata.equilibrium_grid_shape == (7, rho.size)
    assert asdict(metadata)["backend"] == "numpy"


def test_transport_campaign_metadata_rejects_invalid_backend_and_tolerance():
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])

    with pytest.raises(ValueError, match="backend"):
        dt.transport_campaign_metadata(profiles, chi, sources, rho, 1.0e-3, edge_values, backend="")
    with pytest.raises(ValueError, match="gradient_tolerance"):
        dt.transport_campaign_metadata(
            profiles,
            chi,
            sources,
            rho,
            1.0e-3,
            edge_values,
            backend="jax",
            gradient_tolerance=0.0,
        )


def test_transport_campaign_metadata_round_trips_through_json(tmp_path):
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    metadata = dt.transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        1.0e-3,
        edge_values,
        backend="jax",
        gradient_tolerance=1.0e-8,
    )
    path = tmp_path / "transport_campaign_metadata.json"

    dt.save_transport_campaign_metadata(metadata, path)
    loaded = dt.load_transport_campaign_metadata(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == metadata
    assert raw["schema_version"] == 1
    assert raw["metadata"]["backend"] == "jax"
    assert raw["metadata"]["channel_order"] == list(dt.CHANNELS)
    assert raw["metadata"]["gradient_tolerance"] == pytest.approx(1.0e-8)


def test_transport_campaign_metadata_replay_accepts_matching_candidate():
    rho = np.linspace(0.05, 1.0, 18)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    archived = dt.transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        2.0e-3,
        edge_values,
        backend="jax",
        gradient_tolerance=1.0e-8,
    )

    replay = dt.assert_transport_campaign_metadata_replay(
        archived,
        profiles,
        chi,
        sources,
        rho,
        2.0e-3,
        edge_values,
        backend="jax",
        gradient_tolerance=1.0e-8,
    )

    assert replay == archived


def test_transport_campaign_metadata_replay_rejects_grid_boundary_and_closure_drift():
    rho = np.linspace(0.05, 1.0, 18)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    closure = neural_transport_closure_profiles(
        rho,
        profiles[0],
        profiles[1],
        profiles[2],
        1.0 + 2.0 * rho**2,
        0.5 + 1.5 * rho,
        model=NeuralTransportModel(auto_discover=False),
    )
    archived = dt.transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        2.0e-3,
        edge_values,
        backend="jax",
        closure=closure,
        gradient_tolerance=1.0e-8,
    )

    shifted_grid = np.linspace(0.05, 0.95, 18)
    with pytest.raises(ValueError, match="rho_max"):
        dt.assert_transport_campaign_metadata_replay(
            archived,
            profiles,
            chi,
            sources,
            shifted_grid,
            2.0e-3,
            edge_values,
            backend="jax",
            closure=closure,
            gradient_tolerance=1.0e-8,
        )

    with pytest.raises(ValueError, match="edge_values"):
        dt.assert_transport_campaign_metadata_replay(
            archived,
            profiles,
            chi,
            sources,
            rho,
            2.0e-3,
            np.array([0.3, 0.2, 4.0, 0.03]),
            backend="jax",
            closure=closure,
            gradient_tolerance=1.0e-8,
        )

    with pytest.raises(ValueError, match="closure_source"):
        dt.assert_transport_campaign_metadata_replay(
            archived,
            profiles,
            chi,
            sources,
            rho,
            2.0e-3,
            edge_values,
            backend="jax",
            gradient_tolerance=1.0e-8,
        )


def test_transport_campaign_metadata_import_rejects_malformed_payload(tmp_path):
    path = tmp_path / "bad_transport_campaign_metadata.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {
                    "backend": "jax",
                    "dtype": "float64",
                    "channel_order": list(dt.CHANNELS),
                    "n_rho": 2,
                    "rho_min": 1.0,
                    "rho_max": 0.0,
                    "rho_spacing": -0.1,
                    "dt": 0.0,
                    "core_boundary": "zero_gradient",
                    "edge_boundary": "dirichlet",
                    "edge_values": [0.2, 0.2, 4.0, 0.03],
                    "closure_source": None,
                    "closure_weights_checksum": None,
                    "gradient_tolerance": 1.0e-8,
                    "equilibrium_grid_shape": None,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata"):
        dt.load_transport_campaign_metadata(path)


def test_equilibrium_weighted_gradient_fails_closed_without_jax(monkeypatch):
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    target = profiles.copy()
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    psi = np.tile(np.linspace(0.2, 1.0, rho.size), (7, 1))
    monkeypatch.setattr(dt, "_HAS_JAX", False)

    with pytest.raises(RuntimeError, match="equilibrium_weighted_transport_loss_gradient requires JAX"):
        dt.equilibrium_weighted_transport_loss_gradient(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            psi,
        )


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_equilibrium_weighted_transport_gradient_is_finite_with_jax_gs_flux():
    from scpn_control.core.jax_gs_solver import jax_gs_solve

    rho = np.linspace(0.05, 1.0, 17)
    profiles = _profiles(rho)
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
    target[0, 5:12] *= 0.96
    target[1, 5:12] *= 0.98
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    psi = jax_gs_solve(
        NR=17,
        NZ=17,
        Ip_target=1.0e6,
        n_picard=6,
        n_jacobi=12,
        use_jax=True,
    )

    result = dt.equilibrium_weighted_transport_loss_gradient(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge_values,
        psi,
        weights=np.array([1.0, 1.0, 0.25, 0.1]),
    )

    assert np.isfinite(result.loss)
    assert result.chi_gradient.shape == chi.shape
    assert result.equilibrium_gradient.shape == psi.shape
    assert np.all(np.isfinite(result.chi_gradient))
    assert np.all(np.isfinite(result.equilibrium_gradient))
    assert np.any(np.abs(result.chi_gradient) > 0.0)
    assert np.any(np.abs(result.equilibrium_gradient) > 0.0)
    assert np.mean(result.radial_weights) == pytest.approx(1.0)
