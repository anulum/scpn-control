# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport tests

from __future__ import annotations

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

import scpn_control.core.differentiable_transport as dt
from scpn_control.core.gyrokinetic_transport import GyrokineticTransportModel
from scpn_control.core.neural_transport import NeuralTransportModel, neural_transport_closure_profiles


def _profiles(rho: np.ndarray) -> np.ndarray:
    te = 8.0 * np.exp(-((rho - 0.35) ** 2) / 0.03) + 0.2
    ti = 6.0 * np.exp(-((rho - 0.42) ** 2) / 0.04) + 0.2
    ne = 4.0 + 0.8 * (1.0 - rho**2)
    nz = 0.03 + 0.02 * np.exp(-((rho - 0.65) ** 2) / 0.02)
    return np.stack([te, ti, ne, nz])


def _runtime_metadata() -> dt.TransportRuntimeMetadata:
    return dt.TransportRuntimeMetadata(
        schema_version=1,
        measured_at_unix_s=1_717_171_717.0,
        python_version="3.12.0",
        platform="Linux-test",
        machine="x86_64",
        processor="",
        jax_version="0.6.2",
        jaxlib_version="0.6.2",
        jax_default_backend="cpu",
        jax_devices=("TFRT_CPU_0",),
        jax_enable_x64=True,
    )


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


def test_numpy_transport_rollout_advances_source_schedule_with_boundaries():
    rho = np.linspace(0.05, 1.0, 24)
    profiles = _profiles(rho)
    chi = np.stack(
        [
            0.16 + 0.01 * rho,
            0.14 + 0.01 * rho,
            0.035 + 0.003 * rho,
            0.010 + 0.001 * rho,
        ]
    )
    source_sequence = np.zeros((4, dt.CHANNEL_COUNT, rho.size))
    source_sequence[:, 0, 6:11] = np.linspace(0.01, 0.04, 4)[:, None]
    source_sequence[:, 2, 4:9] = np.linspace(0.005, 0.02, 4)[:, None]
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])

    history = dt.differentiable_transport_rollout(
        profiles,
        chi,
        source_sequence,
        rho,
        7.5e-4,
        edge_values,
        use_jax=False,
    )
    no_source_history = dt.differentiable_transport_rollout(
        profiles,
        chi,
        np.zeros_like(source_sequence),
        rho,
        7.5e-4,
        edge_values,
        use_jax=False,
    )
    exact_loss = dt.transport_rollout_tracking_loss(
        profiles,
        chi,
        source_sequence,
        history,
        rho,
        7.5e-4,
        edge_values,
        weights=np.array([1.0, 0.75, 0.25, 0.1]),
        use_jax=False,
    )

    assert history.shape == source_sequence.shape
    assert np.all(np.isfinite(history))
    np.testing.assert_allclose(history[:, :, -1], np.tile(edge_values, (history.shape[0], 1)), atol=1e-12)
    np.testing.assert_allclose(history[:, :, 0], history[:, :, 1], atol=1e-12)
    assert exact_loss == pytest.approx(0.0, abs=1e-24)
    assert np.mean(history[-1, 0, 6:11] - no_source_history[-1, 0, 6:11]) > 0.0
    assert np.mean(history[-1, 2, 4:9] - no_source_history[-1, 2, 4:9]) > 0.0


def test_transport_rollout_rejects_malformed_source_history_and_grid():
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    source_sequence = np.zeros((3, dt.CHANNEL_COUNT, rho.size))
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])

    with pytest.raises(ValueError, match="source_sequence"):
        dt.differentiable_transport_rollout(profiles, chi, np.zeros_like(profiles), rho, 1.0e-3, edge_values)
    with pytest.raises(ValueError, match="target_history"):
        dt.transport_rollout_tracking_loss(
            profiles,
            chi,
            source_sequence,
            np.zeros((2, dt.CHANNEL_COUNT, rho.size)),
            rho,
            1.0e-3,
            edge_values,
            use_jax=False,
        )
    with pytest.raises(ValueError, match="uniform"):
        dt.differentiable_transport_rollout(
            profiles,
            chi,
            source_sequence,
            rho**1.2,
            1.0e-3,
            edge_values,
            use_jax=False,
        )


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


def test_transport_rollout_source_gradients_fail_closed_without_jax(monkeypatch):
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    source_sequence = np.zeros((3, dt.CHANNEL_COUNT, rho.size))
    target_history = source_sequence.copy()
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    monkeypatch.setattr(dt, "_HAS_JAX", False)

    with pytest.raises(RuntimeError, match="transport_rollout_source_gradients requires JAX"):
        dt.transport_rollout_source_gradients(
            profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            1.0e-3,
            edge_values,
        )


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
def test_transport_rollout_source_gradients_are_finite_with_jax():
    rho = np.linspace(0.05, 1.0, 18)
    profiles = _profiles(rho)
    chi = np.stack(
        [
            0.18 + 0.02 * rho,
            0.15 + 0.02 * rho,
            0.04 + 0.004 * rho,
            0.012 + 0.001 * rho,
        ]
    )
    source_sequence = np.zeros((4, dt.CHANNEL_COUNT, rho.size))
    source_sequence[:, 0, 5:10] = 0.01
    source_sequence[:, 2, 4:9] = 0.005
    target_history = np.asarray(
        dt.differentiable_transport_rollout(
            profiles,
            chi,
            source_sequence,
            rho,
            8.0e-4,
            np.array([0.2, 0.2, 4.0, 0.03]),
            use_jax=False,
        )
    )
    target_history[:, 0, 6:11] += 0.015
    target_history[:, 2, 5:10] += 0.006

    result = dt.transport_rollout_source_gradients(
        profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        8.0e-4,
        np.array([0.2, 0.2, 4.0, 0.03]),
        weights=np.array([1.0, 0.75, 0.25, 0.1]),
    )

    assert isinstance(result, dt.TransportRolloutSourceGradients)
    assert np.isfinite(result.loss)
    assert result.source_gradient.shape == source_sequence.shape
    assert result.final_profiles.shape == profiles.shape
    assert np.all(np.isfinite(result.source_gradient))
    assert np.all(np.isfinite(result.final_profiles))
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


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_transport_gradient_latency_report_times_audited_admission_path(tmp_path):
    rho = np.linspace(0.05, 1.0, 17)
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
    target[0, 6:12] += 0.02
    target[1, 5:11] -= 0.015
    target[2, 4:10] += 0.01
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])

    report = dt.benchmark_transport_parameter_gradient_latency(
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
        sample_indices=((0, 5), (1, 9), (2, 7), (3, 12)),
        warmup_runs=0,
        timed_runs=2,
    )
    path = tmp_path / "transport_gradient_latency.json"
    dt.save_transport_gradient_latency_report(report, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(report, dt.TransportGradientLatencyReport)
    assert report.audit.passed is True
    assert report.backend == "jax"
    assert report.n_rho == rho.size
    assert report.channel_count == dt.CHANNEL_COUNT
    assert report.timed_runs == 2
    assert report.p50_ms > 0.0
    assert report.p95_ms >= report.p50_ms
    assert report.max_ms >= report.p95_ms
    assert payload["audit"]["passed"] is True
    assert payload["claim_status"].startswith("local audited gradient-admission latency")


def test_transport_gradient_latency_report_rejects_invalid_run_counts():
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    sources = np.zeros_like(profiles)
    target = profiles.copy()
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])

    with pytest.raises(ValueError, match="timed_runs"):
        dt.benchmark_transport_parameter_gradient_latency(
            profiles,
            chi,
            sources,
            target,
            rho,
            1.0e-3,
            edge_values,
            timed_runs=0,
        )


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_transport_rollout_gradient_latency_report_times_audited_admission_path(tmp_path):
    rho = np.linspace(0.05, 1.0, 13)
    profiles = _profiles(rho)
    chi = np.stack(
        [
            0.16 + 0.01 * rho,
            0.14 + 0.01 * rho,
            0.035 + 0.003 * rho,
            0.01 + 0.001 * rho,
        ]
    )
    source_sequence = np.zeros((3, 4, rho.size), dtype=np.float64)
    source_sequence[:, 0, 3:7] = 0.02
    source_sequence[:, 2, 5:9] = -0.008
    desired_sources = source_sequence.copy()
    desired_sources[:, 0, 4:8] += 0.01
    desired_sources[:, 1, 3:7] -= 0.006
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    target_history = np.asarray(
        dt.differentiable_transport_rollout(
            profiles,
            chi,
            desired_sources,
            rho,
            7.0e-4,
            edge_values,
            use_jax=False,
        ),
        dtype=np.float64,
    )

    report = dt.benchmark_transport_rollout_source_gradient_latency(
        profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        7.0e-4,
        edge_values,
        weights=np.array([1.0, 0.75, 0.25, 0.1]),
        epsilon=5.0e-5,
        tolerance=2.0e-3,
        sample_indices=((0, 0, 4), (1, 1, 5), (2, 2, 7)),
        warmup_runs=0,
        timed_runs=2,
    )
    path = tmp_path / "transport_rollout_gradient_latency.json"
    dt.save_transport_rollout_gradient_latency_report(report, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(report, dt.TransportRolloutGradientLatencyReport)
    assert report.audit.passed is True
    assert report.backend == "jax"
    assert report.n_rho == rho.size
    assert report.n_steps == source_sequence.shape[0]
    assert report.channel_count == dt.CHANNEL_COUNT
    assert report.timed_runs == 2
    assert report.p50_ms > 0.0
    assert report.p95_ms >= report.p50_ms
    assert report.max_ms >= report.p95_ms
    assert report.audit.checked_indices == ((0, 0, 4), (1, 1, 5), (2, 2, 7))
    assert payload["audit"]["passed"] is True
    assert payload["claim_status"].startswith("local audited rollout source-gradient latency")


def test_transport_rollout_gradient_latency_report_rejects_invalid_run_counts():
    rho = np.linspace(0.05, 1.0, 8)
    profiles = _profiles(rho)
    chi = 0.04 * np.ones_like(profiles)
    source_sequence = np.zeros((2, 4, rho.size), dtype=np.float64)
    target_history = np.repeat(profiles[None, :, :], 2, axis=0)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])

    with pytest.raises(ValueError, match="timed_runs"):
        dt.benchmark_transport_rollout_source_gradient_latency(
            profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            1.0e-3,
            edge_values,
            timed_runs=0,
        )

    with pytest.raises(ValueError, match="sample_indices"):
        dt.benchmark_transport_rollout_source_gradient_latency(
            profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            1.0e-3,
            edge_values,
            sample_indices=((0, 0),),
        )


def test_equilibrium_weighted_transport_rollout_loss_uses_flux_radial_weight():
    rho = np.linspace(0.05, 1.0, 9)
    profiles = _profiles(rho)
    chi = 0.03 * np.ones_like(profiles)
    source_sequence = np.zeros((3, 4, rho.size), dtype=np.float64)
    source_sequence[:, 0, 2:5] = 0.02
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    target_history = np.asarray(
        dt.differentiable_transport_rollout(
            profiles,
            chi,
            source_sequence,
            rho,
            8.0e-4,
            edge_values,
            use_jax=False,
        ),
        dtype=np.float64,
    )
    target_history[:, 0, -3:] += 0.04
    psi = np.tile(np.linspace(0.2, 3.0, rho.size), (rho.size, 1))

    weighted_loss = dt.equilibrium_weighted_transport_rollout_tracking_loss(
        profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        8.0e-4,
        edge_values,
        psi,
        weights=np.array([1.0, 0.5, 0.25, 0.1]),
        use_jax=False,
    )
    history = dt.differentiable_transport_rollout(
        profiles,
        chi,
        source_sequence,
        rho,
        8.0e-4,
        edge_values,
        use_jax=False,
    )
    residual = history - target_history
    radial_weights = dt.equilibrium_radial_weights(psi, rho.size)
    expected = np.mean(np.array([1.0, 0.5, 0.25, 0.1])[None, :, None] * radial_weights[None, None, :] * residual**2)

    assert weighted_loss == pytest.approx(expected)
    assert radial_weights[-1] > radial_weights[0]


def test_equilibrium_weighted_rollout_gradient_fails_closed_without_jax(monkeypatch):
    rho = np.linspace(0.05, 1.0, 8)
    profiles = _profiles(rho)
    chi = 0.03 * np.ones_like(profiles)
    source_sequence = np.zeros((2, 4, rho.size), dtype=np.float64)
    target_history = np.repeat(profiles[None, :, :], 2, axis=0)
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    psi = np.tile(np.linspace(0.2, 2.0, rho.size), (rho.size, 1))
    monkeypatch.setattr(dt, "_HAS_JAX", False)
    monkeypatch.setattr(dt, "jax", None)
    monkeypatch.setattr(dt, "jnp", None)

    with pytest.raises(RuntimeError, match="equilibrium_weighted_transport_rollout_source_gradient requires JAX"):
        dt.equilibrium_weighted_transport_rollout_source_gradient(
            profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            8.0e-4,
            edge_values,
            psi,
        )


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_equilibrium_weighted_transport_rollout_gradient_is_finite_with_jax_gs_flux():
    rho = np.linspace(0.05, 1.0, 8)
    profiles = _profiles(rho)
    chi = 0.03 * np.ones_like(profiles)
    source_sequence = np.zeros((3, 4, rho.size), dtype=np.float64)
    source_sequence[:, 0, 2:5] = 0.02
    edge_values = np.array([0.2, 0.2, 4.0, 0.03])
    desired_sources = source_sequence.copy()
    desired_sources[:, 0, 2:5] += 0.01
    target_history = np.asarray(
        dt.differentiable_transport_rollout(
            profiles,
            chi,
            desired_sources,
            rho,
            8.0e-4,
            edge_values,
            use_jax=False,
        ),
        dtype=np.float64,
    )
    psi = np.tile(np.linspace(0.2, 2.0, rho.size), (rho.size, 1))

    result = dt.equilibrium_weighted_transport_rollout_source_gradient(
        profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        8.0e-4,
        edge_values,
        psi,
        weights=np.array([1.0, 0.75, 0.25, 0.1]),
    )

    assert isinstance(result, dt.EquilibriumWeightedTransportRolloutGradient)
    assert np.isfinite(result.loss)
    assert result.source_gradient.shape == source_sequence.shape
    assert result.equilibrium_gradient.shape == psi.shape
    assert result.radial_weights.shape == rho.shape
    assert result.final_profiles.shape == profiles.shape
    assert np.all(np.isfinite(result.source_gradient))
    assert np.all(np.isfinite(result.equilibrium_gradient))


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


def test_gyrokinetic_transport_closure_maps_to_four_channel_coefficients():
    rho = np.linspace(0.05, 1.0, 16)
    profiles = _profiles(rho)
    gk_profiles = {
        "R0": 2.0,
        "a": 0.5,
        "B0": 5.0,
        "q": 1.2 + 1.5 * rho,
        "s_hat": 0.4 + 1.2 * rho,
        "Te": profiles[0],
        "Ti": profiles[1],
        "ne": profiles[2],
        "dTe_dr": np.gradient(profiles[0], rho),
        "dTi_dr": np.gradient(profiles[1], rho),
        "dne_dr": np.gradient(profiles[2], rho),
        "nu_star": np.full(rho.shape, 0.1),
        "beta_e": np.full(rho.shape, 0.01),
        "alpha_MHD": np.zeros_like(rho),
        "Z_eff": np.full(rho.shape, 1.5),
    }
    closure = dt.gyrokinetic_transport_closure_profiles(
        GyrokineticTransportModel(n_modes=4),
        rho,
        gk_profiles,
    )

    chi = dt.transport_coefficients_from_gyrokinetic_closure(
        closure,
        impurity_diffusivity_fraction=0.35,
        chi_floor=1.0e-7,
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

    assert closure.source == "reduced_gyrokinetic"
    assert closure.weights_checksum is None
    assert closure.channel_weights.shape == (3, rho.size)
    np.testing.assert_allclose(closure.channel_weights.sum(axis=0), 1.0)
    assert chi.shape == profiles.shape
    np.testing.assert_allclose(chi[0], np.maximum(closure.chi_e, 1.0e-7))
    np.testing.assert_allclose(chi[1], np.maximum(closure.chi_i, 1.0e-7))
    np.testing.assert_allclose(chi[2], np.maximum(closure.d_e, 1.0e-7))
    np.testing.assert_allclose(chi[3], np.maximum(0.35 * closure.d_e, 1.0e-7))
    assert np.all(np.isfinite(stepped))


def test_gyrokinetic_transport_closure_rejects_bad_model_contract():
    rho = np.linspace(0.05, 1.0, 8)
    with pytest.raises(ValueError, match="evaluate_profile"):
        dt.gyrokinetic_transport_closure_profiles(object(), rho, {})

    class BadModel:
        def evaluate_profile(self, rho, profiles):
            return np.ones(rho.size), np.ones(rho.size - 1), np.ones(rho.size)

    with pytest.raises(ValueError, match="match rho shape"):
        dt.gyrokinetic_transport_closure_profiles(BadModel(), rho, {})


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


def test_transport_campaign_metadata_import_rejects_unreadable_and_invalid_schema(tmp_path):
    not_json = tmp_path / "not_transport_metadata.json"
    not_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="readable JSON"):
        dt.load_transport_campaign_metadata(not_json)

    bad_schema = tmp_path / "bad_schema_transport_metadata.json"
    bad_schema.write_text(json.dumps({"schema_version": 99, "metadata": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        dt.load_transport_campaign_metadata(bad_schema)

    bad_payload = tmp_path / "bad_payload_transport_metadata.json"
    bad_payload.write_text(json.dumps({"schema_version": 1, "metadata": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="payload"):
        dt.load_transport_campaign_metadata(bad_payload)


def test_transport_differentiability_evidence_rejects_missing_tolerance_and_bad_controller_digest():
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
        gradient_tolerance=None,
    )
    audit = dt.TransportGradientAudit(
        loss=0.0,
        epsilon=1.0e-5,
        tolerance=1.0e-6,
        checked_indices=((0, 1),),
        chi_max_abs_error=0.0,
        source_max_abs_error=0.0,
        passed=True,
    )

    with pytest.raises(ValueError, match="gradient_tolerance"):
        dt.transport_differentiability_evidence(metadata, audit)

    metadata_with_tolerance = dt.transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        1.0e-3,
        edge_values,
        backend="jax",
        gradient_tolerance=1.0e-6,
    )
    with pytest.raises(ValueError, match="controller_formal_artifact_sha256"):
        dt.transport_differentiability_evidence(
            metadata_with_tolerance,
            audit,
            controller_formal_artifact_sha256="not-a-digest",
        )


def test_transport_latency_report_persistence_rejects_invalid_percentile_contract(tmp_path):
    audit = dt.TransportGradientAudit(
        loss=0.0,
        epsilon=1.0e-5,
        tolerance=1.0e-6,
        checked_indices=((0, 1),),
        chi_max_abs_error=0.0,
        source_max_abs_error=0.0,
        passed=True,
    )
    report = dt.TransportGradientLatencyReport(
        schema_version=1,
        backend="jax",
        dtype="float64",
        n_rho=16,
        channel_count=dt.CHANNEL_COUNT,
        warmup_runs=0,
        timed_runs=2,
        p50_ms=3.0,
        p95_ms=2.0,
        max_ms=4.0,
        runtime_metadata=_runtime_metadata(),
        audit=audit,
        claim_status="local audited gradient-admission latency only; not a real-time control-loop guarantee",
    )

    with pytest.raises(ValueError, match="percentiles"):
        dt.save_transport_gradient_latency_report(report, tmp_path / "bad_latency.json")


def test_transport_rollout_latency_report_persistence_rejects_invalid_audit_indices(tmp_path):
    audit = dt.TransportRolloutGradientAudit(
        loss=0.0,
        epsilon=1.0e-5,
        tolerance=1.0e-6,
        checked_indices=((0, 0, 99),),
        source_max_abs_error=0.0,
        passed=True,
    )
    report = dt.TransportRolloutGradientLatencyReport(
        schema_version=1,
        backend="jax",
        dtype="float64",
        n_rho=16,
        n_steps=2,
        channel_count=dt.CHANNEL_COUNT,
        warmup_runs=0,
        timed_runs=2,
        p50_ms=1.0,
        p95_ms=2.0,
        max_ms=3.0,
        runtime_metadata=_runtime_metadata(),
        audit=audit,
        claim_status="local audited rollout source-gradient latency only; not a real-time control-loop guarantee",
    )

    with pytest.raises(ValueError, match="checked_indices"):
        dt.save_transport_rollout_gradient_latency_report(report, tmp_path / "bad_rollout_latency.json")


def test_transport_differentiability_evidence_binds_campaign_audit_and_controller_proof():
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
        gradient_tolerance=1.0e-6,
        equilibrium_psi=np.tile(np.linspace(0.2, 1.0, rho.size), (5, 1)),
    )
    audit = dt.TransportRolloutGradientAudit(
        loss=0.125,
        epsilon=1.0e-5,
        tolerance=1.0e-6,
        checked_indices=((0, 0, 1),),
        source_max_abs_error=2.0e-7,
        passed=True,
    )

    evidence = dt.transport_differentiability_evidence(
        metadata,
        audit,
        controller_formal_artifact_sha256="b" * 64,
    )

    assert evidence.backend == "jax"
    assert evidence.equilibrium_coupled
    assert len(evidence.campaign_sha256) == 64
    assert len(evidence.gradient_audit_sha256) == 64
    dt.assert_transport_differentiability_claim_admissible(evidence, metadata, audit)


def test_transport_differentiability_evidence_rejects_tampering_and_failed_audit():
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
        gradient_tolerance=1.0e-6,
    )
    audit = dt.TransportRolloutGradientAudit(
        loss=0.125,
        epsilon=1.0e-5,
        tolerance=1.0e-6,
        checked_indices=((0, 0, 1),),
        source_max_abs_error=2.0e-7,
        passed=True,
    )
    evidence = dt.transport_differentiability_evidence(metadata, audit)
    tampered = dt.transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        2.0e-3,
        edge_values,
        backend="jax",
        gradient_tolerance=1.0e-6,
    )
    failed_audit = dt.TransportRolloutGradientAudit(
        loss=audit.loss,
        epsilon=audit.epsilon,
        tolerance=audit.tolerance,
        checked_indices=audit.checked_indices,
        source_max_abs_error=2.0e-6,
        passed=False,
    )

    with pytest.raises(ValueError, match="campaign_sha256"):
        dt.assert_transport_differentiability_claim_admissible(evidence, tampered, audit)
    with pytest.raises(ValueError, match="passed audit"):
        dt.assert_transport_differentiability_claim_admissible(evidence, metadata, failed_audit)


def test_transport_differentiability_evidence_rejects_malformed_audit_semantics():
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
        gradient_tolerance=1.0e-6,
    )

    with pytest.raises(ValueError, match="tolerance"):
        dt.transport_differentiability_evidence(
            metadata,
            dt.TransportRolloutGradientAudit(
                loss=0.125,
                epsilon=1.0e-5,
                tolerance=2.0e-6,
                checked_indices=((0, 0, 1),),
                source_max_abs_error=2.0e-7,
                passed=True,
            ),
        )

    with pytest.raises(ValueError, match="unique"):
        dt.transport_differentiability_evidence(
            metadata,
            dt.TransportRolloutGradientAudit(
                loss=0.125,
                epsilon=1.0e-5,
                tolerance=1.0e-6,
                checked_indices=((0, 0, 1), (0, 0, 1)),
                source_max_abs_error=2.0e-7,
                passed=True,
            ),
        )

    with pytest.raises(ValueError, match="out of campaign bounds"):
        dt.transport_differentiability_evidence(
            metadata,
            dt.TransportGradientAudit(
                loss=0.125,
                epsilon=1.0e-5,
                tolerance=1.0e-6,
                checked_indices=((0, rho.size),),
                chi_max_abs_error=2.0e-7,
                source_max_abs_error=2.0e-7,
                passed=True,
            ),
        )

    with pytest.raises(ValueError, match="passed flag"):
        dt.transport_differentiability_evidence(
            metadata,
            dt.TransportRolloutGradientAudit(
                loss=0.125,
                epsilon=1.0e-5,
                tolerance=1.0e-6,
                checked_indices=((0, 0, 1),),
                source_max_abs_error=2.0e-7,
                passed=False,
            ),
        )


def test_transport_latency_report_persistence_rejects_malformed_timing_and_audit(tmp_path):
    audit = dt.TransportGradientAudit(
        loss=0.125,
        epsilon=1.0e-5,
        tolerance=1.0e-6,
        checked_indices=((0, 1),),
        chi_max_abs_error=2.0e-7,
        source_max_abs_error=2.0e-7,
        passed=True,
    )
    good = dt.TransportGradientLatencyReport(
        schema_version=1,
        backend="jax",
        dtype="float64",
        n_rho=16,
        channel_count=dt.CHANNEL_COUNT,
        warmup_runs=0,
        timed_runs=1,
        p50_ms=1.0,
        p95_ms=1.2,
        max_ms=1.4,
        runtime_metadata=_runtime_metadata(),
        audit=audit,
        claim_status="local audited gradient-admission latency only",
    )

    dt.save_transport_gradient_latency_report(good, tmp_path / "good.json")

    bad_timing = dt.TransportGradientLatencyReport(
        schema_version=1,
        backend="jax",
        dtype="float64",
        n_rho=16,
        channel_count=dt.CHANNEL_COUNT,
        warmup_runs=0,
        timed_runs=1,
        p50_ms=1.3,
        p95_ms=1.2,
        max_ms=1.4,
        runtime_metadata=_runtime_metadata(),
        audit=audit,
        claim_status="local audited gradient-admission latency only",
    )
    with pytest.raises(ValueError, match="p50 <= p95 <= max"):
        dt.save_transport_gradient_latency_report(bad_timing, tmp_path / "bad_timing.json")

    bad_audit = dt.TransportGradientLatencyReport(
        schema_version=1,
        backend="jax",
        dtype="float64",
        n_rho=16,
        channel_count=dt.CHANNEL_COUNT,
        warmup_runs=0,
        timed_runs=1,
        p50_ms=1.0,
        p95_ms=1.2,
        max_ms=1.4,
        runtime_metadata=_runtime_metadata(),
        audit=dt.TransportGradientAudit(
            loss=0.125,
            epsilon=1.0e-5,
            tolerance=1.0e-6,
            checked_indices=((0, 20),),
            chi_max_abs_error=2.0e-7,
            source_max_abs_error=2.0e-7,
            passed=True,
        ),
        claim_status="local audited gradient-admission latency only",
    )
    with pytest.raises(ValueError, match="out of campaign bounds"):
        dt.save_transport_gradient_latency_report(bad_audit, tmp_path / "bad_audit.json")


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


def _transport_readiness_fixture():
    metadata = dt.TransportCampaignMetadata(
        backend="jax",
        dtype="float64",
        channel_order=dt.CHANNELS,
        n_rho=17,
        rho_min=0.0,
        rho_max=1.0,
        rho_spacing=1.0 / 16.0,
        dt=1.0e-3,
        core_boundary="zero_gradient",
        edge_boundary="dirichlet",
        edge_values=(0.2, 0.2, 4.0, 0.03),
        closure_source="validated-gk-profile",
        closure_weights_checksum="c" * 64,
        gradient_tolerance=5.0e-4,
        equilibrium_grid_shape=(17, 17),
    )
    gradient_audit = dt.TransportGradientAudit(
        loss=1.0e-3,
        epsilon=1.0e-5,
        tolerance=5.0e-4,
        checked_indices=((0, 1), (1, 8), (2, 15), (3, 8)),
        chi_max_abs_error=1.0e-7,
        source_max_abs_error=2.0e-7,
        passed=True,
    )
    gradient_report = dt.TransportGradientLatencyReport(
        schema_version=1,
        backend="jax",
        dtype="float64",
        n_rho=17,
        channel_count=dt.CHANNEL_COUNT,
        warmup_runs=1,
        timed_runs=3,
        p50_ms=3.0,
        p95_ms=4.0,
        max_ms=4.5,
        runtime_metadata=_runtime_metadata(),
        audit=gradient_audit,
        claim_status="local audited gradient-admission latency only",
    )
    rollout_audit = dt.TransportRolloutGradientAudit(
        loss=2.0e-3,
        epsilon=1.0e-5,
        tolerance=5.0e-4,
        checked_indices=((0, 0, 1), (1, 1, 8), (2, 2, 15), (2, 3, 8)),
        source_max_abs_error=3.0e-7,
        passed=True,
    )
    rollout_report = dt.TransportRolloutGradientLatencyReport(
        schema_version=1,
        backend="jax",
        dtype="float64",
        n_rho=17,
        n_steps=3,
        channel_count=dt.CHANNEL_COUNT,
        warmup_runs=1,
        timed_runs=3,
        p50_ms=6.0,
        p95_ms=8.0,
        max_ms=9.0,
        runtime_metadata=_runtime_metadata(),
        audit=rollout_audit,
        claim_status="local audited rollout source-gradient latency only",
    )
    return metadata, gradient_report, rollout_report


def test_transport_full_fidelity_readiness_fails_closed_without_external_admission():
    metadata, gradient_report, rollout_report = _transport_readiness_fixture()

    evidence = dt.transport_full_fidelity_readiness_evidence(
        metadata,
        gradient_report,
        rollout_report=rollout_report,
        controller_formal_artifact_sha256="a" * 64,
    )

    assert not evidence.full_fidelity_claim_admissible
    assert "external_reference_artifact_sha256" in evidence.blocked_reasons
    with pytest.raises(ValueError, match="external reference"):
        dt.assert_transport_full_fidelity_claim_ready(
            evidence,
            metadata,
            gradient_report,
            rollout_report=rollout_report,
        )


def test_transport_full_fidelity_readiness_binds_reports_digests_and_controller_proof():
    from dataclasses import replace

    metadata, gradient_report, rollout_report = _transport_readiness_fixture()

    evidence = dt.transport_full_fidelity_readiness_evidence(
        metadata,
        gradient_report,
        rollout_report=rollout_report,
        external_reference_artifact_sha256="b" * 64,
        external_reference_admitted=True,
        controller_formal_artifact_sha256="a" * 64,
    )

    assert evidence.full_fidelity_claim_admissible
    assert evidence.rollout_steps == 3
    assert evidence.gradient_latency_report_sha256
    assert evidence.rollout_latency_report_sha256
    assert (
        dt.assert_transport_full_fidelity_claim_ready(
            evidence,
            metadata,
            gradient_report,
            rollout_report=rollout_report,
        )
        is evidence
    )

    tampered_report = replace(gradient_report, n_rho=19)
    with pytest.raises(ValueError, match="campaign metadata"):
        dt.transport_full_fidelity_readiness_evidence(
            metadata,
            tampered_report,
            rollout_report=rollout_report,
            external_reference_artifact_sha256="b" * 64,
            external_reference_admitted=True,
            controller_formal_artifact_sha256="a" * 64,
        )

    with pytest.raises(ValueError, match="SHA-256"):
        dt.transport_full_fidelity_readiness_evidence(
            metadata,
            gradient_report,
            rollout_report=rollout_report,
            external_reference_artifact_sha256="not-a-digest",
            external_reference_admitted=True,
            controller_formal_artifact_sha256="a" * 64,
        )


# --------------------------------------------------------------------------- #
# Input-validation contracts                                                   #
# --------------------------------------------------------------------------- #


def _uniform_rho(n: int = 12) -> np.ndarray:
    return np.linspace(0.05, 1.0, n)


def _valid_chi(rho: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            0.20 + 0.02 * rho,
            0.16 + 0.02 * rho,
            0.04 + 0.005 * rho,
            0.012 + 0.001 * rho,
        ]
    )


def _one_step_setup(n: int = 12):
    rho = _uniform_rho(n)
    profiles = _profiles(rho)
    chi = _valid_chi(rho)
    sources = np.zeros_like(profiles)
    edge = np.array([0.2, 0.2, 4.0, 0.03])
    return profiles, chi, sources, rho, 1.0e-3, edge


def _rollout_setup(n: int = 12, n_steps: int = 3):
    profiles, chi, _, rho, dt_value, edge = _one_step_setup(n)
    source_sequence = np.zeros((n_steps, dt.CHANNEL_COUNT, rho.size))
    source_sequence[:, 0, 3:7] = 0.01
    source_sequence[:, 2, 2:6] = 0.004
    target_history = np.asarray(
        dt.differentiable_transport_rollout(profiles, chi, source_sequence, rho, dt_value, edge, use_jax=False)
    )
    target_history[:, 0, 4:8] += 0.01
    return profiles, chi, source_sequence, target_history, rho, dt_value, edge


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("profiles_1d", "profiles must have shape"),
        ("profiles_wrong_channels", "profiles must have shape"),
        ("profiles_nan", "must contain only finite values"),
        ("chi_shape", "chi must match profiles shape"),
        ("chi_negative", "chi must be non-negative"),
        ("sources_shape", "sources must match profiles shape"),
        ("rho_length", "same radial length"),
        ("rho_2d", "one-dimensional"),
        ("rho_not_increasing", "strictly increasing"),
        ("rho_non_uniform", "uniform normalised radial spacing"),
        ("edge_shape", "edge_values must have shape"),
        ("dt_zero", "dt must be positive and finite"),
        ("dt_inf", "dt must be positive and finite"),
    ],
)
def test_differentiable_transport_step_rejects_invalid_inputs(mutation, match):
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup()
    if mutation == "profiles_1d":
        profiles = profiles[0]
    elif mutation == "profiles_wrong_channels":
        profiles = profiles[:3]
    elif mutation == "profiles_nan":
        profiles = profiles.copy()
        profiles[0, 0] = np.nan
    elif mutation == "chi_shape":
        chi = chi[:, :-1]
    elif mutation == "chi_negative":
        chi = chi.copy()
        chi[0, 0] = -1.0
    elif mutation == "sources_shape":
        sources = sources[:, :-1]
    elif mutation == "rho_length":
        rho = rho[:-1]
    elif mutation == "rho_2d":
        rho = rho.reshape(2, 6)
    elif mutation == "rho_not_increasing":
        rho = rho.copy()
        rho[5] = rho[4] - 0.01
    elif mutation == "rho_non_uniform":
        rho = rho.copy()
        rho[6] += 0.002
    elif mutation == "edge_shape":
        edge = edge[:3]
    elif mutation == "dt_zero":
        dt_value = 0.0
    elif mutation == "dt_inf":
        dt_value = np.inf

    with pytest.raises(ValueError, match=match):
        dt.differentiable_transport_step(profiles, chi, sources, rho, dt_value, edge, use_jax=False)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("target_shape", "target_profiles must match profiles shape"),
        ("target_none", "target_profiles is required"),
        ("weights_shape", "weights must have shape"),
        ("weights_negative", "weights must be non-negative"),
    ],
)
def test_transport_tracking_loss_rejects_invalid_target_and_weights(mutation, match):
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup()
    target = profiles.copy()
    weights = None
    if mutation == "target_shape":
        target = target[:, :-1]
    elif mutation == "target_none":
        target = None
    elif mutation == "weights_shape":
        weights = np.ones(3)
    elif mutation == "weights_negative":
        weights = np.array([-1.0, 1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match=match):
        dt.transport_tracking_loss(profiles, chi, sources, target, rho, dt_value, edge, weights=weights, use_jax=False)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("profiles_1d", "initial_profiles must have shape"),
        ("chi_shape", "chi must match initial_profiles shape"),
        ("rho_length", "same radial length"),
        ("edge_shape", "edge_values must have shape"),
        ("dt_zero", "dt must be positive and finite"),
        ("chi_negative", "chi must be non-negative"),
        ("rho_not_increasing", "strictly increasing"),
    ],
)
def test_differentiable_transport_rollout_rejects_invalid_inputs(mutation, match):
    profiles, chi, source_sequence, _, rho, dt_value, edge = _rollout_setup()
    if mutation == "profiles_1d":
        profiles = profiles[0]
    elif mutation == "chi_shape":
        chi = chi[:, :-1]
    elif mutation == "rho_length":
        rho = rho[:-1]
    elif mutation == "edge_shape":
        edge = edge[:3]
    elif mutation == "dt_zero":
        dt_value = 0.0
    elif mutation == "chi_negative":
        chi = chi.copy()
        chi[1, 2] = -0.5
    elif mutation == "rho_not_increasing":
        rho = rho.copy()
        rho[5] = rho[4] - 0.01

    with pytest.raises(ValueError, match=match):
        dt.differentiable_transport_rollout(profiles, chi, source_sequence, rho, dt_value, edge, use_jax=False)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("target_none", "target_history is required"),
        ("weights_shape", "weights must have shape"),
        ("weights_negative", "weights must be non-negative"),
    ],
)
def test_transport_rollout_tracking_loss_rejects_invalid_target_and_weights(mutation, match):
    profiles, chi, source_sequence, target_history, rho, dt_value, edge = _rollout_setup()
    weights = None
    if mutation == "target_none":
        target_history = None
    elif mutation == "weights_shape":
        weights = np.ones(3)
    elif mutation == "weights_negative":
        weights = np.array([1.0, -1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match=match):
        dt.transport_rollout_tracking_loss(
            profiles,
            chi,
            source_sequence,
            target_history,
            rho,
            dt_value,
            edge,
            weights=weights,
            use_jax=False,
        )


def test_transport_rollout_tracking_loss_defaults_unit_weights_numpy():
    profiles, chi, source_sequence, target_history, rho, dt_value, edge = _rollout_setup()
    loss = dt.transport_rollout_tracking_loss(
        profiles, chi, source_sequence, target_history, rho, dt_value, edge, use_jax=False
    )
    assert np.isfinite(loss)
    assert loss > 0.0


# --------------------------------------------------------------------------- #
# Closure adapter validation                                                   #
# --------------------------------------------------------------------------- #


def _closure_ns(chi_e, chi_i, d_e, channel_weights, *, source="neural", checksum=None):
    return SimpleNamespace(
        chi_e=np.asarray(chi_e, dtype=float),
        chi_i=np.asarray(chi_i, dtype=float),
        d_e=np.asarray(d_e, dtype=float),
        channel_weights=np.asarray(channel_weights, dtype=float),
        source=source,
        weights_checksum=checksum,
    )


def _valid_closure(n: int = 5):
    profile = np.linspace(0.5, 1.5, n)
    weights = np.full((3, n), 1.0 / 3.0)
    return _closure_ns(profile, profile, profile, weights)


@pytest.mark.parametrize(
    ("kwargs", "channel_override", "match"),
    [
        ({"impurity_diffusivity_fraction": 1.5}, None, "impurity_diffusivity_fraction"),
        ({"impurity_diffusivity_fraction": -0.1}, None, "impurity_diffusivity_fraction"),
        ({"chi_floor": -1.0e-6}, None, "chi_floor"),
    ],
)
def test_neural_closure_coefficients_reject_invalid_scalars(kwargs, channel_override, match):
    closure = _valid_closure()
    with pytest.raises(ValueError, match=match):
        dt.transport_coefficients_from_neural_closure(closure, **kwargs)


def test_neural_closure_coefficients_reject_short_profile():
    closure = _closure_ns([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], np.full((3, 2), 1.0 / 3.0))
    with pytest.raises(ValueError, match="at least three points"):
        dt.transport_coefficients_from_neural_closure(closure)


def test_neural_closure_coefficients_reject_negative_profile():
    closure = _closure_ns([-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], np.full((3, 3), 1.0 / 3.0))
    with pytest.raises(ValueError, match="must be non-negative"):
        dt.transport_coefficients_from_neural_closure(closure)


def test_neural_closure_coefficients_reject_mismatched_profile_shapes():
    closure = _closure_ns([1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0], np.full((3, 3), 1.0 / 3.0))
    with pytest.raises(ValueError, match="same shape"):
        dt.transport_coefficients_from_neural_closure(closure)


@pytest.mark.parametrize(
    ("channel_weights", "match"),
    [
        (np.full((2, 5), 0.5), "shape"),
        (np.array([[-1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]), "non-negative"),
        (np.full((3, 5), 0.5), "sum to one"),
    ],
)
def test_neural_closure_coefficients_reject_bad_channel_weights(channel_weights, match):
    profile = np.linspace(0.5, 1.5, 5)
    closure = _closure_ns(profile, profile, profile, channel_weights)
    with pytest.raises(ValueError, match=match):
        dt.transport_coefficients_from_neural_closure(closure)


@pytest.mark.parametrize(
    ("rho", "match"),
    [
        (np.array([0.1, 0.2]), "at least three points"),
        (np.array([0.1, 0.3, 0.2, 0.4]), "strictly increasing"),
    ],
)
def test_gyrokinetic_closure_rejects_bad_rho(rho, match):
    with pytest.raises(ValueError, match=match):
        dt.gyrokinetic_transport_closure_profiles(object(), rho, {})


def test_gyrokinetic_closure_uses_uniform_weights_for_degenerate_total():
    rho = np.linspace(0.1, 0.9, 6)

    class _ZeroModel:
        def evaluate_profile(self, rho, profiles):
            zero = np.zeros(rho.size)
            return zero, zero, zero

    closure = dt.gyrokinetic_transport_closure_profiles(_ZeroModel(), rho, {})
    np.testing.assert_allclose(closure.channel_weights, 1.0 / 3.0)
    np.testing.assert_allclose(closure.channel_weights.sum(axis=0), 1.0)


# --------------------------------------------------------------------------- #
# Campaign metadata mapping validation                                         #
# --------------------------------------------------------------------------- #


def _valid_metadata_payload():
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup()
    psi = np.outer(np.linspace(1.0, 0.2, 5), np.linspace(1.0, 0.3, rho.size))
    metadata = dt.transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        dt_value,
        edge,
        backend="jax",
        gradient_tolerance=5.0e-4,
        equilibrium_psi=psi,
    )
    return asdict(metadata)


def test_campaign_metadata_mapping_accepts_none_tolerance_and_equilibrium():
    payload = _valid_metadata_payload()
    payload["gradient_tolerance"] = None
    restored = dt._transport_campaign_metadata_from_mapping(payload)
    assert restored.gradient_tolerance is None
    assert restored.equilibrium_grid_shape == (5, 12)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("backend", "torch", "backend is invalid"),
        ("channel_order", ["a", "b", "c", "d"], "channel_order is invalid"),
        ("n_rho", 2, "n_rho must be >= 3"),
        ("rho_max", -1.0, "rho bounds are invalid"),
        ("edge_values", [0.1, 0.2], "edge_values length is invalid"),
        ("core_boundary", "fixed", "boundary contract is invalid"),
        ("equilibrium_grid_shape", [5], "equilibrium_grid_shape is invalid"),
        ("equilibrium_grid_shape", [2, 2], "must be >= 3 in both dimensions"),
        ("rho_min", float("inf"), "payload is malformed"),
    ],
)
def test_campaign_metadata_mapping_rejects_invalid_fields(key, value, match):
    payload = _valid_metadata_payload()
    payload[key] = value
    with pytest.raises(ValueError, match=match):
        dt._transport_campaign_metadata_from_mapping(payload)


def test_metadata_field_matches_distinguishes_tuple_lengths():
    assert dt._metadata_field_matches((1.0, 2.0), (1.0, 2.0))
    assert not dt._metadata_field_matches((1.0, 2.0), (1.0, 2.0, 3.0))
    assert dt._metadata_field_matches(None, None)
    assert not dt._metadata_field_matches(None, 1.0)


def test_campaign_metadata_replay_rejects_non_metadata_archive():
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup()
    with pytest.raises(ValueError, match="must be TransportCampaignMetadata"):
        dt.assert_transport_campaign_metadata_replay(
            "not-metadata", profiles, chi, sources, rho, dt_value, edge, backend="numpy"
        )


# --------------------------------------------------------------------------- #
# Gradient-audit and index validators                                          #
# --------------------------------------------------------------------------- #


def _audit_metadata(tolerance=5.0e-4, n_rho=12):
    return dt.TransportCampaignMetadata(
        backend="jax",
        dtype="float64",
        channel_order=dt.CHANNELS,
        n_rho=n_rho,
        rho_min=0.0,
        rho_max=1.0,
        rho_spacing=1.0 / float(n_rho - 1),
        dt=1.0e-3,
        core_boundary="zero_gradient",
        edge_boundary="dirichlet",
        edge_values=(0.2, 0.2, 4.0, 0.03),
        closure_source=None,
        closure_weights_checksum=None,
        gradient_tolerance=tolerance,
        equilibrium_grid_shape=None,
    )


def _parameter_audit(**overrides):
    base = dict(
        loss=1.0e-3,
        epsilon=1.0e-5,
        tolerance=5.0e-4,
        checked_indices=((0, 1), (1, 6)),
        chi_max_abs_error=1.0e-7,
        source_max_abs_error=2.0e-7,
        passed=True,
    )
    base.update(overrides)
    return dt.TransportGradientAudit(**base)


@pytest.mark.parametrize(
    ("metadata_kwargs", "audit_overrides", "match"),
    [
        ({"tolerance": None}, {}, "requires metadata.gradient_tolerance"),
        ({}, {"loss": -1.0}, "loss must be finite and non-negative"),
        ({}, {"epsilon": 0.0}, "epsilon must be positive and finite"),
        ({}, {"tolerance": 0.0, "passed": False}, "tolerance must be positive and finite"),
        ({}, {"tolerance": 1.0e-3}, "must match campaign metadata"),
        ({}, {"chi_max_abs_error": -1.0}, "chi_max_abs_error must be finite and non-negative"),
        ({}, {"passed": False}, "inconsistent with tolerance"),
    ],
)
def test_validate_transport_gradient_audit_rejects(metadata_kwargs, audit_overrides, match):
    metadata = _audit_metadata(**metadata_kwargs)
    audit = _parameter_audit(**audit_overrides)
    with pytest.raises(ValueError, match=match):
        dt._validate_transport_gradient_audit(metadata, audit)


def test_validate_transport_gradient_audit_rejects_non_bool_passed():
    metadata = _audit_metadata()
    audit = _parameter_audit(passed=1)
    with pytest.raises(ValueError, match="passed flag must be boolean"):
        dt._validate_transport_gradient_audit(metadata, audit)


@pytest.mark.parametrize(
    ("indices", "match"),
    [
        ((), "must not be empty"),
        (((0, 1), (0, 1)), "must be unique"),
        (((0, 99),), "out of campaign bounds"),
    ],
)
def test_validate_parameter_audit_indices_rejects(indices, match):
    with pytest.raises(ValueError, match=match):
        dt._validate_parameter_audit_indices(indices, 12)


@pytest.mark.parametrize(
    ("indices", "match"),
    [
        ((), "must not be empty"),
        (((0, 0, 1), (0, 0, 1)), "must be unique"),
        (((0, 0, 99),), "out of campaign bounds"),
    ],
)
def test_validate_rollout_audit_indices_rejects(indices, match):
    with pytest.raises(ValueError, match=match):
        dt._validate_rollout_audit_indices(indices, 12)


def test_assert_latency_report_matches_campaign_rejects_drift():
    metadata, gradient_report, _ = _transport_readiness_fixture()
    with pytest.raises(ValueError, match="backend mismatch"):
        dt._assert_latency_report_matches_campaign(
            replace(metadata, backend="numpy"), gradient_report, report_name="gradient latency report"
        )
    with pytest.raises(ValueError, match="dtype mismatch"):
        dt._assert_latency_report_matches_campaign(
            replace(metadata, dtype="float32"), gradient_report, report_name="gradient latency report"
        )
    with pytest.raises(ValueError, match="channel contract mismatch"):
        dt._assert_latency_report_matches_campaign(
            replace(metadata, channel_order=("a", "b", "c", "d")),
            gradient_report,
            report_name="gradient latency report",
        )
    with pytest.raises(ValueError, match="gradient_tolerance is required"):
        dt._assert_latency_report_matches_campaign(
            replace(metadata, gradient_tolerance=None), gradient_report, report_name="gradient latency report"
        )
    with pytest.raises(ValueError, match="audit tolerance mismatch"):
        dt._assert_latency_report_matches_campaign(
            replace(metadata, gradient_tolerance=1.0e-3), gradient_report, report_name="gradient latency report"
        )


# --------------------------------------------------------------------------- #
# Differentiability evidence guards                                            #
# --------------------------------------------------------------------------- #


def _admissible_evidence_fixture():
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup(16)
    target = profiles.copy()
    target[0, 5:11] *= 0.97
    target[1, 5:11] *= 0.98
    metadata = dt.transport_campaign_metadata(
        profiles, chi, sources, rho, dt_value, edge, backend="jax", gradient_tolerance=5.0e-4
    )
    audit = dt.audit_transport_parameter_gradients(
        profiles, chi, sources, target, rho, dt_value, edge, tolerance=5.0e-4
    )
    evidence = dt.transport_differentiability_evidence(metadata, audit, controller_formal_artifact_sha256="a" * 64)
    return metadata, audit, evidence


def test_transport_differentiability_evidence_rejects_wrong_types():
    metadata = _audit_metadata()
    audit = _parameter_audit()
    with pytest.raises(ValueError, match="metadata must be TransportCampaignMetadata"):
        dt.transport_differentiability_evidence("not-metadata", audit)
    with pytest.raises(ValueError, match="audit must be a transport gradient audit"):
        dt.transport_differentiability_evidence(metadata, object())


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("evidence_type", "evidence must be TransportDifferentiabilityEvidence"),
        ("schema", "schema_version is unsupported"),
        ("backend", "requires JAX backend"),
        ("tolerance_none", "requires metadata.gradient_tolerance"),
        ("audit_sha", "gradient_audit_sha256 mismatch"),
        ("channel_order", "channel_order mismatch"),
        ("n_rho", "n_rho mismatch"),
        ("equilibrium", "equilibrium_coupled mismatch"),
        ("tolerance_value", "gradient_tolerance mismatch"),
    ],
)
def test_assert_transport_differentiability_claim_admissible_rejects(kind, match):
    metadata, audit, evidence = _admissible_evidence_fixture()
    if kind == "evidence_type":
        with pytest.raises(ValueError, match=match):
            dt.assert_transport_differentiability_claim_admissible(object(), metadata, audit)
        return
    if kind == "schema":
        evidence = replace(evidence, schema_version=2)
    elif kind == "backend":
        metadata = replace(metadata, backend="numpy")
    elif kind == "tolerance_none":
        metadata = replace(metadata, gradient_tolerance=None)
    elif kind == "audit_sha":
        evidence = replace(evidence, gradient_audit_sha256="0" * 64)
    elif kind == "channel_order":
        evidence = replace(evidence, channel_order=("x", "y", "z", "w"))
    elif kind == "n_rho":
        evidence = replace(evidence, n_rho=evidence.n_rho + 1)
    elif kind == "equilibrium":
        evidence = replace(evidence, equilibrium_coupled=not evidence.equilibrium_coupled)
    elif kind == "tolerance_value":
        evidence = replace(evidence, gradient_tolerance=evidence.gradient_tolerance * 2.0)
    with pytest.raises(ValueError, match=match):
        dt.assert_transport_differentiability_claim_admissible(evidence, metadata, audit)


# --------------------------------------------------------------------------- #
# Full-fidelity readiness blocked reasons and guards                           #
# --------------------------------------------------------------------------- #


def test_full_fidelity_readiness_blocks_missing_equilibrium_and_rollout_and_proof():
    metadata, gradient_report, _ = _transport_readiness_fixture()
    no_equilibrium = replace(metadata, equilibrium_grid_shape=None)
    evidence = dt.transport_full_fidelity_readiness_evidence(
        no_equilibrium,
        gradient_report,
        rollout_report=None,
        external_reference_artifact_sha256="b" * 64,
        external_reference_admitted=True,
        controller_formal_artifact_sha256=None,
    )
    assert not evidence.full_fidelity_claim_admissible
    assert "equilibrium_coupled_campaign" in evidence.blocked_reasons
    assert "rollout_latency_report" in evidence.blocked_reasons
    assert "controller_formal_artifact_sha256" in evidence.blocked_reasons


def test_full_fidelity_readiness_blocks_unadmitted_external_reference():
    metadata, gradient_report, rollout_report = _transport_readiness_fixture()
    evidence = dt.transport_full_fidelity_readiness_evidence(
        metadata,
        gradient_report,
        rollout_report=rollout_report,
        external_reference_artifact_sha256="b" * 64,
        external_reference_admitted=False,
        controller_formal_artifact_sha256="a" * 64,
    )
    assert "external_reference_admission" in evidence.blocked_reasons
    with pytest.raises(ValueError, match="external reference admission"):
        dt.assert_transport_full_fidelity_claim_ready(
            evidence, metadata, gradient_report, rollout_report=rollout_report
        )


def test_full_fidelity_claim_ready_rejects_generic_block_without_external_reasons():
    metadata, gradient_report, _ = _transport_readiness_fixture()
    evidence = dt.transport_full_fidelity_readiness_evidence(
        metadata,
        gradient_report,
        rollout_report=None,
        external_reference_artifact_sha256="b" * 64,
        external_reference_admitted=True,
        controller_formal_artifact_sha256="a" * 64,
    )
    assert "rollout_latency_report" in evidence.blocked_reasons
    assert "external_reference_artifact_sha256" not in evidence.blocked_reasons
    with pytest.raises(ValueError, match="not ready"):
        dt.assert_transport_full_fidelity_claim_ready(evidence, metadata, gradient_report, rollout_report=None)


def test_full_fidelity_claim_ready_rejects_type_schema_and_digest():
    metadata, gradient_report, rollout_report = _transport_readiness_fixture()
    evidence = dt.transport_full_fidelity_readiness_evidence(
        metadata,
        gradient_report,
        rollout_report=rollout_report,
        external_reference_artifact_sha256="b" * 64,
        external_reference_admitted=True,
        controller_formal_artifact_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="must be TransportFullFidelityReadinessEvidence"):
        dt.assert_transport_full_fidelity_claim_ready(
            object(), metadata, gradient_report, rollout_report=rollout_report
        )
    with pytest.raises(ValueError, match="schema_version is unsupported"):
        dt.assert_transport_full_fidelity_claim_ready(
            replace(evidence, schema_version=2), metadata, gradient_report, rollout_report=rollout_report
        )
    with pytest.raises(ValueError, match="digest mismatch"):
        dt.assert_transport_full_fidelity_claim_ready(
            replace(evidence, claim_status="tampered"), metadata, gradient_report, rollout_report=rollout_report
        )


def test_full_fidelity_readiness_rejects_invalid_argument_types():
    metadata, gradient_report, rollout_report = _transport_readiness_fixture()
    with pytest.raises(ValueError, match="metadata must be TransportCampaignMetadata"):
        dt.transport_full_fidelity_readiness_evidence("x", gradient_report)
    with pytest.raises(ValueError, match="gradient_report must be TransportGradientLatencyReport"):
        dt.transport_full_fidelity_readiness_evidence(metadata, object())
    with pytest.raises(ValueError, match="rollout_report must be TransportRolloutGradientLatencyReport"):
        dt.transport_full_fidelity_readiness_evidence(metadata, gradient_report, rollout_report=object())
    with pytest.raises(ValueError, match="external_reference_admitted must be boolean"):
        dt.transport_full_fidelity_readiness_evidence(metadata, gradient_report, external_reference_admitted="yes")


# --------------------------------------------------------------------------- #
# Latency-report and runtime-metadata validators                              #
# --------------------------------------------------------------------------- #


def test_latency_report_validators_reject_wrong_types():
    with pytest.raises(ValueError, match="must be TransportGradientLatencyReport"):
        dt._validate_transport_gradient_latency_report(object())
    with pytest.raises(ValueError, match="must be TransportRolloutGradientLatencyReport"):
        dt._validate_transport_rollout_gradient_latency_report(object())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema_version": 2}, "schema_version is unsupported"),
        ({"backend": "numpy"}, "requires JAX backend"),
        ({"channel_count": 3}, "channel_count is invalid"),
        ({"p50_ms": 9.0}, "p50 <= p95 <= max"),
    ],
)
def test_gradient_latency_report_validation_rejects(overrides, match):
    _, gradient_report, _ = _transport_readiness_fixture()
    with pytest.raises(ValueError, match=match):
        dt._validate_transport_gradient_latency_report(replace(gradient_report, **overrides))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema_version": 2}, "schema_version is unsupported"),
        ({"measured_at_unix_s": -1.0}, "must be finite and non-negative"),
        ({"jax_version": ""}, "must be a non-empty string"),
        ({"processor": 5}, "processor must be a string"),
        ({"jax_devices": ()}, "must be a non-empty tuple"),
        ({"jax_devices": ("",)}, "must contain non-empty strings"),
        ({"jax_enable_x64": "yes"}, "must be boolean"),
    ],
)
def test_runtime_metadata_validation_rejects(overrides, match):
    metadata = _runtime_metadata()
    with pytest.raises(ValueError, match=match):
        dt._validate_transport_runtime_metadata(replace(metadata, **overrides))


def test_runtime_metadata_validation_rejects_wrong_type():
    with pytest.raises(ValueError, match="runtime_metadata is invalid"):
        dt._validate_transport_runtime_metadata(object())


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_require_nonnegative_finite_rejects(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        dt._require_nonnegative_finite("x", value)


@pytest.mark.parametrize("value", [True, 1.5, "3"])
def test_require_int_rejects_non_integers(value):
    with pytest.raises(ValueError, match="must be an integer"):
        dt._require_int("x", value, minimum=0)


def test_percentile_handles_empty_single_and_integer_rank():
    with pytest.raises(ValueError, match="must not be empty"):
        dt._percentile([], 0.5)
    assert dt._percentile([4.2], 0.95) == pytest.approx(4.2)
    assert dt._percentile([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)
    assert dt._percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# Equilibrium radial weighting                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("psi", "match"),
    [
        (np.ones((2, 5)), "two-dimensional flux map"),
        (np.ones(5), "two-dimensional flux map"),
    ],
)
def test_validate_equilibrium_psi_rejects(psi, match):
    with pytest.raises(ValueError, match=match):
        dt._validate_equilibrium_psi(psi)


def test_equilibrium_radial_weights_rejects_bad_n_rho():
    psi = np.outer(np.linspace(1.0, 0.2, 5), np.linspace(1.0, 0.3, 8))
    with pytest.raises(ValueError, match="n_rho must be an integer >= 3"):
        dt.equilibrium_radial_weights(psi, 2)
    with pytest.raises(ValueError, match="n_rho must be an integer >= 3"):
        dt.equilibrium_radial_weights(psi, True)


def test_equilibrium_radial_weights_interpolates_to_requested_length():
    psi = np.outer(np.linspace(1.0, 0.2, 5), np.linspace(1.0, 0.3, 7))
    weights = dt.equilibrium_radial_weights(psi, 12)
    assert weights.shape == (12,)
    assert np.mean(weights) == pytest.approx(1.0)
    assert np.all(weights >= 0.0)


def test_equilibrium_radial_weights_returns_ones_for_degenerate_flux():
    weights = dt.equilibrium_radial_weights(np.zeros((4, 6)), 6)
    np.testing.assert_allclose(weights, 1.0)


# --------------------------------------------------------------------------- #
# JAX-unavailable fail-closed contracts for private helpers                    #
# --------------------------------------------------------------------------- #


def test_private_jax_helpers_fail_closed_when_jnp_missing(monkeypatch):
    monkeypatch.setattr(dt, "jnp", None)
    monkeypatch.setattr(dt, "jax", None)
    rho = _uniform_rho(6)
    profiles = _profiles(rho)
    chi = _valid_chi(rho)
    sources = np.zeros_like(profiles)
    edge = np.array([0.2, 0.2, 4.0, 0.03])
    with pytest.raises(RuntimeError, match="JAX"):
        dt._transport_step_jax(profiles, chi, sources, rho, 1.0e-3, edge)
    with pytest.raises(RuntimeError, match="JAX"):
        dt._transport_rollout_jax(profiles, chi, sources[None], rho, 1.0e-3, edge)
    with pytest.raises(RuntimeError, match="JAX"):
        dt._tracking_loss_jax(profiles, chi, sources, profiles, rho, 1.0e-3, edge, np.ones(4))
    with pytest.raises(RuntimeError, match="JAX"):
        dt._equilibrium_radial_weights_jax(np.ones((4, 6)), 6)
    with pytest.raises(RuntimeError, match="JAX"):
        dt._equilibrium_weighted_tracking_loss_jax(
            profiles, chi, sources, profiles, rho, 1.0e-3, edge, np.ones((4, 6)), np.ones(4)
        )
    with pytest.raises(RuntimeError, match="JAX"):
        dt._equilibrium_weighted_rollout_tracking_loss_jax(
            profiles, chi, sources[None], profiles[None], rho, 1.0e-3, edge, np.ones((4, 6)), np.ones(4)
        )


def test_resolve_use_jax_allows_explicit_numpy_fallback(monkeypatch):
    monkeypatch.setattr(dt, "_HAS_JAX", False)
    monkeypatch.setattr(dt._jax_solvers, "_HAS_JAX", False)
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup()
    stepped = dt.differentiable_transport_step(
        profiles,
        chi,
        sources,
        rho,
        dt_value,
        edge,
        use_jax=True,
        allow_numpy_fallback=True,
        allow_legacy_numpy_fallback=True,
    )
    assert np.all(np.isfinite(stepped))


# --------------------------------------------------------------------------- #
# JAX compute paths and default-weight branches                                #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_jax_step_and_losses_match_numpy_within_tolerance():
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup(20)
    target = profiles.copy()
    target[0, 6:12] *= 0.97
    psi = np.outer(np.linspace(1.0, 0.2, 6), np.linspace(1.0, 0.3, rho.size))

    step_jax = np.asarray(dt.differentiable_transport_step(profiles, chi, sources, rho, dt_value, edge, use_jax=True))
    step_numpy = dt.differentiable_transport_step(profiles, chi, sources, rho, dt_value, edge, use_jax=False)
    np.testing.assert_allclose(step_jax, step_numpy, rtol=1e-9, atol=1e-11)

    loss_jax = float(dt.transport_tracking_loss(profiles, chi, sources, target, rho, dt_value, edge, use_jax=True))
    loss_numpy = dt.transport_tracking_loss(profiles, chi, sources, target, rho, dt_value, edge, use_jax=False)
    assert loss_jax == pytest.approx(loss_numpy, rel=1e-7, abs=1e-12)

    eq_loss_jax = float(
        dt.equilibrium_weighted_transport_tracking_loss(
            profiles, chi, sources, target, rho, dt_value, edge, psi, use_jax=True
        )
    )
    eq_loss_numpy = dt.equilibrium_weighted_transport_tracking_loss(
        profiles, chi, sources, target, rho, dt_value, edge, psi, use_jax=False
    )
    assert eq_loss_jax == pytest.approx(eq_loss_numpy, rel=1e-6, abs=1e-12)


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_jax_rollout_losses_match_numpy_and_interpolate_equilibrium():
    profiles, chi, source_sequence, target_history, rho, dt_value, edge = _rollout_setup(18, 3)
    psi = np.outer(np.linspace(1.0, 0.2, 5), np.linspace(1.0, 0.3, 7))

    loss_jax = float(
        dt.transport_rollout_tracking_loss(
            profiles, chi, source_sequence, target_history, rho, dt_value, edge, use_jax=True
        )
    )
    loss_numpy = dt.transport_rollout_tracking_loss(
        profiles, chi, source_sequence, target_history, rho, dt_value, edge, use_jax=False
    )
    assert loss_jax == pytest.approx(loss_numpy, rel=1e-7, abs=1e-12)

    eq_loss_jax = float(
        dt.equilibrium_weighted_transport_rollout_tracking_loss(
            profiles, chi, source_sequence, target_history, rho, dt_value, edge, psi, use_jax=True
        )
    )
    eq_loss_numpy = dt.equilibrium_weighted_transport_rollout_tracking_loss(
        profiles, chi, source_sequence, target_history, rho, dt_value, edge, psi, use_jax=False
    )
    assert eq_loss_jax == pytest.approx(eq_loss_numpy, rel=1e-6, abs=1e-12)


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_jax_gradient_apis_default_unit_weights():
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup(16)
    target = profiles.copy()
    target[0, 5:11] *= 0.97
    psi = np.outer(np.linspace(1.0, 0.2, 6), np.linspace(1.0, 0.3, rho.size))

    loss, gradient = dt.transport_loss_gradient(profiles, chi, sources, target, rho, dt_value, edge)
    assert np.isfinite(loss)
    assert gradient.shape == chi.shape

    params = dt.transport_parameter_gradients(profiles, chi, sources, target, rho, dt_value, edge)
    assert np.all(np.isfinite(params.chi_gradient))

    eq = dt.equilibrium_weighted_transport_loss_gradient(profiles, chi, sources, target, rho, dt_value, edge, psi)
    assert eq.chi_gradient.shape == chi.shape
    assert eq.equilibrium_gradient.shape == psi.shape


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_jax_rollout_gradient_apis_default_unit_weights():
    profiles, chi, source_sequence, target_history, rho, dt_value, edge = _rollout_setup(16, 3)
    psi = np.outer(np.linspace(1.0, 0.2, 6), np.linspace(1.0, 0.3, rho.size))

    rollout = dt.transport_rollout_source_gradients(profiles, chi, source_sequence, target_history, rho, dt_value, edge)
    assert rollout.source_gradient.shape == source_sequence.shape

    eq_rollout = dt.equilibrium_weighted_transport_rollout_source_gradient(
        profiles, chi, source_sequence, target_history, rho, dt_value, edge, psi
    )
    assert eq_rollout.source_gradient.shape == source_sequence.shape
    assert eq_rollout.equilibrium_gradient.shape == psi.shape


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
@pytest.mark.parametrize(
    ("func", "target_is_history"),
    [
        (dt.transport_loss_gradient, False),
        (dt.transport_parameter_gradients, False),
        (dt.transport_rollout_source_gradients, True),
    ],
)
def test_jax_gradient_apis_require_target(func, target_is_history):
    if target_is_history:
        profiles, chi, sequence, _, rho, dt_value, edge = _rollout_setup(12, 2)
        with pytest.raises(ValueError, match="target_history is required"):
            func(profiles, chi, sequence, None, rho, dt_value, edge)
    else:
        profiles, chi, sources, rho, dt_value, edge = _one_step_setup()
        with pytest.raises(ValueError, match="target_profiles is required"):
            func(profiles, chi, sources, None, rho, dt_value, edge)


# --------------------------------------------------------------------------- #
# Audit helpers, fail-closed asserts, and benchmark latency paths              #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"epsilon": 0.0}, "epsilon must be positive and finite"),
        ({"tolerance": 0.0}, "tolerance must be positive and finite"),
    ],
)
def test_audit_rollout_source_gradients_rejects_bad_audit_scalars(kwargs, match):
    profiles, chi, sequence, target_history, rho, dt_value, edge = _rollout_setup(14, 2)
    with pytest.raises(ValueError, match=match):
        dt.audit_transport_rollout_source_gradients(
            profiles, chi, sequence, target_history, rho, dt_value, edge, **kwargs
        )


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
@pytest.mark.parametrize(
    ("sample_indices", "match"),
    [
        ([(0, 0)], "three-part rollout source indices"),
        ([(0, 0, 99)], "out-of-range rollout source index"),
    ],
)
def test_audit_rollout_source_gradients_rejects_bad_sample_indices(sample_indices, match):
    profiles, chi, sequence, target_history, rho, dt_value, edge = _rollout_setup(14, 2)
    with pytest.raises(ValueError, match=match):
        dt.audit_transport_rollout_source_gradients(
            profiles, chi, sequence, target_history, rho, dt_value, edge, sample_indices=sample_indices
        )


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_assert_rollout_source_gradients_consistent_fails_closed_on_tiny_tolerance():
    profiles, chi, sequence, target_history, rho, dt_value, edge = _rollout_setup(14, 2)
    with pytest.raises(ValueError, match="rollout source-gradient audit failed"):
        dt.assert_transport_rollout_source_gradients_consistent(
            profiles, chi, sequence, target_history, rho, dt_value, edge, tolerance=1.0e-30
        )


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_assert_parameter_gradients_consistent_fails_closed_on_tiny_tolerance():
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup(14)
    target = profiles.copy()
    target[0, 4:10] *= 0.97
    with pytest.raises(ValueError, match="parameter gradient audit failed"):
        dt.assert_transport_parameter_gradients_consistent(
            profiles, chi, sources, target, rho, dt_value, edge, tolerance=1.0e-30
        )


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_parameter_gradient_audit_rejects_bad_sample_indices():
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup(14)
    target = profiles.copy()
    target[0, 4:10] *= 0.97
    with pytest.raises(ValueError, match="\\(channel, radial\\) pairs"):
        dt.audit_transport_parameter_gradients(
            profiles, chi, sources, target, rho, dt_value, edge, sample_indices=[(0,)]
        )
    with pytest.raises(ValueError, match="out-of-bounds transport index"):
        dt.audit_transport_parameter_gradients(
            profiles, chi, sources, target, rho, dt_value, edge, sample_indices=[(0, 999)]
        )
    with pytest.raises(ValueError, match="tolerance must be positive"):
        dt.audit_transport_parameter_gradients(profiles, chi, sources, target, rho, dt_value, edge, tolerance=0.0)


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_benchmark_parameter_gradient_latency_runs_warmup_and_defaults_weights():
    profiles, chi, sources, rho, dt_value, edge = _one_step_setup(14)
    target = profiles.copy()
    target[0, 4:10] *= 0.97
    report = dt.benchmark_transport_parameter_gradient_latency(
        profiles, chi, sources, target, rho, dt_value, edge, warmup_runs=1, timed_runs=2
    )
    assert report.backend == "jax"
    assert report.timed_runs == 2
    assert report.audit.passed
    assert report.p50_ms <= report.p95_ms <= report.max_ms
    with pytest.raises(ValueError, match="target_profiles is required"):
        dt.benchmark_transport_parameter_gradient_latency(
            profiles, chi, sources, None, rho, dt_value, edge, warmup_runs=0, timed_runs=1
        )


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_benchmark_rollout_gradient_latency_runs_warmup_and_defaults_weights():
    profiles, chi, sequence, target_history, rho, dt_value, edge = _rollout_setup(14, 2)
    report = dt.benchmark_transport_rollout_source_gradient_latency(
        profiles, chi, sequence, target_history, rho, dt_value, edge, warmup_runs=1, timed_runs=2
    )
    assert report.backend == "jax"
    assert report.n_steps == 2
    assert report.audit.passed
    assert report.p50_ms <= report.p95_ms <= report.max_ms
    with pytest.raises(ValueError, match="target_history is required"):
        dt.benchmark_transport_rollout_source_gradient_latency(
            profiles, chi, sequence, None, rho, dt_value, edge, warmup_runs=0, timed_runs=1
        )


# --------------------------------------------------------------------------- #
# Remaining edge branches: digests, blocked-audit reports, index helpers       #
# --------------------------------------------------------------------------- #


def test_evidence_rejects_non_hex_proof_digest():
    metadata = _audit_metadata()
    audit = _parameter_audit()
    with pytest.raises(ValueError, match="SHA-256 hex digest"):
        dt.transport_differentiability_evidence(metadata, audit, controller_formal_artifact_sha256="g" * 64)


def test_full_fidelity_readiness_blocks_failing_gradient_and_rollout_audits():
    metadata, gradient_report, rollout_report = _transport_readiness_fixture()

    failing_gradient = replace(
        gradient_report,
        audit=replace(gradient_report.audit, chi_max_abs_error=1.0e-3, passed=False),
    )
    evidence = dt.transport_full_fidelity_readiness_evidence(
        metadata,
        failing_gradient,
        rollout_report=rollout_report,
        external_reference_artifact_sha256="b" * 64,
        external_reference_admitted=True,
        controller_formal_artifact_sha256="a" * 64,
    )
    assert "gradient_latency_audit" in evidence.blocked_reasons

    failing_rollout = replace(
        rollout_report,
        audit=replace(rollout_report.audit, source_max_abs_error=1.0e-3, passed=False),
    )
    evidence = dt.transport_full_fidelity_readiness_evidence(
        metadata,
        gradient_report,
        rollout_report=failing_rollout,
        external_reference_artifact_sha256="b" * 64,
        external_reference_admitted=True,
        controller_formal_artifact_sha256="a" * 64,
    )
    assert "rollout_latency_audit" in evidence.blocked_reasons


def test_full_fidelity_readiness_rejects_rollout_audit_indices_out_of_bounds():
    metadata, gradient_report, rollout_report = _transport_readiness_fixture()
    out_of_bounds = replace(
        rollout_report,
        audit=replace(
            rollout_report.audit,
            checked_indices=((9, 0, 1),),
            source_max_abs_error=2.0e-7,
            passed=True,
        ),
    )
    with pytest.raises(ValueError, match="exceed campaign metadata bounds"):
        dt.transport_full_fidelity_readiness_evidence(
            metadata,
            gradient_report,
            rollout_report=out_of_bounds,
            external_reference_artifact_sha256="b" * 64,
            external_reference_admitted=True,
            controller_formal_artifact_sha256="a" * 64,
        )


def test_validate_gradient_audit_rejects_negative_rollout_source_error():
    metadata = _audit_metadata()
    rollout_audit = dt.TransportRolloutGradientAudit(
        loss=1.0e-3,
        epsilon=1.0e-5,
        tolerance=5.0e-4,
        checked_indices=((0, 0, 1),),
        source_max_abs_error=-1.0,
        passed=True,
    )
    with pytest.raises(ValueError, match="source_max_abs_error must be finite and non-negative"):
        dt._validate_transport_gradient_audit(metadata, rollout_audit)


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
def test_differentiable_transport_rollout_jax_matches_numpy():
    profiles, chi, sequence, _, rho, dt_value, edge = _rollout_setup(16, 3)
    out_jax = np.asarray(
        dt.differentiable_transport_rollout(profiles, chi, sequence, rho, dt_value, edge, use_jax=True)
    )
    out_numpy = np.asarray(
        dt.differentiable_transport_rollout(profiles, chi, sequence, rho, dt_value, edge, use_jax=False)
    )
    np.testing.assert_allclose(out_jax, out_numpy, rtol=1e-9, atol=1e-11)


def test_rollout_gradient_audit_indices_reject_bad_shapes_and_empty():
    with pytest.raises(ValueError, match="must have shape"):
        dt._rollout_gradient_audit_indices((4, 5), None)
    with pytest.raises(ValueError, match="n_rho >= 3"):
        dt._rollout_gradient_audit_indices((0, 4, 5), None)
    with pytest.raises(ValueError, match="at least one rollout source index"):
        dt._rollout_gradient_audit_indices((3, 4, 5), [])


def test_gradient_audit_indices_reject_empty_samples():
    with pytest.raises(ValueError, match="at least one transport index"):
        dt._gradient_audit_indices((4, 12), [])


def test_audit_rollout_source_gradients_requires_target():
    profiles, chi, sequence, _, rho, dt_value, edge = _rollout_setup(12, 2)
    with pytest.raises(ValueError, match="target_history is required"):
        dt.audit_transport_rollout_source_gradients(profiles, chi, sequence, None, rho, dt_value, edge)


@pytest.mark.parametrize(
    ("func", "uses_history"),
    [
        (dt.equilibrium_weighted_transport_tracking_loss, False),
        (dt.equilibrium_weighted_transport_rollout_tracking_loss, True),
    ],
)
def test_equilibrium_weighted_tracking_losses_require_target(func, uses_history):
    psi = np.outer(np.linspace(1.0, 0.2, 5), np.linspace(1.0, 0.3, 12))
    if uses_history:
        profiles, chi, sequence, _, rho, dt_value, edge = _rollout_setup(12, 2)
        with pytest.raises(ValueError, match="target_history is required"):
            func(profiles, chi, sequence, None, rho, dt_value, edge, psi, use_jax=False)
    else:
        profiles, chi, sources, rho, dt_value, edge = _one_step_setup()
        with pytest.raises(ValueError, match="target_profiles is required"):
            func(profiles, chi, sources, None, rho, dt_value, edge, psi, use_jax=False)


@pytest.mark.skipif(not dt.has_jax(), reason="JAX optional dependency is not installed")
@pytest.mark.parametrize(
    ("func", "uses_history"),
    [
        (dt.equilibrium_weighted_transport_loss_gradient, False),
        (dt.equilibrium_weighted_transport_rollout_source_gradient, True),
    ],
)
def test_equilibrium_weighted_gradients_require_target(func, uses_history):
    psi = np.outer(np.linspace(1.0, 0.2, 5), np.linspace(1.0, 0.3, 12))
    if uses_history:
        profiles, chi, sequence, _, rho, dt_value, edge = _rollout_setup(12, 2)
        with pytest.raises(ValueError, match="target_history is required"):
            func(profiles, chi, sequence, None, rho, dt_value, edge, psi)
    else:
        profiles, chi, sources, rho, dt_value, edge = _one_step_setup()
        with pytest.raises(ValueError, match="target_profiles is required"):
            func(profiles, chi, sources, None, rho, dt_value, edge, psi)
