# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-surface tests for transport latency benchmarks

"""Drive production latency helpers through leaf and facade entry points."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_control.core.differentiable_transport as facade
import scpn_control.core.differentiable_transport_latency as latency


def _profiles(n_rho: int = 8) -> tuple[NDArray[np.float64], ...]:
    rho = np.linspace(0.0, 1.0, n_rho, dtype=np.float64)
    profiles = np.stack(
        [
            1.0 + 0.1 * rho,
            0.9 + 0.05 * rho,
            0.5 + 0.02 * rho,
            0.05 + 0.01 * rho,
        ]
    )
    chi = 0.2 * np.ones((4, n_rho), dtype=np.float64)
    sources = 0.01 * np.ones((4, n_rho), dtype=np.float64)
    target = profiles + 0.02
    edge = profiles[:, -1].copy()
    return profiles, chi, sources, target, rho, edge


def test_percentile_empty_fails_closed() -> None:
    """Empty latency sample lists are rejected by the production helper."""
    with pytest.raises(ValueError, match="must not be empty"):
        latency._percentile([], 0.5)


def test_percentile_interpolates_sorted_samples() -> None:
    """Linear percentile matches a known two-point interpolation."""
    samples = [1.0, 3.0]
    assert latency._percentile(samples, 0.0) == pytest.approx(1.0)
    assert latency._percentile(samples, 1.0) == pytest.approx(3.0)
    assert latency._percentile(samples, 0.5) == pytest.approx(2.0)


def test_public_latency_symbols_bind_to_leaf() -> None:
    """Facade re-exports are the production latency leaf objects."""
    assert facade.transport_runtime_metadata is latency.transport_runtime_metadata
    assert (
        facade.benchmark_transport_parameter_gradient_latency is latency.benchmark_transport_parameter_gradient_latency
    )
    assert (
        facade.benchmark_transport_rollout_source_gradient_latency
        is latency.benchmark_transport_rollout_source_gradient_latency
    )


def test_parameter_gradient_latency_report_via_facade() -> None:
    """Timed parameter-gradient admission returns a validated local latency report."""
    if not facade.has_jax():
        pytest.skip("JAX not installed in this CI lane (optional transport backend)")
    profiles, chi, sources, target, rho, edge = _profiles()
    report = facade.benchmark_transport_parameter_gradient_latency(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge,
        warmup_runs=0,
        timed_runs=1,
    )
    assert report.timed_runs == 1
    assert report.n_rho == rho.size
    assert report.p50_ms >= 0.0
    assert report.p95_ms >= report.p50_ms
    assert "local audited" in report.claim_status
    leaf_report = latency.benchmark_transport_parameter_gradient_latency(
        profiles,
        chi,
        sources,
        target,
        rho,
        1.0e-3,
        edge,
        warmup_runs=0,
        timed_runs=1,
    )
    assert leaf_report.schema_version == report.schema_version
    assert leaf_report.backend == "jax"


def test_parameter_gradient_latency_rejects_missing_target() -> None:
    """Fail-closed when target profiles are omitted from the admission path."""
    if not facade.has_jax():
        pytest.skip("JAX not installed in this CI lane (optional transport backend)")
    profiles, chi, sources, _target, rho, edge = _profiles()
    with pytest.raises(ValueError, match="target_profiles"):
        latency.benchmark_transport_parameter_gradient_latency(
            profiles,
            chi,
            sources,
            None,
            rho,
            1.0e-3,
            edge,
            warmup_runs=0,
            timed_runs=1,
        )
