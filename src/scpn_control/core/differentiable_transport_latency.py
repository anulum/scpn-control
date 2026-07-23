# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport latency benchmarks

"""Local latency measurement for audited JAX transport gradient admission.

This leaf owns runtime provenance for latency reports and the timed admission
benchmarks. Numerical AD and array validation remain on the facade; this module
lazy-imports those production call sites so the measured path is the same
fail-closed contract controllers use. Report serialisation lives in the
evidence leaf. Public symbols are re-exported from
:mod:`scpn_control.core.differentiable_transport`.
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Any, cast

import numpy as np

from scpn_control.core.differentiable_transport_evidence import (
    CHANNEL_COUNT,
    TransportGradientAudit,
    TransportGradientLatencyReport,
    TransportRolloutGradientAudit,
    TransportRolloutGradientLatencyReport,
    TransportRuntimeMetadata,
    _require_int,
)

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except Exception:
    jax = cast(Any, None)
    _HAS_JAX = False


def transport_runtime_metadata() -> TransportRuntimeMetadata:
    """Return runtime provenance for audited JAX transport latency reports."""
    if not _HAS_JAX or jax is None:
        raise RuntimeError("transport runtime metadata requires JAX")
    try:
        import jaxlib

        jaxlib_version = str(getattr(jaxlib, "__version__", "unknown"))
    except ImportError:
        jaxlib_version = "unknown"
    return TransportRuntimeMetadata(
        schema_version=1,
        measured_at_unix_s=float(time.time()),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        jax_version=str(getattr(jax, "__version__", "unknown")),
        jaxlib_version=jaxlib_version,
        jax_default_backend=str(jax.default_backend()),
        jax_devices=tuple(str(device) for device in jax.devices()),
        jax_enable_x64=bool(jax.config.read("jax_enable_x64")),
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Return a linear-interpolated percentile of a sorted sample list."""
    if not sorted_values:
        raise ValueError("latency samples must not be empty")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def benchmark_transport_parameter_gradient_latency(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    epsilon: float = 1.0e-5,
    tolerance: float = 5.0e-4,
    sample_indices: Any | None = None,
    warmup_runs: int = 1,
    timed_runs: int = 5,
) -> TransportGradientLatencyReport:
    """Measure audited JAX gradient-admission latency for controller tuning.

    The measured path is intentionally the fail-closed admission contract:
    JAX gradients for transport coefficients and source schedules plus sampled
    independent finite-difference audit. The report is local timing evidence,
    not a real-time control-loop guarantee.
    """
    # Lazy import avoids a load-time cycle with the facade, which re-exports us.
    from scpn_control.core import differentiable_transport as facade

    warmups = _require_int("warmup_runs", warmup_runs, minimum=0)
    repetitions = _require_int("timed_runs", timed_runs, minimum=1)
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        facade._validate_transport_inputs(
            profiles,
            chi,
            sources,
            rho,
            dt,
            edge_values,
            target_profiles=target_profiles,
            weights=weights,
        )
    )
    if target_array is None:
        raise ValueError("target_profiles is required")
    if weight_array is None:
        weight_array = np.ones(CHANNEL_COUNT)

    def run_audit() -> TransportGradientAudit:
        return facade.assert_transport_parameter_gradients_consistent(
            profile_array,
            chi_array,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            weights=weight_array,
            epsilon=epsilon,
            tolerance=tolerance,
            sample_indices=sample_indices,
        )

    audit = run_audit()
    for _ in range(warmups):
        audit = run_audit()

    latencies_ms: list[float] = []
    for _ in range(repetitions):
        start_ns = time.perf_counter_ns()
        audit = run_audit()
        latencies_ms.append((time.perf_counter_ns() - start_ns) / 1.0e6)

    sorted_latencies = sorted(latencies_ms)
    metadata = facade.transport_campaign_metadata(
        profile_array,
        chi_array,
        source_array,
        rho_array,
        float(dt),
        edge_array,
        backend="jax",
        gradient_tolerance=tolerance,
    )
    return TransportGradientLatencyReport(
        schema_version=1,
        backend=metadata.backend,
        dtype=metadata.dtype,
        n_rho=metadata.n_rho,
        channel_count=CHANNEL_COUNT,
        warmup_runs=warmups,
        timed_runs=repetitions,
        p50_ms=_percentile(sorted_latencies, 0.50),
        p95_ms=_percentile(sorted_latencies, 0.95),
        max_ms=float(max(sorted_latencies)),
        runtime_metadata=transport_runtime_metadata(),
        audit=audit,
        claim_status="local audited gradient-admission latency only; not a real-time control-loop guarantee",
    )


def benchmark_transport_rollout_source_gradient_latency(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    epsilon: float = 1.0e-5,
    tolerance: float = 5.0e-4,
    sample_indices: Any | None = None,
    warmup_runs: int = 1,
    timed_runs: int = 5,
) -> TransportRolloutGradientLatencyReport:
    """Measure audited multi-step source-rollout gradient latency.

    The measured path is the controller-admission contract for source schedules:
    JAX rollout gradients with a sampled independent NumPy finite-difference
    audit. The report is local timing evidence, not a real-time control-loop or
    externally validated transport claim.
    """
    from scpn_control.core import differentiable_transport as facade

    warmups = _require_int("warmup_runs", warmup_runs, minimum=0)
    repetitions = _require_int("timed_runs", timed_runs, minimum=1)
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        facade._validate_transport_rollout_inputs(
            initial_profiles,
            chi,
            source_sequence,
            rho,
            dt,
            edge_values,
            target_history=target_history,
            weights=weights,
        )
    )
    if target_array is None:
        raise ValueError("target_history is required")
    if weight_array is None:
        weight_array = np.ones(CHANNEL_COUNT)
    checked_sample_indices = facade._rollout_gradient_audit_indices(source_array.shape, sample_indices)

    def run_audit() -> TransportRolloutGradientAudit:
        return facade.assert_transport_rollout_source_gradients_consistent(
            profile_array,
            chi_array,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            weights=weight_array,
            epsilon=epsilon,
            tolerance=tolerance,
            sample_indices=checked_sample_indices,
        )

    audit = run_audit()
    for _ in range(warmups):
        audit = run_audit()

    latencies_ms: list[float] = []
    for _ in range(repetitions):
        start_ns = time.perf_counter_ns()
        audit = run_audit()
        latencies_ms.append((time.perf_counter_ns() - start_ns) / 1.0e6)

    sorted_latencies = sorted(latencies_ms)
    metadata = facade.transport_campaign_metadata(
        profile_array,
        chi_array,
        source_array[0],
        rho_array,
        float(dt),
        edge_array,
        backend="jax",
        gradient_tolerance=tolerance,
    )
    return TransportRolloutGradientLatencyReport(
        schema_version=1,
        backend=metadata.backend,
        dtype=metadata.dtype,
        n_rho=metadata.n_rho,
        n_steps=int(source_array.shape[0]),
        channel_count=CHANNEL_COUNT,
        warmup_runs=warmups,
        timed_runs=repetitions,
        p50_ms=_percentile(sorted_latencies, 0.50),
        p95_ms=_percentile(sorted_latencies, 0.95),
        max_ms=float(max(sorted_latencies)),
        runtime_metadata=transport_runtime_metadata(),
        audit=audit,
        claim_status="local audited rollout source-gradient latency only; not a real-time control-loop guarantee",
    )
