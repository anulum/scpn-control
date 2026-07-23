# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport core step / rollout

"""Four-channel Crank-Nicolson transport step and multi-step rollout numerics.

This leaf owns the NumPy and JAX one-step primitives and the multi-step source
schedule rollout. Input validation, campaign metadata, and AD/audit contracts
remain on sibling modules; this module lazy-imports facade validators and the
facade JAX gate so public re-exports and test monkeypatches of
``scpn_control.core.differentiable_transport`` remain authoritative.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core import jax_solvers as _jax_solvers
from scpn_control.core.differentiable_transport_evidence import CHANNEL_COUNT


def _facade() -> Any:
    """Return the public numerical facade (lazy to avoid import cycles)."""
    from scpn_control.core import differentiable_transport as facade

    return facade


def _transport_step_numpy(
    profiles: AnyFloatArray,
    chi: AnyFloatArray,
    sources: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
) -> FloatArray:
    """Advance four channels by one Crank-Nicolson step on the NumPy path."""
    drho = float(rho[1] - rho[0])
    return np.stack(
        [
            _jax_solvers.crank_nicolson_step(
                profiles[channel],
                chi[channel],
                sources[channel],
                rho,
                drho,
                float(dt),
                float(edge_values[channel]),
                use_jax=False,
            )
            for channel in range(CHANNEL_COUNT)
        ]
    )


def _transport_step_jax(
    profiles: Any,
    chi: Any,
    sources: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
) -> Any:
    """Advance four channels by one Crank-Nicolson step inside the JAX graph."""
    facade = _facade()
    if facade.jnp is None or facade.jax is None:
        raise RuntimeError("JAX transport step requested but JAX is unavailable")
    rho_jax = facade.jnp.asarray(rho, dtype=facade.jnp.float64)
    drho = rho_jax[1] - rho_jax[0]
    step = facade.jax.vmap(_jax_solvers._cn_step_jax, in_axes=(0, 0, 0, None, None, None, 0))
    return step(
        facade.jnp.asarray(profiles, dtype=facade.jnp.float64),
        facade.jnp.asarray(chi, dtype=facade.jnp.float64),
        facade.jnp.asarray(sources, dtype=facade.jnp.float64),
        rho_jax,
        drho,
        float(dt),
        facade.jnp.asarray(edge_values, dtype=facade.jnp.float64),
    )


def _transport_rollout_numpy(
    initial_profiles: AnyFloatArray,
    chi: AnyFloatArray,
    source_sequence: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
) -> FloatArray:
    """Advance a multi-step source schedule on the NumPy path."""
    current = initial_profiles
    history: list[FloatArray] = []
    for source_step in source_sequence:
        current = _transport_step_numpy(current, chi, source_step, rho, dt, edge_values)
        history.append(current)
    return np.stack(history)


def _transport_rollout_jax(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
) -> Any:
    """Advance a multi-step source schedule inside the JAX graph."""
    facade = _facade()
    if facade.jnp is None or facade.jax is None:
        raise RuntimeError("JAX transport rollout requested but JAX is unavailable")
    chi_jax = facade.jnp.asarray(chi, dtype=facade.jnp.float64)
    rho_jax = facade.jnp.asarray(rho, dtype=facade.jnp.float64)
    edge_jax = facade.jnp.asarray(edge_values, dtype=facade.jnp.float64)

    def body(carry: Any, source_step: Any) -> tuple[Any, Any]:
        next_profiles = _transport_step_jax(carry, chi_jax, source_step, rho_jax, dt, edge_jax)
        return next_profiles, next_profiles

    _, history = facade.jax.lax.scan(
        body,
        facade.jnp.asarray(initial_profiles, dtype=facade.jnp.float64),
        facade.jnp.asarray(source_sequence, dtype=facade.jnp.float64),
    )
    return history


def differentiable_transport_step(
    profiles: Any,
    chi: Any,
    sources: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Advance four transport channels by one differentiable radial step.

    Channel order is electron temperature, ion temperature, electron density,
    and impurity density. The radial coordinate is a strictly increasing,
    uniformly spaced normalised axis from core-side interior to edge. The core
    boundary uses the inherited zero-gradient condition, and each channel uses
    its supplied Dirichlet edge value.
    """
    facade = _facade()
    profile_array, chi_array, source_array, rho_array, edge_array, _, _ = facade._validate_transport_inputs(
        profiles,
        chi,
        sources,
        rho,
        dt,
        edge_values,
    )
    use_jax_runtime = facade._resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="differentiable_transport_step",
    )
    if use_jax_runtime:
        return _transport_step_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return _transport_step_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)


def differentiable_transport_rollout(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Advance a time-series of four-channel transport source schedules.

    The returned array has shape ``(n_steps, 4, n_rho)``. Transport
    coefficients are held fixed over the rollout, while ``source_sequence``
    supplies differentiable additive heating, fuelling, and impurity-source
    schedules at each step. This is a bounded controller-tuning primitive, not
    an externally validated integrated-transport campaign.
    """
    facade = _facade()
    profile_array, chi_array, source_array, rho_array, edge_array, _, _ = facade._validate_transport_rollout_inputs(
        initial_profiles,
        chi,
        source_sequence,
        rho,
        dt,
        edge_values,
    )
    use_jax_runtime = facade._resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="differentiable_transport_rollout",
    )
    if use_jax_runtime:
        return _transport_rollout_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return _transport_rollout_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
