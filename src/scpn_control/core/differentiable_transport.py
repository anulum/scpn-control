# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport facade

"""JAX-first differentiable multi-channel transport facade.

The facade advances electron temperature, ion temperature, electron density,
and impurity density through the existing cylindrical Crank-Nicolson transport
primitive. JAX mode keeps the full step and tracking loss inside the traced
graph so controller tuning can differentiate transport-coefficient schedules.
The NumPy path is deterministic and intentionally does not claim gradients.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_control.core import jax_solvers as _jax_solvers

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    _HAS_JAX = False

CHANNEL_COUNT = 4
CHANNELS = ("electron_temperature", "ion_temperature", "electron_density", "impurity_density")


def has_jax() -> bool:
    """Return whether the differentiable JAX transport path is available."""
    return _HAS_JAX


def _as_float_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_transport_inputs(
    profiles: Any,
    chi: Any,
    sources: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    target_profiles: Any | None = None,
    weights: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    profile_array = _as_float_array("profiles", profiles)
    chi_array = _as_float_array("chi", chi)
    source_array = _as_float_array("sources", sources)
    rho_array = _as_float_array("rho", rho)
    edge_array = _as_float_array("edge_values", edge_values)

    if profile_array.ndim != 2 or profile_array.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"profiles must have shape ({CHANNEL_COUNT}, n_rho)")
    if chi_array.shape != profile_array.shape:
        raise ValueError("chi must match profiles shape")
    if source_array.shape != profile_array.shape:
        raise ValueError("sources must match profiles shape")
    if rho_array.ndim != 1 or rho_array.shape[0] != profile_array.shape[1] or rho_array.shape[0] < 3:
        raise ValueError("rho must be one-dimensional with the same radial length as profiles")
    if edge_array.shape != (CHANNEL_COUNT,):
        raise ValueError(f"edge_values must have shape ({CHANNEL_COUNT},)")
    if float(dt) <= 0.0 or not np.isfinite(float(dt)):
        raise ValueError("dt must be positive and finite")
    if np.any(chi_array < 0.0):
        raise ValueError("chi must be non-negative")

    rho_steps = np.diff(rho_array)
    if np.any(rho_steps <= 0.0):
        raise ValueError("rho must be strictly increasing")
    if not np.allclose(rho_steps, rho_steps[0], rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("rho must use a uniform normalised radial spacing")

    target_array = None
    if target_profiles is not None:
        target_array = _as_float_array("target_profiles", target_profiles)
        if target_array.shape != profile_array.shape:
            raise ValueError("target_profiles must match profiles shape")

    weight_array = None
    if weights is not None:
        weight_array = _as_float_array("weights", weights)
        if weight_array.shape != (CHANNEL_COUNT,):
            raise ValueError(f"weights must have shape ({CHANNEL_COUNT},)")
        if np.any(weight_array < 0.0):
            raise ValueError("weights must be non-negative")

    return profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array


def _resolve_use_jax(
    use_jax: bool,
    *,
    allow_numpy_fallback: bool,
    allow_legacy_numpy_fallback: bool,
    context: str,
) -> bool:
    if not use_jax:
        return False
    if _HAS_JAX:
        return True
    return _jax_solvers._resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context=context,
    )


def _transport_step_numpy(
    profiles: np.ndarray,
    chi: np.ndarray,
    sources: np.ndarray,
    rho: np.ndarray,
    dt: float,
    edge_values: np.ndarray,
) -> np.ndarray:
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
    if jnp is None or jax is None:
        raise RuntimeError("JAX transport step requested but JAX is unavailable")
    rho_jax = jnp.asarray(rho, dtype=jnp.float64)
    drho = rho_jax[1] - rho_jax[0]
    step = jax.vmap(_jax_solvers._cn_step_jax, in_axes=(0, 0, 0, None, None, None, 0))
    return step(
        jnp.asarray(profiles, dtype=jnp.float64),
        jnp.asarray(chi, dtype=jnp.float64),
        jnp.asarray(sources, dtype=jnp.float64),
        rho_jax,
        drho,
        float(dt),
        jnp.asarray(edge_values, dtype=jnp.float64),
    )


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
    profile_array, chi_array, source_array, rho_array, edge_array, _, _ = _validate_transport_inputs(
        profiles,
        chi,
        sources,
        rho,
        dt,
        edge_values,
    )
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="differentiable_transport_step",
    )
    if use_jax_runtime:
        return _transport_step_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return _transport_step_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)


def _tracking_loss_jax(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    weights: Any,
) -> Any:
    if jnp is None:
        raise RuntimeError("JAX tracking loss requested but JAX is unavailable")
    predicted = _transport_step_jax(profiles, chi, sources, rho, dt, edge_values)
    residual = predicted - jnp.asarray(target_profiles, dtype=jnp.float64)
    return jnp.mean(jnp.asarray(weights, dtype=jnp.float64)[:, None] * residual * residual)


def transport_tracking_loss(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Return weighted one-step transport tracking loss for controller tuning."""
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = _validate_transport_inputs(
        profiles,
        chi,
        sources,
        rho,
        dt,
        edge_values,
        target_profiles=target_profiles,
        weights=weights,
    )
    if target_array is None:
        raise ValueError("target_profiles is required")
    if weight_array is None:
        weight_array = np.ones(CHANNEL_COUNT)
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="transport_tracking_loss",
    )
    if use_jax_runtime:
        return _tracking_loss_jax(
            profile_array,
            chi_array,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            weight_array,
        )
    predicted = _transport_step_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    residual = predicted - target_array
    return float(np.mean(weight_array[:, None] * residual * residual))


def transport_loss_gradient(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
) -> tuple[float, np.ndarray]:
    """Return the tracking loss and JAX gradient with respect to ``chi``."""
    if not _HAS_JAX or jax is None or jnp is None:
        raise RuntimeError("transport_loss_gradient requires JAX")
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = _validate_transport_inputs(
        profiles,
        chi,
        sources,
        rho,
        dt,
        edge_values,
        target_profiles=target_profiles,
        weights=weights,
    )
    if target_array is None:
        raise ValueError("target_profiles is required")
    if weight_array is None:
        weight_array = np.ones(CHANNEL_COUNT)

    def loss_for_chi(chi_candidate: Any) -> Any:
        return _tracking_loss_jax(
            profile_array,
            chi_candidate,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            weight_array,
        )

    loss, gradient = jax.value_and_grad(loss_for_chi)(jnp.asarray(chi_array, dtype=jnp.float64))
    return float(np.asarray(loss)), np.asarray(gradient)
