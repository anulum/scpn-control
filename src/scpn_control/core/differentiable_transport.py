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

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
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
_TRANSPORT_METADATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EquilibriumWeightedTransportGradient:
    """JAX gradient of equilibrium-weighted transport tracking loss."""

    loss: float
    chi_gradient: np.ndarray
    equilibrium_gradient: np.ndarray
    radial_weights: np.ndarray


@dataclass(frozen=True)
class TransportParameterGradients:
    """JAX gradients of transport tracking loss for tunable transport inputs."""

    loss: float
    chi_gradient: np.ndarray
    source_gradient: np.ndarray


@dataclass(frozen=True)
class TransportRolloutSourceGradients:
    """JAX gradients of multi-step transport loss for source schedules."""

    loss: float
    source_gradient: np.ndarray
    final_profiles: np.ndarray


@dataclass(frozen=True)
class TransportGradientAudit:
    """Finite-difference audit of differentiable transport tuning gradients."""

    loss: float
    epsilon: float
    tolerance: float
    checked_indices: tuple[tuple[int, int], ...]
    chi_max_abs_error: float
    source_max_abs_error: float
    passed: bool


@dataclass(frozen=True)
class TransportGradientLatencyReport:
    """Latency evidence for audited differentiable transport tuning gradients."""

    schema_version: int
    backend: str
    dtype: str
    n_rho: int
    channel_count: int
    warmup_runs: int
    timed_runs: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    audit: TransportGradientAudit
    claim_status: str


@dataclass(frozen=True)
class TransportCampaignMetadata:
    """Validated provenance for a differentiable transport tuning campaign."""

    backend: str
    dtype: str
    channel_order: tuple[str, ...]
    n_rho: int
    rho_min: float
    rho_max: float
    rho_spacing: float
    dt: float
    core_boundary: str
    edge_boundary: str
    edge_values: tuple[float, ...]
    closure_source: str | None
    closure_weights_checksum: str | None
    gradient_tolerance: float | None
    equilibrium_grid_shape: tuple[int, int] | None


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


def _closure_profile(name: str, value: Any) -> np.ndarray:
    arr = _as_float_array(name, value)
    if arr.ndim != 1 or arr.size < 3:
        raise ValueError(f"{name} must be a one-dimensional profile with at least three points")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def transport_coefficients_from_neural_closure(
    closure: Any,
    *,
    impurity_diffusivity_fraction: float = 1.0,
    chi_floor: float = 0.0,
) -> np.ndarray:
    """Map a neural transport closure into four facade coefficient channels.

    The facade channel order is electron temperature, ion temperature, electron
    density, and impurity density.  Neural transport closures provide electron
    heat, ion heat, and particle diffusivity profiles; the electron-density
    channel uses the particle diffusivity, while the impurity-density channel
    uses a declared bounded fraction of the same particle diffusivity until
    species-resolved impurity transport is externally validated.
    """
    fraction = float(impurity_diffusivity_fraction)
    floor = float(chi_floor)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("impurity_diffusivity_fraction must be finite and in [0, 1]")
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("chi_floor must be non-negative and finite")

    chi_e = _closure_profile("closure.chi_e", closure.chi_e)
    chi_i = _closure_profile("closure.chi_i", closure.chi_i)
    d_e = _closure_profile("closure.d_e", closure.d_e)
    if chi_i.shape != chi_e.shape or d_e.shape != chi_e.shape:
        raise ValueError("closure chi_e, chi_i, and d_e profiles must have the same shape")
    channel_weights = _as_float_array("closure.channel_weights", closure.channel_weights)
    if channel_weights.shape != (3, chi_e.size):
        raise ValueError("closure.channel_weights must have shape (3, n_rho)")
    if np.any(channel_weights < 0.0):
        raise ValueError("closure.channel_weights must be non-negative")
    if not np.allclose(channel_weights.sum(axis=0), 1.0, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError("closure.channel_weights must sum to one at each radius")

    coefficients = np.stack(
        [
            chi_e,
            chi_i,
            d_e,
            fraction * d_e,
        ]
    )
    return np.asarray(np.maximum(floor, coefficients), dtype=float)


def transport_campaign_metadata(
    profiles: Any,
    chi: Any,
    sources: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    backend: str,
    closure: Any | None = None,
    gradient_tolerance: float | None = None,
    equilibrium_psi: Any | None = None,
) -> TransportCampaignMetadata:
    """Return serialisable provenance for differentiable transport campaigns."""
    backend_value = str(backend).strip().lower()
    if backend_value not in {"numpy", "jax"}:
        raise ValueError("backend must be either 'numpy' or 'jax'")
    tolerance_value: float | None = None
    if gradient_tolerance is not None:
        tolerance_value = float(gradient_tolerance)
        if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
            raise ValueError("gradient_tolerance must be positive and finite")

    profile_array, chi_array, source_array, rho_array, edge_array, _, _ = _validate_transport_inputs(
        profiles,
        chi,
        sources,
        rho,
        dt,
        edge_values,
    )
    dtype_name = np.result_type(
        profile_array.dtype,
        chi_array.dtype,
        source_array.dtype,
        rho_array.dtype,
        edge_array.dtype,
    ).name
    closure_source: str | None = None
    closure_weights_checksum: str | None = None
    if closure is not None:
        closure_source = str(closure.source)
        checksum = closure.weights_checksum
        closure_weights_checksum = None if checksum is None else str(checksum)
    equilibrium_grid_shape: tuple[int, int] | None = None
    if equilibrium_psi is not None:
        psi_array = _validate_equilibrium_psi(equilibrium_psi)
        equilibrium_grid_shape = (int(psi_array.shape[0]), int(psi_array.shape[1]))

    return TransportCampaignMetadata(
        backend=backend_value,
        dtype=dtype_name,
        channel_order=CHANNELS,
        n_rho=int(rho_array.size),
        rho_min=float(rho_array[0]),
        rho_max=float(rho_array[-1]),
        rho_spacing=float(rho_array[1] - rho_array[0]),
        dt=float(dt),
        core_boundary="zero_gradient",
        edge_boundary="dirichlet",
        edge_values=tuple(float(x) for x in edge_array),
        closure_source=closure_source,
        closure_weights_checksum=closure_weights_checksum,
        gradient_tolerance=tolerance_value,
        equilibrium_grid_shape=equilibrium_grid_shape,
    )


def _finite_float_field(name: str, value: Any, *, positive: bool = False) -> float:
    field = float(value)
    if not np.isfinite(field):
        raise ValueError(f"metadata field {name} must be finite")
    if positive and field <= 0.0:
        raise ValueError(f"metadata field {name} must be positive")
    return field


def _optional_positive_float_field(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float_field(name, value, positive=True)


def _transport_campaign_metadata_from_mapping(payload: dict[str, Any]) -> TransportCampaignMetadata:
    try:
        backend = str(payload["backend"]).strip().lower()
        dtype = str(payload["dtype"])
        channel_order = tuple(str(channel) for channel in payload["channel_order"])
        n_rho = int(payload["n_rho"])
        rho_min = _finite_float_field("rho_min", payload["rho_min"])
        rho_max = _finite_float_field("rho_max", payload["rho_max"])
        rho_spacing = _finite_float_field("rho_spacing", payload["rho_spacing"], positive=True)
        dt_value = _finite_float_field("dt", payload["dt"], positive=True)
        core_boundary = str(payload["core_boundary"])
        edge_boundary = str(payload["edge_boundary"])
        edge_values = tuple(_finite_float_field("edge_values", value) for value in payload["edge_values"])
        closure_source_value = payload["closure_source"]
        closure_checksum_value = payload["closure_weights_checksum"]
        tolerance = _optional_positive_float_field("gradient_tolerance", payload["gradient_tolerance"])
        equilibrium_shape_value = payload["equilibrium_grid_shape"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transport campaign metadata payload is malformed") from exc

    if backend not in {"numpy", "jax"}:
        raise ValueError("transport campaign metadata backend is invalid")
    if channel_order != CHANNELS:
        raise ValueError("transport campaign metadata channel_order is invalid")
    if n_rho < 3:
        raise ValueError("transport campaign metadata n_rho must be >= 3")
    if rho_max <= rho_min:
        raise ValueError("transport campaign metadata rho bounds are invalid")
    if len(edge_values) != CHANNEL_COUNT:
        raise ValueError("transport campaign metadata edge_values length is invalid")
    if core_boundary != "zero_gradient" or edge_boundary != "dirichlet":
        raise ValueError("transport campaign metadata boundary contract is invalid")
    closure_source = None if closure_source_value is None else str(closure_source_value)
    closure_weights_checksum = None if closure_checksum_value is None else str(closure_checksum_value)
    equilibrium_grid_shape: tuple[int, int] | None = None
    if equilibrium_shape_value is not None:
        if not isinstance(equilibrium_shape_value, list | tuple) or len(equilibrium_shape_value) != 2:
            raise ValueError("transport campaign metadata equilibrium_grid_shape is invalid")
        equilibrium_grid_shape = (int(equilibrium_shape_value[0]), int(equilibrium_shape_value[1]))
        if min(equilibrium_grid_shape) < 3:
            raise ValueError("transport campaign metadata equilibrium_grid_shape must be >= 3 in both dimensions")

    return TransportCampaignMetadata(
        backend=backend,
        dtype=dtype,
        channel_order=channel_order,
        n_rho=n_rho,
        rho_min=rho_min,
        rho_max=rho_max,
        rho_spacing=rho_spacing,
        dt=dt_value,
        core_boundary=core_boundary,
        edge_boundary=edge_boundary,
        edge_values=edge_values,
        closure_source=closure_source,
        closure_weights_checksum=closure_weights_checksum,
        gradient_tolerance=tolerance,
        equilibrium_grid_shape=equilibrium_grid_shape,
    )


def save_transport_campaign_metadata(metadata: TransportCampaignMetadata, path: str | Path) -> None:
    """Persist transport campaign metadata as schema-versioned JSON."""
    destination = Path(path)
    payload = {
        "schema_version": _TRANSPORT_METADATA_SCHEMA_VERSION,
        "metadata": asdict(metadata),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_transport_campaign_metadata(path: str | Path) -> TransportCampaignMetadata:
    """Load and validate schema-versioned transport campaign metadata JSON."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("transport campaign metadata file is not readable JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != _TRANSPORT_METADATA_SCHEMA_VERSION:
        raise ValueError("transport campaign metadata schema_version is unsupported")
    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise ValueError("transport campaign metadata payload is malformed")
    return _transport_campaign_metadata_from_mapping(metadata_payload)


def _metadata_field_matches(archived: Any, current: Any) -> bool:
    if archived is None or current is None:
        return archived is current
    if isinstance(archived, float | int) and isinstance(current, float | int):
        return bool(np.isclose(float(archived), float(current), rtol=1.0e-12, atol=1.0e-15))
    if isinstance(archived, tuple) and isinstance(current, tuple):
        if len(archived) != len(current):
            return False
        return all(_metadata_field_matches(left, right) for left, right in zip(archived, current, strict=True))
    return bool(archived == current)


def assert_transport_campaign_metadata_replay(
    archived: TransportCampaignMetadata,
    profiles: Any,
    chi: Any,
    sources: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    backend: str,
    closure: Any | None = None,
    gradient_tolerance: float | None = None,
    equilibrium_psi: Any | None = None,
) -> TransportCampaignMetadata:
    """Validate that a candidate transport setup matches archived metadata.

    This guard is intended for replaying differentiable transport tuning
    campaigns. It fails closed on backend, dtype, grid, timestep, boundary,
    closure-provenance, gradient-tolerance, or equilibrium-shape drift before a
    controller rerun can silently compare against a different physics setup.
    """
    if not isinstance(archived, TransportCampaignMetadata):
        raise ValueError("archived transport campaign metadata must be TransportCampaignMetadata")
    current = transport_campaign_metadata(
        profiles,
        chi,
        sources,
        rho,
        dt,
        edge_values,
        backend=backend,
        closure=closure,
        gradient_tolerance=gradient_tolerance,
        equilibrium_psi=equilibrium_psi,
    )
    archived_fields = asdict(archived)
    current_fields = asdict(current)
    mismatches = [
        field_name
        for field_name, archived_value in archived_fields.items()
        if not _metadata_field_matches(archived_value, current_fields[field_name])
    ]
    if mismatches:
        raise ValueError("transport campaign metadata replay mismatch: " + ", ".join(mismatches))
    return current


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


def _validate_transport_rollout_inputs(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    target_history: Any | None = None,
    weights: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    profile_array = _as_float_array("initial_profiles", initial_profiles)
    chi_array = _as_float_array("chi", chi)
    source_array = _as_float_array("source_sequence", source_sequence)
    rho_array = _as_float_array("rho", rho)
    edge_array = _as_float_array("edge_values", edge_values)

    if profile_array.ndim != 2 or profile_array.shape[0] != CHANNEL_COUNT:
        raise ValueError(f"initial_profiles must have shape ({CHANNEL_COUNT}, n_rho)")
    if chi_array.shape != profile_array.shape:
        raise ValueError("chi must match initial_profiles shape")
    if source_array.ndim != 3 or source_array.shape[1:] != profile_array.shape or source_array.shape[0] < 1:
        raise ValueError("source_sequence must have shape (n_steps, 4, n_rho) with n_steps >= 1")
    if rho_array.ndim != 1 or rho_array.shape[0] != profile_array.shape[1] or rho_array.shape[0] < 3:
        raise ValueError("rho must be one-dimensional with the same radial length as initial_profiles")
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
    if target_history is not None:
        target_array = _as_float_array("target_history", target_history)
        if target_array.shape != source_array.shape:
            raise ValueError("target_history must match source_sequence shape")

    weight_array = None
    if weights is not None:
        weight_array = _as_float_array("weights", weights)
        if weight_array.shape != (CHANNEL_COUNT,):
            raise ValueError(f"weights must have shape ({CHANNEL_COUNT},)")
        if np.any(weight_array < 0.0):
            raise ValueError("weights must be non-negative")

    return profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array


def _transport_rollout_numpy(
    initial_profiles: np.ndarray,
    chi: np.ndarray,
    source_sequence: np.ndarray,
    rho: np.ndarray,
    dt: float,
    edge_values: np.ndarray,
) -> np.ndarray:
    current = initial_profiles
    history: list[np.ndarray] = []
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
    if jnp is None or jax is None:
        raise RuntimeError("JAX transport rollout requested but JAX is unavailable")
    chi_jax = jnp.asarray(chi, dtype=jnp.float64)
    rho_jax = jnp.asarray(rho, dtype=jnp.float64)
    edge_jax = jnp.asarray(edge_values, dtype=jnp.float64)

    def body(carry: Any, source_step: Any) -> tuple[Any, Any]:
        next_profiles = _transport_step_jax(carry, chi_jax, source_step, rho_jax, dt, edge_jax)
        return next_profiles, next_profiles

    _, history = jax.lax.scan(
        body,
        jnp.asarray(initial_profiles, dtype=jnp.float64),
        jnp.asarray(source_sequence, dtype=jnp.float64),
    )
    return history


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
    profile_array, chi_array, source_array, rho_array, edge_array, _, _ = _validate_transport_rollout_inputs(
        initial_profiles,
        chi,
        source_sequence,
        rho,
        dt,
        edge_values,
    )
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="differentiable_transport_rollout",
    )
    if use_jax_runtime:
        return _transport_rollout_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return _transport_rollout_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)


def transport_rollout_tracking_loss(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Return weighted multi-step transport tracking loss."""
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_rollout_inputs(
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
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="transport_rollout_tracking_loss",
    )
    if use_jax_runtime:
        history = _transport_rollout_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
        residual = history - jnp.asarray(target_array, dtype=jnp.float64)
        return jnp.mean(jnp.asarray(weight_array, dtype=jnp.float64)[None, :, None] * residual * residual)
    history = _transport_rollout_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    residual = history - target_array
    return float(np.mean(weight_array[None, :, None] * residual * residual))


def transport_rollout_source_gradients(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
) -> TransportRolloutSourceGradients:
    """Return JAX gradients for a multi-step transport source schedule."""
    if not _HAS_JAX or jax is None or jnp is None:
        raise RuntimeError("transport_rollout_source_gradients requires JAX")
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_rollout_inputs(
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

    def loss_for_sources(source_candidate: Any) -> Any:
        return transport_rollout_tracking_loss(
            profile_array,
            chi_array,
            source_candidate,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            weights=weight_array,
            use_jax=True,
        )

    loss, gradient = jax.value_and_grad(loss_for_sources)(jnp.asarray(source_array, dtype=jnp.float64))
    history = _transport_rollout_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return TransportRolloutSourceGradients(
        loss=float(np.asarray(loss)),
        source_gradient=np.asarray(gradient),
        final_profiles=np.asarray(history[-1]),
    )


def _validate_equilibrium_psi(equilibrium_psi: Any) -> np.ndarray:
    psi = _as_float_array("equilibrium_psi", equilibrium_psi)
    if psi.ndim != 2 or min(psi.shape) < 3:
        raise ValueError("equilibrium_psi must be a finite two-dimensional flux map with both dimensions >= 3")
    return psi


def equilibrium_radial_weights(equilibrium_psi: Any, n_rho: int) -> np.ndarray:
    """Return positive mean-one radial weights from a Grad-Shafranov flux map."""
    psi = _validate_equilibrium_psi(equilibrium_psi)
    if isinstance(n_rho, bool) or int(n_rho) != n_rho or int(n_rho) < 3:
        raise ValueError("n_rho must be an integer >= 3")
    radial_profile = np.mean(np.abs(psi), axis=0)
    if radial_profile.size != int(n_rho):
        src = np.linspace(0.0, 1.0, radial_profile.size)
        dst = np.linspace(0.0, 1.0, int(n_rho))
        radial_profile = np.interp(dst, src, radial_profile)
    radial_profile = np.maximum(radial_profile, 0.0)
    mean_profile = float(np.mean(radial_profile))
    if mean_profile <= 1.0e-30:
        return np.ones(int(n_rho))
    weights = np.asarray(radial_profile / mean_profile, dtype=float)
    return weights


def _equilibrium_radial_weights_jax(equilibrium_psi: Any, n_rho: int) -> Any:
    if jnp is None:
        raise RuntimeError("JAX equilibrium weighting requested but JAX is unavailable")
    psi = jnp.asarray(equilibrium_psi, dtype=jnp.float64)
    radial_profile = jnp.mean(jnp.abs(psi), axis=0)
    if int(radial_profile.shape[0]) != int(n_rho):
        src = jnp.linspace(0.0, 1.0, int(radial_profile.shape[0]))
        dst = jnp.linspace(0.0, 1.0, int(n_rho))
        radial_profile = jnp.interp(dst, src, radial_profile)
    radial_profile = jnp.maximum(radial_profile, 0.0)
    mean_profile = jnp.mean(radial_profile)
    return jnp.where(mean_profile <= 1.0e-30, jnp.ones(int(n_rho)), radial_profile / mean_profile)


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
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_inputs(
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


def _equilibrium_weighted_tracking_loss_jax(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    weights: Any,
) -> Any:
    if jnp is None:
        raise RuntimeError("JAX equilibrium-weighted tracking loss requested but JAX is unavailable")
    predicted = _transport_step_jax(profiles, chi, sources, rho, dt, edge_values)
    radial_weights = _equilibrium_radial_weights_jax(equilibrium_psi, int(predicted.shape[1]))
    residual = predicted - jnp.asarray(target_profiles, dtype=jnp.float64)
    channel_weights = jnp.asarray(weights, dtype=jnp.float64)[:, None]
    return jnp.mean(channel_weights * radial_weights[None, :] * residual * residual)


def equilibrium_weighted_transport_tracking_loss(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    *,
    weights: Any | None = None,
    use_jax: bool = True,
    allow_numpy_fallback: bool = False,
    allow_legacy_numpy_fallback: bool = False,
) -> Any:
    """Return transport tracking loss weighted by GS-equilibrium flux geometry."""
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_inputs(
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
    psi_array = _validate_equilibrium_psi(equilibrium_psi)
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="equilibrium_weighted_transport_tracking_loss",
    )
    if use_jax_runtime:
        return _equilibrium_weighted_tracking_loss_jax(
            profile_array,
            chi_array,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            psi_array,
            weight_array,
        )
    predicted = _transport_step_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    radial_weights = equilibrium_radial_weights(psi_array, profile_array.shape[1])
    residual = predicted - target_array
    return float(np.mean(weight_array[:, None] * radial_weights[None, :] * residual * residual))


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
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_inputs(
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


def transport_parameter_gradients(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    *,
    weights: Any | None = None,
) -> TransportParameterGradients:
    """Return JAX gradients with respect to ``chi`` and source schedules.

    This is the controller-tuning primitive for differentiable auxiliary
    heating, fuelling, and impurity-source schedules.  It keeps the same
    four-channel Crank-Nicolson, source-term, core zero-gradient, and edge
    Dirichlet contracts as :func:`differentiable_transport_step`; unlike
    :func:`transport_loss_gradient`, it exposes gradients for both turbulent
    transport coefficients and additive source terms.
    """
    if not _HAS_JAX or jax is None or jnp is None:
        raise RuntimeError("transport_parameter_gradients requires JAX")
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_inputs(
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

    def loss_for_chi_and_sources(chi_candidate: Any, source_candidate: Any) -> Any:
        return _tracking_loss_jax(
            profile_array,
            chi_candidate,
            source_candidate,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            weight_array,
        )

    loss, gradients = jax.value_and_grad(loss_for_chi_and_sources, argnums=(0, 1))(
        jnp.asarray(chi_array, dtype=jnp.float64),
        jnp.asarray(source_array, dtype=jnp.float64),
    )
    chi_gradient, source_gradient = gradients
    return TransportParameterGradients(
        loss=float(np.asarray(loss)),
        chi_gradient=np.asarray(chi_gradient),
        source_gradient=np.asarray(source_gradient),
    )


def _gradient_audit_indices(
    shape: tuple[int, int],
    sample_indices: Any | None,
) -> tuple[tuple[int, int], ...]:
    if sample_indices is None:
        radial_indices = sorted({1, shape[1] // 2, shape[1] - 2})
        return tuple((channel, radial) for channel in range(shape[0]) for radial in radial_indices)
    indices: list[tuple[int, int]] = []
    for raw_index in sample_indices:
        try:
            channel = int(raw_index[0])
            radial = int(raw_index[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("sample_indices must contain (channel, radial) pairs") from exc
        if channel < 0 or channel >= shape[0] or radial < 0 or radial >= shape[1]:
            raise ValueError("sample_indices contains an out-of-bounds transport index")
        indices.append((channel, radial))
    if not indices:
        raise ValueError("sample_indices must contain at least one transport index")
    return tuple(indices)


def _central_difference_parameter(
    base_loss: float,
    parameter_array: np.ndarray,
    index: tuple[int, int],
    epsilon: float,
    loss_fn: Any,
) -> float:
    if parameter_array[index] - epsilon < 0.0:
        plus = parameter_array.copy()
        plus[index] += epsilon
        return (float(loss_fn(plus)) - base_loss) / epsilon
    plus = parameter_array.copy()
    minus = parameter_array.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    return (float(loss_fn(plus)) - float(loss_fn(minus))) / (2.0 * epsilon)


def audit_transport_parameter_gradients(
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
) -> TransportGradientAudit:
    """Audit JAX transport gradients against independent finite differences.

    The audit is intentionally sampled rather than exhaustive so it can run in
    controller-tuning admission checks. It evaluates deterministic interior
    radial points for every channel by default and compares JAX gradients for
    both turbulent transport coefficients and additive source schedules against
    independently perturbed NumPy losses.
    """
    epsilon_value = float(epsilon)
    tolerance_value = float(tolerance)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    gradient_result = transport_parameter_gradients(
        profiles,
        chi,
        sources,
        target_profiles,
        rho,
        dt,
        edge_values,
        weights=weights,
    )
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_inputs(
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
    indices = _gradient_audit_indices(chi_array.shape, sample_indices)

    def loss_for_chi(candidate: np.ndarray) -> float:
        return float(
            transport_tracking_loss(
                profile_array,
                candidate,
                source_array,
                target_array,
                rho_array,
                float(dt),
                edge_array,
                weights=weight_array,
                use_jax=False,
            )
        )

    def loss_for_sources(candidate: np.ndarray) -> float:
        return float(
            transport_tracking_loss(
                profile_array,
                chi_array,
                candidate,
                target_array,
                rho_array,
                float(dt),
                edge_array,
                weights=weight_array,
                use_jax=False,
            )
        )

    base_loss = loss_for_chi(chi_array)
    chi_errors: list[float] = []
    source_errors: list[float] = []
    for index in indices:
        chi_fd = _central_difference_parameter(base_loss, chi_array, index, epsilon_value, loss_for_chi)
        source_fd = _central_difference_parameter(base_loss, source_array, index, epsilon_value, loss_for_sources)
        chi_errors.append(abs(float(gradient_result.chi_gradient[index]) - chi_fd))
        source_errors.append(abs(float(gradient_result.source_gradient[index]) - source_fd))
    chi_max_error = max(chi_errors)
    source_max_error = max(source_errors)
    passed = bool(chi_max_error <= tolerance_value and source_max_error <= tolerance_value)
    return TransportGradientAudit(
        loss=gradient_result.loss,
        epsilon=epsilon_value,
        tolerance=tolerance_value,
        checked_indices=indices,
        chi_max_abs_error=float(chi_max_error),
        source_max_abs_error=float(source_max_error),
        passed=passed,
    )


def assert_transport_parameter_gradients_consistent(
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
) -> TransportGradientAudit:
    """Return the gradient audit or fail closed when consistency is violated."""
    audit = audit_transport_parameter_gradients(
        profiles,
        chi,
        sources,
        target_profiles,
        rho,
        dt,
        edge_values,
        weights=weights,
        epsilon=epsilon,
        tolerance=tolerance,
        sample_indices=sample_indices,
    )
    if not audit.passed:
        raise ValueError(
            "transport parameter gradient audit failed: "
            f"chi_max_abs_error={audit.chi_max_abs_error:.6g}, "
            f"source_max_abs_error={audit.source_max_abs_error:.6g}, "
            f"tolerance={audit.tolerance:.6g}"
        )
    return audit


def _percentile(sorted_values: list[float], percentile: float) -> float:
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
    warmups = int(warmup_runs)
    repetitions = int(timed_runs)
    if warmups < 0:
        raise ValueError("warmup_runs must be non-negative")
    if repetitions <= 0:
        raise ValueError("timed_runs must be positive")
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_inputs(
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
        return assert_transport_parameter_gradients_consistent(
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
    metadata = transport_campaign_metadata(
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
        audit=audit,
        claim_status="local audited gradient-admission latency only; not a real-time control-loop guarantee",
    )


def save_transport_gradient_latency_report(report: TransportGradientLatencyReport, path: str | Path) -> None:
    """Persist differentiable transport gradient-latency evidence as JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def equilibrium_weighted_transport_loss_gradient(
    profiles: Any,
    chi: Any,
    sources: Any,
    target_profiles: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    *,
    weights: Any | None = None,
) -> EquilibriumWeightedTransportGradient:
    """Return JAX gradients of equilibrium-weighted transport loss.

    The returned gradients are with respect to the transport coefficients and
    the supplied equilibrium flux map. If the flux map was produced inside an
    outer JAX graph by the Grad-Shafranov solver, this loss is compatible with
    further chain-rule propagation through that equilibrium solve.
    """
    if not _HAS_JAX or jax is None or jnp is None:
        raise RuntimeError("equilibrium_weighted_transport_loss_gradient requires JAX")
    profile_array, chi_array, source_array, rho_array, edge_array, target_array, weight_array = (
        _validate_transport_inputs(
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
    psi_array = _validate_equilibrium_psi(equilibrium_psi)

    def loss_for_chi_and_equilibrium(chi_candidate: Any, psi_candidate: Any) -> Any:
        return _equilibrium_weighted_tracking_loss_jax(
            profile_array,
            chi_candidate,
            source_array,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            psi_candidate,
            weight_array,
        )

    loss, gradients = jax.value_and_grad(loss_for_chi_and_equilibrium, argnums=(0, 1))(
        jnp.asarray(chi_array, dtype=jnp.float64),
        jnp.asarray(psi_array, dtype=jnp.float64),
    )
    chi_gradient, equilibrium_gradient = gradients
    return EquilibriumWeightedTransportGradient(
        loss=float(np.asarray(loss)),
        chi_gradient=np.asarray(chi_gradient),
        equilibrium_gradient=np.asarray(equilibrium_gradient),
        radial_weights=equilibrium_radial_weights(psi_array, profile_array.shape[1]),
    )
