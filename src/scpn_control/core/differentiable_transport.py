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

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core import jax_solvers as _jax_solvers

try:
    import jax

    jax.config.update("jax_enable_x64", True)
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
    chi_gradient: FloatArray
    equilibrium_gradient: FloatArray
    radial_weights: FloatArray


@dataclass(frozen=True)
class EquilibriumWeightedTransportRolloutGradient:
    """JAX gradient of equilibrium-weighted multi-step transport rollout loss."""

    loss: float
    source_gradient: FloatArray
    equilibrium_gradient: FloatArray
    radial_weights: FloatArray
    final_profiles: FloatArray


@dataclass(frozen=True)
class GyrokineticTransportClosureResult:
    """Reduced gyrokinetic closure profiles for differentiable transport."""

    chi_e: FloatArray
    chi_i: FloatArray
    d_e: FloatArray
    channel_weights: FloatArray
    source: str
    weights_checksum: str | None


@dataclass(frozen=True)
class TransportParameterGradients:
    """JAX gradients of transport tracking loss for tunable transport inputs."""

    loss: float
    chi_gradient: FloatArray
    source_gradient: FloatArray


@dataclass(frozen=True)
class TransportRolloutSourceGradients:
    """JAX gradients of multi-step transport loss for source schedules."""

    loss: float
    source_gradient: FloatArray
    final_profiles: FloatArray


@dataclass(frozen=True)
class TransportRolloutGradientAudit:
    """Finite-difference audit of multi-step source-rollout gradients."""

    loss: float
    epsilon: float
    tolerance: float
    checked_indices: tuple[tuple[int, int, int], ...]
    source_max_abs_error: float
    passed: bool


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
class TransportRuntimeMetadata:
    """Runtime provenance for differentiable transport latency evidence."""

    schema_version: int
    measured_at_unix_s: float
    python_version: str
    platform: str
    machine: str
    processor: str
    jax_version: str
    jaxlib_version: str
    jax_default_backend: str
    jax_devices: tuple[str, ...]
    jax_enable_x64: bool


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
    runtime_metadata: TransportRuntimeMetadata
    audit: TransportGradientAudit
    claim_status: str


@dataclass(frozen=True)
class TransportRolloutGradientLatencyReport:
    """Latency evidence for audited multi-step source-rollout gradients."""

    schema_version: int
    backend: str
    dtype: str
    n_rho: int
    n_steps: int
    channel_count: int
    warmup_runs: int
    timed_runs: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    runtime_metadata: TransportRuntimeMetadata
    audit: TransportRolloutGradientAudit
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


@dataclass(frozen=True)
class TransportDifferentiabilityEvidence:
    """Tamper-evident admission evidence for differentiable transport gradients."""

    schema_version: int
    backend: str
    campaign_sha256: str
    gradient_audit_sha256: str
    gradient_tolerance: float
    audit_kind: str
    audit_passed: bool
    n_rho: int
    channel_order: tuple[str, ...]
    equilibrium_coupled: bool
    controller_formal_artifact_sha256: str | None
    claim_status: str


@dataclass(frozen=True)
class TransportFullFidelityReadinessEvidence:
    """Fail-closed readiness evidence for full-fidelity transport claims."""

    schema_version: int
    backend: str
    campaign_sha256: str
    gradient_latency_report_sha256: str
    gradient_audit_sha256: str
    rollout_latency_report_sha256: str | None
    rollout_audit_sha256: str | None
    external_reference_artifact_sha256: str | None
    external_reference_admitted: bool
    controller_formal_artifact_sha256: str | None
    n_rho: int
    rollout_steps: int | None
    channel_order: tuple[str, ...]
    equilibrium_coupled: bool
    full_fidelity_claim_admissible: bool
    blocked_reasons: tuple[str, ...]
    claim_status: str


def has_jax() -> bool:
    """Return whether the differentiable JAX transport path is available."""
    return _HAS_JAX


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


def _canonical_sha256(value: Any) -> str:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _is_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_optional_sha256(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _is_sha256_hex(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value.lower()


def _as_float_array(name: str, value: Any) -> FloatArray:
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
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray | None, FloatArray | None]:
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


def _closure_profile(name: str, value: Any) -> FloatArray:
    arr = _as_float_array(name, value)
    if arr.ndim != 1 or arr.size < 3:
        raise ValueError(f"{name} must be a one-dimensional profile with at least three points")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def _closure_channel_weights(name: str, value: Any, n_rho: int) -> FloatArray:
    channel_weights = _as_float_array(name, value)
    if channel_weights.shape != (3, n_rho):
        raise ValueError(f"{name} must have shape (3, n_rho)")
    if np.any(channel_weights < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if not np.allclose(channel_weights.sum(axis=0), 1.0, rtol=1.0e-9, atol=1.0e-12):
        raise ValueError(f"{name} must sum to one at each radius")
    return channel_weights


def _three_channel_transport_coefficients_from_closure(
    closure: Any,
    *,
    closure_name: str,
    impurity_diffusivity_fraction: float,
    chi_floor: float,
) -> FloatArray:
    fraction = float(impurity_diffusivity_fraction)
    floor = float(chi_floor)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("impurity_diffusivity_fraction must be finite and in [0, 1]")
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("chi_floor must be non-negative and finite")

    chi_e = _closure_profile(f"{closure_name}.chi_e", closure.chi_e)
    chi_i = _closure_profile(f"{closure_name}.chi_i", closure.chi_i)
    d_e = _closure_profile(f"{closure_name}.d_e", closure.d_e)
    if chi_i.shape != chi_e.shape or d_e.shape != chi_e.shape:
        raise ValueError(f"{closure_name} chi_e, chi_i, and d_e profiles must have the same shape")
    _closure_channel_weights(f"{closure_name}.channel_weights", closure.channel_weights, chi_e.size)

    coefficients = np.stack(
        [
            chi_e,
            chi_i,
            d_e,
            fraction * d_e,
        ]
    )
    return np.asarray(np.maximum(floor, coefficients), dtype=float)


def transport_coefficients_from_neural_closure(
    closure: Any,
    *,
    impurity_diffusivity_fraction: float = 1.0,
    chi_floor: float = 0.0,
) -> FloatArray:
    """Map a neural transport closure into four facade coefficient channels.

    The facade channel order is electron temperature, ion temperature, electron
    density, and impurity density.  Neural transport closures provide electron
    heat, ion heat, and particle diffusivity profiles; the electron-density
    channel uses the particle diffusivity, while the impurity-density channel
    uses a declared bounded fraction of the same particle diffusivity until
    species-resolved impurity transport is externally validated.
    """
    return _three_channel_transport_coefficients_from_closure(
        closure,
        closure_name="closure",
        impurity_diffusivity_fraction=impurity_diffusivity_fraction,
        chi_floor=chi_floor,
    )


def gyrokinetic_transport_closure_profiles(
    model: Any,
    rho: Any,
    profiles: dict[str, Any],
) -> GyrokineticTransportClosureResult:
    """Wrap a reduced gyrokinetic profile closure for controller tuning.

    ``model`` must provide the existing ``evaluate_profile(rho, profiles)``
    contract and return ion heat, electron heat, and particle diffusivity
    profiles. This adapter validates only the CONTROL boundary and does not
    promote the reduced GK model to an externally validated transport claim.
    """
    rho_array = _as_float_array("rho", rho)
    if rho_array.ndim != 1 or rho_array.size < 3:
        raise ValueError("rho must be a one-dimensional profile with at least three points")
    if np.any(np.diff(rho_array) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    if not hasattr(model, "evaluate_profile"):
        raise ValueError("model must provide evaluate_profile(rho, profiles)")
    chi_i_raw, chi_e_raw, d_e_raw = model.evaluate_profile(rho_array, profiles)
    chi_i = _closure_profile("gyrokinetic_closure.chi_i", chi_i_raw)
    chi_e = _closure_profile("gyrokinetic_closure.chi_e", chi_e_raw)
    d_e = _closure_profile("gyrokinetic_closure.d_e", d_e_raw)
    if chi_i.shape != rho_array.shape or chi_e.shape != rho_array.shape or d_e.shape != rho_array.shape:
        raise ValueError("gyrokinetic closure profiles must match rho shape")
    total = chi_e + chi_i + d_e
    safe_total = np.where(total > 0.0, total, 1.0)
    channel_weights = np.stack([chi_e / safe_total, chi_i / safe_total, d_e / safe_total])
    stable_mask = total <= 0.0
    if np.any(stable_mask):
        channel_weights[:, stable_mask] = 1.0 / 3.0
    return GyrokineticTransportClosureResult(
        chi_e=chi_e,
        chi_i=chi_i,
        d_e=d_e,
        channel_weights=np.asarray(channel_weights, dtype=float),
        source="reduced_gyrokinetic",
        weights_checksum=None,
    )


def transport_coefficients_from_gyrokinetic_closure(
    closure: GyrokineticTransportClosureResult,
    *,
    impurity_diffusivity_fraction: float = 1.0,
    chi_floor: float = 0.0,
) -> FloatArray:
    """Map a reduced gyrokinetic closure into four facade channels."""
    return _three_channel_transport_coefficients_from_closure(
        closure,
        closure_name="gyrokinetic_closure",
        impurity_diffusivity_fraction=impurity_diffusivity_fraction,
        chi_floor=chi_floor,
    )


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


def transport_differentiability_evidence(
    metadata: TransportCampaignMetadata,
    audit: TransportGradientAudit | TransportRolloutGradientAudit,
    *,
    controller_formal_artifact_sha256: str | None = None,
) -> TransportDifferentiabilityEvidence:
    """Build tamper-evident evidence for audited differentiable transport."""
    if not isinstance(metadata, TransportCampaignMetadata):
        raise ValueError("metadata must be TransportCampaignMetadata")
    if not isinstance(audit, TransportGradientAudit | TransportRolloutGradientAudit):
        raise ValueError("audit must be a transport gradient audit result")
    if metadata.gradient_tolerance is None:
        raise ValueError("metadata.gradient_tolerance is required for differentiability evidence")
    _validate_transport_gradient_audit(metadata, audit)
    proof_digest = _validate_optional_sha256(
        "controller_formal_artifact_sha256",
        controller_formal_artifact_sha256,
    )
    audit_kind = "rollout_source_gradient" if isinstance(audit, TransportRolloutGradientAudit) else "parameter_gradient"
    return TransportDifferentiabilityEvidence(
        schema_version=1,
        backend=metadata.backend,
        campaign_sha256=_canonical_sha256(metadata),
        gradient_audit_sha256=_canonical_sha256(audit),
        gradient_tolerance=float(metadata.gradient_tolerance),
        audit_kind=audit_kind,
        audit_passed=bool(audit.passed),
        n_rho=int(metadata.n_rho),
        channel_order=metadata.channel_order,
        equilibrium_coupled=metadata.equilibrium_grid_shape is not None,
        controller_formal_artifact_sha256=proof_digest,
        claim_status=(
            "bounded audited differentiable transport evidence only; "
            "not external transport validation or hardware timing evidence"
        ),
    )


def assert_transport_differentiability_claim_admissible(
    evidence: TransportDifferentiabilityEvidence,
    metadata: TransportCampaignMetadata,
    audit: TransportGradientAudit | TransportRolloutGradientAudit,
) -> TransportDifferentiabilityEvidence:
    """Fail closed unless differentiable transport evidence matches inputs."""
    if not isinstance(evidence, TransportDifferentiabilityEvidence):
        raise ValueError("evidence must be TransportDifferentiabilityEvidence")
    if evidence.schema_version != 1:
        raise ValueError("transport differentiability evidence schema_version is unsupported")
    if metadata.backend != "jax" or evidence.backend != "jax":
        raise ValueError("transport differentiability evidence requires JAX backend")
    if not audit.passed or not evidence.audit_passed:
        raise ValueError("transport differentiability evidence requires a passed audit")
    if metadata.gradient_tolerance is None:
        raise ValueError("transport differentiability evidence requires metadata.gradient_tolerance")
    _validate_transport_gradient_audit(metadata, audit)
    if evidence.campaign_sha256 != _canonical_sha256(metadata):
        raise ValueError("transport differentiability evidence campaign_sha256 mismatch")
    if evidence.gradient_audit_sha256 != _canonical_sha256(audit):
        raise ValueError("transport differentiability evidence gradient_audit_sha256 mismatch")
    if evidence.channel_order != metadata.channel_order or evidence.channel_order != CHANNELS:
        raise ValueError("transport differentiability evidence channel_order mismatch")
    if evidence.n_rho != metadata.n_rho:
        raise ValueError("transport differentiability evidence n_rho mismatch")
    if evidence.equilibrium_coupled != (metadata.equilibrium_grid_shape is not None):
        raise ValueError("transport differentiability evidence equilibrium_coupled mismatch")
    if not np.isclose(evidence.gradient_tolerance, metadata.gradient_tolerance, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("transport differentiability evidence gradient_tolerance mismatch")
    _validate_optional_sha256("controller_formal_artifact_sha256", evidence.controller_formal_artifact_sha256)
    # Defence-in-depth: _validate_transport_gradient_audit (called above) already
    # enforces audit.passed == (max_abs_error <= tolerance), and a passed audit is
    # required earlier, so these error-versus-tolerance re-checks cannot fire. They
    # are retained as a redundant fail-closed guard but are unreachable for coverage.
    if isinstance(audit, TransportRolloutGradientAudit):  # pragma: no cover
        if audit.source_max_abs_error > audit.tolerance:
            raise ValueError("transport differentiability rollout source-gradient error exceeds tolerance")
    else:  # pragma: no cover
        if audit.chi_max_abs_error > audit.tolerance or audit.source_max_abs_error > audit.tolerance:
            raise ValueError("transport differentiability parameter-gradient error exceeds tolerance")
    return evidence


def transport_full_fidelity_readiness_evidence(
    metadata: TransportCampaignMetadata,
    gradient_report: TransportGradientLatencyReport,
    *,
    rollout_report: TransportRolloutGradientLatencyReport | None = None,
    external_reference_artifact_sha256: str | None = None,
    external_reference_admitted: bool = False,
    controller_formal_artifact_sha256: str | None = None,
) -> TransportFullFidelityReadinessEvidence:
    """Build fail-closed readiness evidence for full-fidelity transport claims.

    Local gradient audits and timing reports are necessary but not sufficient
    for a full-fidelity claim. This certificate requires a JAX campaign,
    equilibrium coupling, one-step and rollout audit reports, a controller proof
    digest, and an independently admitted external reference artefact.
    """

    if not isinstance(metadata, TransportCampaignMetadata):
        raise ValueError("metadata must be TransportCampaignMetadata")
    if not isinstance(gradient_report, TransportGradientLatencyReport):
        raise ValueError("gradient_report must be TransportGradientLatencyReport")
    if rollout_report is not None and not isinstance(rollout_report, TransportRolloutGradientLatencyReport):
        raise ValueError("rollout_report must be TransportRolloutGradientLatencyReport")
    if not isinstance(external_reference_admitted, bool):
        raise ValueError("external_reference_admitted must be boolean")

    _validate_transport_gradient_latency_report(gradient_report)
    _assert_latency_report_matches_campaign(metadata, gradient_report, report_name="gradient latency report")
    _validate_transport_gradient_audit(metadata, gradient_report.audit)

    rollout_steps: int | None = None
    rollout_report_sha256: str | None = None
    rollout_audit_sha256: str | None = None
    if rollout_report is not None:
        _validate_transport_rollout_gradient_latency_report(rollout_report)
        _assert_latency_report_matches_campaign(metadata, rollout_report, report_name="rollout latency report")
        _validate_transport_gradient_audit(metadata, rollout_report.audit)
        for step, _, rho_index in rollout_report.audit.checked_indices:
            if step < 0 or step >= rollout_report.n_steps or rho_index < 0 or rho_index >= metadata.n_rho:
                raise ValueError("rollout latency report audit indices exceed campaign metadata bounds")
        rollout_steps = int(rollout_report.n_steps)
        rollout_report_sha256 = _canonical_sha256(rollout_report)
        rollout_audit_sha256 = _canonical_sha256(rollout_report.audit)

    external_digest = _validate_optional_sha256(
        "external_reference_artifact_sha256",
        external_reference_artifact_sha256,
    )
    proof_digest = _validate_optional_sha256(
        "controller_formal_artifact_sha256",
        controller_formal_artifact_sha256,
    )

    blocked_reasons: list[str] = []
    # Defence-in-depth: _validate_transport_gradient_latency_report requires a JAX
    # backend for each report and _assert_latency_report_matches_campaign requires
    # the campaign backend to match, so by this point every backend is "jax". This
    # block is a redundant fail-closed guard and cannot add the reason in practice.
    if (  # pragma: no cover
        metadata.backend != "jax"
        or gradient_report.backend != "jax"
        or (rollout_report is not None and rollout_report.backend != "jax")
    ):
        blocked_reasons.append("jax_backend")
    if metadata.equilibrium_grid_shape is None:
        blocked_reasons.append("equilibrium_coupled_campaign")
    if not gradient_report.audit.passed:
        blocked_reasons.append("gradient_latency_audit")
    if rollout_report is None:
        blocked_reasons.append("rollout_latency_report")
    elif not rollout_report.audit.passed:
        blocked_reasons.append("rollout_latency_audit")
    if proof_digest is None:
        blocked_reasons.append("controller_formal_artifact_sha256")
    if external_digest is None:
        blocked_reasons.append("external_reference_artifact_sha256")
    elif not external_reference_admitted:
        blocked_reasons.append("external_reference_admission")

    full_fidelity_admissible = len(blocked_reasons) == 0
    return TransportFullFidelityReadinessEvidence(
        schema_version=1,
        backend=metadata.backend,
        campaign_sha256=_canonical_sha256(metadata),
        gradient_latency_report_sha256=_canonical_sha256(gradient_report),
        gradient_audit_sha256=_canonical_sha256(gradient_report.audit),
        rollout_latency_report_sha256=rollout_report_sha256,
        rollout_audit_sha256=rollout_audit_sha256,
        external_reference_artifact_sha256=external_digest,
        external_reference_admitted=external_reference_admitted,
        controller_formal_artifact_sha256=proof_digest,
        n_rho=int(metadata.n_rho),
        rollout_steps=rollout_steps,
        channel_order=metadata.channel_order,
        equilibrium_coupled=metadata.equilibrium_grid_shape is not None,
        full_fidelity_claim_admissible=full_fidelity_admissible,
        blocked_reasons=tuple(blocked_reasons),
        claim_status=(
            "full-fidelity differentiable transport claim admitted"
            if full_fidelity_admissible
            else "bounded differentiable transport readiness only; full-fidelity claim remains blocked"
        ),
    )


def assert_transport_full_fidelity_claim_ready(
    evidence: TransportFullFidelityReadinessEvidence,
    metadata: TransportCampaignMetadata,
    gradient_report: TransportGradientLatencyReport,
    *,
    rollout_report: TransportRolloutGradientLatencyReport | None = None,
) -> TransportFullFidelityReadinessEvidence:
    """Fail closed unless readiness evidence admits a full-fidelity claim."""

    if not isinstance(evidence, TransportFullFidelityReadinessEvidence):
        raise ValueError("evidence must be TransportFullFidelityReadinessEvidence")
    if evidence.schema_version != 1:
        raise ValueError("transport full-fidelity readiness schema_version is unsupported")
    expected = transport_full_fidelity_readiness_evidence(
        metadata,
        gradient_report,
        rollout_report=rollout_report,
        external_reference_artifact_sha256=evidence.external_reference_artifact_sha256,
        external_reference_admitted=evidence.external_reference_admitted,
        controller_formal_artifact_sha256=evidence.controller_formal_artifact_sha256,
    )
    if evidence != expected:
        raise ValueError("transport full-fidelity readiness evidence digest mismatch")
    if not evidence.full_fidelity_claim_admissible:
        reasons = ", ".join(evidence.blocked_reasons)
        if "external_reference_artifact_sha256" in evidence.blocked_reasons:
            raise ValueError(f"transport full-fidelity claim requires external reference evidence: {reasons}")
        if "external_reference_admission" in evidence.blocked_reasons:
            raise ValueError(f"transport full-fidelity claim requires external reference admission: {reasons}")
        raise ValueError(f"transport full-fidelity claim is not ready: {reasons}")
    return evidence


def _assert_latency_report_matches_campaign(
    metadata: TransportCampaignMetadata,
    report: TransportGradientLatencyReport | TransportRolloutGradientLatencyReport,
    *,
    report_name: str,
) -> None:
    if report.backend != metadata.backend:
        raise ValueError(f"campaign metadata and {report_name} backend mismatch")
    if report.dtype != metadata.dtype:
        raise ValueError(f"campaign metadata and {report_name} dtype mismatch")
    if report.n_rho != metadata.n_rho:
        raise ValueError(f"campaign metadata and {report_name} n_rho mismatch")
    if report.channel_count != len(metadata.channel_order) or metadata.channel_order != CHANNELS:
        raise ValueError(f"campaign metadata and {report_name} channel contract mismatch")
    if metadata.gradient_tolerance is None:
        raise ValueError("campaign metadata gradient_tolerance is required for latency evidence")
    if not np.isclose(report.audit.tolerance, metadata.gradient_tolerance, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError(f"campaign metadata and {report_name} audit tolerance mismatch")


def _validate_transport_gradient_audit(
    metadata: TransportCampaignMetadata,
    audit: TransportGradientAudit | TransportRolloutGradientAudit,
) -> None:
    if metadata.gradient_tolerance is None:
        raise ValueError("transport differentiability evidence requires metadata.gradient_tolerance")
    loss = float(audit.loss)
    epsilon = float(audit.epsilon)
    tolerance = float(audit.tolerance)
    if not np.isfinite(loss) or loss < 0.0:
        raise ValueError("transport gradient audit loss must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("transport gradient audit epsilon must be positive and finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("transport gradient audit tolerance must be positive and finite")
    if not np.isclose(tolerance, metadata.gradient_tolerance, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("transport gradient audit tolerance must match campaign metadata")
    if not isinstance(audit.passed, bool):
        raise ValueError("transport gradient audit passed flag must be boolean")
    if isinstance(audit, TransportRolloutGradientAudit):
        _validate_rollout_audit_indices(audit.checked_indices, metadata.n_rho)
        max_error = float(audit.source_max_abs_error)
    else:
        _validate_parameter_audit_indices(audit.checked_indices, metadata.n_rho)
        chi_error = float(audit.chi_max_abs_error)
        source_error = float(audit.source_max_abs_error)
        if not np.isfinite(chi_error) or chi_error < 0.0:
            raise ValueError("transport gradient audit chi_max_abs_error must be finite and non-negative")
        max_error = max(chi_error, source_error)
    if not np.isfinite(max_error) or max_error < 0.0:
        raise ValueError("transport gradient audit source_max_abs_error must be finite and non-negative")
    if audit.passed != bool(max_error <= tolerance):
        raise ValueError("transport gradient audit passed flag is inconsistent with tolerance")


def _validate_parameter_audit_indices(indices: tuple[tuple[int, int], ...], n_rho: int) -> None:
    if not indices:
        raise ValueError("transport gradient audit checked_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("transport gradient audit checked_indices must be unique")
    for channel, radius in indices:
        if not (0 <= int(channel) < CHANNEL_COUNT and 0 <= int(radius) < int(n_rho)):
            raise ValueError("transport gradient audit checked_indices out of campaign bounds")


def _validate_rollout_audit_indices(indices: tuple[tuple[int, int, int], ...], n_rho: int) -> None:
    if not indices:
        raise ValueError("transport rollout gradient audit checked_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("transport rollout gradient audit checked_indices must be unique")
    for step, channel, radius in indices:
        if int(step) < 0 or not (0 <= int(channel) < CHANNEL_COUNT and 0 <= int(radius) < int(n_rho)):
            raise ValueError("transport rollout gradient audit checked_indices out of campaign bounds")


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
    profiles: AnyFloatArray,
    chi: AnyFloatArray,
    sources: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
) -> FloatArray:
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
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray | None, FloatArray | None]:
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
    initial_profiles: AnyFloatArray,
    chi: AnyFloatArray,
    source_sequence: AnyFloatArray,
    rho: AnyFloatArray,
    dt: float,
    edge_values: AnyFloatArray,
) -> FloatArray:
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
        history = _transport_rollout_jax(
            profile_array,
            chi_array,
            source_candidate,
            rho_array,
            float(dt),
            edge_array,
        )
        residual = history - jnp.asarray(target_array, dtype=jnp.float64)
        return jnp.mean(jnp.asarray(weight_array, dtype=jnp.float64)[None, :, None] * residual * residual)

    loss, gradient = jax.value_and_grad(loss_for_sources)(jnp.asarray(source_array, dtype=jnp.float64))
    history = _transport_rollout_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return TransportRolloutSourceGradients(
        loss=float(np.asarray(loss)),
        source_gradient=np.asarray(gradient),
        final_profiles=np.asarray(history[-1]),
    )


def _rollout_gradient_audit_indices(
    source_shape: tuple[int, ...],
    sample_indices: Any | None,
) -> tuple[tuple[int, int, int], ...]:
    if len(source_shape) != 3:
        raise ValueError("source_sequence must have shape (n_steps, 4, n_rho)")
    n_steps, n_channels, n_rho = source_shape
    if n_steps < 1 or n_channels != CHANNEL_COUNT or n_rho < 3:
        raise ValueError("source_sequence must have shape (n_steps, 4, n_rho) with n_rho >= 3")
    if sample_indices is None:
        candidates: tuple[tuple[int, int, int], ...] = (
            (0, 0, 1),
            (n_steps - 1, 1, n_rho // 2),
            (n_steps // 2, 2, n_rho - 2),
            (n_steps - 1, 3, max(1, n_rho // 3)),
        )
    else:
        try:
            parsed = []
            for raw_index in sample_indices:
                index = tuple(int(part) for part in raw_index)
                if len(index) != 3:
                    raise ValueError
                parsed.append((index[0], index[1], index[2]))
            candidates = tuple(parsed)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_indices must contain three-part rollout source indices") from exc
    unique: list[tuple[int, int, int]] = []
    for step, channel, radius in candidates:
        if not (0 <= step < n_steps and 0 <= channel < n_channels and 0 <= radius < n_rho):
            raise ValueError("sample_indices contain an out-of-range rollout source index")
        index = (int(step), int(channel), int(radius))
        if index not in unique:
            unique.append(index)
    if not unique:
        raise ValueError("sample_indices must contain at least one rollout source index")
    return tuple(unique)


def audit_transport_rollout_source_gradients(
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
) -> TransportRolloutGradientAudit:
    """Compare JAX rollout source gradients with sampled finite differences."""
    epsilon_float = float(epsilon)
    tolerance_float = float(tolerance)
    if not np.isfinite(epsilon_float) or epsilon_float <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(tolerance_float) or tolerance_float <= 0.0:
        raise ValueError("tolerance must be positive and finite")
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
    gradient_result = transport_rollout_source_gradients(
        profile_array,
        chi_array,
        source_array,
        target_array,
        rho_array,
        float(dt),
        edge_array,
        weights=weight_array,
    )
    indices = _rollout_gradient_audit_indices(source_array.shape, sample_indices)
    max_abs_error = 0.0
    for index in indices:
        plus_sources = source_array.copy()
        minus_sources = source_array.copy()
        plus_sources[index] += epsilon_float
        minus_sources[index] -= epsilon_float
        plus_loss = float(
            transport_rollout_tracking_loss(
                profile_array,
                chi_array,
                plus_sources,
                target_array,
                rho_array,
                float(dt),
                edge_array,
                weights=weight_array,
                use_jax=False,
            )
        )
        minus_loss = float(
            transport_rollout_tracking_loss(
                profile_array,
                chi_array,
                minus_sources,
                target_array,
                rho_array,
                float(dt),
                edge_array,
                weights=weight_array,
                use_jax=False,
            )
        )
        finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon_float)
        max_abs_error = max(max_abs_error, abs(float(gradient_result.source_gradient[index]) - finite_difference))
    return TransportRolloutGradientAudit(
        loss=float(gradient_result.loss),
        epsilon=epsilon_float,
        tolerance=tolerance_float,
        checked_indices=indices,
        source_max_abs_error=float(max_abs_error),
        passed=bool(max_abs_error <= tolerance_float),
    )


def assert_transport_rollout_source_gradients_consistent(
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
) -> TransportRolloutGradientAudit:
    """Return rollout source-gradient audit evidence or fail closed."""
    audit = audit_transport_rollout_source_gradients(
        initial_profiles,
        chi,
        source_sequence,
        target_history,
        rho,
        dt,
        edge_values,
        weights=weights,
        epsilon=epsilon,
        tolerance=tolerance,
        sample_indices=sample_indices,
    )
    if not audit.passed:
        raise ValueError("transport rollout source-gradient audit failed")
    return audit


def _equilibrium_weighted_rollout_tracking_loss_jax(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    weights: Any,
) -> Any:
    if jnp is None:
        raise RuntimeError("JAX equilibrium-weighted rollout tracking loss requested but JAX is unavailable")
    history = _transport_rollout_jax(initial_profiles, chi, source_sequence, rho, dt, edge_values)
    radial_weights = _equilibrium_radial_weights_jax(equilibrium_psi, int(history.shape[2]))
    residual = history - jnp.asarray(target_history, dtype=jnp.float64)
    channel_weights = jnp.asarray(weights, dtype=jnp.float64)[None, :, None]
    return jnp.mean(channel_weights * radial_weights[None, None, :] * residual * residual)


def equilibrium_weighted_transport_rollout_tracking_loss(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
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
    """Return multi-step transport rollout loss weighted by GS flux geometry."""
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
    psi_array = _validate_equilibrium_psi(equilibrium_psi)
    use_jax_runtime = _resolve_use_jax(
        use_jax,
        allow_numpy_fallback=allow_numpy_fallback,
        allow_legacy_numpy_fallback=allow_legacy_numpy_fallback,
        context="equilibrium_weighted_transport_rollout_tracking_loss",
    )
    if use_jax_runtime:
        return _equilibrium_weighted_rollout_tracking_loss_jax(
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
    history = _transport_rollout_numpy(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    radial_weights = equilibrium_radial_weights(psi_array, profile_array.shape[1])
    residual = history - target_array
    return float(np.mean(weight_array[None, :, None] * radial_weights[None, None, :] * residual * residual))


def equilibrium_weighted_transport_rollout_source_gradient(
    initial_profiles: Any,
    chi: Any,
    source_sequence: Any,
    target_history: Any,
    rho: Any,
    dt: float,
    edge_values: Any,
    equilibrium_psi: Any,
    *,
    weights: Any | None = None,
) -> EquilibriumWeightedTransportRolloutGradient:
    """Return JAX gradients of GS-weighted rollout loss.

    The returned gradients are with respect to the full source schedule and the
    supplied equilibrium flux map. If the flux map was produced inside an outer
    JAX graph by the Grad-Shafranov solver, this loss is compatible with
    chain-rule propagation through that equilibrium solve.
    """
    if not _HAS_JAX or jax is None or jnp is None:
        raise RuntimeError("equilibrium_weighted_transport_rollout_source_gradient requires JAX")
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
    psi_array = _validate_equilibrium_psi(equilibrium_psi)

    def loss_for_sources_and_equilibrium(source_candidate: Any, psi_candidate: Any) -> Any:
        return _equilibrium_weighted_rollout_tracking_loss_jax(
            profile_array,
            chi_array,
            source_candidate,
            target_array,
            rho_array,
            float(dt),
            edge_array,
            psi_candidate,
            weight_array,
        )

    loss, gradients = jax.value_and_grad(loss_for_sources_and_equilibrium, argnums=(0, 1))(
        jnp.asarray(source_array, dtype=jnp.float64),
        jnp.asarray(psi_array, dtype=jnp.float64),
    )
    source_gradient, equilibrium_gradient = gradients
    history = _transport_rollout_jax(profile_array, chi_array, source_array, rho_array, float(dt), edge_array)
    return EquilibriumWeightedTransportRolloutGradient(
        loss=float(np.asarray(loss)),
        source_gradient=np.asarray(source_gradient),
        equilibrium_gradient=np.asarray(equilibrium_gradient),
        radial_weights=equilibrium_radial_weights(psi_array, profile_array.shape[1]),
        final_profiles=np.asarray(history[-1]),
    )


def _validate_equilibrium_psi(equilibrium_psi: Any) -> FloatArray:
    psi = _as_float_array("equilibrium_psi", equilibrium_psi)
    if psi.ndim != 2 or min(psi.shape) < 3:
        raise ValueError("equilibrium_psi must be a finite two-dimensional flux map with both dimensions >= 3")
    return psi


def equilibrium_radial_weights(equilibrium_psi: Any, n_rho: int) -> FloatArray:
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
) -> tuple[float, FloatArray]:
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
    shape: tuple[int, ...],
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
    parameter_array: AnyFloatArray,
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
    # Defence-in-depth: transport_parameter_gradients (called above) already rejects
    # a missing target_profiles, so this re-check after input validation is a
    # redundant fail-closed guard that cannot be reached.
    if target_array is None:  # pragma: no cover
        raise ValueError("target_profiles is required")
    if weight_array is None:
        weight_array = np.ones(CHANNEL_COUNT)
    indices = _gradient_audit_indices(chi_array.shape, sample_indices)

    def loss_for_chi(candidate: AnyFloatArray) -> float:
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

    def loss_for_sources(candidate: AnyFloatArray) -> float:
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
    warmups = _require_int("warmup_runs", warmup_runs, minimum=0)
    repetitions = _require_int("timed_runs", timed_runs, minimum=1)
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
    warmups = _require_int("warmup_runs", warmup_runs, minimum=0)
    repetitions = _require_int("timed_runs", timed_runs, minimum=1)
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
    checked_sample_indices = _rollout_gradient_audit_indices(source_array.shape, sample_indices)

    def run_audit() -> TransportRolloutGradientAudit:
        return assert_transport_rollout_source_gradients_consistent(
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
    metadata = transport_campaign_metadata(
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


def save_transport_gradient_latency_report(report: TransportGradientLatencyReport, path: str | Path) -> None:
    """Persist differentiable transport gradient-latency evidence as JSON."""

    _validate_transport_gradient_latency_report(report)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_transport_rollout_gradient_latency_report(
    report: TransportRolloutGradientLatencyReport,
    path: str | Path,
) -> None:
    """Persist rollout source-gradient latency evidence as JSON."""

    _validate_transport_rollout_gradient_latency_report(report)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_transport_gradient_latency_report(report: TransportGradientLatencyReport) -> None:
    if not isinstance(report, TransportGradientLatencyReport):
        raise ValueError("transport gradient latency report must be TransportGradientLatencyReport")
    _validate_latency_common(
        schema_version=report.schema_version,
        backend=report.backend,
        n_rho=report.n_rho,
        channel_count=report.channel_count,
        warmup_runs=report.warmup_runs,
        timed_runs=report.timed_runs,
        p50_ms=report.p50_ms,
        p95_ms=report.p95_ms,
        max_ms=report.max_ms,
    )
    _validate_transport_runtime_metadata(report.runtime_metadata)
    metadata = TransportCampaignMetadata(
        backend=report.backend,
        dtype=report.dtype,
        channel_order=CHANNELS,
        n_rho=report.n_rho,
        rho_min=0.0,
        rho_max=1.0,
        rho_spacing=1.0 / float(report.n_rho - 1),
        dt=1.0,
        core_boundary="zero_gradient",
        edge_boundary="dirichlet",
        edge_values=(0.0, 0.0, 0.0, 0.0),
        closure_source=None,
        closure_weights_checksum=None,
        gradient_tolerance=report.audit.tolerance,
        equilibrium_grid_shape=None,
    )
    _validate_transport_gradient_audit(metadata, report.audit)


def _validate_transport_rollout_gradient_latency_report(report: TransportRolloutGradientLatencyReport) -> None:
    if not isinstance(report, TransportRolloutGradientLatencyReport):
        raise ValueError("transport rollout gradient latency report must be TransportRolloutGradientLatencyReport")
    _validate_latency_common(
        schema_version=report.schema_version,
        backend=report.backend,
        n_rho=report.n_rho,
        channel_count=report.channel_count,
        warmup_runs=report.warmup_runs,
        timed_runs=report.timed_runs,
        p50_ms=report.p50_ms,
        p95_ms=report.p95_ms,
        max_ms=report.max_ms,
    )
    _validate_transport_runtime_metadata(report.runtime_metadata)
    _require_int("n_steps", report.n_steps, minimum=1)
    metadata = TransportCampaignMetadata(
        backend=report.backend,
        dtype=report.dtype,
        channel_order=CHANNELS,
        n_rho=report.n_rho,
        rho_min=0.0,
        rho_max=1.0,
        rho_spacing=1.0 / float(report.n_rho - 1),
        dt=1.0,
        core_boundary="zero_gradient",
        edge_boundary="dirichlet",
        edge_values=(0.0, 0.0, 0.0, 0.0),
        closure_source=None,
        closure_weights_checksum=None,
        gradient_tolerance=report.audit.tolerance,
        equilibrium_grid_shape=None,
    )
    _validate_transport_gradient_audit(metadata, report.audit)


def _validate_latency_common(
    *,
    schema_version: int,
    backend: str,
    n_rho: int,
    channel_count: int,
    warmup_runs: int,
    timed_runs: int,
    p50_ms: float,
    p95_ms: float,
    max_ms: float,
) -> None:
    if schema_version != 1:
        raise ValueError("transport latency report schema_version is unsupported")
    if backend != "jax":
        raise ValueError("transport latency report requires JAX backend")
    _require_int("n_rho", n_rho, minimum=3)
    if _require_int("channel_count", channel_count, minimum=1) != CHANNEL_COUNT:
        raise ValueError("transport latency report channel_count is invalid")
    _require_int("warmup_runs", warmup_runs, minimum=0)
    _require_int("timed_runs", timed_runs, minimum=1)
    p50 = _require_nonnegative_finite("p50_ms", p50_ms)
    p95 = _require_nonnegative_finite("p95_ms", p95_ms)
    maximum = _require_nonnegative_finite("max_ms", max_ms)
    if not (p50 <= p95 <= maximum):
        raise ValueError("transport latency report percentiles must satisfy p50 <= p95 <= max")


def _validate_transport_runtime_metadata(metadata: TransportRuntimeMetadata) -> None:
    if not isinstance(metadata, TransportRuntimeMetadata):
        raise ValueError("transport latency report runtime_metadata is invalid")
    if metadata.schema_version != 1:
        raise ValueError("transport runtime metadata schema_version is unsupported")
    _require_nonnegative_finite("measured_at_unix_s", metadata.measured_at_unix_s)
    for name in (
        "python_version",
        "platform",
        "machine",
        "jax_version",
        "jaxlib_version",
        "jax_default_backend",
    ):
        value = getattr(metadata, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"transport runtime metadata {name} must be a non-empty string")
    if not isinstance(metadata.processor, str):
        raise ValueError("transport runtime metadata processor must be a string")
    if not isinstance(metadata.jax_devices, tuple) or not metadata.jax_devices:
        raise ValueError("transport runtime metadata jax_devices must be a non-empty tuple")
    if not all(isinstance(device, str) and device for device in metadata.jax_devices):
        raise ValueError("transport runtime metadata jax_devices must contain non-empty strings")
    if not isinstance(metadata.jax_enable_x64, bool):
        raise ValueError("transport runtime metadata jax_enable_x64 must be boolean")


def _require_nonnegative_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _require_int(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


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
