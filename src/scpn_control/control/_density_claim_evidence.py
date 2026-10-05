# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density-control claim evidence and admission

"""Density-control claim evidence and admission."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from scpn_control._typing import AnyFloatArray
from scpn_control.control._density_control_runtime import ActuatorCommand, DensityController
from scpn_control.control._density_transport import ParticleTransportModel

_DENSITY_CLAIM_SCHEMA_VERSION = 2
_FACILITY_DENSITY_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "measured_discharge", "external_particle_balance", "facility_replay"}
)
_BOUNDED_DENSITY_REFERENCE_SOURCES = frozenset({"synthetic_regression_reference", *_FACILITY_DENSITY_REFERENCE_SOURCES})


@dataclass(frozen=True)
class DensityControlClaimEvidence:
    """Serialisable provenance and reference-comparison evidence for density-control claims."""

    schema_version: int
    source: str
    source_id: str
    geometry_source: str
    transport_source: str
    actuator_source: str
    diagnostic_source: str
    model_id: str
    n_rho: int
    R0_m: float
    a_m: float
    dt_requested_s: float
    dt_cfl_s: float | None
    cfl_limited: bool
    greenwald_fraction: float
    below_iter_greenwald_margin: bool
    gas_puff_rate: float
    pellet_frequency: float
    pellet_speed: float
    cryopump_speed: float
    total_source_particles_per_s: float
    particle_inventory_delta: float
    greenwald_fraction_abs_error: float | None
    inventory_delta_relative_error: float | None
    greenwald_fraction_abs_tolerance: float
    inventory_delta_relative_tolerance: float
    reference_comparison_passed: bool
    facility_density_claim_allowed: bool
    claim_status: str


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _relative_error(name: str, observed: float, reference: float | None) -> float | None:
    if reference is None:
        return None
    ref = DensityController._validate_non_negative_scalar(reference, name)
    obs = DensityController._validate_non_negative_scalar(observed, f"observed_{name}")
    return abs(obs - ref) / max(abs(ref), 1e-30)


def density_control_claim_evidence(
    model: ParticleTransportModel,
    controller: DensityController,
    *,
    source: str,
    source_id: str,
    geometry_source: str,
    transport_source: str,
    actuator_source: str,
    diagnostic_source: str,
    ne_before: AnyFloatArray,
    ne_after: AnyFloatArray,
    sources: AnyFloatArray,
    command: ActuatorCommand,
    dt_requested_s: float,
    model_id: str = "bounded_density_control",
    reference_greenwald_fraction: float | None = None,
    reference_inventory_delta: float | None = None,
    greenwald_fraction_abs_tolerance: float = 0.02,
    inventory_delta_relative_tolerance: float = 0.05,
) -> DensityControlClaimEvidence:
    """Build fail-closed density-control evidence for bounded or facility claims."""
    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_DENSITY_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_DENSITY_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")

    ne_before_arr = controller._validate_density_profile(ne_before, "ne_before")
    ne_after_arr = controller._validate_density_profile(ne_after, "ne_after")
    sources_arr = model._validate_profile(sources, "sources")
    if np.any(sources_arr < 0.0):
        raise ValueError("sources must be non-negative")

    dt_requested = DensityController._validate_positive_scalar(dt_requested_s, "dt_requested_s")
    greenwald_tol = DensityController._validate_positive_scalar(
        greenwald_fraction_abs_tolerance, "greenwald_fraction_abs_tolerance"
    )
    inventory_tol = DensityController._validate_positive_scalar(
        inventory_delta_relative_tolerance, "inventory_delta_relative_tolerance"
    )

    d_max = float(np.max(model.D))
    dt_cfl = None
    cfl_limited = False
    if d_max > 0.0:
        dt_cfl = float((model.drho * model.a) ** 2 / (2.0 * d_max))
        cfl_limited = dt_requested > dt_cfl

    greenwald_fraction = controller.greenwald_fraction(
        ne_after_arr, I_p_MA=controller.n_GW * math.pi * model.a**2 / 1e20, a=model.a
    )
    below_margin = controller.below_greenwald_safety_margin(ne_after_arr)
    volume_weights = model.V_prime * model.drho
    total_source = float(np.sum(sources_arr * volume_weights))
    inventory_delta = float(np.sum((ne_after_arr - ne_before_arr) * volume_weights))
    greenwald_error = None
    if reference_greenwald_fraction is not None:
        ref_gw = DensityController._validate_non_negative_scalar(
            reference_greenwald_fraction, "reference_greenwald_fraction"
        )
        greenwald_error = abs(greenwald_fraction - ref_gw)
    inventory_error = _relative_error(
        "reference_inventory_delta",
        abs(inventory_delta),
        None if reference_inventory_delta is None else abs(reference_inventory_delta),
    )

    references_pass = (
        greenwald_error is not None
        and inventory_error is not None
        and greenwald_error <= greenwald_tol
        and inventory_error <= inventory_tol
        and below_margin
    )
    # A caller can supply both the reference values and their provenance labels.
    # Numerical agreement alone cannot establish an independent facility witness.
    claim_status = (
        "bounded_density_reference_matched"
        if source_clean in _FACILITY_DENSITY_REFERENCE_SOURCES and references_pass
        else "bounded_density_control_evidence"
    )

    return DensityControlClaimEvidence(
        schema_version=_DENSITY_CLAIM_SCHEMA_VERSION,
        source=source_clean,
        source_id=_non_empty_text("source_id", source_id),
        geometry_source=_non_empty_text("geometry_source", geometry_source),
        transport_source=_non_empty_text("transport_source", transport_source),
        actuator_source=_non_empty_text("actuator_source", actuator_source),
        diagnostic_source=_non_empty_text("diagnostic_source", diagnostic_source),
        model_id=_non_empty_text("model_id", model_id),
        n_rho=model.n_rho,
        R0_m=float(model.R0),
        a_m=float(model.a),
        dt_requested_s=dt_requested,
        dt_cfl_s=dt_cfl,
        cfl_limited=bool(cfl_limited),
        greenwald_fraction=float(greenwald_fraction),
        below_iter_greenwald_margin=bool(below_margin),
        gas_puff_rate=DensityController._validate_non_negative_scalar(command.gas_puff_rate, "gas_puff_rate"),
        pellet_frequency=DensityController._validate_non_negative_scalar(command.pellet_freq, "pellet_freq"),
        pellet_speed=DensityController._validate_non_negative_scalar(command.pellet_speed, "pellet_speed"),
        cryopump_speed=DensityController._validate_non_negative_scalar(command.cryo_pump_speed, "cryo_pump_speed"),
        total_source_particles_per_s=total_source,
        particle_inventory_delta=inventory_delta,
        greenwald_fraction_abs_error=greenwald_error,
        inventory_delta_relative_error=inventory_error,
        greenwald_fraction_abs_tolerance=greenwald_tol,
        inventory_delta_relative_tolerance=inventory_tol,
        reference_comparison_passed=bool(references_pass),
        facility_density_claim_allowed=False,
        claim_status=claim_status,
    )


def assert_density_control_facility_claim_admissible(evidence: DensityControlClaimEvidence) -> None:
    """Reject facility promotion until an independent witness can be verified."""
    raise ValueError("facility density-control claim requires independently verified reference evidence")


def save_density_control_claim_evidence(evidence: DensityControlClaimEvidence, path: str | Path) -> None:
    """Persist density-control claim evidence as deterministic JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
