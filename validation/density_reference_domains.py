# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Density reference artifact validator

"""Check declared density geometry, units, actuator domains and caller-supplied metric tolerances.

These original predicates authenticate no mesh, shot, reference bytes or model run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard

_REQUIRED_GRID_FIELDS = ("n_rho", "major_radius_m", "minor_radius_m")


_REQUIRED_ACTUATOR_FIELDS = (
    "gas_puff_rate_particles_s",
    "pellet_radius_mm",
    "pellet_speed_m_s",
    "nbi_energy_keV",
    "nbi_power_MW",
    "cryopump_speed_m3_s",
    "recycling_coefficient",
)


_REQUIRED_UNITS = {
    "density": "m^-3",
    "particle_rate": "s^-1",
    "radius": "m",
    "diffusivity": "m^2/s",
    "pinch_velocity": "m/s",
    "time": "s",
    "greenwald_fraction": "1",
}


_MAXIMUM_ERROR_METRICS = (
    "pellet_deposition_rmse",
    "recycling_source_relative_error",
    "greenwald_fraction_abs_error",
    "density_profile_relative_error",
)


def _validate_metric_block(path: Path, metrics: object, tolerances: object, errors: list[dict[str, object]]) -> None:
    """Compare the four required declared finite errors to their positive declared tolerances."""
    if not isinstance(metrics, dict):
        errors.append({"path": str(path), "field": "metrics", "error": "metrics must be an object"})
        return
    if not isinstance(tolerances, dict):
        errors.append({"path": str(path), "field": "tolerances", "error": "tolerances must be an object"})
        return
    for field in _MAXIMUM_ERROR_METRICS:
        metric = metrics.get(field)
        tolerance = tolerances.get(field)
        if not _is_nonnegative_finite(metric):
            errors.append({"path": str(path), "field": field, "error": "metric must be finite and non-negative"})
            continue
        if not _is_positive_finite(tolerance):
            errors.append({"path": str(path), "field": field, "error": "tolerance must be finite and positive"})
            continue
        if float(metric) > float(tolerance):
            errors.append({"path": str(path), "field": field, "error": "metric exceeds declared tolerance"})


def _valid_radial_grid(value: object) -> bool:
    """Require integer n_rho >= 2 and positive finite radii; do not infer mesh samples or a < R."""
    if not isinstance(value, dict):
        return False
    n_rho = value.get("n_rho")
    if isinstance(n_rho, bool) or not isinstance(n_rho, int) or n_rho < 2:
        return False
    return all(_is_positive_finite(value.get(field)) for field in _REQUIRED_GRID_FIELDS if field != "n_rho")


def _valid_actuator_metadata(value: object) -> bool:
    """Require finite nonnegative actuator values and a recycling fraction at most one."""
    if not isinstance(value, dict):
        return False
    if not all(_is_nonnegative_finite(value.get(field)) for field in _REQUIRED_ACTUATOR_FIELDS):
        return False
    recycling = value.get("recycling_coefficient")
    return _is_nonnegative_finite(recycling) and float(recycling) <= 1.0


def _valid_units(value: object) -> bool:
    """Require the seven literal unit labels without conversions or validation of unused keys."""
    return isinstance(value, dict) and all(value.get(field) == unit for field, unit in _REQUIRED_UNITS.items())


def _has_public_reference(payload: dict[str, object]) -> bool:
    """Accept any nonblank declared DOI or URL string without resolving or validating its syntax."""
    return any(_has_nonempty_str(payload, field) for field in ("reference_url", "reference_doi"))


def _has_nonempty_str(payload: dict[str, object], field: str) -> bool:
    """Check only that the selected declaration field is a string containing nonwhitespace text."""
    value = payload.get(field)
    return isinstance(value, str) and bool(value.strip())


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Recognize nonboolean int/float values whose float conversion is finite and representable."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_nonnegative_finite(value: object) -> TypeGuard[int | float]:
    """Recognize representable finite real numbers at least zero, rejecting booleans."""
    return _is_finite_number(value) and float(value) >= 0.0


def _is_positive_finite(value: object) -> TypeGuard[int | float]:
    """Recognize representable finite real numbers greater than zero, rejecting booleans."""
    return _is_finite_number(value) and float(value) > 0.0
