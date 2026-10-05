#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK geometry reference case comparisons


"""Compare persisted local Miller cases on the original four-angle grid without changing equations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from scpn_control.core.gk_geometry import miller_geometry
from validation.gk_geometry_reference_contracts import _ABS_TOLERANCE, _REL_TOLERANCE, _json_sha256

_REQUIRED_CASES = {"circular_cyclone_limit", "shaped_positive_triangularity", "high_shear_local_equilibrium"}

_REQUIRED_SAMPLE_FIELDS = (
    "theta",
    "R",
    "Z",
    "jacobian",
    "g_rr",
    "g_rt",
    "g_tt",
    "B_toroidal",
    "b_dot_grad_theta",
)

_GEOMETRY_PARAMETERS = {
    "R0",
    "a",
    "rho",
    "kappa",
    "delta",
    "s_kappa",
    "s_delta",
    "q",
    "s_hat",
    "alpha_MHD",
    "dR_dr",
    "B0",
}


def _validate_case(
    path: Path,
    index: int,
    case_payload: object,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    """Compare original named case metadata and nine sample fields at nearest original-grid theta; invalid values or physical domains become findings, with tolerances unchanged."""
    if not isinstance(case_payload, dict):
        errors.append({"path": str(path), "index": index, "field": "case", "error": "case must be an object"})
        return None
    case_name = case_payload.get("case")
    if not isinstance(case_name, str) or not case_name.strip():
        errors.append({"path": str(path), "index": index, "field": "case", "error": "case must be a non-empty string"})
        return None
    parameters = case_payload.get("parameters")
    sample_points = case_payload.get("sample_points")
    if not isinstance(parameters, dict):
        errors.append(
            {"path": str(path), "index": index, "field": "parameters", "error": "parameters must be an object"}
        )
        return None
    if not isinstance(sample_points, list) or not sample_points:
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "sample_points",
                "error": "sample_points must be a non-empty array",
            }
        )
        return None
    geometry_args = _geometry_arguments(path, index, parameters, errors)
    if geometry_args is None:
        return None
    try:
        geometry = miller_geometry(**geometry_args, n_theta=4, n_period=1)
    except (ValueError, OverflowError):
        errors.append(
            {
                "path": str(path),
                "index": index,
                "field": "parameters",
                "error": "parameters must satisfy the local Miller equilibrium domain",
            }
        )
        return None
    max_abs_error = 0.0
    for sample_index, sample in enumerate(sample_points):
        if not isinstance(sample, dict):
            errors.append(
                {"path": str(path), "index": index, "field": "sample_points", "error": "sample point must be an object"}
            )
            continue
        theta = _finite_sample(sample.get("theta"))
        if theta is None:
            errors.append(
                {"path": str(path), "index": index, "field": "theta", "error": "sample field must be a finite number"}
            )
            continue
        actual = _actual_values_at_theta(
            float(geometry_args["B0"]),
            float(geometry_args["R0"]),
            geometry,
            theta,
        )
        for field in _REQUIRED_SAMPLE_FIELDS:
            expected_value = _finite_sample(sample.get(field))
            if expected_value is None or not math.isfinite(actual[field]):
                errors.append(
                    {
                        "path": str(path),
                        "index": index,
                        "field": field,
                        "error": "sample and computed field must be finite numbers",
                    }
                )
                continue
            abs_error = abs(actual[field] - float(expected_value))
            max_abs_error = max(max_abs_error, abs_error)
            if not np.isclose(actual[field], float(expected_value), rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE):
                errors.append(
                    {
                        "path": str(path),
                        "index": index,
                        "sample_index": sample_index,
                        "field": field,
                        "error": "geometry value exceeds declared comparison tolerances",
                    }
                )
    if any(error.get("index") == index for error in errors):
        return None
    return {
        "case": case_name,
        "samples": len(sample_points),
        "max_abs_error": max_abs_error,
        "case_sha256": _json_sha256(case_payload),
    }


def _parameters_are_numeric(
    path: Path,
    index: int,
    parameters: dict[object, object],
    errors: list[dict[str, object]],
) -> bool:
    """Require the original eight nonboolean numeric parameters; recognized optional float-coercible fields and unknown extras retain their separate domains."""
    required_parameters = {"R0", "a", "rho", "kappa", "delta", "dR_dr", "q", "B0"}
    ok = True
    for field in sorted(required_parameters):
        value = parameters.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append({"path": str(path), "index": index, "field": field, "error": "parameter must be numeric"})
            ok = False
    return ok


def _geometry_arguments(
    path: Path, index: int, parameters: dict[object, object], errors: list[dict[str, object]]
) -> dict[str, float] | None:
    """Convert original admitted argument fields once, refusing malformed/nonfinite/overflow values without changing optional coercion rules."""
    if not _parameters_are_numeric(path, index, parameters, errors):
        return None
    arguments: dict[str, float] = {}
    valid = True
    for key, value in parameters.items():
        if key not in _GEOMETRY_PARAMETERS:
            continue
        try:
            number = float(cast(Any, value))
        except (TypeError, ValueError, OverflowError):
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": str(key),
                    "error": "parameter must be convertible to a finite number",
                }
            )
            valid = False
            continue
        if not math.isfinite(number):
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": str(key),
                    "error": "parameter must be convertible to a finite number",
                }
            )
            valid = False
            continue
        arguments[str(key)] = number
    return arguments if valid else None


def _finite_sample(value: object) -> float | None:
    """Return a finite binary64 sample from original nonboolean int/float values, refusing overflow."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _actual_values_at_theta(b0: float, r0: float, geometry: Any, theta: float) -> dict[str, float]:
    """Select the nearest original theta-grid node and original nine geometric values, including B0 R0/R toroidal field without changing equations."""
    index = int(np.argmin(np.abs(geometry.theta - theta)))
    R = float(geometry.R[index])
    return {
        "theta": float(geometry.theta[index]),
        "R": R,
        "Z": float(geometry.Z[index]),
        "jacobian": float(geometry.jacobian[index]),
        "g_rr": float(geometry.g_rr[index]),
        "g_rt": float(geometry.g_rt[index]),
        "g_tt": float(geometry.g_tt[index]),
        "B_toroidal": b0 * r0 / R,
        "b_dot_grad_theta": float(geometry.b_dot_grad_theta[index]),
    }
