# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — GK species Bessel quadrature and pitch operator comparisons

"""GK species Bessel quadrature and pitch operator comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_control.core.gk_species import VelocityGrid, bessel_j0, pitch_angle_operator
from validation.gk_species_reference_contracts import _ABS_TOLERANCE, _REL_TOLERANCE
from validation.gk_species_reference_numeric import _numeric_scalar, _object_fields_are_numeric, _relative_error


def _validate_operator_checks(
    path: Path,
    payload: object,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    """Compare original Bessel, quadrature and pitch-angle operator declarations."""
    if not isinstance(payload, dict):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks",
                "error": "operator_checks must be an object",
            }
        )
        return None

    report: dict[str, object] = {}
    _validate_bessel_checks(path, payload.get("bessel_j0"), report, errors)
    _validate_velocity_grid_check(path, payload.get("velocity_grid"), report, errors)
    _validate_pitch_angle_check(path, payload.get("pitch_angle_operator"), report, errors)
    return report if report else None


def _validate_bessel_checks(
    path: Path,
    payload: object,
    report: dict[str, object],
    errors: list[dict[str, object]],
) -> None:
    """Compare finite Bessel arguments and reference values using original tolerances."""
    if not isinstance(payload, list) or not payload:
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.bessel_j0",
                "error": "bessel_j0 must be a non-empty array",
            }
        )
        return
    entries: list[dict[str, float]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": "bessel_j0",
                    "error": "case must be an object",
                }
            )
            continue
        if not _object_fields_are_numeric(path, index, item, ("argument", "expected"), errors):
            continue
        argument = float(item["argument"])
        expected = float(item["expected"])
        actual = float(bessel_j0(np.asarray([argument], dtype=np.float64))[0])
        rel_error = _relative_error(actual, expected)
        if not np.isclose(actual, expected, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE):
            errors.append(
                {
                    "path": str(path),
                    "index": index,
                    "field": "operator_checks.bessel_j0",
                    "error": "Bessel J0 drifted beyond declared tolerance",
                }
            )
        entries.append({"argument": argument, "actual": actual, "relative_error": rel_error})
    report["bessel_j0"] = entries


def _validate_velocity_grid_check(
    path: Path,
    payload: object,
    report: dict[str, object],
    errors: list[dict[str, object]],
) -> None:
    """Compare original int-coerced quadrature counts, sums and endpoints."""
    if not isinstance(payload, dict):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.velocity_grid",
                "error": "velocity_grid must be an object",
            }
        )
        return
    if not _object_fields_are_numeric(
        path,
        -1,
        payload,
        (
            "n_energy",
            "n_lambda",
            "energy_weight_sum",
            "lambda_weight_sum",
            "energy_min",
            "energy_max",
        ),
        errors,
    ):
        return
    try:
        grid = VelocityGrid(n_energy=int(payload["n_energy"]), n_lambda=int(payload["n_lambda"]))
    except (ValueError, ArithmeticError, MemoryError):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.velocity_grid",
                "error": "operator parameters are outside the computed finite domain",
            }
        )
        return
    actual = {
        "energy_weight_sum": float(np.sum(grid.energy_weights)),
        "lambda_weight_sum": float(np.sum(grid.lambda_weights)),
        "energy_min": float(grid.energy[0]),
        "energy_max": float(grid.energy[-1]),
    }
    max_relative_error = 0.0
    for field, actual_value in actual.items():
        expected = float(payload[field])
        max_relative_error = max(max_relative_error, _relative_error(actual_value, expected))
        if not np.isclose(actual_value, expected, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE):
            errors.append(
                {
                    "path": str(path),
                    "field": f"operator_checks.velocity_grid.{field}",
                    "error": "velocity-grid reference drifted beyond declared tolerance",
                }
            )
    report["velocity_grid"] = {
        "n_energy": grid.n_energy,
        "n_lambda": grid.n_lambda,
        "actual": actual,
        "max_relative_error": max_relative_error,
    }


def _validate_pitch_angle_check(
    path: Path,
    payload: object,
    report: dict[str, object],
    errors: list[dict[str, object]],
) -> None:
    """Compare original nullspace and sparsity of the producer-constructed tridiagonal pitch operator."""
    if not isinstance(payload, dict):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator",
                "error": "pitch_angle_operator must be an object",
            }
        )
        return
    lam_payload = payload.get("lambda_grid")
    if not isinstance(lam_payload, list) or not lam_payload:
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator.lambda_grid",
                "error": "lambda_grid must be a non-empty array",
            }
        )
        return
    try:
        lam = np.asarray([_numeric_scalar(value) for value in lam_payload], dtype=np.float64)
        b_ratio = _numeric_scalar(payload.get("B_ratio"))
        expected_constant = _numeric_scalar(payload.get("constant_nullspace_max_abs"))
        expected_nonzero = int(_numeric_scalar(payload.get("tridiagonal_nonzero_entries")))
    except (ValueError, ArithmeticError, MemoryError):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator",
                "error": "operator parameters are outside the computed finite domain",
            }
        )
        return
    try:
        matrix = pitch_angle_operator(len(lam), lam, B_ratio=b_ratio)
    except (ValueError, ArithmeticError, MemoryError):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator",
                "error": "operator parameters are outside the computed finite domain",
            }
        )
        return
    constant_residual = float(np.max(np.abs(matrix @ np.ones_like(lam))))
    nonzero_entries = int(np.count_nonzero(np.abs(matrix) > 0.0))
    if not np.isclose(constant_residual, expected_constant, rtol=_REL_TOLERANCE, atol=_ABS_TOLERANCE):
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator.constant_nullspace_max_abs",
                "error": "pitch-angle nullspace drifted beyond declared tolerance",
            }
        )
    if nonzero_entries != expected_nonzero:
        errors.append(
            {
                "path": str(path),
                "field": "operator_checks.pitch_angle_operator.tridiagonal_nonzero_entries",
                "error": "pitch-angle sparsity drifted from declared entry count",
            }
        )
    report["pitch_angle_operator"] = {
        "n_lambda": len(lam),
        "B_ratio": b_ratio,
        "constant_nullspace_max_abs": constant_residual,
        "tridiagonal_nonzero_entries": nonzero_entries,
    }
