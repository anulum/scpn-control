# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport profile coordinates and finite comparison.

"""Compare observed temperature profiles on a common normalized radial grid.

Returned metrics are arithmetic diagnostics. Neither finite arrays nor their
self-digests authenticate a provider, match transport closures, or validate
physical accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


def _canonical_json(value: Any) -> str:
    """Encode JSON with sorted keys, compact separators and no nonfinite floats.

    Parameters
    ----------
    value : object
        JSON-serializable payload; NumPy arrays and scalars require conversion.

    Returns
    -------
    str
        Deterministic Unicode JSON text.

    Raises
    ------
    TypeError, ValueError
        A value is not JSON-compatible or contains NaN/infinity.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_payload(value: Any) -> str:
    """Hash canonical UTF-8 JSON, without establishing producer authenticity."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _payload_without_digest(report: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy a report and omit only its self-referential digest field."""
    payload = dict(report)
    payload.pop("payload_sha256", None)
    return payload


def verify_payload_digest(report: dict[str, Any]) -> bool:
    """Compare the declared digest to canonical bytes, without admitting science.

    Malformed JSON-compatible payloads return False. This checks consistency
    only: a caller can recompute a digest after changing every scientific field.
    """
    digest = report.get("payload_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    try:
        return digest == _sha256_payload(_payload_without_digest(report))
    except (TypeError, ValueError, OverflowError):
        return False


_verify_payload_digest = verify_payload_digest


def _finite_number(value: Any) -> bool:
    """Accept finite real Python/NumPy numbers, excluding booleans and overflow."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_vector(value: Any) -> bool:
    """Accept a nonempty one-dimensional real sequence representable in float64.

    Numeric strings, complex values, booleans and nested arrays are refused.
    Inspect list/tuple/NumPy elements before coercion, preserving booleans in
    mixed sequences. Scalar conversion overflow is refused by _finite_number.
    """
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return False
    elif not isinstance(value, (list, tuple)):
        return False
    return len(value) > 0 and all(_finite_number(item) for item in value)


def _profile_coordinates_are_valid(result: dict[str, Any], fields: tuple[str, ...]) -> bool:
    """Check strictly increasing rho in [0,1] and equally sized finite vectors."""
    if not _finite_vector(result.get("rho")):
        return False
    rho = np.asarray(result["rho"], dtype=np.float64)
    return (
        len(rho) >= 2
        and bool(np.all((rho >= 0.0) & (rho <= 1.0)))
        and bool(np.all(np.diff(rho) > 0.0))
        and all(_finite_vector(result.get(k)) and len(result[k]) == len(rho) for k in fields)
    )


def _benchmark_numeric_payload_is_finite(result: dict[str, Any]) -> bool:
    """Check full temperature profiles and optional finite scalar diagnostics.

    Coordinate ordering, domain and shape are checked; temperatures need not
    be positive. This is a numerical payload contract, not an identity gate.
    """
    if not _profile_coordinates_are_valid(result, ("Te_final", "Ti_final")):
        return False
    optional_scalars = ("Te_avg", "Ti_avg", "wall_time_s", "tau_E_s", "W_thermal_total_J")
    return all(field not in result or _finite_number(result[field]) for field in optional_scalars)


def compare_transport_profiles(scpn: dict[str, Any], torax: dict[str, Any] | None) -> dict[str, Any]:
    """Compare declared final profiles using external rho, even at equal lengths.

    Parameters
    ----------
    scpn : dict
        Local rho and Te_final vectors; Ti_final is compared when both inputs
        contain it. Normalized rho is strictly increasing in [0,1].
    torax : dict or None
        Reference vectors in keV and their actual rho coordinates. None returns
        empty metrics. The reference domain must cover every local coordinate;
        endpoint extrapolation and fabricated coordinate defaults are refused.

    Returns
    -------
    dict
        Original payloads under scpn_control/torax, numerical comparison metrics
        and authored comparison_findings. RMSE uses equal weight per local
        sample, not volume weighting. Maximum difference is absolute keV.
        Optional reference tau_E_s is passed through as seconds.
        Invalid profiles or arithmetic leave comparison empty.

    Notes
    -----
    Inputs are declarations and are not mutated. Linear interpolation is applied
    to reference temperatures on the local grid. Values and coordinates may
    differ in count; equal counts do not imply equal coordinates. No physical
    agreement threshold, runtime authenticity or admission follows.
    """
    result: dict[str, Any] = {"scpn_control": scpn, "torax": torax, "comparison": {}, "comparison_findings": []}
    if torax is None:
        return result
    fields = ("Te_final", "Ti_final") if "Ti_final" in scpn and "Ti_final" in torax else ("Te_final",)
    if not _profile_coordinates_are_valid(scpn, fields):
        result["comparison_findings"].append("scpn_profile_coordinates_or_values")
        return result
    if not _profile_coordinates_are_valid(torax, fields):
        result["comparison_findings"].append("torax_profile_coordinates_or_values")
        return result
    rho = np.asarray(scpn["rho"], dtype=np.float64)
    reference_rho = np.asarray(torax["rho"], dtype=np.float64)
    if rho[0] < reference_rho[0] or rho[-1] > reference_rho[-1]:
        result["comparison_findings"].append("reference_grid_does_not_cover_local_grid")
        return result
    for field in fields:
        local = np.asarray(scpn[field], dtype=np.float64)
        reference = np.interp(rho, reference_rho, np.asarray(torax[field], dtype=np.float64))
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(local - reference)
        if not bool(np.all(np.isfinite(difference))):
            result["comparison"] = {}
            result["comparison_findings"].append("temperature_difference_not_representable")
            return result
        maximum = float(np.max(difference))
        rmse = maximum * float(np.sqrt(np.mean((difference / maximum) ** 2))) if maximum else 0.0
        channel = field.removesuffix("_final")
        result["comparison"][channel + "_rmse_keV"] = rmse
        result["comparison"][channel + "_max_diff_keV"] = maximum
    if "tau_E_s" in torax:
        if not _finite_number(torax["tau_E_s"]):
            result["comparison"] = {}
            result["comparison_findings"].append("torax_tau_E_not_finite")
            return result
        result["comparison"]["torax_tau_E_s"] = float(torax["tau_E_s"])
    return result


_compare_results = compare_transport_profiles
