# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MAST EFM neural-equilibrium dataset builder

"""Derive the existing12 features from selected reference arrays without granting predictive admission."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from validation.neural_equilibrium_dataset_contracts import FEATURE_NAMES, FEATURE_SOURCE_POLICY, integer


def _per_equilibrium_array(data: dict[str, NDArray[Any]], key: str, count: int, fallback: float) -> NDArray[np.float64]:
    """Broadcast one scalar or require N flattened values; missing/nonfinite entries use the declared fallback."""
    raw = data.get(key)
    if raw is None:
        return np.full(count, fallback, dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 0:
        values = np.full(count, float(arr), dtype=np.float64)
    else:
        flat = arr.reshape(-1)
        if flat.size != count:
            raise ValueError(f"{key} must contain {count} per-equilibrium values")
        values = flat.astype(np.float64)
    values[~np.isfinite(values)] = fallback
    return values


def _last_finite_profile_value(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    """Use the last finite masked profile observation, or the documented q95 fallback4 when no observation exists."""
    valid = mask & np.isfinite(values)
    if not np.any(valid):
        return 4.0
    indices = np.flatnonzero(valid)
    return float(values[indices[-1]])


def _profile_last_values(
    data: dict[str, NDArray[Any]], value_key: str, mask_key: str, count: int, fallback: float
) -> NDArray[np.float64]:
    """Broadcast one profile across rows or require aligned rows and masks, then derive per-row last observations."""
    raw = data.get(value_key)
    if raw is None:
        return np.full(count, fallback, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim not in (1, 2):
        raise ValueError("profile arrays must have one or two dimensions")
    if values.ndim == 1:
        values = np.tile(values.reshape(1, -1), (count, 1))
    if values.shape[0] != count:
        raise ValueError(f"{value_key} row count must match psirz_Wb_per_rad")
    raw_mask = data.get(mask_key)
    if raw_mask is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.ndim == 1:
            mask = np.tile(mask.reshape(1, -1), (count, 1))
        if mask.shape != values.shape:
            raise ValueError(f"{mask_key} shape must match {value_key}")
    return np.asarray([_last_finite_profile_value(values[row], mask[row]) for row in range(count)], dtype=np.float64)


def _pprime_scales(data: dict[str, NDArray[Any]], count: int) -> NDArray[np.float64]:
    """Normalise observed mean absolute pressure gradients by the shot median and clip0.25..4; unobserved rows retain default1."""
    raw = data.get("pprime_Pa_per_Wb_rad")
    if raw is None:
        return np.ones(count, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim not in (1, 2):
        raise ValueError("profile arrays must have one or two dimensions")
    if values.ndim == 1:
        values = np.tile(values.reshape(1, -1), (count, 1))
    if values.shape[0] != count:
        raise ValueError("pprime_Pa_per_Wb_rad row count must match psirz_Wb_per_rad")
    mask = np.asarray(data.get("pprime_valid_mask", np.ones(values.shape, dtype=bool)), dtype=bool)
    if mask.ndim == 1:
        mask = np.tile(mask.reshape(1, -1), (count, 1))
    if mask.shape != values.shape:
        raise ValueError("pprime_valid_mask shape must match pprime_Pa_per_Wb_rad")
    magnitudes = np.ones(count, dtype=np.float64)
    for row in range(count):
        valid = mask[row] & np.isfinite(values[row])
        if np.any(valid):
            absolute = np.abs(values[row, valid])
            peak = float(np.max(absolute))
            magnitudes[row] = peak * float(np.mean(absolute / peak)) if peak > 0.0 else 0.0
    positive = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]
    reference = _positive_median(positive) if positive.size else 1.0
    return np.clip(magnitudes / reference, 0.25, 4.0).astype(np.float64)


def _ffprime_rms_values(data: dict[str, NDArray[Any]], count: int) -> NDArray[np.float64] | None:
    """Require finite positive sourced RMS for every row, returning None only when the key is absent."""
    raw = data.get("ffprime_rms_T_rad")
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size != count:
        raise ValueError("ffprime_rms_T_rad row count must match psirz_Wb_per_rad")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("ffprime_rms_T_rad must contain finite positive per-equilibrium values")
    return values


def _campaign_ffprime_reference(rows: list[dict[str, NDArray[Any]]]) -> float | None:
    """Use a positive campaign median only when every selected shot supplies complete sourced RMS values."""
    values: list[NDArray[np.float64]] = []
    for row in rows:
        psi = np.asarray(row["psirz_Wb_per_rad"])
        count = int(psi.shape[0])
        rms = _ffprime_rms_values(row, count)
        if rms is not None:
            values.append(rms)
    if not values or len(values) != len(rows):
        return None
    # Each selected array is nonempty, finite and positive by reference validation
    # and _ffprime_rms_values. Averaging the two middle values must not overflow.
    return _positive_median(np.concatenate(values))


def _positive_median(values: NDArray[np.float64]) -> float:
    """Return the median of a nonempty validated finite positive vector without even-count addition overflow."""
    ordered = np.sort(values)
    middle = ordered.size // 2
    if ordered.size % 2:
        return float(ordered[middle])
    lower = float(ordered[middle - 1])
    upper = float(ordered[middle])
    return lower + (upper - lower) / 2.0


def _public_feature_availability(
    rows: list[dict[str, NDArray[Any]]], ffprime_reference: float | None
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    """Declare only Ip/Bt/FF-prime provenance available on all selected references; otherwise retain explicit fallback names."""
    fallback: list[str] = []
    policy: dict[str, dict[str, Any]] = {}
    for feature in ("Ip_MA", "Bt_T"):
        key = str(FEATURE_SOURCE_POLICY[feature]["source_key"])
        if all(key in row for row in rows):
            policy[feature] = dict(FEATURE_SOURCE_POLICY[feature])
        else:
            fallback.append(feature)
    if ffprime_reference is not None and all("ffprime_rms_T_rad" in row for row in rows):
        policy["ffprime_scale"] = {**FEATURE_SOURCE_POLICY["ffprime_scale"], "campaign_reference": ffprime_reference}
    else:
        fallback.append("ffprime_scale")
    return tuple(fallback), policy


def _sourced_scalar_feature(
    data: dict[str, NDArray[Any]], key: str, count: int, fallback: float
) -> NDArray[np.float64]:
    """Use finite per-equilibrium public-source scalars; a wholly absent key uses its documented engineering fallback."""
    raw = data.get(key)
    if raw is None:
        return np.full(count, fallback, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size != count:
        raise ValueError(f"{key} row count must match psirz_Wb_per_rad")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} contains non-finite values")
    return values


def _lcfs_geometry_features(
    data: dict[str, NDArray[Any]], axis_r: NDArray[np.float64], axis_z: NDArray[np.float64], count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Derive clipped elongation/triangularities from jointly valid LCFS points; insufficient geometry retains declared defaults."""
    raw_r = data.get("lcfs_r_m")
    raw_z = data.get("lcfs_z_m")
    if raw_r is None or raw_z is None:
        return (
            np.full(count, 1.7, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
        )
    r_values = np.asarray(raw_r, dtype=np.float64)
    z_values = np.asarray(raw_z, dtype=np.float64)
    if r_values.ndim not in (1, 2) or z_values.ndim not in (1, 2):
        raise ValueError("LCFS arrays must have one or two dimensions")
    if r_values.ndim == 1:
        r_values = np.tile(r_values.reshape(1, -1), (count, 1))
    if z_values.ndim == 1:
        z_values = np.tile(z_values.reshape(1, -1), (count, 1))
    if r_values.shape[0] != count or z_values.shape != r_values.shape:
        raise ValueError("LCFS arrays must align with psirz_Wb_per_rad")
    mask = np.asarray(data.get("lcfs_valid_mask", np.ones(r_values.shape, dtype=bool)), dtype=bool)
    if mask.ndim == 1:
        mask = np.tile(mask.reshape(1, -1), (count, 1))
    if mask.shape != r_values.shape:
        raise ValueError("lcfs_valid_mask shape must match lcfs_r_m")
    kappa = np.full(count, 1.7, dtype=np.float64)
    delta_upper = np.zeros(count, dtype=np.float64)
    delta_lower = np.zeros(count, dtype=np.float64)
    for row in range(count):
        valid = mask[row] & np.isfinite(r_values[row]) & np.isfinite(z_values[row])
        if np.count_nonzero(valid) < 3:
            continue
        r_row = r_values[row, valid]
        z_row = z_values[row, valid]
        radial_minor = float(np.max(np.abs(r_row - axis_r[row])))
        if radial_minor <= 1e-9 or not np.isfinite(radial_minor):
            continue
        z_relative = z_row - axis_z[row]
        vertical_minor = float(max(np.max(z_relative), -np.min(z_relative)))
        if vertical_minor > 0.0 and np.isfinite(vertical_minor):
            kappa[row] = float(np.clip(vertical_minor / radial_minor, 0.5, 3.5))
        upper_index = int(np.argmax(z_relative))
        lower_index = int(np.argmin(z_relative))
        delta_upper[row] = float(np.clip((axis_r[row] - r_row[upper_index]) / radial_minor, -1.0, 1.0))
        delta_lower[row] = float(np.clip((axis_r[row] - r_row[lower_index]) / radial_minor, -1.0, 1.0))
    return kappa, delta_upper, delta_lower


def validate_feature_matrix(values: NDArray[Any], *, row_count: int) -> NDArray[np.float64]:
    """Require the declared row count and12 real finite columns in the actual float64 computation representation.

    Producer and trainer share this final matrix contract. Row count must be a
    genuine positive integer; the caller owns named-column identity. Conversion rejects
    values representable only in a wider storage dtype; existing float64 input
    can return a view. Concurrent caller mutation is not frozen by this check.

    >>> validate_feature_matrix(np.asarray([1.0]), row_count=1)
    Traceback (most recent call last):
        ...
    ValueError: features must be finite real values with shape (1, 12)
    """
    shape = (integer(row_count, "feature row_count"), len(FEATURE_NAMES))
    if values.dtype.kind not in "fiu" or values.shape != shape:
        raise ValueError(f"features must be finite real values with shape {shape}")
    features = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("feature matrix contains non-finite values")
    return features


def build_feature_matrix(
    data: dict[str, NDArray[Any]], *, ffprime_reference: float | None = None
) -> NDArray[np.float64]:
    """Derive the12 ordered reference feature columns without predictive admission.

    Missing axes/flux scalars and absent source keys have explicit defaults;
    nonfinite scalar fallback entries are replaced. Profiles/LCFS geometry use
    valid observations and documented fallback4/default geometry when absent.
    FF-prime normalisation requires a finite positive campaign reference.
    These reference-derived input features are not independent predictive
    validation of the same target profiles/geometry or authenticated physics.
    Profile and LCFS inputs accept a shared vector or aligned two-dimensional
    rows; scalar and higher-rank arrays refuse through ValueError.

    >>> build_feature_matrix({})
    Traceback (most recent call last):
        ...
    ValueError: feature input must contain psirz_Wb_per_rad
    """
    if not isinstance(data, dict) or "psirz_Wb_per_rad" not in data:
        raise ValueError("feature input must contain psirz_Wb_per_rad")
    numeric = {
        "psirz_Wb_per_rad",
        "magnetic_axis_r_m",
        "magnetic_axis_z_m",
        "psi_axis_Wb_per_rad",
        "psi_boundary_Wb_per_rad",
        "Ip_MA",
        "Bt_T",
        "ffprime_rms_T_rad",
        "pprime_Pa_per_Wb_rad",
        "q_profile",
        "lcfs_r_m",
        "lcfs_z_m",
    }
    masks = {"pprime_valid_mask", "q_profile_valid_mask", "lcfs_valid_mask"}
    for key, values in data.items():
        if key in numeric and (not isinstance(values, np.ndarray) or values.dtype.kind not in "fiu"):
            raise ValueError(f"{key} feature input must be a real numeric array")
        if key in masks and (not isinstance(values, np.ndarray) or values.dtype != np.bool_):
            raise ValueError(f"{key} feature input must be a boolean array")
    if ffprime_reference is not None:
        if isinstance(ffprime_reference, bool) or not isinstance(ffprime_reference, int | float):
            raise ValueError("ffprime_reference must be finite and positive")
        try:
            ffprime_reference = float(ffprime_reference)
        except OverflowError as exc:
            raise ValueError("ffprime_reference must be finite and positive") from exc
        if not math.isfinite(ffprime_reference) or ffprime_reference <= 0:
            raise ValueError("ffprime_reference must be finite and positive")
    psi = np.asarray(data["psirz_Wb_per_rad"])
    if psi.dtype.kind not in "fiu" or psi.ndim != 3 or psi.shape[0] < 1:
        raise ValueError("psirz_Wb_per_rad must have shape (n_equilibria, nz, nr)")
    count = int(psi.shape[0])
    axis_r = _per_equilibrium_array(data, "magnetic_axis_r_m", count, 1.0)
    axis_z = _per_equilibrium_array(data, "magnetic_axis_z_m", count, 0.0)
    psi_axis = _per_equilibrium_array(data, "psi_axis_Wb_per_rad", count, 0.0)
    psi_boundary = _per_equilibrium_array(data, "psi_boundary_Wb_per_rad", count, 1.0)
    pprime_scale = _pprime_scales(data, count)
    ffprime_rms = _ffprime_rms_values(data, count)
    if ffprime_rms is not None and ffprime_reference is not None:
        ffprime_scale = np.clip(ffprime_rms / ffprime_reference, 0.25, 4.0).astype(np.float64)
    else:
        ffprime_scale = np.ones(count, dtype=np.float64)
    q95 = _profile_last_values(data, "q_profile", "q_profile_valid_mask", count, 4.0)
    kappa, delta_upper, delta_lower = _lcfs_geometry_features(data, axis_r, axis_z, count)
    columns = {
        "Ip_MA": _sourced_scalar_feature(data, "Ip_MA", count, 8.0),
        "Bt_T": _sourced_scalar_feature(data, "Bt_T", count, 5.0),
        "R_axis_m": axis_r,
        "Z_axis_m": axis_z,
        "pprime_scale": pprime_scale,
        "ffprime_scale": ffprime_scale,
        "simag_Wb": psi_axis,
        "sibry_Wb": psi_boundary,
        "kappa": kappa,
        "delta_upper": delta_upper,
        "delta_lower": delta_lower,
        "q95": q95,
    }
    return validate_feature_matrix(np.column_stack([columns[name] for name in FEATURE_NAMES]), row_count=count)
