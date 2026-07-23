# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Differentiable transport closure adapters

"""Map neural and reduced gyrokinetic closures into facade coefficient channels.

This leaf owns only closure → four-channel transport-coefficient conversion.
Step, rollout, automatic differentiation, and claim/evidence contracts remain
on the numerical facade and evidence modules. The public import path
:mod:`scpn_control.core.differentiable_transport` re-exports these symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_control._typing import FloatArray


@dataclass(frozen=True)
class GyrokineticTransportClosureResult:
    """Reduced gyrokinetic closure profiles for differentiable transport."""

    chi_e: FloatArray
    chi_i: FloatArray
    d_e: FloatArray
    channel_weights: FloatArray
    source: str
    weights_checksum: str | None


def _as_float_array(name: str, value: Any) -> FloatArray:
    """Coerce ``value`` to a finite floating array or raise."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _closure_profile(name: str, value: Any) -> FloatArray:
    """Validate a non-negative one-dimensional transport-coefficient profile."""
    arr = _as_float_array(name, value)
    if arr.ndim != 1 or arr.size < 3:
        raise ValueError(f"{name} must be a one-dimensional profile with at least three points")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def _closure_channel_weights(name: str, value: Any, n_rho: int) -> FloatArray:
    """Validate three-channel weights that sum to one at each radius."""
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
    """Stack electron/ion heat and particle diffusivities into four facade channels."""
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
