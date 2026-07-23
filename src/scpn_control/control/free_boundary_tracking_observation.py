# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Free-boundary tracking observation vector builders

"""Pure target/measurement vector builders for free-boundary tracking.

This leaf owns objective-block topology, target-vector construction from a
:class:`~scpn_control.core.fusion_kernel.CoilSet`, measurement-offset vector
resolution, and control-objective weighting. Shot orchestration, kernel
observation, and latency-state machines stay on
:class:`~scpn_control.control.free_boundary_tracking.FreeBoundaryTrackingController`
(R3-S3/S4). Claims remain in ``free_boundary_tracking_claims``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.core.fusion_kernel import CoilSet


@dataclass(frozen=True)
class ObjectiveBlock:
    """Contiguous slice of the free-boundary tracking objective vector."""

    name: str
    start: int
    stop: int


def build_target_vector(coils: CoilSet) -> tuple[FloatArray, tuple[ObjectiveBlock, ...]]:
    """Build the stacked objective target vector and block map from a coil set."""
    values: list[float] = []
    blocks: list[ObjectiveBlock] = []
    start = 0

    if coils.target_flux_points is not None and coils.target_flux_values is not None:
        target_flux = np.asarray(coils.target_flux_values, dtype=np.float64).reshape(-1)
        values.extend(float(v) for v in target_flux)
        stop = start + target_flux.size
        blocks.append(ObjectiveBlock("shape_flux", start, stop))
        start = stop

    if coils.x_point_target is not None:
        x_target = np.asarray(coils.x_point_target, dtype=np.float64).reshape(2)
        values.extend((float(x_target[0]), float(x_target[1])))
        stop = start + 2
        blocks.append(ObjectiveBlock("x_point_position", start, stop))
        start = stop
        if coils.x_point_flux_target is not None:
            values.append(float(coils.x_point_flux_target))
            stop = start + 1
            blocks.append(ObjectiveBlock("x_point_flux", start, stop))
            start = stop
    elif coils.x_point_flux_target is not None:
        raise ValueError("x_point_flux_target requires x_point_target for free-boundary tracking.")

    if coils.divertor_strike_points is not None and coils.divertor_flux_values is not None:
        divertor_flux = np.asarray(coils.divertor_flux_values, dtype=np.float64).reshape(-1)
        values.extend(float(v) for v in divertor_flux)
        stop = start + divertor_flux.size
        blocks.append(ObjectiveBlock("divertor_flux", start, stop))
        start = stop

    return np.asarray(values, dtype=np.float64), tuple(blocks)


def resolve_measurement_vector(
    raw_value: Any,
    *,
    objective_blocks: Sequence[ObjectiveBlock],
    target_size: int,
    name: str,
) -> FloatArray:
    """Resolve a per-block measurement offset/bias vector aligned to the target."""
    vector = np.zeros(target_size, dtype=np.float64)
    if raw_value is None:
        return cast(FloatArray, vector)
    if not isinstance(raw_value, dict):
        raise ValueError(f"{name} must be a mapping of objective block names to finite scalars or vectors.")

    block_map = {block.name: block for block in objective_blocks}
    allowed_keys = ", ".join(sorted(block_map))
    for key, raw_block_value in raw_value.items():
        block = block_map.get(key)
        if block is None:
            raise ValueError(f"Unknown {name} key {key!r}. Allowed keys: {allowed_keys}.")
        width = block.stop - block.start
        if np.isscalar(raw_block_value):
            block_values = np.full(width, float(cast(Any, raw_block_value)), dtype=np.float64)
        else:
            block_values = np.asarray(raw_block_value, dtype=np.float64).reshape(-1)
            if block_values.size == 1:
                block_values = np.full(width, float(block_values[0]), dtype=np.float64)
        if block_values.shape != (width,):
            raise ValueError(f"{name}.{key} must be a scalar or contain exactly {width} entries.")
        if np.any(~np.isfinite(block_values)):
            raise ValueError(f"{name}.{key} must contain only finite values.")
        vector[block.start : block.stop] = block_values
    return cast(FloatArray, np.asarray(vector, dtype=np.float64))


def weight_from_tolerances(objective_tolerances: Mapping[str, float], *keys: str) -> float:
    """Map objective tolerances to a control weight (larger weight for tighter tol)."""
    weight = 1.0
    for key in keys:
        tol = objective_tolerances.get(key)
        if tol is None:
            continue
        weight = max(weight, 1.0 / max(float(tol), 1.0e-12))
    return float(weight)


def build_control_objective_weights(
    target_size: int,
    objective_blocks: Sequence[ObjectiveBlock],
    objective_tolerances: Mapping[str, float],
) -> FloatArray:
    """Build per-entry control weights from objective blocks and tolerances."""
    weights = np.ones(target_size, dtype=np.float64)
    for block in objective_blocks:
        if block.name == "shape_flux":
            block_weight = weight_from_tolerances(objective_tolerances, "shape_rms", "shape_max_abs")
        elif block.name == "x_point_position":
            block_weight = weight_from_tolerances(objective_tolerances, "x_point_position")
        elif block.name == "x_point_flux":
            block_weight = weight_from_tolerances(objective_tolerances, "x_point_flux")
        elif block.name == "divertor_flux":
            block_weight = weight_from_tolerances(objective_tolerances, "divertor_rms", "divertor_max_abs")
        else:
            raise ValueError(f"Unknown objective block {block.name!r}.")
        weights[block.start : block.stop] = block_weight
    return cast(FloatArray, np.asarray(weights, dtype=np.float64))


def current_measurement_offset(
    measurement_bias_vector: FloatArray,
    measurement_drift_state: FloatArray,
    measurement_correction_bias: FloatArray,
    measurement_correction_drift_state: FloatArray,
) -> FloatArray:
    """Combine bias/drift and correction channels into the net measurement offset."""
    return cast(
        FloatArray,
        np.asarray(
            measurement_bias_vector
            + measurement_drift_state
            - measurement_correction_bias
            - measurement_correction_drift_state,
            dtype=np.float64,
        ),
    )
