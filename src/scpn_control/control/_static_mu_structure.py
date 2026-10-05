# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Static Mu Structure

"""Static mu structure for bounded structured-uncertainty analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpn_control._typing import FloatArray

_VALID_BLOCK_TYPES = frozenset({"real_scalar", "complex_scalar", "full"})


def _finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass
class UncertaintyBlock:
    """Single block in the structured uncertainty set Δ.

    Attributes
    ----------
    name : str
        Physical label, e.g. "plasma_position".
    size : int
        Block dimension (number of channels).
    bound : float
        Norm bound on this block (||Δ_i|| ≤ bound).
    block_type : str
        "real_scalar" | "complex_scalar" | "full".
        Tokamak usage: parametric deviations are "real_scalar";
        unmodelled dynamics are "full" (Ariola & Pironti 2008, Ch. 7).
    """

    name: str
    size: int
    bound: float
    block_type: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        self.size = _positive_int("size", self.size)
        self.bound = _finite_scalar("bound", self.bound, positive=True)
        if self.block_type not in _VALID_BLOCK_TYPES:
            raise ValueError(f"block_type must be one of {sorted(_VALID_BLOCK_TYPES)}")


class StructuredUncertainty:
    """Ordered collection of UncertaintyBlock objects defining Δ_struct."""

    def __init__(self, blocks: list[UncertaintyBlock]):
        if not blocks:
            raise ValueError("blocks must contain at least one uncertainty block")
        self.blocks = blocks

    def build_delta_structure(self) -> list[tuple[int, str]]:
        """Return the uncertainty structure as ``(size, block_type)`` per block."""
        return [(b.size, b.block_type) for b in self.blocks]

    def total_size(self) -> int:
        """Total dimension of the block-diagonal uncertainty structure Δ."""
        return sum(b.size for b in self.blocks)

    def bound_matrix(self) -> FloatArray:
        """Block-diagonal bound matrix mapping normalised Δ blocks to physical Δ."""
        bounds = np.concatenate([np.full(block.size, block.bound, dtype=float) for block in self.blocks])
        return np.diag(bounds)
