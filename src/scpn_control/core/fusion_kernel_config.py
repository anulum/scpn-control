# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fusion kernel configuration schemas

"""Pydantic configuration models and parse/dump helpers for FusionKernel.

This leaf owns only Grad-Shafranov runtime JSON configuration contracts:
dimensions, physics, solver, coil entries, and the root fusion-kernel schema,
plus pure parse/dump helpers. The CONTROL product surface
:class:`~scpn_control.core.fusion_kernel.FusionKernel` remains first-class and
re-exports these symbols so existing import paths stay stable under dual-home C.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


def normalize_boundary_variant(variant: str | None) -> str:
    """Normalise free/fixed boundary variant labels to the canonical pair."""
    raw = "fixed_boundary" if variant is None else str(variant).strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")

    if raw in {"fixed", "fixed_boundary", "fixedboundary"}:
        return "fixed_boundary"
    if raw in {"free", "free_boundary", "freeboundary"}:
        return "free_boundary"

    raise ValueError("solver.boundary_variant must be one of: 'fixed_boundary', 'free_boundary', 'fixed', 'free'.")


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a mapping while failing closed on duplicate JSON object keys."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate JSON key in fusion kernel config: {key}")
        out[key] = value
    return out


# Private aliases retained for call sites that still use underscore names.
_normalize_boundary_variant = normalize_boundary_variant
_reject_duplicate_json_keys = reject_duplicate_json_keys


class DimensionsConfig(BaseModel):
    """Pydantic v2 schema for Grad-Shafranov domain bounds."""

    model_config = ConfigDict(extra="allow")

    R_min: float
    R_max: float
    Z_min: float
    Z_max: float

    @field_validator("R_min", "R_max", "Z_min", "Z_max")
    @classmethod
    def _finite_bound(cls, value: float) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("dimension bounds must be finite")
        return value

    @model_validator(mode="after")
    def _ordered_bounds(self) -> DimensionsConfig:
        if self.R_min <= 0.0 or self.R_max <= self.R_min:
            raise ValueError("dimensions must satisfy 0 < R_min < R_max")
        if self.Z_max <= self.Z_min:
            raise ValueError("dimensions must satisfy Z_min < Z_max")
        return self


class PhysicsConfig(BaseModel):
    """Pydantic v2 schema for scalar physics controls."""

    model_config = ConfigDict(extra="allow")

    plasma_current_target: float = Field(..., description="Target plasma current magnitude in MA or configured units")
    plasma_current_sign: float | None = None
    vacuum_permeability: float | None = None

    @field_validator("plasma_current_target")
    @classmethod
    def _positive_current(cls, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("plasma_current_target must be positive finite")
        return value

    @field_validator("plasma_current_sign")
    @classmethod
    def _valid_current_sign(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value not in {-1.0, 1.0}:
            raise ValueError("plasma_current_sign must be -1.0 or 1.0 when configured")
        return value

    @field_validator("vacuum_permeability")
    @classmethod
    def _positive_permeability(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("vacuum_permeability must be positive finite")
        return value


class SolverConfig(BaseModel):
    """Pydantic v2 schema for solver controls, preserving extension keys."""

    model_config = ConfigDict(extra="allow")

    boundary_variant: str = "fixed_boundary"

    @field_validator("boundary_variant", mode="before")
    @classmethod
    def _normalised_boundary_variant(cls, value: str | None) -> str:
        return normalize_boundary_variant(value)


class CoilConfig(BaseModel):
    """Pydantic v2 schema for coil entries, preserving extension keys."""

    model_config = ConfigDict(extra="allow")


class FusionKernelConfig(BaseModel):
    """Pydantic v2 schema for Grad-Shafranov runtime JSON configuration."""

    model_config = ConfigDict(extra="allow")

    reactor_name: str = Field(..., min_length=1)
    dimensions: DimensionsConfig
    grid_resolution: tuple[StrictInt, StrictInt]
    physics: PhysicsConfig
    solver: SolverConfig = Field(default_factory=SolverConfig)
    coils: list[CoilConfig] = Field(default_factory=list)

    @field_validator("reactor_name")
    @classmethod
    def _non_empty_reactor_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("fusion kernel config requires non-empty 'reactor_name'")
        return value

    @field_validator("grid_resolution")
    @classmethod
    def _valid_grid_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        if any(isinstance(entry, bool) or int(entry) < 3 for entry in value):
            raise ValueError("grid_resolution entries must be integers >= 3")
        return value


def parse_fusion_kernel_config(raw: Any) -> FusionKernelConfig:
    """Validate a raw JSON object as a fusion-kernel configuration model."""
    if not isinstance(raw, dict):
        raise ValueError("fusion kernel config root must be a JSON object.")
    return FusionKernelConfig.model_validate(raw)


def fusion_kernel_config_dump(config: FusionKernelConfig) -> dict[str, Any]:
    """Dump a validated config model to a plain Python dict for runtime use."""
    validated = config.model_dump(mode="python", exclude_none=True)
    validated["grid_resolution"] = list(validated["grid_resolution"])
    return validated


# Underscore aliases for historical private import paths on the owner module.
_parse_fusion_kernel_config = parse_fusion_kernel_config
_fusion_kernel_config_dump = fusion_kernel_config_dump
