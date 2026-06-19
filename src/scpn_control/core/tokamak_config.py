# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tokamak Config
"""Tokamak machine parameter container with named presets.

Provides a frozen dataclass holding the geometric, magnetic, and kinetic
parameters that define a tokamak operating point.  Class methods create
presets for ITER, SPARC, DIII-D, and JET.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TokamakConfigSchema(BaseModel):
    """Pydantic v2 schema for validated tokamak operating-point configs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    R0: float = Field(..., gt=0.0, description="Major radius [m]")
    a: float = Field(..., gt=0.0, description="Minor radius [m]")
    B0: float = Field(..., gt=0.0, description="Toroidal field on axis [T]")
    Ip: float = Field(..., gt=0.0, description="Plasma current [MA]")
    kappa: float = Field(..., gt=0.0, description="Elongation")
    delta: float = Field(..., description="Triangularity")
    n_e: float = Field(..., gt=0.0, description="Line-averaged electron density [1e19 m^-3]")
    T_e: float = Field(..., gt=0.0, description="Central electron temperature [keV]")
    P_aux: float = Field(..., ge=0.0, description="Auxiliary heating power [MW]")

    @field_validator("R0", "a", "B0", "Ip", "kappa", "delta", "n_e", "T_e", "P_aux")
    @classmethod
    def _finite_scalar(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("tokamak scalar fields must be finite")
        return value

    @field_validator("name")
    @classmethod
    def _normalise_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("name must be non-empty")
        return name

    @model_validator(mode="after")
    def _physics_bounds(self) -> "TokamakConfigSchema":
        if self.a >= self.R0:
            raise ValueError("minor radius must be smaller than major radius")
        if abs(self.delta) >= 1.0:
            raise ValueError("triangularity magnitude must be < 1")
        return self


@dataclass(frozen=True)
class TokamakConfig:
    """Tokamak equilibrium operating point.

    Units: lengths [m], field [T], current [MA], density [1e19 m^-3],
    temperature [keV], power [MW].
    """

    name: str
    R0: float  # major radius [m]
    a: float  # minor radius [m]
    B0: float  # toroidal field on axis [T]
    Ip: float  # plasma current [MA]
    kappa: float  # elongation
    delta: float  # triangularity
    n_e: float  # line-averaged electron density [1e19 m^-3]
    T_e: float  # central electron temperature [keV]
    P_aux: float  # auxiliary heating power [MW]

    def __post_init__(self) -> None:
        validated = TokamakConfigSchema.model_validate(self.__dict__)
        for field_name, value in validated.model_dump().items():
            object.__setattr__(self, field_name, value)

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        """Return the JSON Schema for serialized tokamak configurations."""
        return TokamakConfigSchema.model_json_schema()

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio ``R0 / a`` (major over minor radius)."""
        return self.R0 / self.a

    @property
    def epsilon(self) -> float:
        """Inverse aspect ratio ``a / R0``."""
        return self.a / self.R0

    # ── Named presets ─────────────────────────────────────────────────

    @classmethod
    def iter(cls) -> TokamakConfig:
        """ITER baseline H-mode. ITER Research Plan (2018)."""
        return cls(
            name="ITER",
            R0=6.2,
            a=2.0,
            B0=5.3,
            Ip=15.0,
            kappa=1.7,
            delta=0.33,
            n_e=10.1,
            T_e=25.0,
            P_aux=50.0,
        )

    @classmethod
    def sparc(cls) -> TokamakConfig:
        """SPARC V2C. Creely et al., J. Plasma Phys. 86, 865860502 (2020)."""
        return cls(
            name="SPARC",
            R0=1.85,
            a=0.57,
            B0=12.2,
            Ip=8.7,
            kappa=1.97,
            delta=0.54,
            n_e=30.0,
            T_e=21.0,
            P_aux=25.0,
        )

    @classmethod
    def diiid(cls) -> TokamakConfig:
        """DIII-D typical H-mode. Luxon, Nucl. Fusion 42, 614 (2002)."""
        return cls(
            name="DIII-D",
            R0=1.67,
            a=0.67,
            B0=2.1,
            Ip=1.5,
            kappa=1.8,
            delta=0.4,
            n_e=5.0,
            T_e=4.0,
            P_aux=10.0,
        )

    @classmethod
    def jet(cls) -> TokamakConfig:
        """JET with ITER-like wall. Joffrin et al., Nucl. Fusion 59 (2019)."""
        return cls(
            name="JET",
            R0=2.96,
            a=1.25,
            B0=3.45,
            Ip=3.5,
            kappa=1.68,
            delta=0.3,
            n_e=7.0,
            T_e=8.0,
            P_aux=25.0,
        )
