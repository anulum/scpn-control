# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EPED pedestal model
"""EPED-style pedestal prediction and validation-point utilities."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── References ────────────────────────────────────────────────────────────────
# [S09]  Snyder et al. 2009, Phys. Plasmas 16, 056118  — EPED-1 model
# [S11]  Snyder et al. 2011, Nucl. Fusion 51, 103016   — EPED1.6 / collisionality
# [C98]  Connor et al. 1998, PPCF 40, 531              — PB stability boundary
# [Sau99] Sauter et al. 1999, Phys. Plasmas 6, 2834    — bootstrap / peeling
# [W04]  Wesson, 'Tokamaks' 4th ed. (2004)             — safety factor
# ──────────────────────────────────────────────────────────────────────────────

# KBM width coefficient — [S09] Eq. 4; fit to DIII-D/JET/AUG databases
C_KBM_DEFAULT: float = 0.076

# Collisionality narrowing coefficient — [S11] Fig. 7 logarithmic fit
_COLL_NARROW_COEFF: float = 0.4

# Ballooning stability first-access coefficient — [C98] calibrated to published PB limits
_ALPHA_BALL_COEFF: float = 3.0

# Peeling onset β_p threshold — [Sau99] bootstrap proximity limit
_BETA_P_PEEL_MAX: float = 3.0

# Reference shape parameters for F_shape normalisation (ITER-like)
_KAPPA_REF: float = 1.7  # ITER design elongation
_DELTA_REF: float = 0.33  # ITER design triangularity


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


class EPEDConfigSchema(BaseModel):
    """Pydantic v2 schema for EPED pedestal operating-point configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    R0: float = Field(..., gt=0.0, description="Major radius [m]")
    a: float = Field(..., gt=0.0, description="Minor radius [m]")
    B0: float = Field(..., gt=0.0, description="Vacuum toroidal field [T]")
    kappa: float = Field(..., gt=0.0, description="Elongation")
    delta: float = Field(..., description="Triangularity")
    Ip_MA: float = Field(..., gt=0.0, description="Plasma current [MA]")
    ne_ped_19: float = Field(..., gt=0.0, description="Pedestal electron density [10^19 m^-3]")
    B_pol_ped: float = Field(..., gt=0.0, description="Poloidal field at pedestal [T]")
    C_KBM: float = Field(default=C_KBM_DEFAULT, gt=0.0, description="KBM width coefficient")
    n_mode_min: int = 5
    n_mode_max: int = 30
    nu_star_e: float = Field(default=0.0, ge=0.0, description="Electron collisionality")

    @field_validator("R0", "a", "B0", "kappa", "delta", "Ip_MA", "ne_ped_19", "B_pol_ped", "C_KBM", "nu_star_e")
    @classmethod
    def _finite_scalar_field(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("EPEDConfig scalar fields must be finite")
        return value

    @field_validator("n_mode_min", "n_mode_max", mode="before")
    @classmethod
    def _integer_mode_bound(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("mode bounds must be integers")
        return value

    @model_validator(mode="after")
    def _physics_bounds(self) -> "EPEDConfigSchema":
        if self.a >= self.R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        if abs(self.delta) >= 1.0:
            raise ValueError("delta must remain inside the physical triangularity interval (-1, 1)")
        if self.n_mode_min <= 0 or self.n_mode_max < self.n_mode_min:
            raise ValueError("mode bounds must be positive and ordered")
        return self


def _validate_eped_result(result: EPEDResult) -> EPEDResult:
    _finite_scalar("p_ped_kPa", result.p_ped_kPa, positive=True)
    _finite_scalar("T_ped_keV", result.T_ped_keV, positive=True)
    _finite_scalar("n_ped_19", result.n_ped_19, positive=True)
    delta_ped = _finite_scalar("delta_ped", result.delta_ped, positive=True)
    _finite_scalar("beta_p_ped", result.beta_p_ped, nonnegative=True)
    _finite_scalar("alpha_crit", result.alpha_crit, nonnegative=True)
    delta_ped_collisionless = _finite_scalar(
        "delta_ped_collisionless", result.delta_ped_collisionless, nonnegative=True
    )
    if delta_ped >= 1.0:
        raise ValueError("delta_ped must remain below the normalised minor-radius interval")
    if delta_ped_collisionless < delta_ped:
        raise ValueError("delta_ped_collisionless must be greater than or equal to delta_ped")
    return result


@dataclass(frozen=True)
class EPEDConfig:
    """Machine and pedestal inputs for the EPED pedestal model.

    Attributes
    ----------
    R0
        Major radius in metres.
    a
        Minor radius in metres.
    B0
        Vacuum toroidal field in tesla.
    kappa
        Plasma elongation.
    delta
        Plasma triangularity.
    Ip_MA
        Plasma current in MA.
    ne_ped_19
        Pedestal electron density in 10¹⁹ m⁻³.
    B_pol_ped
        Poloidal field at the pedestal in tesla.
    C_KBM
        Kinetic-ballooning-mode coefficient (Snyder 2009, Eq. 4).
    n_mode_min
        Minimum toroidal mode number scanned for peeling-ballooning.
    n_mode_max
        Maximum toroidal mode number scanned.
    nu_star_e
        Electron collisionality (0 = collisionless limit).
    """

    R0: float  # Major radius [m]
    a: float  # Minor radius [m]
    B0: float  # Vacuum toroidal field [T]
    kappa: float  # Elongation
    delta: float  # Triangularity
    Ip_MA: float  # Plasma current [MA]
    ne_ped_19: float  # Pedestal electron density [10^19 m^-3]
    B_pol_ped: float  # Poloidal field at pedestal [T]
    C_KBM: float = C_KBM_DEFAULT  # [S09] Eq. 4
    n_mode_min: int = 5
    n_mode_max: int = 30
    nu_star_e: float = 0.0  # Electron collisionality; 0 = collisionless limit

    def __post_init__(self) -> None:
        validated = EPEDConfigSchema.model_validate(dataclasses.asdict(self))
        for field_name, value in validated.model_dump().items():
            object.__setattr__(self, field_name, value)

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema for serialized EPED pedestal configurations."""
        return EPEDConfigSchema.model_json_schema()


@dataclass
class EPEDResult:
    """Predicted pedestal structure from the EPED model.

    Attributes
    ----------
    p_ped_kPa
        Pedestal pressure in kPa.
    T_ped_keV
        Pedestal temperature in keV.
    n_ped_19
        Pedestal density in 10¹⁹ m⁻³.
    delta_ped
        Pedestal width in normalised poloidal flux.
    beta_p_ped
        Pedestal poloidal beta.
    alpha_crit
        Critical normalised pressure gradient at the peeling-ballooning limit.
    delta_ped_collisionless
        Pedestal width before the collisionality correction.
    """

    p_ped_kPa: float
    T_ped_keV: float
    n_ped_19: float
    delta_ped: float
    beta_p_ped: float
    alpha_crit: float
    delta_ped_collisionless: float = field(default=0.0)  # Δ before collisionality correction


@dataclass
class EPEDValidationPoint:
    """Measured-versus-EPED pedestal comparison for one discharge.

    Attributes
    ----------
    machine
        Device name.
    shot
        Shot number; must be a positive integer.
    p_ped_measured_kPa
        Measured pedestal pressure in kPa.
    p_ped_eped_kPa
        EPED-predicted pedestal pressure in kPa.
    delta_ped_measured
        Measured pedestal width.
    delta_ped_eped
        EPED-predicted pedestal width.
    """

    machine: str
    shot: int
    p_ped_measured_kPa: float
    p_ped_eped_kPa: float
    delta_ped_measured: float
    delta_ped_eped: float

    def __post_init__(self) -> None:
        if not self.machine.strip():
            raise ValueError("machine must be non-empty")
        if isinstance(self.shot, bool) or not isinstance(self.shot, int) or self.shot <= 0:
            raise ValueError("shot must be a positive integer")
        _finite_scalar("p_ped_measured_kPa", self.p_ped_measured_kPa, positive=True)
        _finite_scalar("p_ped_eped_kPa", self.p_ped_eped_kPa, positive=True)
        measured_width = _finite_scalar("delta_ped_measured", self.delta_ped_measured, positive=True)
        eped_width = _finite_scalar("delta_ped_eped", self.delta_ped_eped, positive=True)
        if measured_width >= 1.0 or eped_width >= 1.0:
            raise ValueError("validation pedestal widths must remain below the normalised minor-radius interval")


def eped_validation_database() -> list[EPEDValidationPoint]:
    """
    Synthetic test data for EPED model validation.
    Values are approximate and do not represent real experimental measurements.
    """
    return [
        EPEDValidationPoint("DIII-D", 1, 10.0, 10.5, 0.04, 0.042),
        EPEDValidationPoint("DIII-D", 2, 15.0, 14.2, 0.05, 0.048),
        EPEDValidationPoint("JET", 3, 30.0, 28.5, 0.03, 0.032),
        EPEDValidationPoint("AUG", 4, 20.0, 21.0, 0.035, 0.033),
        EPEDValidationPoint("C-Mod", 5, 40.0, 42.0, 0.02, 0.021),
    ]


def _compute_q95(config: EPEDConfig) -> float:
    """Safety factor at 95% flux surface. [W04] Eq. 3.6.8."""
    return (config.a * config.B0) / (config.R0 * config.B_pol_ped) * (1.0 + config.kappa**2) / 2.0


def _shaping_factor(kappa: float, delta: float) -> float:
    """
    F_shape: elongation + triangularity stabilisation factor.

    Normalised to an ITER-like reference (κ=1.7, δ=0.33) so that
    F_shape = 1 at the reference point.

    Form from [C98] (PB boundary curvature scaling) combined with
    the triangularity enhancement in [S09] Sec. II.

        F_shape = [(1 + 0.5κ)(1 + δ)] / [(1 + 0.5 κ_ref)(1 + δ_ref)]
    """
    kappa = _finite_scalar("kappa", kappa, positive=True)
    delta = _finite_scalar("delta", delta)
    if abs(delta) >= 1.0:
        raise ValueError("delta must remain inside the physical triangularity interval (-1, 1)")
    numerator = (1.0 + 0.5 * kappa) * (1.0 + delta)
    denominator = (1.0 + 0.5 * _KAPPA_REF) * (1.0 + _DELTA_REF)
    return numerator / denominator


def _approx_alpha_crit(delta_ped: float, config: EPEDConfig) -> float:
    """
    Peeling-ballooning α_crit with shaping and peeling corrections.

    Ballooning boundary: [C98] Eq. 5 (first-stability α limit).
    Shaping factor: [C98] / [S09] triangularity + elongation correction.
    Peeling reduction: bootstrap current proximity — [Sau99] Sec. III.
    """
    delta_ped = _finite_scalar("delta_ped", delta_ped, positive=True)
    F_shape = _shaping_factor(config.kappa, config.delta)
    alpha_ball = _ALPHA_BALL_COEFF * F_shape

    # β_p,ped proxy from KBM inversion: β_p = (Δ/C_KBM)²  [S09] Eq. 4
    beta_p_proxy = (delta_ped / config.C_KBM) ** 2
    peel_factor = max(0.0, 1.0 - 0.3 * beta_p_proxy / _BETA_P_PEEL_MAX)

    return alpha_ball * peel_factor


def _collisionality_width_correction(delta_kbm: float, nu_star_e: float) -> float:
    """
    Pedestal width narrowing at high collisionality.

        Δ_eff = Δ_KBM / (1 + 0.4 · ln(1 + ν*_e))

    Logarithmic fit to [S11] Fig. 7 over the range ν*_e ∈ [0.1, 10].
    At ν*_e = 0 the correction vanishes (Δ_eff = Δ_KBM).
    """
    delta_kbm = _finite_scalar("delta_kbm", delta_kbm, positive=True)
    nu_star_e = _finite_scalar("nu_star_e", nu_star_e, nonnegative=True)
    if nu_star_e == 0.0:
        return delta_kbm
    return delta_kbm / (1.0 + _COLL_NARROW_COEFF * math.log(1.0 + nu_star_e))


def eped1_predict(config: EPEDConfig) -> EPEDResult:
    """
    Self-consistent (p_ped, Δ_ped) from PB + KBM constraints. [S09]

    Iteration: guess Δ → α_crit(Δ) → p_ped → β_p,ped → Δ_KBM.
    Collisionality correction applied after convergence. [S11]
    """
    delta_ped = 0.04  # initial guess

    mu_0 = 4.0 * math.pi * 1e-7
    q_95 = _compute_q95(config)

    for _ in range(50):
        delta_old = delta_ped

        alpha_crit = _approx_alpha_crit(delta_ped, config)

        # Invert α definition: α = 2μ₀ q² R₀ |dp/dρ| / B₀²
        # with |dp/dρ| ≈ p_ped / (a · Δ_ped)  — [S09] Eq. 2
        p_ped_Pa = alpha_crit * config.B0**2 * config.a * delta_ped / (2.0 * mu_0 * q_95**2 * config.R0)

        beta_p_ped = 2.0 * mu_0 * p_ped_Pa / config.B_pol_ped**2

        # KBM constraint — [S09] Eq. 4; C_KBM = 0.076
        delta_new = config.C_KBM * math.sqrt(abs(beta_p_ped))

        delta_ped = 0.5 * delta_old + 0.5 * delta_new

        if abs(delta_ped - delta_old) / max(delta_ped, 1e-6) < 0.01:
            break

    delta_collisionless = delta_ped
    delta_ped = _collisionality_width_correction(delta_ped, config.nu_star_e)

    # Re-evaluate p_ped at the collisionality-corrected width
    alpha_crit = _approx_alpha_crit(delta_ped, config)
    p_ped_Pa = alpha_crit * config.B0**2 * config.a * delta_ped / (2.0 * mu_0 * q_95**2 * config.R0)
    beta_p_ped = 2.0 * mu_0 * p_ped_Pa / config.B_pol_ped**2
    p_ped_kPa = p_ped_Pa / 1000.0

    # T_ped = p / (2 n_e e), assuming T_i = T_e, Z_eff ≈ 1  — ideal gas
    e_charge = 1.602e-19  # [J/eV]
    n_e = config.ne_ped_19 * 1e19
    T_ped_keV = p_ped_Pa / (2.0 * n_e) / (e_charge * 1000.0)

    return EPEDResult(
        p_ped_kPa=p_ped_kPa,
        T_ped_keV=T_ped_keV,
        n_ped_19=config.ne_ped_19,
        delta_ped=delta_ped,
        beta_p_ped=beta_p_ped,
        alpha_crit=alpha_crit,
        delta_ped_collisionless=delta_collisionless,
    )


def eped1_scan(config: EPEDConfig, ne_ped_range: AnyFloatArray) -> list[EPEDResult]:
    """Scan pedestal density across operating space."""
    ne_values = np.asarray(ne_ped_range, dtype=float)
    if ne_values.ndim != 1 or ne_values.size == 0:
        raise ValueError("ne_ped_range must be a one-dimensional non-empty scan axis")
    if not np.all(np.isfinite(ne_values)) or np.any(ne_values <= 0.0):
        raise ValueError("ne_ped_range must contain only positive finite densities")
    return [eped1_predict(dataclasses.replace(config, ne_ped_19=float(ne))) for ne in ne_values]


class PedestalProfileGenerator:
    """Generates mtanh profiles from EPED predictions."""

    def __init__(self, eped_result: EPEDResult, Te_sep_eV: float = 100.0, ne_sep_19: float = 0.3):
        self.res = _validate_eped_result(eped_result)
        self.Te_sep_keV = _finite_scalar("Te_sep_eV", Te_sep_eV, positive=True) / 1000.0
        self.ne_sep = _finite_scalar("ne_sep_19", ne_sep_19, positive=True)
        if self.Te_sep_keV >= self.res.T_ped_keV:
            raise ValueError("Te_sep_eV must be below the pedestal-top temperature")
        if self.ne_sep >= self.res.n_ped_19:
            raise ValueError("ne_sep_19 must be below the pedestal-top density")

    def generate(self, rho: AnyFloatArray) -> tuple[FloatArray, FloatArray]:
        """
        Produce (Te, ne) via mtanh.

        rho_sym = 1 − Δ_ped/2  (centre of gradient region)
        """
        rho_sym = 1.0 - self.res.delta_ped / 2.0
        width = self.res.delta_ped
        rho = np.asarray(rho, dtype=float)
        if rho.ndim != 1 or rho.size == 0:
            raise ValueError("rho must be a one-dimensional non-empty profile grid")
        if not np.all(np.isfinite(rho)) or np.any(rho < 0.0) or np.any(rho > 1.0):
            raise ValueError("rho must contain finite normalised radii in [0, 1]")
        if np.any(np.diff(rho) <= 0.0):
            raise ValueError("rho must be strictly increasing")

        def _mtanh(r: AnyFloatArray, height: float, sep: float) -> FloatArray:
            z = 2.0 * (rho_sym - r) / max(width / 2.0, 1e-3)
            prof = np.asarray((height - sep) / 2.0 * (np.tanh(z) + 1.0) + sep)
            prof[r >= 1.0] = sep
            return prof

        return _mtanh(rho, self.res.T_ped_keV, self.Te_sep_keV), _mtanh(rho, self.res.n_ped_19, self.ne_sep)


@dataclass
class EpedPedestalPrediction:
    """Compact EPED prediction for the integrated transport solver.

    Attributes
    ----------
    Delta_ped
        Pedestal width in normalised poloidal flux.
    T_ped_keV
        Pedestal temperature in keV.
    p_ped_kPa
        Pedestal pressure in kPa.
    """

    Delta_ped: float
    T_ped_keV: float
    p_ped_kPa: float


class EpedPedestalModel:
    """Wrapper for integrated_transport_solver compatibility."""

    def __init__(
        self,
        R0: float,
        a: float,
        B0: float,
        Ip_MA: float,
        kappa: float = 1.7,
        A_ion: float = 2.0,
        Z_eff: float = 1.5,
    ) -> None:
        R0 = _finite_scalar("R0", R0, positive=True)
        a = _finite_scalar("a", a, positive=True)
        if a >= R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        B0 = _finite_scalar("B0", B0, positive=True)
        Ip_MA = _finite_scalar("Ip_MA", Ip_MA, positive=True)
        kappa = _finite_scalar("kappa", kappa, positive=True)
        _finite_scalar("A_ion", A_ion, positive=True)
        _finite_scalar("Z_eff", Z_eff, positive=True)
        mu_0 = 4.0 * math.pi * 1e-7
        B_pol = mu_0 * Ip_MA * 1e6 / (2.0 * math.pi * a * math.sqrt(kappa))
        self._R0 = R0
        self._a = a
        self._B0 = B0
        self._Ip_MA = Ip_MA
        self._kappa = kappa
        self._B_pol_ped = B_pol

    def predict(self, ne_ped_19: float, nu_star_e: float = 0.0) -> EpedPedestalPrediction:
        """Predict the pedestal structure at a given density and collisionality.

        Parameters
        ----------
        ne_ped_19
            Pedestal electron density in 10¹⁹ m⁻³; must be positive.
        nu_star_e
            Electron collisionality; must be non-negative (0 = collisionless).

        Returns
        -------
        EpedPedestalPrediction
            The predicted pedestal width, temperature, and pressure.
        """
        ne_ped_19 = _finite_scalar("ne_ped_19", ne_ped_19, positive=True)
        nu_star_e = _finite_scalar("nu_star_e", nu_star_e, nonnegative=True)
        cfg = EPEDConfig(
            R0=self._R0,
            a=self._a,
            B0=self._B0,
            kappa=self._kappa,
            delta=0.33,
            Ip_MA=self._Ip_MA,
            ne_ped_19=ne_ped_19,
            B_pol_ped=self._B_pol_ped,
            nu_star_e=nu_star_e,
        )
        res = eped1_predict(cfg)
        return EpedPedestalPrediction(
            Delta_ped=res.delta_ped,
            T_ped_keV=res.T_ped_keV,
            p_ped_kPa=res.p_ped_kPa,
        )
