# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# Contact: protoscience@anulum.li  ORCID: 0009-0009-3560-0851
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np


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

# Ballooning stability first-access coefficient — [C98] calibrated to ELITE
_ALPHA_BALL_COEFF: float = 3.0

# Peeling onset β_p threshold — [Sau99] bootstrap proximity limit
_BETA_P_PEEL_MAX: float = 3.0

# Reference shape parameters for F_shape normalisation (ITER-like)
_KAPPA_REF: float = 1.7  # ITER design elongation
_DELTA_REF: float = 0.33  # ITER design triangularity


@dataclass
class EPEDConfig:
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


@dataclass
class EPEDResult:
    p_ped_kPa: float
    T_ped_keV: float
    n_ped_19: float
    delta_ped: float
    beta_p_ped: float
    alpha_crit: float
    delta_ped_collisionless: float = field(default=0.0)  # Δ before collisionality correction


@dataclass
class EPEDValidationPoint:
    machine: str
    shot: int
    p_ped_measured_kPa: float
    p_ped_eped_kPa: float
    delta_ped_measured: float
    delta_ped_eped: float


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
    if nu_star_e <= 0.0:
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


def eped1_scan(config: EPEDConfig, ne_ped_range: np.ndarray) -> list[EPEDResult]:
    """Scan pedestal density across operating space."""
    return [eped1_predict(dataclasses.replace(config, ne_ped_19=float(ne))) for ne in ne_ped_range]


class PedestalProfileGenerator:
    """Generates mtanh profiles from EPED predictions."""

    def __init__(self, eped_result: EPEDResult, Te_sep_eV: float = 100.0, ne_sep_19: float = 0.3):
        self.res = eped_result
        self.Te_sep_keV = Te_sep_eV / 1000.0
        self.ne_sep = ne_sep_19

    def generate(self, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Produce (Te, ne) via mtanh.

        rho_sym = 1 − Δ_ped/2  (centre of gradient region)
        """
        rho_sym = 1.0 - self.res.delta_ped / 2.0
        width = self.res.delta_ped

        def _mtanh(r: np.ndarray, height: float, sep: float) -> np.ndarray:
            z = 2.0 * (rho_sym - r) / max(width / 2.0, 1e-3)
            prof = np.asarray((height - sep) / 2.0 * (np.tanh(z) + 1.0) + sep)
            prof[r >= 1.0] = sep
            return prof

        return _mtanh(rho, self.res.T_ped_keV, self.Te_sep_keV), _mtanh(rho, self.res.n_ped_19, self.ne_sep)


@dataclass
class EpedPedestalPrediction:
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
        mu_0 = 4.0 * math.pi * 1e-7
        B_pol = mu_0 * Ip_MA * 1e6 / (2.0 * math.pi * a * math.sqrt(kappa))
        self._R0 = R0
        self._a = a
        self._B0 = B0
        self._Ip_MA = Ip_MA
        self._kappa = kappa
        self._B_pol_ped = B_pol

    def predict(self, ne_ped_19: float, nu_star_e: float = 0.0) -> EpedPedestalPrediction:
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
