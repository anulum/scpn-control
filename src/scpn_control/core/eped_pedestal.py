# ──────────────────────────────────────────────────────────────────────
# SCPN Control — EPED Pedestal Prediction Model
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class EPEDConfig:
    R0: float
    a: float
    B0: float
    kappa: float
    delta: float
    Ip_MA: float
    ne_ped_19: float
    B_pol_ped: float
    C_KBM: float = 0.076
    n_mode_min: int = 5
    n_mode_max: int = 30


@dataclass
class EPEDResult:
    p_ped_kPa: float
    T_ped_keV: float
    n_ped_19: float
    delta_ped: float
    beta_p_ped: float
    alpha_crit: float


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
    """Safety factor at 95% flux surface. Wesson, 'Tokamaks' 4th ed., Eq. 3.6.8."""
    return (config.a * config.B0) / (config.R0 * config.B_pol_ped) * math.sqrt((1.0 + config.kappa ** 2) / 2.0)


def _approx_alpha_crit(delta_ped: float, config: EPEDConfig) -> float:
    """
    Ballooning stability α_crit with shaping correction.
    Connor et al., PPCF 40, 531 (1998).
    Shaping factor: Sauter et al., Phys. Plasmas 6, 2834 (1999).
    Peeling: bootstrap current proximity to kink limit reduces α_crit.
    """
    kappa = config.kappa
    delta = config.delta

    # Shaping stabilization (elongation + triangularity)
    F_shape = (1.0 + kappa ** 2 * (1.0 + 2.0 * delta ** 2)) / (2.0 * kappa)

    # Ballooning α_crit calibrated to ELITE for ITER-class devices
    ALPHA_BALL_COEFF = 3.0  # dimensionless, first-stability access coefficient
    alpha_ball = ALPHA_BALL_COEFF * F_shape

    # Peeling correction: wider pedestal → more bootstrap current → destabilization
    # β_p,ped ≈ (Δ_ped / C_KBM)² from KBM constraint
    beta_p_proxy = (delta_ped / config.C_KBM) ** 2
    BETA_P_PEEL_MAX = 3.0  # peeling onset threshold, dimensionless
    peel_factor = max(0.0, 1.0 - 0.3 * beta_p_proxy / BETA_P_PEEL_MAX)

    return alpha_ball * peel_factor


def eped1_predict(config: EPEDConfig) -> EPEDResult:
    """
    Solve for self-consistent (p_ped, delta_ped) satisfying PB and KBM constraints.
    """
    delta_ped = 0.04  # Initial guess

    mu_0 = 4.0 * math.pi * 1e-7
    q_95 = _compute_q95(config)

    # Iterate to convergence
    for _ in range(50):
        delta_old = delta_ped

        # 1. PB constraint gives alpha_crit
        alpha_crit = _approx_alpha_crit(delta_ped, config)

        # Standard α definition: α = 2μ₀ q² R₀ (dp/dr) / B₀²
        # At pedestal: dp/dr ≈ p_ped / (a Δ_ped)
        # Invert: p_ped = α_crit × B₀² × a × Δ_ped / (2μ₀ q₉₅² R₀)
        p_ped_Pa = alpha_crit * config.B0 ** 2 * config.a * delta_ped / (2.0 * mu_0 * q_95 ** 2 * config.R0)

        # 2. Compute beta_p_ped
        beta_p_ped = 2.0 * mu_0 * p_ped_Pa / (config.B_pol_ped**2)

        # 3. KBM constraint gives new delta_ped
        delta_new = config.C_KBM * math.sqrt(abs(beta_p_ped))

        # Relaxation
        delta_ped = 0.5 * delta_old + 0.5 * delta_new

        if abs(delta_ped - delta_old) / max(delta_ped, 1e-6) < 0.01:
            break

    # Finalize
    p_ped_kPa = p_ped_Pa / 1000.0

    # T_ped = p_ped / (2 * n_ped * e)
    # p = 2 * n_e * T_e (assume Ti = Te, Z=1)
    e_charge = 1.602e-19
    n_e = config.ne_ped_19 * 1e19
    T_ped_J = p_ped_Pa / (2.0 * n_e)
    T_ped_keV = T_ped_J / (e_charge * 1000.0)

    return EPEDResult(
        p_ped_kPa=p_ped_kPa,
        T_ped_keV=T_ped_keV,
        n_ped_19=config.ne_ped_19,
        delta_ped=delta_ped,
        beta_p_ped=beta_p_ped,
        alpha_crit=alpha_crit,
    )


def eped1_scan(config: EPEDConfig, ne_ped_range: np.ndarray) -> list[EPEDResult]:
    """
    Scan pedestal density to map out operating space.
    """
    results = []
    for ne in ne_ped_range:
        cfg = dataclasses.replace(config, ne_ped_19=float(ne))
        results.append(eped1_predict(cfg))
    return results


class PedestalProfileGenerator:
    """
    Generates mtanh profiles from EPED predictions.
    """

    def __init__(self, eped_result: EPEDResult, Te_sep_eV: float = 100.0, ne_sep_19: float = 0.3):
        self.res = eped_result
        self.Te_sep_keV = Te_sep_eV / 1000.0
        self.ne_sep = ne_sep_19

    def generate(self, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Produce (Te, ne) using mtanh.
        rho_ped = 1 - delta_ped/2 (approx center of gradient region)
        width = delta_ped
        """
        rho_sym = 1.0 - self.res.delta_ped / 2.0
        width = self.res.delta_ped

        def mtanh(r: np.ndarray, height: float, sep: float) -> np.ndarray:
            z = 2.0 * (rho_sym - r) / max(width / 2.0, 1e-3)
            prof = np.asarray((height - sep) / 2.0 * (np.tanh(z) + 1.0) + sep)
            prof[r >= 1.0] = sep
            return prof

        Te = mtanh(rho, self.res.T_ped_keV, self.Te_sep_keV)
        ne = mtanh(rho, self.res.n_ped_19, self.ne_sep)

        return Te, ne


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

    def predict(self, ne_ped_19: float) -> EpedPedestalPrediction:
        cfg = EPEDConfig(
            R0=self._R0,
            a=self._a,
            B0=self._B0,
            kappa=self._kappa,
            delta=0.33,
            Ip_MA=self._Ip_MA,
            ne_ped_19=ne_ped_19,
            B_pol_ped=self._B_pol_ped,
        )
        res = eped1_predict(cfg)
        return EpedPedestalPrediction(
            Delta_ped=res.delta_ped,
            T_ped_keV=res.T_ped_keV,
            p_ped_kPa=res.p_ped_kPa,
        )
