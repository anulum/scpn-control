# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Confinement Scaling Laws
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Energy confinement time scaling laws for tokamak H-mode plasmas.

Implements the IPB98(y,2) empirical scaling law from the ITER Physics
Basis (Nuclear Fusion 39, 1999, 2175) with updated coefficients from
Verdoolaege et al. (Nuclear Fusion 61, 2021, 076006).

The IPB98(y,2) law is a multi-variate power-law regression fitted to
5920 H-mode data points from 18 tokamaks in the ITPA global confinement
database.  It predicts the thermal energy confinement time τ_E as:

    τ_E = C · Ip^α_I · B_T^α_B · n̄_e19^α_n · P_loss^α_P
              · R^α_R · κ^α_κ · ε^α_ε · M^α_M

Coefficients are loaded from the JSON file shipped in
``validation/reference_data/itpa/ipb98y2_coefficients.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scpn_control.core._validators import require_finite_float, require_positive_float

logger = logging.getLogger(__name__)

# ── Default coefficient path ──────────────────────────────────────────

_DEFAULT_COEFF_PATH = (
    Path(__file__).resolve().parents[3] / "validation" / "reference_data" / "itpa" / "ipb98y2_coefficients.json"
)

_REQUIRED_EXPONENT_KEYS = (
    "Ip_MA",
    "BT_T",
    "ne19_1e19m3",
    "Ploss_MW",
    "R_m",
    "kappa",
    "epsilon",
    "M_AMU",
)


# ── Data container ────────────────────────────────────────────────────


@dataclass
class TransportBenchmarkResult:
    """Result of an IPB98(y,2) benchmark comparison."""

    machine: str
    shot: str
    tau_e_measured: float
    tau_e_predicted: float
    h_factor: float
    relative_error: float


# ── Core functions ────────────────────────────────────────────────────


def _require_finite_number(name: str, value: Any) -> float:
    """Validate generic numeric metadata loaded from coefficient files.

    Wraps non-numeric inputs (strings, None) into ValueError before
    delegating the finiteness check to require_finite_float.
    """
    try:
        float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    return require_finite_float(name, value)


def _validate_ipb98y2_coefficients(raw: Any) -> dict[str, Any]:
    """Validate and normalize an IPB98(y,2) coefficient mapping."""
    if not isinstance(raw, dict):
        raise ValueError("IPB98(y,2) coefficients must be a JSON object")

    if "C" not in raw:
        raise ValueError("IPB98(y,2) coefficients missing required key 'C'")
    C = _require_finite_number("C", raw["C"])
    if C <= 0.0:
        raise ValueError(f"C must be > 0, got {C!r}")

    exp_raw = raw.get("exponents")
    if not isinstance(exp_raw, dict):
        raise ValueError("IPB98(y,2) coefficients key 'exponents' must be an object")

    exponents: dict[str, float] = {}
    for key in _REQUIRED_EXPONENT_KEYS:
        if key not in exp_raw:
            raise ValueError(f"IPB98(y,2) coefficients missing exponent key '{key}'")
        exponents[key] = _require_finite_number(f"exponents.{key}", exp_raw[key])

    normalized: dict[str, Any] = dict(raw)
    normalized["C"] = C
    normalized["exponents"] = exponents

    if "sigma_lnC" in normalized:
        sigma_lnc = _require_finite_number("sigma_lnC", normalized["sigma_lnC"])
        if sigma_lnc < 0.0:
            raise ValueError(f"sigma_lnC must be >= 0, got {sigma_lnc!r}")
        normalized["sigma_lnC"] = sigma_lnc

    if "uncertainties_1sigma" in normalized:
        unc_raw = normalized["uncertainties_1sigma"]
        if not isinstance(unc_raw, dict):
            raise ValueError("uncertainties_1sigma must be an object")
        uncertainties: dict[str, float] = {}
        for key, value in unc_raw.items():
            sigma = _require_finite_number(f"uncertainties_1sigma.{key}", value)
            if sigma < 0.0:
                raise ValueError(f"uncertainties_1sigma.{key} must be >= 0, got {sigma!r}")
            uncertainties[key] = sigma
        normalized["uncertainties_1sigma"] = uncertainties

    return normalized


def load_ipb98y2_coefficients(
    path: str | Path | None = None,
) -> dict:
    """Load IPB98(y,2) coefficients from the JSON reference file.

    Parameters
    ----------
    path : str or Path, optional
        Override path.  Defaults to the file shipped with this package.

    Returns
    -------
    dict
        Parsed JSON with keys ``"C"``, ``"exponents"``, etc.
    """
    p = Path(path) if path else _DEFAULT_COEFF_PATH
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return _validate_ipb98y2_coefficients(raw)


def ipb98y2_tau_e(
    Ip: float,
    BT: float,
    ne19: float,
    Ploss: float,
    R: float,
    kappa: float,
    epsilon: float,
    M: float = 2.5,  # D-T effective ion mass [AMU]; ITER Physics Basis, NF 39 (1999)
    *,
    coefficients: dict | None = None,
) -> float:
    """Evaluate the IPB98(y,2) confinement time scaling law.

    Parameters
    ----------
    Ip : float
        Plasma current [MA].
    BT : float
        Toroidal magnetic field [T].
    ne19 : float
        Line-averaged electron density [10^19 m^-3].
    Ploss : float
        Loss power [MW].  Must be > 0.
    R : float
        Major radius [m].
    kappa : float
        Elongation (κ).
    epsilon : float
        Inverse aspect ratio a/R.
    M : float
        Effective ion mass [AMU].  Default 2.5 (D-T).
    coefficients : dict, optional
        Pre-loaded coefficient dict.  If *None*, loaded from disk.

    Returns
    -------
    float
        Predicted thermal energy confinement time τ_E [s].

    Raises
    ------
    ValueError
        If any input is non-finite or non-positive.
    """
    Ip = require_positive_float("Ip", Ip)
    BT = require_positive_float("BT", BT)
    ne19 = require_positive_float("ne19", ne19)
    Ploss = require_positive_float("Ploss", Ploss)
    R = require_positive_float("R", R)
    kappa = require_positive_float("kappa", kappa)
    epsilon = require_positive_float("epsilon", epsilon)
    M = require_positive_float("M", M)

    if coefficients is None:
        coefficients = load_ipb98y2_coefficients()
    else:
        coefficients = _validate_ipb98y2_coefficients(coefficients)

    C = coefficients["C"]
    exp = coefficients["exponents"]

    tau = (
        C
        * Ip ** exp["Ip_MA"]
        * BT ** exp["BT_T"]
        * ne19 ** exp["ne19_1e19m3"]
        * Ploss ** exp["Ploss_MW"]
        * R ** exp["R_m"]
        * kappa ** exp["kappa"]
        * epsilon ** exp["epsilon"]
        * M ** exp["M_AMU"]
    )
    tau_f = float(tau)
    if not np.isfinite(tau_f) or tau_f <= 0:
        raise ValueError(f"Computed IPB98(y,2) confinement time is invalid (tau={tau_f!r}) for supplied inputs.")
    return tau_f


def ipb98y2_with_uncertainty(
    Ip: float,
    BT: float,
    ne19: float,
    Ploss: float,
    R: float,
    kappa: float,
    epsilon: float,
    M: float = 2.5,  # D-T effective ion mass [AMU]; ITER Physics Basis, NF 39 (1999)
    *,
    coefficients: dict | None = None,
) -> tuple[float, float]:
    """Evaluate IPB98(y,2) with log-linear error propagation.

    Uses published exponent uncertainties from Verdoolaege et al.,
    Nuclear Fusion 61, 076006 (2021) for 95% confidence interval
    estimation via log-linear error propagation.

    Parameters
    ----------
    (same as ipb98y2_tau_e)

    Returns
    -------
    tuple[float, float]
        (tau_E, sigma_tau_E) — predicted confinement time and its
        1-sigma uncertainty [s].
    """
    if coefficients is None:
        coeff = load_ipb98y2_coefficients()
    else:
        coeff = _validate_ipb98y2_coefficients(coefficients)

    tau = ipb98y2_tau_e(
        Ip,
        BT,
        ne19,
        Ploss,
        R,
        kappa,
        epsilon,
        M,
        coefficients=coeff,
    )

    # Published exponent uncertainties (Verdoolaege et al. NF 2021)
    # These are 1-sigma uncertainties on the log-space exponents.
    exp_unc = coeff.get(
        "uncertainties_1sigma",
        {
            "Ip_MA": 0.03,
            "BT_T": 0.05,
            "ne19_1e19m3": 0.04,
            "Ploss_MW": 0.02,
            "R_m": 0.08,
            "kappa": 0.07,
            "epsilon": 0.06,
            "M_AMU": 0.04,
        },
    )

    # Log-linear error propagation:
    # ln(tau) = ln(C) + sum_k alpha_k * ln(x_k)
    # sigma_ln_tau^2 = sum_k (ln(x_k))^2 * sigma_alpha_k^2 + sigma_ln_C^2
    sigma_lnC = float(coeff.get("sigma_lnC", 0.14))

    inputs = {
        "Ip_MA": Ip,
        "BT_T": BT,
        "ne19_1e19m3": ne19,
        "Ploss_MW": Ploss,
        "R_m": R,
        "kappa": kappa,
        "epsilon": epsilon,
        "M_AMU": M,
    }

    var_ln_tau = sigma_lnC**2
    max_f64 = np.finfo(np.float64).max
    for key, val in inputs.items():
        if val > 0 and key in exp_unc:
            log_val = abs(float(np.log(val)))
            sigma_alpha = abs(float(exp_unc[key]))
            if not np.isfinite(log_val) or not np.isfinite(sigma_alpha):
                var_ln_tau = float("inf")
                break

            # Avoid runtime overflow warnings in extreme uncertainty metadata.
            if log_val > 0.0 and sigma_alpha > np.sqrt(max_f64) / log_val:
                var_ln_tau = float("inf")
                break

            var_ln_tau += (log_val * sigma_alpha) ** 2

    sigma_ln_tau = np.sqrt(var_ln_tau)
    # Convert from log-space: sigma_tau ≈ tau * sigma_ln_tau
    sigma_tau = float(tau * sigma_ln_tau)
    if not np.isfinite(sigma_tau) or sigma_tau < 0.0:
        raise ValueError(f"Computed IPB98(y,2) uncertainty is invalid (sigma_tau={sigma_tau!r}) for supplied inputs.")

    return float(tau), sigma_tau


def greenwald_limit(Ip_MA: float, a_m: float) -> float:
    """Compute the Greenwald density limit.

    Reference: Greenwald, M., Plasma Phys. Control. Fusion 44 (2002) R27.
    n_GW = Ip / (pi * a^2) [10^20 m^-3].

    Parameters
    ----------
    Ip_MA : float
        Plasma current in Mega-Amperes.
    a_m : float
        Minor radius in meters.
    """
    Ip = require_positive_float("Ip_MA", Ip_MA)
    a = require_positive_float("a_m", a_m)
    return float(Ip / (np.pi * a**2))


def compute_betan(beta_t_pct: float, a_m: float, BT_T: float, Ip_MA: float) -> float:
    """Compute the normalised beta (Troyon beta).

    Reference: Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209.
    beta_N = beta_t (%) / (Ip / (a * BT)).

    Parameters
    ----------
    beta_t_pct : float
        Toroidal beta in percent.
    a_m : float
        Minor radius in meters.
    BT_T : float
        Toroidal magnetic field in Tesla.
    Ip_MA : float
        Plasma current in Mega-Amperes.
    """
    bt = require_positive_float("beta_t_pct", beta_t_pct)
    a = require_positive_float("a_m", a_m)
    B = require_positive_float("BT_T", BT_T)
    Ip = require_positive_float("Ip_MA", Ip_MA)

    return float(bt / (Ip / (a * B)))


def compute_h_factor(tau_actual: float, tau_predicted: float) -> float:
    """Compute the H-factor (enhancement factor over scaling law).

    Parameters
    ----------
    tau_actual : float
        Measured or simulated confinement time [s].
    tau_predicted : float
        IPB98(y,2) predicted confinement time [s].

    Returns
    -------
    float
        H98(y,2) = tau_actual / tau_predicted.
    """
    tau_actual_f = _require_finite_number("tau_actual", tau_actual)
    tau_predicted_f = _require_finite_number("tau_predicted", tau_predicted)

    if tau_predicted_f <= 0:
        return float("inf")
    return tau_actual_f / tau_predicted_f
