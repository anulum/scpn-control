# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Bayesian Uncertainty Quantification
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Bayesian uncertainty quantification for fusion performance predictions.

Provides error bars on confinement time, fusion power, and Q-factor by
Monte Carlo sampling over scaling-law parameter uncertainties. Uses the
IPB98(y,2) H-mode confinement scaling as the baseline model.

References
----------
- ITER Physics Basis, Nucl. Fusion 39 (1999) 2175
- Verdoolaege et al., Nucl. Fusion 61 (2021) 076006
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scpn_control.core.scaling_laws import load_ipb98y2_coefficients

# Load central values and uncertainties from the global registry
_COEFFS = load_ipb98y2_coefficients()
IPB98_CENTRAL = {
    "C": _COEFFS["C"],
    "alpha_I": _COEFFS["exponents"]["Ip_MA"],
    "alpha_B": _COEFFS["exponents"]["BT_T"],
    "alpha_P": _COEFFS["exponents"]["Ploss_MW"],
    "alpha_n": _COEFFS["exponents"]["ne19_1e19m3"],
    "alpha_R": _COEFFS["exponents"]["R_m"],
    "alpha_A": -_COEFFS["exponents"]["epsilon"],  # ε = a/R = 1/A
    "alpha_kappa": _COEFFS["exponents"]["kappa"],
    "alpha_M": _COEFFS["exponents"]["M_AMU"],
}

IPB98_SIGMA = {
    "C": _COEFFS["uncertainties_1sigma"]["C"],
    "alpha_I": _COEFFS["uncertainties_1sigma"]["Ip_MA"],
    "alpha_B": _COEFFS["uncertainties_1sigma"]["BT_T"],
    "alpha_P": _COEFFS["uncertainties_1sigma"]["Ploss_MW"],
    "alpha_n": _COEFFS["uncertainties_1sigma"]["ne19_1e19m3"],
    "alpha_R": _COEFFS["uncertainties_1sigma"]["R_m"],
    "alpha_A": _COEFFS["uncertainties_1sigma"]["epsilon"],
    "alpha_kappa": _COEFFS["uncertainties_1sigma"]["kappa"],
    "alpha_M": _COEFFS["uncertainties_1sigma"]["M_AMU"],
}


def _validate_n_samples(n_samples: int) -> int:
    """Validate Monte Carlo sample count and normalise to int."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError("n_samples must be an integer >= 1")
    parsed = int(n_samples)
    if parsed < 1:
        raise ValueError("n_samples must be an integer >= 1")
    return parsed


@dataclass
class PlasmaScenario:
    """Input plasma parameters for a confinement prediction."""

    I_p: float  # Plasma current (MA)
    B_t: float  # Toroidal field (T)
    P_heat: float  # Total heating power (MW)
    n_e: float  # Line-average electron density (10^19 m^-3)
    R: float  # Major radius (m)
    A: float  # Aspect ratio R/a
    kappa: float  # Elongation
    M: float = 2.5  # Effective ion mass (AMU, 2.5 for D-T)


@dataclass
class UQResult:
    """Uncertainty-quantified prediction result."""

    # Central estimates
    tau_E: float  # Confinement time (s)
    P_fusion: float  # Fusion power (MW)
    Q: float  # Fusion gain Q = P_fus / P_heat

    # Uncertainties (1-sigma)
    tau_E_sigma: float
    P_fusion_sigma: float
    Q_sigma: float

    # Percentiles [5%, 25%, 50%, 75%, 95%]
    tau_E_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    P_fusion_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    Q_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # Raw samples (for custom analysis)
    n_samples: int = 0


def ipb98_tau_e(scenario: PlasmaScenario, params: dict | None = None) -> float:
    """
    Compute IPB98(y,2) confinement time for given plasma parameters.

    Parameters
    ----------
    scenario : PlasmaScenario
        Plasma parameters.
    params : dict, optional
        Scaling law coefficients. Defaults to IPB98 central values.

    Returns
    -------
    float — confinement time in seconds.
    """
    p = params or IPB98_CENTRAL
    return float(
        p["C"]
        * scenario.I_p ** p["alpha_I"]
        * scenario.B_t ** p["alpha_B"]
        * scenario.P_heat ** p["alpha_P"]
        * scenario.n_e ** p["alpha_n"]
        * scenario.R ** p["alpha_R"]
        * scenario.A ** p["alpha_A"]
        * scenario.kappa ** p["alpha_kappa"]
        * scenario.M ** p["alpha_M"]
    )


def bosch_hale_reactivity(T_i_kev: float | np.ndarray) -> float | np.ndarray:
    """
    Compute D-T fusion reactivity <σv> using Bosch-Hale parameterization.

    Reference: Bosch, H.S. & Hale, G.M. (1992). "Improved formulas for fusion
    cross-sections and thermal reactivities." Nucl. Fusion 32, 611.

    Parameters
    ----------
    T_i_kev : float | np.ndarray
        Ion temperature in keV.

    Returns
    -------
    float | np.ndarray — Reactivity in m^3/s.
    """
    T = np.maximum(T_i_kev, 0.1)
    # D-T coefficients from Table VII
    B_G = 34.3827
    m_rc2 = 1124656.0
    C1 = 1.17302e-9
    C2 = 1.51361e-2
    C3 = 7.51886e-2
    C4 = 4.60643e-3
    C5 = 1.35302e-2
    C6 = -1.06750e-4
    C7 = 1.36600e-5

    theta = T / (1.0 - T * (C2 + T * (C4 + T * C6)) / (1.0 + T * (C3 + T * (C5 + T * C7))))
    xi = (B_G**2 / (4.0 * theta)) ** (1.0 / 3.0)
    sig_v = C1 * theta * np.sqrt(xi / (m_rc2 * T**3)) * np.exp(-3.0 * xi)

    if isinstance(T_i_kev, np.ndarray):
        return np.asarray(sig_v * 1.0e-6)
    return float(sig_v * 1.0e-6)


def fusion_power_from_tau(scenario: PlasmaScenario, tau_E: float) -> float:
    """
    Estimate fusion power using energy balance and Bosch-Hale reactivity.

    Based on the simplified fusion power model:
    P_fus = 1/4 * n_e^2 * <σv>(T_i) * V * E_fus
    Ref: Wesson, J. (2011). Tokamaks. 4th Edition, Chapter 1.

    Where T_i is estimated from P_heat, tau_E and V.
    """
    # 1. Estimate average temperature from energy balance
    # W = 3 * n_e * T_avg * V = P_heat * tau_E
    # T_avg [keV] = (P_heat [MW] * tau_E [s]) / (3 * n_e [10^19] * V [m^3] * e_J_per_kev * 1e19 * 1e-6)
    e_J_per_kev = 1.602176634e-16
    a = scenario.R / scenario.A
    V = 2.0 * np.pi**2 * scenario.R * a**2 * scenario.kappa
    n_m3 = scenario.n_e * 1e19

    T_avg = (scenario.P_heat * 1e6 * tau_E) / (3.0 * n_m3 * V * e_J_per_kev)

    # 2. Compute reactivity
    sig_v = bosch_hale_reactivity(T_avg)

    # 3. Fusion power: P_fus = 1/4 * n_e^2 * <σv> * V * E_fus
    # E_fus = 17.6 MeV = 2.82e-12 J
    E_fus_J = 2.82e-12
    P_fus_W = 0.25 * (n_m3**2) * sig_v * V * E_fus_J
    return float(P_fus_W * 1e-6)  # MW


def compute_fusion_sensitivities(scenario: PlasmaScenario, tau_E: float) -> dict[str, float]:
    """
    Compute derivatives of P_fusion with respect to temperature and density.

    Returns
    -------
    dict: {'dP_dT': ..., 'dP_dn': ...} in MW/keV and MW/(10^19 m^-3).
    """
    # Numerical derivatives
    eps = 1e-3

    def get_p(n_e: float, p_heat: float) -> float:
        sc = PlasmaScenario(scenario.I_p, scenario.B_t, p_heat, n_e, scenario.R, scenario.A, scenario.kappa, scenario.M)
        return fusion_power_from_tau(sc, tau_E)

    # dP/dn (fixed tau_E and P_heat)
    p_plus = get_p(scenario.n_e * (1 + eps), scenario.P_heat)
    p_minus = get_p(scenario.n_e * (1 - eps), scenario.P_heat)
    dP_dn = (p_plus - p_minus) / (2 * scenario.n_e * eps)

    # dP/dT is more indirect via P_heat (T ~ P_heat * tau / n)
    p_plus_t = get_p(scenario.n_e, scenario.P_heat * (1 + eps))
    p_minus_t = get_p(scenario.n_e, scenario.P_heat * (1 - eps))
    # T = c * P_heat -> dT = c * dP_heat -> dP/dT = (dP/dP_heat) / c
    # c = tau / (3 * n * V * e)
    e_J_per_kev = 1.602176634e-16
    a = scenario.R / scenario.A
    V = 2.0 * np.pi**2 * scenario.R * a**2 * scenario.kappa
    c = 1e6 * tau_E / (3.0 * scenario.n_e * 1e19 * V * e_J_per_kev)
    dP_dT = (p_plus_t - p_minus_t) / (2 * scenario.P_heat * eps) / c

    return {"dP_dT": float(dP_dT), "dP_dn": float(dP_dn)}


@dataclass
class EquilibriumUncertainty:
    """Uncertainty contributions from equilibrium reconstruction.

    Captures how boundary perturbations propagate into psi-field and
    magnetic-axis location uncertainty.
    """

    psi_nrmse_mean: float = 0.0  # Mean normalised RMSE of psi reconstruction
    psi_nrmse_sigma: float = 0.01  # 1-sigma spread from boundary perturbation
    R_axis_sigma: float = 0.02  # Magnetic axis R location uncertainty (m)
    Z_axis_sigma: float = 0.01  # Magnetic axis Z location uncertainty (m)


@dataclass
class TransportUncertainty:
    """Uncertainty contributions from transport model coefficients.

    Captures the dominant sources of uncertainty in the gyro-Bohm diffusivity
    model and the EPED pedestal prediction.
    """

    chi_gB_factor_sigma: float = 0.3  # Gyro-Bohm coefficient fractional uncertainty (30%)
    pedestal_height_sigma: float = 0.2  # EPED pedestal height fractional uncertainty (20%)


@dataclass
class FullChainUQResult:
    """Extended uncertainty-quantified prediction covering the full
    equilibrium -> transport -> fusion power chain.

    All ``*_bands`` fields are length-3 arrays: [5th, 50th, 95th] percentiles.
    """

    # Central estimates (medians)
    tau_E: float
    P_fusion: float
    Q: float

    # 1-sigma spreads
    tau_E_sigma: float
    P_fusion_sigma: float
    Q_sigma: float

    # Percentile bands [5%, 50%, 95%]
    psi_nrmse_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tau_E_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P_fusion_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Q_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    beta_N_bands: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Legacy-compatible percentiles [5, 25, 50, 75, 95]
    tau_E_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    P_fusion_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))
    Q_percentiles: np.ndarray = field(default_factory=lambda: np.zeros(5))

    n_samples: int = 0


def _build_ipb98_covariance() -> np.ndarray:
    """Build the covariance matrix for IPB98(y,2) coefficients.

    Incorporates known physical correlations (Verdoolaege et al. 2021).
    """
    keys = ["C", "alpha_I", "alpha_B", "alpha_P", "alpha_n", "alpha_R", "alpha_A", "alpha_kappa", "alpha_M"]
    sigmas = np.array([IPB98_SIGMA[k] for k in keys])

    # Start with diagonal covariance (uncorrelated)
    cov = np.diag(sigmas**2)

    # Add known physical correlations
    # Correlation between prefactor C and R_major exponent is typically ~ -0.7
    idx_c = 0
    idx_r = 5
    corr_cr = -0.7
    cov[idx_c, idx_r] = corr_cr * sigmas[idx_c] * sigmas[idx_r]
    cov[idx_r, idx_c] = cov[idx_c, idx_r]

    # Correlation between Ip and BT exponents is typically ~ 0.4
    idx_i = 1
    idx_b = 2
    corr_ib = 0.4
    cov[idx_i, idx_b] = corr_ib * sigmas[idx_i] * sigmas[idx_b]
    cov[idx_b, idx_i] = cov[idx_i, idx_b]

    return cov


def quantify_full_chain(
    scenario: PlasmaScenario,
    n_samples: int = 5000,
    seed: int | None = None,
    chi_gB_sigma: float = 0.3,
    pedestal_sigma: float = 0.2,
    boundary_sigma: float = 0.02,
) -> FullChainUQResult:
    """
    Full-chain Monte Carlo uncertainty propagation:
    equilibrium -> transport -> fusion power -> gain.

    Now uses correlated sampling for IPB98 coefficients.
    """
    n_samples = _validate_n_samples(n_samples)

    def _validate_sigma(name: str, value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and >= 0") from exc
        if not np.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
        return parsed

    chi_gB_sigma = _validate_sigma("chi_gB_sigma", chi_gB_sigma)
    pedestal_sigma = _validate_sigma("pedestal_sigma", pedestal_sigma)
    boundary_sigma = _validate_sigma("boundary_sigma", boundary_sigma)

    rng = np.random.default_rng(seed)

    tau_samples = np.zeros(n_samples)
    pfus_samples = np.zeros(n_samples)
    q_samples = np.zeros(n_samples)
    beta_n_samples = np.zeros(n_samples)
    psi_nrmse_samples = np.zeros(n_samples)

    # 1. Pre-sample correlated IPB98 coefficients
    keys = ["C", "alpha_I", "alpha_B", "alpha_P", "alpha_n", "alpha_R", "alpha_A", "alpha_kappa", "alpha_M"]
    means = np.array([IPB98_CENTRAL[k] for k in keys])
    cov = _build_ipb98_covariance()
    ipb_samples = rng.multivariate_normal(means, cov, size=n_samples)

    for i in range(n_samples):
        # --- (a) Extract perturbed IPB98 scaling-law coefficients ---
        params = {keys[j]: ipb_samples[i, j] for j in range(len(keys))}
        params["C"] = max(params["C"], 1e-4)
        params["alpha_P"] = min(params["alpha_P"], -0.1)

        # --- (b) Perturb gyro-Bohm transport coefficient ---
        chi_factor = rng.lognormal(0.0, chi_gB_sigma)

        # --- (c) Perturb pedestal height ---
        ped_factor = rng.normal(1.0, pedestal_sigma)
        ped_factor = max(ped_factor, 0.1)  # floor to 10% of nominal

        # --- (d) Perturb equilibrium boundary (major radius) ---
        R_pert = scenario.R * (1.0 + rng.normal(0.0, boundary_sigma))
        R_pert = max(R_pert, 0.5)  # physical floor

        # --- (i) psi NRMSE from boundary perturbation ---
        # Boundary displacement maps roughly linearly to psi reconstruction error
        psi_nrmse = abs(R_pert - scenario.R) / scenario.R
        psi_nrmse_samples[i] = psi_nrmse

        # --- (e) Compute tau_E with perturbed chi applied multiplicatively ---
        # Use perturbed R for the scaling law
        pert_scenario = PlasmaScenario(
            I_p=scenario.I_p,
            B_t=scenario.B_t,
            P_heat=scenario.P_heat,
            n_e=scenario.n_e,
            R=R_pert,
            A=scenario.A,
            kappa=scenario.kappa,
            M=scenario.M,
        )
        tau = ipb98_tau_e(pert_scenario, params)
        # chi_factor > 1 means stronger transport → shorter confinement;
        # pedestal height factor > 1 means better pedestal → longer confinement
        tau = tau * ped_factor / chi_factor
        tau = max(tau, 1e-6)

        # --- (f) Compute P_fusion from perturbed tau_E ---
        pfus = fusion_power_from_tau(pert_scenario, tau)
        pfus = max(pfus, 0.0)

        # --- (g) Compute Q ---
        q = pfus / scenario.P_heat if scenario.P_heat > 0 else 0.0

        # --- (h) Compute normalised beta ---
        # beta_t = (n_e * 1e19 * k_B * T_i) / (B^2 / 2mu0)
        # For a rough estimate, T_i ~ tau_E * P_heat / (n_e * V) gives
        # beta_t ~ C * n_e * tau_E * P_heat / (B^2 * V)
        # Then beta_N = beta_t(%) / (I_p / (a * B_t))
        a_pert = R_pert / scenario.A
        V_approx = 2.0 * np.pi**2 * R_pert * a_pert**2 * scenario.kappa
        # Volume-average pressure proxy: P_heat * tau_E / V  (MW·s / m^3 = MJ/m^3)
        # beta_t = 2 mu0 * <p> / B^2;  mu0 = 4pi*1e-7;  1 MJ/m^3 = 1e6 Pa
        p_avg = scenario.P_heat * tau * 1e6 / V_approx  # Pa
        mu0 = 4.0 * np.pi * 1e-7
        beta_t = 2.0 * mu0 * p_avg / (scenario.B_t**2)  # dimensionless
        beta_t_pct = beta_t * 100.0
        I_N = scenario.I_p / (a_pert * scenario.B_t)  # normalised current (MA / (m·T))
        beta_N = beta_t_pct / I_N if I_N > 1e-6 else 0.0

        tau_samples[i] = tau
        pfus_samples[i] = pfus
        q_samples[i] = q
        beta_n_samples[i] = beta_N

    band_pcts = [5, 50, 95]
    full_pcts = [5, 25, 50, 75, 95]

    return FullChainUQResult(
        tau_E=float(np.median(tau_samples)),
        P_fusion=float(np.median(pfus_samples)),
        Q=float(np.median(q_samples)),
        tau_E_sigma=float(np.std(tau_samples)),
        P_fusion_sigma=float(np.std(pfus_samples)),
        Q_sigma=float(np.std(q_samples)),
        psi_nrmse_bands=np.asarray(
            np.percentile(psi_nrmse_samples, band_pcts),
            dtype=float,
        ),
        tau_E_bands=np.asarray(np.percentile(tau_samples, band_pcts), dtype=float),
        P_fusion_bands=np.asarray(
            np.percentile(pfus_samples, band_pcts),
            dtype=float,
        ),
        Q_bands=np.asarray(np.percentile(q_samples, band_pcts), dtype=float),
        beta_N_bands=np.asarray(np.percentile(beta_n_samples, band_pcts), dtype=float),
        tau_E_percentiles=np.asarray(np.percentile(tau_samples, full_pcts), dtype=float),
        P_fusion_percentiles=np.asarray(
            np.percentile(pfus_samples, full_pcts),
            dtype=float,
        ),
        Q_percentiles=np.asarray(np.percentile(q_samples, full_pcts), dtype=float),
        n_samples=n_samples,
    )


def summarize_uq(result: FullChainUQResult) -> dict:
    """
    Pretty-print a FullChainUQResult as a plain dict suitable for
    ``json.dumps()``.

    All numpy arrays are converted to Python lists; floats are rounded
    to 6 significant figures.

    Parameters
    ----------
    result : FullChainUQResult
        Output of :func:`quantify_full_chain`.

    Returns
    -------
    dict — JSON-serialisable summary.
    """

    def _round(x: float, sig: int = 6) -> float:
        return float(f"{x:.{sig}g}")

    def _arr(a: np.ndarray) -> list:
        return [_round(float(v)) for v in a]

    return {
        "central": {
            "tau_E_s": _round(result.tau_E),
            "P_fusion_MW": _round(result.P_fusion),
            "Q": _round(result.Q),
        },
        "sigma": {
            "tau_E_s": _round(result.tau_E_sigma),
            "P_fusion_MW": _round(result.P_fusion_sigma),
            "Q": _round(result.Q_sigma),
        },
        "bands_5_50_95": {
            "psi_nrmse": _arr(result.psi_nrmse_bands),
            "tau_E_s": _arr(result.tau_E_bands),
            "P_fusion_MW": _arr(result.P_fusion_bands),
            "Q": _arr(result.Q_bands),
            "beta_N": _arr(result.beta_N_bands),
        },
        "n_samples": result.n_samples,
    }


def quantify_uncertainty(scenario: PlasmaScenario, n_samples: int = 10000, seed: int | None = None) -> UQResult:
    """
    Monte Carlo uncertainty quantification for fusion performance.

    Samples scaling-law coefficients from their Gaussian posteriors and
    propagates through the confinement and fusion power models.

    Parameters
    ----------
    scenario : PlasmaScenario
        Plasma parameters (held fixed).
    n_samples : int
        Number of Monte Carlo samples (default 10,000).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    UQResult — central estimates + error bars + percentiles.
    """
    n_samples = _validate_n_samples(n_samples)
    rng = np.random.default_rng(seed)

    tau_samples = np.zeros(n_samples)
    pfus_samples = np.zeros(n_samples)
    q_samples = np.zeros(n_samples)

    for i in range(n_samples):
        # Sample scaling law parameters
        params = {}
        for key in IPB98_CENTRAL:
            params[key] = rng.normal(IPB98_CENTRAL[key], IPB98_SIGMA[key])

        # Ensure physical constraints
        params["C"] = max(params["C"], 1e-4)
        params["alpha_P"] = min(params["alpha_P"], -0.1)  # must be negative

        tau = ipb98_tau_e(scenario, params)
        tau = max(tau, 1e-6)  # floor

        pfus = fusion_power_from_tau(scenario, tau)
        pfus = max(pfus, 0.0)

        q = pfus / scenario.P_heat if scenario.P_heat > 0 else 0.0

        tau_samples[i] = tau
        pfus_samples[i] = pfus
        q_samples[i] = q

    pcts = [5, 25, 50, 75, 95]

    return UQResult(
        tau_E=float(np.median(tau_samples)),
        P_fusion=float(np.median(pfus_samples)),
        Q=float(np.median(q_samples)),
        tau_E_sigma=float(np.std(tau_samples)),
        P_fusion_sigma=float(np.std(pfus_samples)),
        Q_sigma=float(np.std(q_samples)),
        tau_E_percentiles=np.asarray(np.percentile(tau_samples, pcts), dtype=float),
        P_fusion_percentiles=np.asarray(
            np.percentile(pfus_samples, pcts),
            dtype=float,
        ),
        Q_percentiles=np.asarray(np.percentile(q_samples, pcts), dtype=float),
        n_samples=n_samples,
    )
