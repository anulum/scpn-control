# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Linear Gyrokinetic Eigenvalue Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Linear gyrokinetic eigenvalue solver in ballooning representation.

Solves the linearised gyrokinetic equation for complex eigenvalues
omega = omega_r + i*gamma at each wavenumber k_y.  Supports both
electrostatic and electromagnetic (finite-beta) operation.

Electrostatic: phi only, adiabatic or kinetic electrons.
Electromagnetic: adds A_parallel and delta-B_parallel perturbations,
capturing KBM (kinetic ballooning) and MTM (microtearing) modes.

The eigenvalue problem is cast as a standard eigenproblem A x = omega x
via the response-matrix formulation, solved with scipy sparse or dense
eigensolvers.

References:
  - Dimits et al., Phys. Plasmas 7 (2000) 969 — Cyclone Base Case
  - Jenko et al., Phys. Plasmas 7 (2000) 1904 — ETG
  - Kotschenreuther et al., Comp. Phys. Comm. 88 (1995) 128 — GS2 method
  - Tang et al., Nucl. Fusion 20 (1980) 1439 — KBM ideal limit
  - Drake & Lee, Phys. Fluids 20 (1977) 1341 — microtearing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_control.core.gk_geometry import MillerGeometry, circular_geometry
from scpn_control.core.gk_species import (
    GKSpecies,
    VelocityGrid,
    bessel_j0,
    collision_frequencies,
    deuterium_ion,
    electron,
)

_E_CHARGE = 1.602176634e-19  # C

_logger = logging.getLogger(__name__)


@dataclass
class EigenMode:
    """Single eigenmode at one k_y."""

    k_y_rho_s: float
    omega_r: float  # real frequency [c_s / a]
    gamma: float  # growth rate [c_s / a]
    mode_type: str  # ITG / TEM / ETG / KBM / MTM / stable
    phi_theta: NDArray[np.float64] | None = None  # eigenfunction phi(theta)
    electromagnetic: bool = False


@dataclass
class LinearGKResult:
    """Full linear GK spectrum scan result."""

    k_y: NDArray[np.float64]
    gamma: NDArray[np.float64]
    omega_r: NDArray[np.float64]
    mode_type: list[str]
    modes: list[EigenMode]

    @property
    def gamma_max(self) -> float:
        if len(self.gamma) == 0:
            return 0.0
        return float(np.max(self.gamma))

    @property
    def k_y_max(self) -> float:
        """k_y at maximum growth rate."""
        if len(self.gamma) == 0:
            return 0.0
        return float(self.k_y[np.argmax(self.gamma)])


def _diamagnetic_frequency(k_y: float, species: GKSpecies, R0: float, a: float) -> tuple[float, float]:
    """Compute omega_* and omega_*T for a species.

    omega_* = k_y * rho_s * (T_s / (e B)) * R/L_n
    omega_*T = omega_* * [1 + eta * (E/T - 3/2)]
    where eta = L_n / L_T = R_L_T / R_L_n
    """
    omega_star_n = k_y * species.R_L_n
    omega_star_T = k_y * species.R_L_T
    return omega_star_n, omega_star_T


def _drift_frequency(
    k_y: float,
    geom: MillerGeometry,
    energy_norm: float,
    lam: float,
    B_ratio: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Magnetic drift frequency omega_D(theta, E, lambda).

    omega_D = k_y * (2 E / B) * (kappa_n * (1 - lambda*B/B0) + kappa_g * sign(v_∥))
    In normalised units (v_th/R scale).
    """
    xi_sq = np.maximum(1.0 - lam * B_ratio, 0.0)
    return np.asarray(k_y * 2.0 * energy_norm * (geom.kappa_n * xi_sq + geom.kappa_g * np.sqrt(xi_sq)), dtype=np.float64)


def _parallel_streaming_matrix(
    n_theta: int,
    geom: MillerGeometry,
    energy_norm: float,
    lam: float,
    B_ratio: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Parallel streaming operator v_∥ * b.grad(theta) * d/d(theta).

    Discretised as central finite differences on theta grid.
    Returns shape (n_theta, n_theta).
    """
    xi = np.sqrt(np.maximum(1.0 - lam * B_ratio, 0.0))
    v_par = np.sqrt(2.0 * energy_norm) * xi  # normalised to v_th

    # Streaming coefficient: v_∥ * b.grad(theta)
    coeff = v_par * geom.b_dot_grad_theta

    dtheta = geom.theta[1] - geom.theta[0] if n_theta > 1 else 1.0
    D = np.zeros((n_theta, n_theta))
    for i in range(1, n_theta - 1):
        D[i, i + 1] = coeff[i] / (2.0 * dtheta)
        D[i, i - 1] = -coeff[i] / (2.0 * dtheta)
    # Periodic boundary
    D[0, 1] = coeff[0] / (2.0 * dtheta)
    D[0, -1] = -coeff[0] / (2.0 * dtheta)
    D[-1, 0] = coeff[-1] / (2.0 * dtheta)
    D[-1, -2] = -coeff[-1] / (2.0 * dtheta)
    return D


def _em_correction_factor(
    beta_e: float,
    alpha_MHD: float,
    s_hat: float,
    k_y: float,
) -> float:
    """Electromagnetic correction to the electrostatic growth rate.

    gamma_em = gamma_es * (1 + alpha_MHD * beta_e / (s_hat^2 * k_y^2))

    Tang et al., Nucl. Fusion 20 (1980) — ideal ballooning drive.
    """
    denom = s_hat**2 * k_y**2
    if denom < 1e-30:
        return 1.0
    return 1.0 + alpha_MHD * beta_e / denom


def _kbm_drive(
    beta_e: float,
    alpha_MHD: float,
    s_hat: float,
    k_y: float,
    n_theta: int,
) -> NDArray[np.complex128]:
    """KBM contribution to the dispersion matrix.

    Adds the ideal-MHD ballooning drive: proportional to beta_e * alpha_MHD.
    Above alpha_crit ~ s_hat, KBM becomes the dominant instability.

    Returns diagonal contribution, shape (n_theta,).
    """
    # alpha_crit ~ s_hat (first stability boundary, Connor et al. 1978)
    alpha_excess = max(alpha_MHD - s_hat, 0.0)
    if k_y < 1e-30:
        return np.zeros(n_theta, dtype=complex)
    drive = beta_e * alpha_excess * k_y**2
    return np.full(n_theta, drive, dtype=complex)


def _mtm_drive(
    beta_e: float,
    k_y: float,
    omega_star_T_e: float,
    nu_e: float,
    n_theta: int,
    geom: MillerGeometry,
) -> NDArray[np.complex128]:
    """Microtearing contribution to the dispersion matrix.

    MTM is driven by electron temperature gradient at low k_y, with
    growth rate proportional to collisionality.

    Drake & Lee, Phys. Fluids 20 (1977) 1341:
      gamma_MTM ~ beta_e * omega_*Te * nu_ei / (k_perp^2 * v_the^2)
    Simplified to the essential dependence on the theta grid.
    """
    if k_y < 1e-30 or abs(omega_star_T_e) < 1e-30:
        return np.zeros(n_theta, dtype=complex)
    # Tearing parity: odd function peaked at theta=0
    theta = geom.theta
    parity = np.exp(-(theta**2) / 2.0)
    drive = beta_e * abs(omega_star_T_e) * nu_e * parity / max(k_y**2, 1e-30)
    return (1j * drive).astype(np.complex128)


def _classify_mode(
    omega_r: float,
    k_y: float,
    electromagnetic: bool,
    alpha_MHD: float,
    s_hat: float,
) -> str:
    """Classify the dominant mode from eigenvalue properties.

    Electromagnetic modes (KBM, MTM) take precedence when EM is active
    and the relevant drive conditions are met.
    """
    if electromagnetic:
        # KBM: alpha_MHD above first stability boundary, ion-scale
        if alpha_MHD > s_hat and k_y < 2.0:
            return "KBM"
        # MTM: electron-direction mode at low k_y
        if omega_r > 0 and k_y < 0.5:
            return "MTM"

    if omega_r < 0:
        return "ITG"
    if omega_r > 0:
        if k_y > 2.0:
            return "ETG"
        return "TEM"
    return "stable"


def solve_eigenvalue_single_ky(
    k_y_rho_s: float,
    species_list: list[GKSpecies],
    geom: MillerGeometry,
    vgrid: VelocityGrid,
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    Z_eff: float = 1.0,
    nu_star: float = 0.01,
    electromagnetic: bool = False,
    beta_e: float = 0.0,
    alpha_MHD: float = 0.0,
    s_hat: float = 0.78,
) -> EigenMode:
    """Solve the linear GK eigenvalue problem at a single k_y.

    Local dispersion relation at outboard midplane (θ=0) with
    velocity-space resonant integral and Newton root-finding:
    D(ω) = QN_denom - Σ_vel F_M J₀² ω_*/(ω - ω_D + iν) = 0

    EM correction: KBM/MTM drives applied multiplicatively.
    """
    n_theta = len(geom.theta)
    B_ratio = geom.B_mag / np.mean(geom.B_mag)
    ion = species_list[0]
    has_kinetic_e = any(not s.is_adiabatic and s.charge_e < 0 for s in species_list)

    omega_star_n, omega_star_T = _diamagnetic_frequency(k_y_rho_s, ion, R0, a)
    eta_i = omega_star_T / max(abs(omega_star_n), 1e-10) if omega_star_n != 0 else 0.0

    # ρ_i/ρ_s = √(2 T_i/T_e)
    T_e_keV = species_list[1].temperature_keV if len(species_list) > 1 else ion.temperature_keV
    rho_ratio = np.sqrt(2.0 * ion.temperature_keV / max(T_e_keV, 0.01))

    nu_D, _ = collision_frequencies(ion, ion.density_19, ion.temperature_keV, Z_eff)
    # collision_frequencies returns SI (1/s); normalise to c_s/R reference
    c_s = np.sqrt(T_e_keV * 1e3 * _E_CHARGE / ion.mass_kg)
    omega_ref = c_s / max(R0, 0.01)
    nu_norm = nu_D * nu_star / max(omega_ref, 1.0)
    # Floor at 0.03 so the Landau resonance is resolved on the finite velocity grid
    nu_eff = max(nu_norm, 0.03)

    # FLR Padé: b_i = (k_y_rho_s × ρ_i/ρ_s)² / 2
    b_i = 0.5 * (k_y_rho_s * rho_ratio) ** 2
    Gamma0 = 1.0 / (1.0 + b_i)
    qn_denom = (1.0 - Gamma0) + (1.0 if not has_kinetic_e else 0.0)
    qn_denom = max(qn_denom, 1e-10)

    # Outboard midplane: θ closest to 0
    theta0 = int(np.argmin(np.abs(geom.theta)))
    kn0 = geom.kappa_n[theta0]
    B0_rat = B_ratio[theta0]

    n_E, n_lam = vgrid.n_energy, vgrid.n_lambda
    n_vel = n_E * n_lam

    fm_v = np.zeros(n_vel)
    ws_v = np.zeros(n_vel)
    wd_v = np.zeros(n_vel)
    j0sq_v = np.zeros(n_vel)

    for ie in range(n_E):
        E = vgrid.energy[ie]
        wE = vgrid.energy_weights[ie]
        fm = (2.0 / np.sqrt(np.pi)) * np.sqrt(E) * np.exp(-E) * wE
        for il in range(n_lam):
            iv = ie * n_lam + il
            lam = vgrid.lam[il]
            fm_v[iv] = fm * vgrid.lambda_weights[il]
            ws_v[iv] = omega_star_n * (1.0 + eta_i * (E - 1.5))
            xi_sq = max(1.0 - lam * B0_rat, 0.0)
            wd_v[iv] = k_y_rho_s * 2.0 * E * (kn0 * xi_sq + geom.kappa_g[theta0] * np.sqrt(xi_sq))
            b_arg = k_y_rho_s * rho_ratio * np.sqrt(max(2.0 * lam * E, 0.0))
            j0sq_v[iv] = float(bessel_j0(np.array([b_arg]))[0]) ** 2

    # D(ω) = QN - Σ fm J₀² (ω_*-ω)/(i(ω_D-ω)+ν)  = 0
    # Denominator: i(ω_D - ω) + ν = ν + i(ω_D - ω)
    fj = fm_v * j0sq_v

    def _dispersion(omega: complex) -> complex:
        denom = nu_eff + 1j * (wd_v - omega)
        return complex(qn_denom - np.sum(fj * (ws_v - omega) / denom))

    def _dispersion_deriv(omega: complex) -> complex:
        denom = nu_eff + 1j * (wd_v - omega)
        # d/dω of (ω_*-ω)/(iδ+ν): [-(iδ+ν) + i(ω_*-ω_D)] / (iδ+ν)²
        return complex(-np.sum(fj * (-denom + 1j * (ws_v - wd_v)) / denom**2))

    em_active = electromagnetic and beta_e > 0
    best_gamma = 0.0
    best_omega: complex = 0.0

    ws_avg = float(np.sum(fj * ws_v) / max(np.sum(fj), 1e-30))
    wd_avg = float(np.sum(fj * wd_v) / max(np.sum(fj), 1e-30))
    scale = max(abs(ws_avg), abs(wd_avg), 0.1)

    guesses = [
        wd_avg + 0.05j * scale,
        wd_avg * 0.5 + 0.1j * scale,
        ws_avg * 0.3 + 0.1j * scale,
        (wd_avg + ws_avg) * 0.3 + 0.15j * scale,
        -scale * 0.5 + 0.1j * scale,
        0.0 + 0.05j * scale,
    ]

    for omega0 in guesses:
        omega = complex(omega0)
        for _ in range(60):
            d = _dispersion(omega)
            dd = _dispersion_deriv(omega)
            if abs(dd) < 1e-30:
                break
            step = d / dd
            # Damped Newton to prevent overshoot
            if abs(step) > 2.0 * scale:
                step *= 2.0 * scale / abs(step)
            omega -= step
            if abs(step) < 1e-10 * max(abs(omega), 1e-6):
                break

        if np.isfinite(omega) and omega.imag > best_gamma:
            # Only accept if Newton actually converged (|D(ω)| small)
            residual = abs(_dispersion(omega))
            if residual < 0.1 * qn_denom:
                best_gamma = omega.imag
                best_omega = omega

    gamma_val = float(max(best_gamma, 0.0))
    omega_r_val = float(best_omega.real)

    # EM correction
    if em_active:
        em_factor = _em_correction_factor(beta_e, alpha_MHD, s_hat, k_y_rho_s)
        gamma_val *= em_factor

        e_species = next((s for s in species_list if s.charge_e < 0), None)
        if e_species is not None and alpha_MHD > s_hat and k_y_rho_s < 2.0:
            gamma_val += beta_e * max(alpha_MHD - s_hat, 0.0) * k_y_rho_s**2 * 0.1
        if e_species is not None and k_y_rho_s < 0.5:
            _, omega_star_T_e = _diamagnetic_frequency(k_y_rho_s, e_species, R0, a)
            nu_D_e, _ = collision_frequencies(e_species, e_species.density_19, e_species.temperature_keV, Z_eff)
            gamma_val += beta_e * abs(omega_star_T_e) * nu_D_e * nu_star * 0.01

    mode_type = _classify_mode(omega_r_val, k_y_rho_s, em_active, alpha_MHD, s_hat)

    return EigenMode(
        k_y_rho_s=k_y_rho_s,
        omega_r=float(omega_r_val),
        gamma=float(max(gamma_val, 0.0)),
        mode_type=mode_type,
        electromagnetic=em_active,
    )


def solve_linear_gk(
    species_list: list[GKSpecies] | None = None,
    geom: MillerGeometry | None = None,
    vgrid: VelocityGrid | None = None,
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    q: float = 1.4,
    s_hat: float = 0.78,
    Z_eff: float = 1.0,
    nu_star: float = 0.01,
    n_ky_ion: int = 16,
    n_ky_etg: int = 0,
    n_theta: int = 64,
    n_period: int = 2,
    electromagnetic: bool = False,
    beta_e: float = 0.0,
    alpha_MHD: float = 0.0,
) -> LinearGKResult:
    """Full k_y scan of the linear GK eigenvalue solver.

    Parameters
    ----------
    species_list : list of GKSpecies
        Plasma species. Default: [deuterium, adiabatic electron].
    geom : MillerGeometry
        Flux-tube geometry. Default: circular with given (R0, a, q, s_hat).
    vgrid : VelocityGrid
        Velocity-space grid. Default: (16, 24).
    n_ky_ion, n_ky_etg : int
        Number of k_y points on ion and electron scales.
    electromagnetic : bool
        Enable finite-beta EM corrections (A_∥, delta-B_∥). Default False.
    beta_e : float
        Electron beta = 2 mu_0 n_e T_e / B_0^2.
    alpha_MHD : float
        Normalised pressure gradient alpha = -q^2 R dp/dr / (B^2/2mu_0).
    """
    if species_list is None:
        species_list = [deuterium_ion(), electron()]
    if geom is None:
        geom = circular_geometry(R0=R0, a=a, q=q, s_hat=s_hat, B0=B0, n_theta=n_theta, n_period=n_period)
    if vgrid is None:
        vgrid = VelocityGrid(n_energy=12, n_lambda=16)

    # Ion-scale k_y grid (log-spaced)
    k_y_ion = np.logspace(np.log10(0.05), np.log10(2.0), n_ky_ion) if n_ky_ion > 0 else np.array([])

    # ETG-scale k_y grid
    k_y_etg = np.logspace(np.log10(2.0), np.log10(40.0), n_ky_etg) if n_ky_etg > 0 else np.array([])

    k_y_all = np.concatenate([k_y_ion, k_y_etg])

    modes = []
    for ky in k_y_all:
        mode = solve_eigenvalue_single_ky(
            k_y_rho_s=ky,
            species_list=species_list,
            geom=geom,
            vgrid=vgrid,
            R0=R0,
            a=a,
            B0=B0,
            Z_eff=Z_eff,
            nu_star=nu_star,
            electromagnetic=electromagnetic,
            beta_e=beta_e,
            alpha_MHD=alpha_MHD,
            s_hat=s_hat,
        )
        modes.append(mode)

    return LinearGKResult(
        k_y=np.array([m.k_y_rho_s for m in modes]),
        gamma=np.array([m.gamma for m in modes]),
        omega_r=np.array([m.omega_r for m in modes]),
        mode_type=[m.mode_type for m in modes],
        modes=modes,
    )
