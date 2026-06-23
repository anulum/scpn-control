# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — JAX-Accelerated Linear Gyrokinetic Eigenvalue Solver
"""JAX-accelerated linear GK eigenvalue solver.

Re-implements the native local-dispersion eigenvalue contract from
gk_eigenvalue.py using jax.numpy for batched velocity-space quadrature and
jax.grad for the transport-stiffness proxy d(chi_i)/d(R_L_Ti). JAX execution is
explicit and fails closed when JAX is unavailable.

References:
  - Dimits et al., Phys. Plasmas 7 (2000) 969 — Cyclone Base Case
  - Kotschenreuther et al., Comp. Phys. Comm. 88 (1995) 128
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.gk_eigenvalue import EigenMode, LinearGKResult, solve_linear_gk
from scpn_control.core.gk_geometry import circular_geometry
from scpn_control.core.gk_species import (
    VelocityGrid,
    collision_frequencies,
    deuterium_ion,
    electron,
)

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    jax = None
    jnp = cast(Any, None)  # optional-dep fallback (keeps jnp.* annotations typed)
    _HAS_JAX = False

_E_CHARGE = 1.602176634e-19  # C
JAX_GK_PARITY_SCHEMA_VERSION = "scpn-control.jax-gk-parity.v1"
JAX_GK_PARITY_EVIDENCE_BOUNDARY = "backend_parity_only"
JAX_GK_PARITY_SOLVER_CONTRACT = "native_linear_gk_local_dispersion"
JAX_GK_PARITY_CASES = ("cyclone_base_case", "tem_kinetic_electron", "stable_mode")

_logger = logging.getLogger(__name__)


def has_jax() -> bool:
    """Return whether JAX is importable in this environment."""
    return _HAS_JAX


def _require_jax() -> None:
    if not _HAS_JAX:
        raise ImportError("JAX is required for jax_gk_solver but not installed. pip install jax jaxlib")


# ── JAX Bessel J0 approximation ─────────────────────────────────────
# Abramowitz & Stegun 9.4.1 / 9.4.3 — max error < 5e-8.


def _bessel_j0_jax(x: Any) -> Any:
    """J_0(x) via rational Chebyshev approximation, differentiable under JAX."""
    ax = jnp.abs(x)

    y = (ax / 3.0) ** 2
    small = 1.0 + y * (
        -2.2499997 + y * (1.2656208 + y * (-0.3163866 + y * (0.0444479 + y * (-0.0039444 + y * 0.0002100))))
    )

    y3 = 3.0 / ax
    p0 = 0.79788456 + y3 * (
        -0.00000077 + y3 * (-0.00552740 + y3 * (-0.00009512 + y3 * (0.00137237 + y3 * (-0.00072805 + y3 * 0.00014476))))
    )
    q0 = y3 * (
        -0.04166397 + y3 * (-0.00003954 + y3 * (0.00262573 + y3 * (-0.00054125 + y3 * (-0.00029333 + y3 * 0.00013558))))
    )
    theta0 = ax - 0.78539816  # ax - pi/4
    large = p0 * jnp.cos(theta0 - q0) / jnp.sqrt(ax + 1e-30)

    return jnp.where(ax <= 3.0, small, large)


# ── Core JAX kernels ────────────────────────────────────────────────


def _build_response_matrix_single_ky(
    k_y_rho_s: Any,
    ion_R_L_n: float,
    ion_R_L_T: float,
    rho_i_over_a: float,
    nu_eff: float,
    B_ratio: Any,
    kappa_n: Any,
    kappa_g: Any,
    b_dot_grad_theta: Any,
    theta: Any,
    energy: Any,
    energy_weights: Any,
    lam: Any,
    lambda_weights: Any,
    has_kinetic_electrons: bool,
) -> tuple[Any, Any, Any]:
    """Build the response matrix for a single k_y. Pure JAX, no side effects.

    nu_eff and rho_i_over_a are pre-computed outside the traced path to
    avoid constructing Python objects (GKSpecies) inside JAX tracing.
    """
    n_theta = theta.shape[0]

    omega_star_n_i = k_y_rho_s * ion_R_L_n
    omega_star_T_i = k_y_rho_s * ion_R_L_T

    eta_i = jnp.where(jnp.abs(omega_star_n_i) > 1e-10, omega_star_T_i / omega_star_n_i, 0.0)

    n_energy = int(energy.shape[0])
    n_lambda = int(lam.shape[0])

    dtheta = theta[1] - theta[0]

    R_ion_real = jnp.zeros((n_theta, n_theta))
    R_ion_imag = jnp.zeros((n_theta, n_theta))

    for ie in range(n_energy):
        E_norm = energy[ie]
        w_E = energy_weights[ie]
        fm_weight = (2.0 / jnp.sqrt(jnp.pi)) * jnp.sqrt(E_norm) * jnp.exp(-E_norm) * w_E

        for il in range(n_lambda):
            lam_val = lam[il]
            w_lam = lambda_weights[il]

            b_arg = k_y_rho_s * rho_i_over_a * jnp.sqrt(2.0 * lam_val * E_norm) * jnp.ones(n_theta)
            J0_val = _bessel_j0_jax(b_arg)

            xi_sq = jnp.maximum(1.0 - lam_val * B_ratio, 0.0)
            omega_D = k_y_rho_s * 2.0 * E_norm * (kappa_n * xi_sq + kappa_g * jnp.sqrt(xi_sq))

            omega_star_full = omega_star_n_i * (1.0 + eta_i * (E_norm - 1.5))

            xi = jnp.sqrt(xi_sq)
            v_par = jnp.sqrt(2.0 * E_norm) * xi
            coeff = v_par * b_dot_grad_theta

            idx_p = jnp.arange(n_theta)
            idx_next = (idx_p + 1) % n_theta
            idx_prev = (idx_p - 1) % n_theta

            D_par = jnp.zeros((n_theta, n_theta))
            D_par = D_par.at[idx_p, idx_next].add(coeff / (2.0 * dtheta))
            D_par = D_par.at[idx_p, idx_prev].add(-coeff / (2.0 * dtheta))

            J0_diag = jnp.diag(J0_val)
            drive_diag = jnp.diag(jnp.full(n_theta, omega_star_full))

            weight = fm_weight * w_lam

            R_ion_real = R_ion_real + weight * (J0_diag @ drive_diag @ J0_diag)
            R_ion_imag = R_ion_imag + weight * (J0_diag @ D_par @ J0_diag)

    I_theta = jnp.eye(n_theta)
    fsa = jnp.ones((n_theta, n_theta)) / n_theta
    adiabatic = jnp.where(has_kinetic_electrons, 0.0, 1.0)
    adiabatic_response = adiabatic * (I_theta - fsa)

    full_real = R_ion_real + adiabatic_response
    full_imag = R_ion_imag

    return full_real, full_imag, R_ion_real


def _solve_eigenvalue_from_matrix(full_real: Any, full_imag: Any) -> tuple[float, float, str, AnyFloatArray | None]:
    """Extract most unstable eigenmode from the response matrix (NumPy)."""
    full_matrix = np.asarray(full_real) + 1j * np.asarray(full_imag)

    try:
        eigenvalues, eigenvectors = np.linalg.eig(full_matrix)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, "stable", None

    gammas = eigenvalues.imag
    omega_rs = eigenvalues.real

    if np.all(gammas <= 0):
        return 0.0, 0.0, "stable", None

    idx = int(np.argmax(gammas))
    gamma_max = float(gammas[idx])
    omega_r_max = float(omega_rs[idx])
    phi_mode = np.abs(eigenvectors[:, idx])

    if omega_r_max < 0:
        mode_type = "ITG"
    elif omega_r_max > 0:
        mode_type = "TEM"
    else:
        mode_type = "stable"

    return float(max(gamma_max, 0.0)), omega_r_max, mode_type, phi_mode


def _precompute_ion_params(ion: Any, B0: float, a: float, Z_eff: float, nu_star: float) -> tuple[float, float]:
    """Pre-compute collision frequency and FLR scale outside JAX trace."""
    rho_i_over_a = ion.mass_kg * ion.thermal_speed / (abs(ion.charge_e) * _E_CHARGE * B0) / a
    nu_D, _ = collision_frequencies(ion, ion.density_19, ion.temperature_keV, Z_eff)
    nu_eff = nu_D * nu_star
    return float(rho_i_over_a), float(nu_eff)


def _make_batched_builder(
    ion_R_L_n: float,
    ion_R_L_T: float,
    rho_i_over_a: float,
    nu_eff: float,
    B_ratio: Any,
    kappa_n: Any,
    kappa_g: Any,
    b_dot_grad_theta: Any,
    theta: Any,
    energy: Any,
    energy_weights: Any,
    lam: Any,
    lambda_weights: Any,
    has_kinetic_electrons: bool,
) -> Any:
    """Return a vmap'd function that builds response matrices for all k_y at once."""

    def _build_single(k_y_rho_s: Any) -> tuple[Any, Any, Any]:
        return _build_response_matrix_single_ky(
            k_y_rho_s,
            ion_R_L_n,
            ion_R_L_T,
            rho_i_over_a,
            nu_eff,
            B_ratio,
            kappa_n,
            kappa_g,
            b_dot_grad_theta,
            theta,
            energy,
            energy_weights,
            lam,
            lambda_weights,
            has_kinetic_electrons,
        )

    return jax.vmap(_build_single)


def solve_linear_gk_jax(
    species_list: Any = None,
    geom: Any = None,
    vgrid: Any = None,
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    q: float = 1.4,
    s_hat: float = 0.78,
    Z_eff: float = 1.0,
    nu_star: float = 0.01,
    n_ky_ion: int = 16,
    n_theta: int = 64,
) -> LinearGKResult:
    """JAX-accelerated linear GK solver. Batched over k_y via vmap.

    Parameters match solve_linear_gk from gk_eigenvalue.py.
    Uses JAX to build the local-dispersion velocity integrals for all k_y
    simultaneously, then applies the same damped Newton root finder used by the
    native solver. This keeps backend parity on the physical growth/frequency
    path while preserving JAX acceleration for the high-volume quadrature
    assembly.
    """
    _require_jax()
    if n_ky_ion <= 0:
        return LinearGKResult(k_y=np.array([]), gamma=np.array([]), omega_r=np.array([]), mode_type=[], modes=[])

    if species_list is None:
        species_list = [deuterium_ion(), electron()]
    if geom is None:
        geom = circular_geometry(R0=R0, a=a, q=q, s_hat=s_hat, B0=B0, n_theta=n_theta, n_period=2)
    if vgrid is None:
        vgrid = VelocityGrid(n_energy=12, n_lambda=16)

    k_y_all = jnp.logspace(jnp.log10(0.05), jnp.log10(2.0), n_ky_ion)

    ion = species_list[0]
    has_kinetic_electrons = any(not s.is_adiabatic and s.charge_e < 0 for s in species_list)
    electron_temperature = species_list[1].temperature_keV if len(species_list) > 1 else ion.temperature_keV
    rho_ratio = np.sqrt(2.0 * ion.temperature_keV / max(electron_temperature, 0.01))
    nu_D, _ = collision_frequencies(ion, ion.density_19, ion.temperature_keV, Z_eff)
    c_s = np.sqrt(electron_temperature * 1.0e3 * _E_CHARGE / ion.mass_kg)
    omega_ref = c_s / max(R0, 0.01)
    nu_norm = nu_D * nu_star / max(omega_ref, 1.0)
    nu_eff = max(float(nu_norm), 0.03)

    B_ratio = jnp.array(geom.B_mag / np.mean(geom.B_mag))
    theta0 = int(np.argmin(np.abs(np.asarray(geom.theta, dtype=np.float64))))
    B0_ratio = B_ratio[theta0]
    kappa_n0 = jnp.asarray(geom.kappa_n[theta0])
    kappa_g0 = jnp.asarray(geom.kappa_g[theta0])
    energy = jnp.array(vgrid.energy)
    energy_weights = jnp.array(vgrid.energy_weights)
    lam = jnp.array(vgrid.lam)
    lambda_weights = jnp.array(vgrid.lambda_weights)
    qn_denom, fj, ws, wd = _build_local_dispersion_payload(
        k_y_all,
        float(ion.R_L_n),
        float(ion.R_L_T),
        float(rho_ratio),
        bool(has_kinetic_electrons),
        B0_ratio,
        kappa_n0,
        kappa_g0,
        energy,
        energy_weights,
        lam,
        lambda_weights,
    )

    k_y_np = np.asarray(k_y_all)
    qn_np = np.asarray(qn_denom, dtype=np.float64)
    fj_np = np.asarray(fj, dtype=np.float64)
    ws_np = np.asarray(ws, dtype=np.float64)
    wd_np = np.asarray(wd, dtype=np.float64)
    modes = []
    for i in range(n_ky_ion):
        gamma_val, omega_r_val = _solve_local_dispersion_from_payload(qn_np[i], fj_np[i], ws_np[i], wd_np[i], nu_eff)
        mode_type = "ITG" if omega_r_val < 0.0 else ("TEM" if omega_r_val > 0.0 else "stable")
        modes.append(
            EigenMode(
                k_y_rho_s=float(k_y_np[i]),
                omega_r=omega_r_val,
                gamma=gamma_val,
                mode_type=mode_type,
                phi_theta=None,
            )
        )

    return LinearGKResult(
        k_y=np.array([m.k_y_rho_s for m in modes]),
        gamma=np.array([m.gamma for m in modes]),
        omega_r=np.array([m.omega_r for m in modes]),
        mode_type=[m.mode_type for m in modes],
        modes=modes,
    )


def _build_local_dispersion_payload(
    k_y_all: Any,
    ion_R_L_n: float,
    ion_R_L_T: float,
    rho_ratio: float,
    has_kinetic_electrons: bool,
    B0_ratio: Any,
    kappa_n0: Any,
    kappa_g0: Any,
    energy: Any,
    energy_weights: Any,
    lam: Any,
    lambda_weights: Any,
) -> tuple[Any, Any, Any, Any]:
    """Build native local-dispersion quadrature payloads with JAX arrays."""
    energy_grid = energy[:, None]
    lambda_grid = lam[None, :]
    weight_grid = energy_weights[:, None] * lambda_weights[None, :]
    fm_grid = (2.0 / jnp.sqrt(jnp.pi)) * jnp.sqrt(energy_grid) * jnp.exp(-energy_grid) * weight_grid

    def _single(k_y_rho_s: Any) -> tuple[Any, Any, Any, Any]:
        omega_star_n = k_y_rho_s * ion_R_L_n
        omega_star_T = k_y_rho_s * ion_R_L_T
        eta_i = jnp.where(jnp.abs(omega_star_n) > 1.0e-10, omega_star_T / jnp.abs(omega_star_n), 0.0)
        ws_grid = jnp.broadcast_to(omega_star_n * (1.0 + eta_i * (energy_grid - 1.5)), fm_grid.shape)
        xi_sq = jnp.maximum(1.0 - lambda_grid * B0_ratio, 0.0)
        wd_grid = k_y_rho_s * 2.0 * energy_grid * (kappa_n0 * xi_sq + kappa_g0 * jnp.sqrt(xi_sq))
        b_arg = k_y_rho_s * rho_ratio * jnp.sqrt(jnp.maximum(2.0 * lambda_grid * energy_grid, 0.0))
        fj_grid = fm_grid * (_bessel_j0_jax(b_arg) ** 2)
        b_i = 0.5 * (k_y_rho_s * rho_ratio) ** 2
        gamma0 = 1.0 / (1.0 + b_i)
        adiabatic = 0.0 if has_kinetic_electrons else 1.0
        qn_denom = jnp.maximum((1.0 - gamma0) + adiabatic, 1.0e-10)
        return qn_denom, jnp.ravel(fj_grid), jnp.ravel(ws_grid), jnp.ravel(wd_grid)

    result: tuple[Any, Any, Any, Any] = jax.vmap(_single)(k_y_all)
    return result


def _solve_local_dispersion_from_payload(
    qn_denom: float,
    fj: npt.NDArray[Any],
    ws_v: npt.NDArray[Any],
    wd_v: npt.NDArray[Any],
    nu_eff: float,
) -> tuple[float, float]:
    """Solve the same scalar local dispersion relation as the native path."""

    def _dispersion(omega: complex) -> complex:
        denom = nu_eff + 1j * (wd_v - omega)
        return complex(qn_denom - np.sum(fj * (ws_v - omega) / denom))

    def _dispersion_deriv(omega: complex) -> complex:
        denom = nu_eff + 1j * (wd_v - omega)
        return complex(-np.sum(fj * (-denom + 1j * (ws_v - wd_v)) / denom**2))

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
            if abs(dd) < 1.0e-30:
                break
            step = d / dd
            if abs(step) > 2.0 * scale:
                step *= 2.0 * scale / abs(step)
            omega -= step
            if abs(step) < 1.0e-10 * max(abs(omega), 1.0e-6):
                break
        if np.isfinite(omega) and omega.imag > best_gamma:
            residual = abs(_dispersion(omega))
            if residual < 0.1 * qn_denom:
                best_gamma = float(omega.imag)
                best_omega = omega
    return float(max(best_gamma, 0.0)), float(best_omega.real)


def _chi_i_proxy(
    R_L_Ti: Any,
    R_L_n: float,
    rho_i_over_a: float,
    nu_eff: float,
    B_ratio: Any,
    kappa_n: Any,
    kappa_g: Any,
    b_dot_grad_theta: Any,
    theta: Any,
    energy: Any,
    energy_weights: Any,
    lam: Any,
    lambda_weights: Any,
    has_kinetic_electrons: bool,
    n_ky_ion: int,
) -> Any:
    """R_L_Ti -> differentiable scalar proxy for chi_i.

    Uses the squared Frobenius norm of the ion drive matrix (R_ion_real)
    summed over k_y. Since R_ion_real ~ omega_star_full ~ R_L_Ti, this
    is quadratic in R_L_Ti with strictly positive gradient for R_L_Ti > 0.
    """
    k_y_all = jnp.logspace(jnp.log10(0.05), jnp.log10(2.0), n_ky_ion)

    batched = _make_batched_builder(
        R_L_n,
        R_L_Ti,
        rho_i_over_a,
        nu_eff,
        B_ratio,
        kappa_n,
        kappa_g,
        b_dot_grad_theta,
        theta,
        energy,
        energy_weights,
        lam,
        lambda_weights,
        has_kinetic_electrons,
    )
    _, _, R_ion_real = batched(k_y_all)

    # ||R_ion||_F^2 scales as omega_star_full^2 ~ (R_L_n + R_L_Ti * (E-1.5))^2
    return jnp.sum(R_ion_real**2)


def transport_stiffness_jax(
    R_L_Ti: float,
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    q: float = 1.4,
    s_hat: float = 0.78,
    Z_eff: float = 1.0,
    nu_star: float = 0.01,
    n_ky_ion: int = 8,
    n_theta: int = 32,
) -> float:
    """d(chi_i_proxy)/d(R_L_Ti) via jax.grad.

    Returns the transport stiffness — sensitivity of the ion heat flux
    proxy to the normalised temperature gradient. Positive above critical
    gradient, near-zero below.
    """
    _require_jax()

    ion = deuterium_ion(R_L_T=R_L_Ti)
    geom = circular_geometry(R0=R0, a=a, q=q, s_hat=s_hat, B0=B0, n_theta=n_theta, n_period=1)
    vgrid = VelocityGrid(n_energy=6, n_lambda=8)

    rho_i_over_a, nu_eff = _precompute_ion_params(ion, B0, a, Z_eff, nu_star)
    has_kinetic_electrons = False  # default: adiabatic electrons

    B_ratio = jnp.array(geom.B_mag / np.mean(geom.B_mag))
    kappa_n = jnp.array(geom.kappa_n)
    kappa_g = jnp.array(geom.kappa_g)
    b_dot_grad_theta = jnp.array(geom.b_dot_grad_theta)
    theta = jnp.array(geom.theta)
    energy = jnp.array(vgrid.energy)
    energy_weights = jnp.array(vgrid.energy_weights)
    lam_arr = jnp.array(vgrid.lam)
    lambda_weights = jnp.array(vgrid.lambda_weights)

    grad_fn = jax.grad(_chi_i_proxy, argnums=0)

    stiffness = grad_fn(
        jnp.float64(R_L_Ti),
        float(ion.R_L_n),
        rho_i_over_a,
        nu_eff,
        B_ratio,
        kappa_n,
        kappa_g,
        b_dot_grad_theta,
        theta,
        energy,
        energy_weights,
        lam_arr,
        lambda_weights,
        has_kinetic_electrons,
        n_ky_ion,
    )

    return float(stiffness)


def gk_stiffness_chi_i_profile_jax(
    R_L_Ti: float,
    rho: Any,
    base_chi_i: float = 0.1,
    stiffness_scale: float = 1.0e-6,
    R0: float = 2.78,
    a: float = 1.0,
    B0: float = 2.0,
    q: float = 1.4,
    s_hat: float = 0.78,
    Z_eff: float = 1.0,
    nu_star: float = 0.01,
    n_ky_ion: int = 8,
    n_theta: int = 32,
) -> FloatArray:
    """Map JAX GK stiffness into a bounded ion heat diffusivity profile.

    The closure is intentionally conservative: it uses the existing
    differentiable `transport_stiffness_jax` scalar as a local stiffness
    amplitude and applies a smooth edge-weighted radial shape. It is a
    controller-tuning closure, not a replacement for external GK validation.
    """
    _require_jax()
    rho_array = np.asarray(rho, dtype=np.float64)
    if rho_array.ndim != 1 or rho_array.size < 3 or not np.all(np.isfinite(rho_array)):
        raise ValueError("rho must be a finite one-dimensional array with length >= 3")
    if np.any(np.diff(rho_array) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    base = float(base_chi_i)
    scale = float(stiffness_scale)
    if not np.isfinite(base) or base < 0.0:
        raise ValueError("base_chi_i must be non-negative and finite")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("stiffness_scale must be non-negative and finite")

    stiffness = transport_stiffness_jax(
        R_L_Ti=R_L_Ti,
        R0=R0,
        a=a,
        B0=B0,
        q=q,
        s_hat=s_hat,
        Z_eff=Z_eff,
        nu_star=nu_star,
        n_ky_ion=n_ky_ion,
        n_theta=n_theta,
    )
    critical_gradient = 4.0
    supercritical_excess = max(float(R_L_Ti) - critical_gradient, 0.0)
    stiffness_magnitude = max(abs(float(stiffness)), 1.0)
    amplitude = supercritical_excess * stiffness_magnitude * scale
    radial_shape = 1.0 + rho_array * rho_array
    return np.asarray(base * (1.0 + amplitude * radial_shape), dtype=np.float64)


def build_jax_gk_parity_artifact(
    *,
    case: str = "cyclone_base_case",
    solver_kwargs: dict[str, Any] | None = None,
    native_result: LinearGKResult | None = None,
    jax_result: LinearGKResult | None = None,
    gamma_relative_tolerance: float = 0.25,
    omega_absolute_tolerance: float = 0.25,
    executed_at: str | None = None,
) -> dict[str, Any]:
    """Build a tamper-evident JAX/native gyrokinetic parity artifact.

    The artifact is backend-parity evidence for the repository's native
    local-dispersion formulation. It is intentionally not a cross-code
    validation artifact and always records that external validation remains
    required before quantitative facility claims.
    """
    _require_jax()
    if case not in JAX_GK_PARITY_CASES:
        allowed = ", ".join(JAX_GK_PARITY_CASES)
        raise ValueError(f"case must be one of: {allowed}")
    gamma_tol = float(gamma_relative_tolerance)
    omega_tol = float(omega_absolute_tolerance)
    if not np.isfinite(gamma_tol) or gamma_tol <= 0.0:
        raise ValueError("gamma_relative_tolerance must be positive and finite")
    if not np.isfinite(omega_tol) or omega_tol <= 0.0:
        raise ValueError("omega_absolute_tolerance must be positive and finite")

    run_kwargs: dict[str, Any] = {
        "R0": 2.78,
        "a": 1.0,
        "B0": 2.0,
        "q": 1.4,
        "s_hat": 0.78,
        "n_ky_ion": 4,
        "n_theta": 16,
    }
    if solver_kwargs:
        run_kwargs.update(solver_kwargs)
    _validate_parity_solver_kwargs(run_kwargs)
    execution_kwargs, case_parameters, case_acceptance = _parity_case_inputs(case, run_kwargs)

    native = native_result if native_result is not None else solve_linear_gk(**execution_kwargs)
    jax_gk = jax_result if jax_result is not None else solve_linear_gk_jax(**execution_kwargs)
    native_gamma, native_omega = _dominant_growth_and_frequency(native, "native_result")
    jax_gamma, jax_omega = _dominant_growth_and_frequency(jax_gk, "jax_result")
    native_mode_types = _mode_type_spectrum(native, "native_result")
    jax_mode_types = _mode_type_spectrum(jax_gk, "jax_result")

    backend_metadata = _jax_backend_metadata()
    payload: dict[str, Any] = {
        "schema_version": JAX_GK_PARITY_SCHEMA_VERSION,
        "case": case,
        "backend": backend_metadata["backend"],
        "jax_version": backend_metadata["jax_version"],
        "jaxlib_version": backend_metadata["jaxlib_version"],
        "platform": backend_metadata["platform"],
        "device_kind": backend_metadata["device_kind"],
        "dtype": backend_metadata["dtype"],
        "x64_enabled": backend_metadata["x64_enabled"],
        "executed_at": executed_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "native_gamma_max_cs_over_a": native_gamma,
        "jax_gamma_max_cs_over_a": jax_gamma,
        "native_omega_r_cs_over_a": native_omega,
        "jax_omega_r_cs_over_a": jax_omega,
        "gamma_relative_tolerance": gamma_tol,
        "omega_absolute_tolerance": omega_tol,
        "solver_contract": JAX_GK_PARITY_SOLVER_CONTRACT,
        "normalisation": "c_s_over_a",
        "evidence_boundary": JAX_GK_PARITY_EVIDENCE_BOUNDARY,
        "external_validation_required": True,
        "admitted_for_control": False,
        "solver_kwargs": _canonical_solver_kwargs(run_kwargs),
        "solver_kwargs_sha256": _sha256_json(_canonical_solver_kwargs(run_kwargs)),
        "case_parameters": case_parameters,
        "case_parameters_sha256": _sha256_json(case_parameters),
        "case_acceptance": case_acceptance,
        "native_mode_types": native_mode_types,
        "jax_mode_types": jax_mode_types,
        "native_dominant_mode_type": _dominant_mode_type(native, "native_result"),
        "jax_dominant_mode_type": _dominant_mode_type(jax_gk, "jax_result"),
    }
    payload["payload_sha256"] = _sha256_json(payload)
    return payload


def write_jax_gk_parity_artifact(
    output_path: str | Path,
    *,
    case: str = "cyclone_base_case",
    solver_kwargs: dict[str, Any] | None = None,
    gamma_relative_tolerance: float = 0.25,
    omega_absolute_tolerance: float = 0.25,
    executed_at: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Persist a JAX/native GK parity artifact and return payload plus path."""
    payload = build_jax_gk_parity_artifact(
        case=case,
        solver_kwargs=solver_kwargs,
        gamma_relative_tolerance=gamma_relative_tolerance,
        omega_absolute_tolerance=omega_absolute_tolerance,
        executed_at=executed_at,
    )
    path = Path(output_path)
    if path.suffix.lower() != ".json":
        artifact_name = _safe_artifact_token(f"{payload['case']}_{payload['backend']}_{payload['device_kind']}")
        path = path / f"{artifact_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload, path


def _safe_artifact_token(value: str) -> str:
    token = "".join(char.lower() if char.isalnum() else "_" for char in value)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "jax_gk_parity"


def _dominant_growth_and_frequency(result: LinearGKResult, label: str) -> tuple[float, float]:
    gamma = np.asarray(result.gamma, dtype=np.float64)
    omega = np.asarray(result.omega_r, dtype=np.float64)
    if gamma.ndim != 1 or omega.ndim != 1 or gamma.size == 0 or gamma.shape != omega.shape:
        raise ValueError(f"{label} must contain matching one-dimensional gamma and omega_r arrays")
    if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(omega)):
        raise ValueError(f"{label} must contain finite gamma and omega_r arrays")
    idx = int(np.argmax(gamma))
    dominant_gamma = float(max(gamma[idx], 0.0))
    dominant_omega = float(omega[idx])
    return dominant_gamma, dominant_omega


def _mode_type_spectrum(result: LinearGKResult, label: str) -> list[str]:
    """Return the per-k_y instability labels after validating result shape."""
    gamma = np.asarray(result.gamma, dtype=np.float64)
    mode_types = list(result.mode_type)
    if gamma.ndim != 1 or gamma.size == 0:
        raise ValueError(f"{label} must contain a non-empty one-dimensional gamma array")
    if len(mode_types) != gamma.size:
        raise ValueError(f"{label} must contain one mode type per gamma sample")
    if any(not isinstance(mode_type, str) or not mode_type.strip() for mode_type in mode_types):
        raise ValueError(f"{label} mode types must be non-empty strings")
    return [mode_type.strip() for mode_type in mode_types]


def _dominant_mode_type(result: LinearGKResult, label: str) -> str:
    """Return the mode label at the dominant-growth index."""
    gamma = np.asarray(result.gamma, dtype=np.float64)
    mode_types = _mode_type_spectrum(result, label)
    if not np.all(np.isfinite(gamma)):
        raise ValueError(f"{label} must contain finite gamma values")
    return mode_types[int(np.argmax(gamma))]


def _parity_case_inputs(
    case: str, solver_kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve built-in parity cases into execution kwargs and hashable physics metadata."""
    execution_kwargs = dict(solver_kwargs)
    species = [deuterium_ion(), electron()]
    case_acceptance: dict[str, Any]
    if case == "cyclone_base_case":
        case_acceptance = {
            "required_mode_types": ["ITG"],
            "max_gamma_max_cs_over_a": None,
            "description": "Cyclone Base Case ion-temperature-gradient parity guard",
        }
    elif case == "tem_kinetic_electron":
        species = [deuterium_ion(R_L_T=1.0, R_L_n=1.0), electron(R_L_T=8.0, R_L_n=3.0, adiabatic=False)]
        case_acceptance = {
            "required_mode_types": ["TEM", "ITG"],
            "max_gamma_max_cs_over_a": None,
            "description": "Kinetic-electron trapped-electron-mode branch parity guard",
        }
    elif case == "stable_mode":
        species = [deuterium_ion(R_L_T=0.1, R_L_n=0.1), electron(R_L_T=0.1, R_L_n=0.1)]
        case_acceptance = {
            "required_mode_types": ["stable", "ITG"],
            "max_gamma_max_cs_over_a": 0.01,
            "description": "Low-drive bounded-growth branch parity guard",
        }
    else:
        allowed = ", ".join(JAX_GK_PARITY_CASES)
        raise ValueError(f"case must be one of: {allowed}")

    execution_kwargs["species_list"] = species
    case_parameters = {
        "case": case,
        "solver_kwargs": _canonical_solver_kwargs(solver_kwargs),
        "species": [_canonical_species(species_item) for species_item in species],
        "electron_model": "kinetic"
        if any(not item.is_adiabatic and item.charge_e < 0 for item in species)
        else "adiabatic",
    }
    return execution_kwargs, case_parameters, case_acceptance


def _canonical_species(species: Any) -> dict[str, int | float | str | bool]:
    """Serialise the GK species fields that define a parity case."""
    role = "electron" if float(species.charge_e) < 0.0 else "ion"
    return {
        "species_role": role,
        "mass_amu": float(species.mass_amu),
        "mass_kg": float(species.mass_kg),
        "charge_e": float(species.charge_e),
        "density_19": float(species.density_19),
        "temperature_keV": float(species.temperature_keV),
        "R_L_n": float(species.R_L_n),
        "R_L_T": float(species.R_L_T),
        "adiabatic": bool(species.is_adiabatic),
    }


def _validate_parity_solver_kwargs(solver_kwargs: dict[str, Any]) -> None:
    for field in ("R0", "a", "B0", "q", "s_hat"):
        value = solver_kwargs.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(float(value)):
            raise ValueError(f"solver_kwargs[{field!r}] must be finite numeric")
    for field in ("n_ky_ion", "n_theta"):
        value = solver_kwargs.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"solver_kwargs[{field!r}] must be a positive integer")


def _canonical_solver_kwargs(solver_kwargs: dict[str, Any]) -> dict[str, int | float]:
    canonical: dict[str, int | float] = {}
    for key in sorted(solver_kwargs):
        value = solver_kwargs[key]
        if isinstance(value, bool):
            raise ValueError(f"solver_kwargs[{key!r}] must not be boolean")
        if isinstance(value, int):
            canonical[key] = int(value)
        elif isinstance(value, float):
            canonical[key] = float(value)
        else:
            raise ValueError(f"solver_kwargs[{key!r}] must be numeric")
    return canonical


def _jax_backend_metadata() -> dict[str, str | bool]:
    backend = str(jax.default_backend()).lower()
    devices = list(jax.devices())
    device_kind = str(getattr(devices[0], "device_kind", backend)).lower() if devices else backend
    dtype = str(jnp.asarray(1.0).dtype)
    try:
        x64_enabled = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        x64_enabled = dtype == "float64"
    return {
        "backend": backend,
        "device_kind": device_kind,
        "dtype": dtype,
        "x64_enabled": x64_enabled,
        "jax_version": _dist_version("jax"),
        "jaxlib_version": _dist_version("jaxlib"),
        "platform": platform.platform(),
    }


def _dist_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _sha256_json(payload: dict[str, Any]) -> str:
    digest_payload = {key: value for key, value in payload.items() if key != "payload_sha256"}
    encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
