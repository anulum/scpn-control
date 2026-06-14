# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Toroidal Momentum Transport
"""Toroidal momentum-transport and rotation-profile evolution utilities."""

from __future__ import annotations

import numpy as np

# Momentum Prandtl number: χ_φ / χ_i ≈ 0.7 (deuterium, ρ* ~ 0.004–0.007)
# Peeters et al. 2011, Nucl. Fusion 51, 083015, Fig. 5.
PRANDTL_MOMENTUM: float = 0.7


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _finite_array(name: str, values: np.ndarray, *, positive: bool = False, nonnegative: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def _finite_1d_grid(name: str, values: np.ndarray, *, minimum_size: int = 1) -> np.ndarray:
    arr = _finite_array(name, values)
    if arr.ndim != 1 or arr.size < minimum_size:
        if minimum_size == 2:
            raise ValueError(f"{name} must be a one-dimensional grid with at least two points")
        raise ValueError(f"{name} must be a one-dimensional grid with at least {minimum_size} points")
    return arr


def _strictly_increasing_rho_grid(rho: np.ndarray) -> np.ndarray:
    rho = _finite_1d_grid("rho", rho, minimum_size=2)
    if np.any(np.diff(rho) < 0.0):
        raise ValueError("rho must be sorted")
    if np.any(np.diff(rho) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    return rho


def _uniform_axis_to_edge_rho_grid(rho: np.ndarray) -> np.ndarray:
    rho = _strictly_increasing_rho_grid(rho)
    if not np.isclose(rho[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("rho must start at the magnetic axis")
    if not np.isclose(rho[-1], 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("rho must end at the plasma edge")
    spacing = np.diff(rho)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
        raise ValueError("rho must be uniformly spaced for the momentum finite-difference stencil")
    return rho


def nbi_torque(
    P_nbi_profile: np.ndarray,
    R0: float,
    v_beam: float,
    theta_inj_deg: float,
) -> np.ndarray:
    """NBI torque density [N·m/m³].

    T_NBI = P_NBI · R_tang / v_beam  where R_tang = R₀ sin(θ).
    Stacey & Sigmar 1985, Phys. Fluids 28, 2800.
    """
    P_nbi_profile = _finite_array("P_nbi_profile", P_nbi_profile, nonnegative=True)
    R0 = _finite_scalar("R0", R0, positive=True)
    v_beam = _finite_scalar("v_beam", v_beam)
    theta_inj_deg = _finite_scalar("theta_inj_deg", theta_inj_deg)
    if v_beam <= 0.0:
        return np.zeros_like(P_nbi_profile)

    theta_rad = np.radians(theta_inj_deg)
    R_tang = R0 * np.sin(theta_rad)  # tangential injection radius [m]
    return np.asarray(P_nbi_profile * R_tang / v_beam)


def intrinsic_rotation_torque(
    grad_Ti: np.ndarray,
    grad_ne: np.ndarray,
    R0: float,
    a: float,
) -> np.ndarray:
    """Residual-stress torque driving intrinsic counter-current rotation [N·m/m³].

    Residual stress Π_res ∝ −∇T_i → counter-current rotation.
    Empirical Rice scaling: v_φ,intr ∝ W_p / I_p.
    Rice et al. 2007, Nucl. Fusion 47, 1618, Eq. 3.
    """
    grad_Ti = _finite_array("grad_Ti", grad_Ti)
    grad_ne = _finite_array("grad_ne", grad_ne)
    if grad_Ti.shape != grad_ne.shape:
        raise ValueError("grad_Ti and grad_ne must have matching shape")
    R0 = _finite_scalar("R0", R0)
    a = _finite_scalar("a", a)
    if R0 <= 0.0 or a <= 0.0:
        raise ValueError("R0 and a must be positive")
    return np.asarray(-1e-3 * grad_Ti)


def exb_shearing_rate(
    omega_phi: np.ndarray,
    B_theta: np.ndarray,
    B0: float,
    R0: float,
    rho: np.ndarray,
    a: float,
) -> np.ndarray:
    """E×B shearing rate [rad/s].

    ω_E×B = (R B_θ / B) · d/dr (E_r / R B_θ)
    Burrell 1997, Phys. Plasmas 4, 1499, Eq. 1.

    In the rotation-dominated limit E_r ≈ v_φ B_θ = R₀ ω_φ B_θ, so
    E_r / (R₀ B_θ) ≈ ω_φ and ω_E×B ≈ (R₀ B_θ / B) · dω_φ/dr.
    """
    omega_phi = _finite_array("omega_phi", omega_phi)
    B_theta = _finite_array("B_theta", B_theta)
    rho = _strictly_increasing_rho_grid(rho)
    _require_equal_shape("omega_phi, B_theta, and rho", omega_phi, B_theta, rho)
    B0 = _finite_scalar("B0", B0)
    R0 = _finite_scalar("R0", R0)
    a = _finite_scalar("a", a)
    if B0 <= 0.0 or R0 <= 0.0 or a <= 0.0:
        raise ValueError("B0, R0, and a must be positive")
    radius = rho * a
    domega_dr = np.gradient(omega_phi, radius, edge_order=2)
    B_tot = np.sqrt(B0**2 + B_theta**2)
    rate = (R0 * B_theta / np.maximum(B_tot, 1e-6)) * domega_dr
    return np.asarray(np.abs(rate))


def turbulence_suppression_factor(
    omega_ExB: np.ndarray,
    gamma_max: np.ndarray,
) -> np.ndarray:
    """Suppression factor on anomalous transport.

    F = 1 / (1 + (ω_E×B / γ_max)²)
    Biglari, Diamond & Terry 1990, Phys. Fluids B 2, 1.
    """
    omega_ExB = _finite_array("omega_ExB", omega_ExB)
    gamma_max = _finite_array("gamma_max", gamma_max)
    _require_equal_shape("omega_ExB and gamma_max", omega_ExB, gamma_max)
    if np.any(gamma_max < 0.0):
        raise ValueError("gamma_max must be non-negative")
    gamma_safe = np.maximum(gamma_max, 1e-6)
    return np.asarray(1.0 / (1.0 + (omega_ExB / gamma_safe) ** 2))


def radial_electric_field(
    ne: np.ndarray,
    Ti_keV: np.ndarray,
    omega_phi: np.ndarray,
    B_theta: np.ndarray,
    B0: float,
    R0: float,
    rho: np.ndarray,
    a: float,
) -> np.ndarray:
    """E_r [V/m] from radial force balance (neoclassical), Z_i = 1.

    E_r = (1 / Z_i e n_i) dp_i/dr + v_φ B_θ   (v_θ ≈ 0)
    Hinton & Hazeltine 1976, Rev. Mod. Phys. 48, 239, Eq. 2.3.
    """
    ne = _finite_array("ne", ne)
    Ti_keV = _finite_array("Ti_keV", Ti_keV)
    omega_phi = _finite_array("omega_phi", omega_phi)
    B_theta = _finite_array("B_theta", B_theta)
    rho = _strictly_increasing_rho_grid(rho)
    _require_equal_shape("ne, Ti_keV, omega_phi, B_theta, and rho", ne, Ti_keV, omega_phi, B_theta, rho)
    if np.any(ne <= 0.0) or np.any(Ti_keV < 0.0):
        raise ValueError("ne must be positive and Ti_keV must be non-negative")
    B0 = _finite_scalar("B0", B0)
    R0 = _finite_scalar("R0", R0)
    a = _finite_scalar("a", a)
    if B0 <= 0.0 or R0 <= 0.0 or a <= 0.0:
        raise ValueError("B0, R0, and a must be positive")
    e_charge = 1.602e-19  # C
    p_i = ne * 1e19 * Ti_keV * 1e3 * e_charge

    radius = rho * a
    dp_dr = np.gradient(p_i, radius, edge_order=2)

    term1 = dp_dr / np.maximum(e_charge * ne * 1e19, 1e-6)
    term2 = R0 * omega_phi * B_theta  # v_φ B_θ
    return np.asarray(term1 + term2)


def rice_intrinsic_velocity(W_p_MJ: float, I_p_MA: float) -> float:
    """Intrinsic toroidal velocity [km/s] from Rice scaling.

    v_φ,intr ≈ c_Rice · (W_p / I_p)
    Rice et al. 2007, Nucl. Fusion 47, 1618, Eq. 3.

    c_Rice = 3.5 km s⁻¹ MA MJ⁻¹ (empirical, Ohmic + NBI plasmas, Fig. 4).
    """
    W_p_MJ = _finite_scalar("W_p_MJ", W_p_MJ)
    I_p_MA = _finite_scalar("I_p_MA", I_p_MA)
    if W_p_MJ < 0.0:
        raise ValueError("W_p_MJ must be non-negative")
    if I_p_MA <= 0.0:
        raise ValueError("I_p_MA must be positive")
    C_RICE: float = 3.5  # km/s per (MJ/MA), Rice et al. 2007 Fig. 4
    return C_RICE * W_p_MJ / I_p_MA


def _require_equal_shape(label: str, *arrays: np.ndarray) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"{label} must have matching shape")


def _nonnegative_profile_or_scalar(
    name: str,
    values: np.ndarray | float | None,
    shape: tuple[int, ...],
) -> np.ndarray:
    if values is None:
        return np.zeros(shape, dtype=float)

    arr = _finite_array(name, np.asarray(values, dtype=float), nonnegative=True)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have matching shape")
    return arr


class RotationDiagnostics:
    @staticmethod
    def mach_number(
        omega_phi: np.ndarray,
        Ti_keV: np.ndarray,
        R0: float,
    ) -> np.ndarray:
        omega_phi = _finite_array("omega_phi", omega_phi)
        Ti_keV = _finite_array("Ti_keV", Ti_keV)
        _require_equal_shape("omega_phi and Ti_keV", omega_phi, Ti_keV)
        R0 = _finite_scalar("R0", R0)
        if R0 <= 0.0:
            raise ValueError("R0 must be positive")
        if np.any(Ti_keV <= 0.0):
            raise ValueError("Ti_keV must be positive")
        e_charge = 1.602e-19  # C
        m_i = 2.0 * 1.67e-27  # kg, deuterium
        v_phi = omega_phi * R0
        c_s = np.sqrt(Ti_keV * 1e3 * e_charge / m_i)
        return np.asarray(np.abs(v_phi) / np.maximum(c_s, 1e-3))

    @staticmethod
    def rwm_stabilization_criterion(
        omega_phi: np.ndarray,
        tau_wall: float,
    ) -> bool:
        omega_phi = _finite_array("omega_phi", omega_phi)
        if omega_phi.size == 0:
            raise ValueError("omega_phi must not be empty")
        tau_wall = _finite_scalar("tau_wall", tau_wall)
        if tau_wall <= 0.0:
            raise ValueError("tau_wall must be positive")
        # ω τ_wall > O(1) criterion; Bondeson & Ward 1994, Phys. Rev. Lett. 72, 2709.
        return bool(np.abs(omega_phi[0]) * tau_wall > 0.01)


class MomentumTransportSolver:
    def __init__(
        self,
        rho: np.ndarray,
        R0: float,
        a: float,
        B0: float,
        prandtl: float = PRANDTL_MOMENTUM,
    ) -> None:
        # prandtl = χ_φ / χ_i; Peeters et al. 2011, Nucl. Fusion 51, 083015, Fig. 5.
        rho = _uniform_axis_to_edge_rho_grid(rho)
        R0 = _finite_scalar("R0", R0)
        a = _finite_scalar("a", a)
        B0 = _finite_scalar("B0", B0)
        prandtl = _finite_scalar("prandtl", prandtl)
        if R0 <= 0.0 or a <= 0.0 or B0 <= 0.0:
            raise ValueError("R0, a, and B0 must be positive")
        if prandtl <= 0.0:
            raise ValueError("prandtl must be positive")
        self.rho = rho
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.prandtl = prandtl
        self.nr = len(rho)
        self.drho = rho[1] - rho[0]
        self.omega_phi = np.zeros(self.nr)

    def step(
        self,
        dt: float,
        chi_i: np.ndarray,
        ne: np.ndarray,
        Ti_keV: np.ndarray,
        T_nbi: np.ndarray,
        T_intrinsic: np.ndarray,
        *,
        momentum_damping_frequency_s: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Advance rotation profile one step [rad/s].

        Momentum equation (cylindrical approximation):
          ∂(ρ_m R² ω)/∂t + (1/r) ∂/∂r(r Π_φ) = T_tot − ν_φ ρ_m R² ω
          Π_φ = −χ_φ ∂(ρ_m R² ω)/∂r   (pinch neglected)
        Wesson 2011, "Tokamaks", §4.10. The optional ν_φ profile is a
        non-negative bounded linear momentum damping frequency [s⁻¹], treated
        implicitly so zero-source, zero-diffusion cells decay as
        ωⁿ⁺¹ = ωⁿ / (1 + Δt ν_φ).
        """
        import scipy.linalg

        dt = _finite_scalar("dt", dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        chi_i = _finite_array("chi_i", chi_i)
        ne = _finite_array("ne", ne)
        Ti_keV = _finite_array("Ti_keV", Ti_keV)
        T_nbi = _finite_array("T_nbi", T_nbi)
        T_intrinsic = _finite_array("T_intrinsic", T_intrinsic)
        self.omega_phi = _finite_array("omega_phi", self.omega_phi)
        _require_equal_shape(
            "chi_i, ne, Ti_keV, T_nbi, and T_intrinsic",
            chi_i,
            ne,
            Ti_keV,
            T_nbi,
            T_intrinsic,
        )
        nu_phi = _nonnegative_profile_or_scalar(
            "momentum_damping_frequency_s", momentum_damping_frequency_s, chi_i.shape
        )
        if len(chi_i) != self.nr:
            raise ValueError("transport profiles must match the solver rho grid")
        if np.any(chi_i < 0.0):
            raise ValueError("chi_i must be non-negative")
        if np.any(ne <= 0.0) or np.any(Ti_keV < 0.0):
            raise ValueError("ne must be positive and Ti_keV must be non-negative")
        chi_phi = self.prandtl * chi_i  # Peeters et al. 2011

        m_i = 2.0 * 1.67e-27  # kg, deuterium
        rho_m = ne * 1e19 * m_i
        L = rho_m * self.R0**2 * self.omega_phi  # angular momentum density

        T_tot = T_nbi + T_intrinsic
        dr = self.drho * self.a
        if not np.isfinite(dr) or dr <= 0.0 or not np.all(np.isfinite(self.rho)) or not np.all(np.diff(self.rho) > 0.0):
            raise ValueError("rho grid must remain finite and strictly increasing")
        dr = max(float(dr), 1e-12)

        diag = np.zeros(self.nr)
        upper = np.zeros(self.nr)
        lower = np.zeros(self.nr)
        rhs = np.zeros(self.nr)

        # Axis: zero-gradient BC
        diag[0] = 1.0
        upper[0] = -1.0
        rhs[0] = 0.0

        # Edge: no-slip BC
        diag[-1] = 1.0
        rhs[-1] = 0.0

        for i in range(1, self.nr - 1):
            r_val = max(float(self.rho[i] * self.a), 0.5 * dr)
            c_plus = chi_phi[i] / dr**2 + chi_phi[i] / (2.0 * r_val * dr)
            c_minus = chi_phi[i] / dr**2 - chi_phi[i] / (2.0 * r_val * dr)
            c_0 = -2.0 * chi_phi[i] / dr**2

            lower[i] = -dt * c_minus
            diag[i] = 1.0 - dt * c_0 + dt * nu_phi[i]
            upper[i] = -dt * c_plus
            rhs[i] = L[i] + dt * T_tot[i]

        ab = np.zeros((3, self.nr))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]

        L_new = scipy.linalg.solve_banded((1, 1), ab, rhs)
        self.omega_phi = L_new / np.maximum(rho_m * self.R0**2, 1e-12)
        return self.omega_phi
