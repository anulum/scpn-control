# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Impurity Transport and Radiative Losses
"""Impurity transport, tungsten accumulation, and radiative-loss diagnostic utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

# Neoclassical impurity pinch velocity:
#   V_neo = -D_neo (Z/T_i) dT_i/dr
# Hirshman & Sigmar 1981, Nucl. Fusion 21, 1079, Eq. 4.17.
#
# Neoclassical diffusion coefficient (banana regime):
#   D_neo = q² ρ_i² ν_ii / ε^(3/2)
# Hinton & Hazeltine 1976, Rev. Mod. Phys. 48, 239, Eq. 4.62.
#
# Radiation power density:
#   P_rad = n_e n_Z L_Z(T_e)   [W m^-3]
# Post et al. 1977, At. Data Nucl. Data Tables 20, 397.
#
# Cooling curve parameters — peak temperatures and widths:
#   W high-T peak:  T_peak ≈ 1500 eV, σ = 1.5 (log-normal width)  — Putterich et al. 2010, Nucl. Fusion 50, 025012
#   W low-T peak:   T_peak ≈ 50 eV,   σ = 1.0                      — Putterich et al. 2010
#   C:              T_peak ≈ 10 eV,    σ = 0.5                      — Post et al. 1977
#   Ar:             T_peak ≈ 200 eV,   σ = 1.0                      — Post et al. 1977
#   Ne:             T_peak ≈ 50 eV,    σ = 1.0                      — Post et al. 1977


def _positive_finite_profile(name: str, values: AnyFloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive everywhere")
    return arr


def _finite_profile_like(name: str, values: AnyFloatArray, shape: tuple[int, ...]) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must match rho shape")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _nonnegative_profile_like(name: str, values: AnyFloatArray, shape: tuple[int, ...]) -> FloatArray:
    arr = _finite_profile_like(name, values, shape)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def _radiation_rho_grid(rho: AnyFloatArray) -> FloatArray:
    arr = np.asarray(rho, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("rho must be a one-dimensional profile with at least two points")
    if not np.all(np.isfinite(arr)):
        raise ValueError("rho must contain only finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("rho must stay within [0, 1]")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    return arr


def _uniform_axis_to_edge_rho_grid(rho: AnyFloatArray) -> FloatArray:
    arr = _radiation_rho_grid(rho)
    if not np.isclose(arr[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("rho must start at the magnetic axis")
    if not np.isclose(arr[-1], 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("rho must end at the plasma edge")
    spacing = np.diff(arr)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
        raise ValueError("rho must be uniformly spaced for the impurity finite-difference stencil")
    return arr


@dataclass
class ImpuritySpecies:
    """An impurity species for transport modelling.

    Attributes
    ----------
    element
        Element symbol (e.g. ``"W"``, ``"Ne"``).
    Z_nucleus
        Nuclear charge number.
    mass_amu
        Atomic mass in amu.
    source_rate
        Edge source rate in particles/s.
    """

    element: str
    Z_nucleus: int
    mass_amu: float
    source_rate: float = 0.0


class CoolingCurve:
    """
    Parametric cooling rate coefficient L_Z(T_e) [W m³].

    Log-normal fits to:
      W:   Putterich et al. 2010, Nucl. Fusion 50, 025012, Fig. 3.
      C/Ar/Ne: Post et al. 1977, At. Data Nucl. Data Tables 20, 397.
    """

    def __init__(self, element: str):
        self.element = element

    def L_z(self, Te_eV: AnyFloatArray) -> FloatArray:
        """Return cooling coefficients for positive finite electron temperatures [W m³]."""
        Te_eV = _positive_finite_profile("Te_eV", Te_eV)
        log_Te = np.log(Te_eV)
        if self.element == "W":
            # High-T peak: L_max ≈ 1e-31 W m³ at T ≈ 1500 eV, width σ = 1.5
            # Low-T peak:  L_max ≈ 3e-33 W m³ at T ≈ 50 eV,   width σ = 1.0
            # Putterich et al. 2010, Nucl. Fusion 50, 025012.
            L = 1e-31 * np.exp(-(((log_Te - np.log(1500.0)) / 1.5) ** 2))
            L += 3e-33 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2))
            return np.asarray(L)
        if self.element == "C":
            # Peak: L_max ≈ 1e-32 W m³ at T ≈ 10 eV, σ = 0.5
            # Post et al. 1977, ADNDT 20, 397.
            return np.asarray(1e-32 * np.exp(-(((log_Te - np.log(10.0)) / 0.5) ** 2)))
        if self.element == "Ar":
            # Peak: L_max ≈ 1e-32 W m³ at T ≈ 200 eV, σ = 1.0
            # Post et al. 1977, ADNDT 20, 397.
            return np.asarray(1e-32 * np.exp(-(((log_Te - np.log(200.0)) / 1.0) ** 2)))
        if self.element == "Ne":
            # Peak: L_max ≈ 1e-32 W m³ at T ≈ 50 eV, σ = 1.0
            # Post et al. 1977, ADNDT 20, 397.
            return np.asarray(1e-32 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2)))
        return np.zeros_like(Te_eV)


def neoclassical_impurity_pinch(
    Z: int,
    ne: AnyFloatArray,
    Te_eV: AnyFloatArray,
    Ti_eV: AnyFloatArray,
    q: AnyFloatArray,
    rho: AnyFloatArray,
    R0: float,
    a: float,
    epsilon: AnyFloatArray,
) -> FloatArray:
    """
    Neoclassical impurity pinch velocity V_neo [m/s].

    Full formula (banana regime, trace impurity limit):
      V_neo,Z = -D_neo [Z ∇n/n + (Z/2 - H_Z) ∇T_i/T_i]
    where H_Z = 0.5 is the ion screening factor.
    Hirshman & Sigmar 1981, Nucl. Fusion 21, 1079, Eq. 4.17.

    The thermal-gradient term dominates for peaked T_i profiles and
    produces an inward (negative) pinch when dT_i/dr < 0.
    Sign convention: V < 0 → inward (toward axis).

    D_neo = q² ρ_i² ν_ii / ε^(3/2)  (banana regime)
    Hinton & Hazeltine 1976, Rev. Mod. Phys. 48, 239, Eq. 4.62.
    Nominal scale D_neo ≈ 0.1 m²/s for ITER-relevant parameters.
    """
    rho = _radiation_rho_grid(rho)
    shape = rho.shape
    ne = _finite_profile_like("ne", ne, shape)
    Te_eV = _positive_finite_profile("Te_eV", Te_eV)
    Ti_eV = _positive_finite_profile("Ti_eV", Ti_eV)
    q = _finite_profile_like("q", q, shape)
    epsilon = _finite_profile_like("epsilon", epsilon, shape)
    if np.any(ne < 0.0):
        raise ValueError("ne must be non-negative")
    if Te_eV.shape != shape:
        raise ValueError("Te_eV must match rho shape")
    if Ti_eV.shape != shape:
        raise ValueError("Ti_eV must match rho shape")
    R0 = float(R0)
    a = float(a)
    if Z <= 0:
        raise ValueError("Z must be positive")
    if not np.isfinite(R0) or R0 <= 0.0:
        raise ValueError("R0 must be finite and positive")
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("a must be finite and positive")

    radius = rho * a
    grad_ne_over_n = np.gradient(ne, radius, edge_order=2) / np.maximum(ne, 1e-6)
    grad_Ti_over_T = np.gradient(Ti_eV, radius, edge_order=2) / np.maximum(Ti_eV, 1e-6)

    # D_neo = 0.1 m²/s nominal; banana regime, Hinton & Hazeltine 1976 Eq. 4.62
    D_neo = 0.1 * np.ones_like(rho)

    # H_Z = 0.5: banana-regime impurity screening factor, Hirshman & Sigmar 1981 Eq. 4.17
    H_Z = 0.5

    # Inward pinch for peaked T_i: (Z/2 - H_Z) > 0 when Z > 2*H_Z = 1,
    # so grad_Ti_over_T < 0 drives V_neo < 0 (inward).
    V_neo = -D_neo * (Z * grad_ne_over_n + (Z / 2.0 - H_Z) * grad_Ti_over_T)
    return np.asarray(V_neo)


def total_radiated_power(
    ne: AnyFloatArray,
    n_impurity: dict[str, AnyFloatArray],
    Te_eV: AnyFloatArray,
    rho: AnyFloatArray,
    R0: float,
    a: float,
) -> float:
    """
    Total radiated power P_rad [MW] integrated over plasma volume.

    Local power density:
      p_rad(r) = n_e(r) n_Z(r) L_Z(T_e(r))   [W m^-3]
    Post et al. 1977, At. Data Nucl. Data Tables 20, 397.

    Volume element for circular cross-section torus:
      dV = 4π² R0 a² ρ dρ
    """
    rho = _radiation_rho_grid(rho)
    ne = _positive_finite_profile("ne", ne)
    Te_eV = _positive_finite_profile("Te_eV", Te_eV)
    if ne.shape != rho.shape:
        raise ValueError("ne must match rho shape")
    if Te_eV.shape != rho.shape:
        raise ValueError("Te_eV must match rho shape")
    R0 = float(R0)
    a = float(a)
    if not np.isfinite(R0) or R0 <= 0.0:
        raise ValueError("R0 must be finite and positive")
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("a must be finite and positive")

    p_rad_density = np.zeros_like(rho)

    for element, n_Z in n_impurity.items():
        n_Z = _nonnegative_profile_like(f"n_impurity[{element}]", n_Z, rho.shape)
        curve = CoolingCurve(element)
        L = curve.L_z(Te_eV)
        p_rad_density += ne * n_Z * L

    vol_element = 4.0 * np.pi**2 * R0 * a**2 * rho
    _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    P_rad_W = _trapz(p_rad_density * vol_element, rho)

    return float(P_rad_W / 1e6)


def tungsten_accumulation_diagnostic(n_W: AnyFloatArray, ne: AnyFloatArray) -> dict[str, Any]:
    """Assess core tungsten accumulation from density profiles.

    Parameters
    ----------
    n_W
        Tungsten density profile in m⁻³ (core to edge).
    ne
        Electron density profile in m⁻³ on the same grid.

    Returns
    -------
    dict[str, Any]
        Core and edge tungsten concentration, the peaking factor, and a
        ``"danger_level"`` of ``"safe"``, ``"warning"``, or ``"critical"``.
    """
    c_W_core = float(n_W[0] / max(ne[0], 1e-6))
    c_W_edge = float(n_W[-1] / max(ne[-1], 1e-6))

    peaking_factor = c_W_core / max(c_W_edge, 1e-12)

    if c_W_core < 1e-5:
        danger = "safe"
    elif c_W_core < 5e-5:
        danger = "warning"
    else:
        danger = "critical"

    return {"c_W_core": c_W_core, "c_W_edge": c_W_edge, "peaking_factor": peaking_factor, "danger_level": danger}


class ImpurityTransportSolver:
    """Radial impurity-transport solver with diffusion and neoclassical pinch.

    Parameters
    ----------
    rho
        Normalised-radius grid (resampled to an edge grid).
    R0
        Major radius in metres; must be positive.
    a
        Minor radius in metres.
    species
        The impurity species to transport.
    """

    def __init__(self, rho: AnyFloatArray, R0: float, a: float, species: list[ImpuritySpecies]):
        self.rho = _uniform_axis_to_edge_rho_grid(rho)
        self.R0 = float(R0)
        self.a = float(a)
        if not np.isfinite(self.R0) or self.R0 <= 0.0:
            raise ValueError("R0 must be finite and positive")
        if not np.isfinite(self.a) or self.a <= 0.0:
            raise ValueError("a must be finite and positive")
        self.species = species

        self.nr = len(self.rho)
        self.drho = self.rho[1] - self.rho[0]

        self.n_z: dict[str, FloatArray] = {s.element: np.zeros(self.nr) for s in species}

    def step(
        self,
        dt: float,
        ne: AnyFloatArray,
        Te_eV: AnyFloatArray,
        Ti_eV: AnyFloatArray,
        D_anom: float,
        V_pinch: dict[str, AnyFloatArray],
    ) -> dict[str, FloatArray]:
        """
        1D transport advance for each impurity species.

        Flux: Γ = -(D_anom + D_neo) dn/dr + V n
        Upwind differencing for convection; Crank-Nicolson implicit for diffusion.
        """
        import scipy.linalg

        dt = float(dt)
        D_anom = float(D_anom)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive")
        if not np.isfinite(D_anom) or D_anom < 0.0:
            raise ValueError("D_anom must be non-negative")
        shape = self.rho.shape
        _positive_finite_profile("ne", ne)
        _positive_finite_profile("Te_eV", Te_eV)
        _positive_finite_profile("Ti_eV", Ti_eV)
        if np.asarray(ne, dtype=float).shape != shape:
            raise ValueError("ne must match rho shape")
        if np.asarray(Te_eV, dtype=float).shape != shape:
            raise ValueError("Te_eV must match rho shape")
        if np.asarray(Ti_eV, dtype=float).shape != shape:
            raise ValueError("Ti_eV must match rho shape")

        dr = self.drho * self.a

        for s in self.species:
            n = self.n_z[s.element]
            V = _finite_profile_like(f"V_pinch[{s.element}]", V_pinch.get(s.element, np.zeros(self.nr)), shape)

            D = D_anom * np.ones(self.nr)

            diag = np.zeros(self.nr)
            upper = np.zeros(self.nr)
            lower = np.zeros(self.nr)
            rhs = np.zeros(self.nr)

            diag[0] = 1.0
            upper[0] = -1.0
            rhs[0] = 0.0  # dn/dr = 0 at magnetic axis

            diag[-1] = 1.0
            rhs[-1] = s.source_rate * dt / dr

            for i in range(1, self.nr - 1):
                r_val = self.rho[i] * self.a

                coeff_D_plus = D[i] / dr**2 + D[i] / (2.0 * r_val * dr)
                coeff_D_minus = D[i] / dr**2 - D[i] / (2.0 * r_val * dr)
                coeff_D_0 = -2.0 * D[i] / dr**2

                if V[i] > 0:
                    coeff_V_0 = -V[i] / dr - V[i] / r_val
                    coeff_V_minus = V[i] / dr
                    coeff_V_plus = 0.0
                else:
                    coeff_V_0 = V[i] / dr - V[i] / r_val
                    coeff_V_plus = -V[i] / dr
                    coeff_V_minus = 0.0

                lower[i] = -dt * (coeff_D_minus + coeff_V_minus)
                diag[i] = 1.0 - dt * (coeff_D_0 + coeff_V_0)
                upper[i] = -dt * (coeff_D_plus + coeff_V_plus)

                rhs[i] = n[i]

            ab = np.zeros((3, self.nr))
            ab[0, 1:] = upper[:-1]
            ab[1, :] = diag
            ab[2, :-1] = lower[1:]

            n_new = scipy.linalg.solve_banded((1, 1), ab, rhs)
            self.n_z[s.element] = np.maximum(n_new, 0.0)

        return self.n_z
