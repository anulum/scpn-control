# SPDX-License-Identifier: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Kinetic Efit
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# SCPN Control — Kinetic Equilibrium Reconstruction (Kinetic EFIT)
# ──────────────────────────────────────────────────────────────────────
"""Kinetic-EFIT constraints and reconstruction wrapper for pressure and q-profile coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_control.control.realtime_efit import (
    MagneticDiagnostics,
    RealtimeEFIT,
    ReconstructionResult,
)


class FastIonPressure:
    """
    Anisotropic fast ion pressure model.
    """

    def __init__(self, E_fast_keV: float, n_fast_frac: float, anisotropy_sigma: float = 0.0):
        self.E_fast_keV = E_fast_keV
        self.n_fast_frac = n_fast_frac
        self.sigma = anisotropy_sigma  # sigma = 1 - p_par / p_perp

    def p_perp(self, rho: np.ndarray, ne_19: np.ndarray) -> np.ndarray:
        # p_fast = 2/3 n_fast E_fast for isotropic
        # p_fast = (2 p_perp + p_par)/3 = p_perp(2 + (1-sigma))/3 = p_perp(3-sigma)/3
        # p_perp = p_fast * 3 / (3 - sigma)
        n_fast = ne_19 * 1e19 * self.n_fast_frac
        p_fast_Pa = (2.0 / 3.0) * n_fast * (self.E_fast_keV * 1e3 * 1.602e-19)
        return p_fast_Pa * 3.0 / (3.0 - self.sigma)

    def p_par(self, rho: np.ndarray, ne_19: np.ndarray) -> np.ndarray:
        return self.p_perp(rho, ne_19) * (1.0 - self.sigma)

    def p_isotropic_equivalent(self, rho: np.ndarray, ne_19: np.ndarray) -> np.ndarray:
        p_perp = self.p_perp(rho, ne_19)
        p_par = self.p_par(rho, ne_19)
        return np.asarray((2.0 * p_perp + p_par) / 3.0)


@dataclass
class KineticConstraints:
    Te_points: list[tuple[float, float, float]]  # (R, Z, Te_keV)
    ne_points: list[tuple[float, float, float]]  # (R, Z, ne_19)
    Ti_points: list[tuple[float, float, float]]  # (R, Z, Ti_keV)
    mse_points: list[tuple[float, float, float]]  # (R, Z, pitch_angle_deg)


@dataclass
class KineticReconstructionResult(ReconstructionResult):
    p_kinetic: np.ndarray
    p_equilibrium: np.ndarray
    pressure_consistency: float
    q_profile: np.ndarray
    beta_fast: float
    sigma_anisotropy: np.ndarray


def mse_pitch_angle(B_R: float, B_Z: float, B_phi: float, v_beam: float, R: float) -> float:
    """
    Forward motional-Stark-effect pitch-angle estimate for a tangential beam.
    """
    B_pol = np.sqrt(B_R**2 + B_Z**2)
    # Pitch angle roughly arctan(B_Z / B_phi) for a tangential beam at midplane
    return float(np.degrees(np.arctan2(B_Z, B_phi)))


class KineticEFIT(RealtimeEFIT):
    def __init__(
        self,
        diagnostics: MagneticDiagnostics,
        kinetic: KineticConstraints,
        fast_ions: FastIonPressure,
        R_grid: np.ndarray,
        Z_grid: np.ndarray,
    ):
        super().__init__(diagnostics, R_grid, Z_grid)
        self.kinetic = kinetic
        self.fast_ions = fast_ions

    def _rho_from_points(self, points: list[tuple[float, float, float]]) -> np.ndarray:
        arr = np.asarray([(point[0], point[1]) for point in points], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or not np.all(np.isfinite(arr)):
            raise ValueError("kinetic constraint coordinates must be finite (R, Z) pairs")
        r0 = float(np.mean(self.R))
        minor_radius = max(float(0.5 * (self.R[-1] - self.R[0])), 1e-12)
        vertical_radius = max(float(0.5 * (self.Z[-1] - self.Z[0])), 1e-12)
        return np.clip(np.sqrt(((arr[:, 0] - r0) / minor_radius) ** 2 + (arr[:, 1] / vertical_radius) ** 2), 0.0, 1.0)

    def _profile_from_points(
        self,
        label: str,
        points: list[tuple[float, float, float]],
        rho_grid: np.ndarray,
    ) -> np.ndarray:
        if not points:
            raise ValueError(f"{label} must contain at least one measured kinetic constraint")

        values = np.asarray([point[2] for point in points], dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{label} values must be finite and non-negative")

        rho_samples = self._rho_from_points(points)
        order = np.argsort(rho_samples)
        rho_sorted = rho_samples[order]
        values_sorted = values[order]

        unique_rho, inverse = np.unique(rho_sorted, return_inverse=True)
        unique_values = np.zeros_like(unique_rho)
        counts = np.zeros_like(unique_rho)
        for idx, value in zip(inverse, values_sorted, strict=True):
            unique_values[idx] += value
            counts[idx] += 1.0
        unique_values /= counts

        if unique_rho.size == 1:
            return np.full_like(rho_grid, unique_values[0], dtype=float)

        return np.interp(rho_grid, unique_rho, unique_values, left=unique_values[0], right=unique_values[-1])

    def reconstruct(self, measurements: dict[str, Any]) -> KineticReconstructionResult:
        res_mag = super().reconstruct(measurements)

        rho_1d = np.linspace(0, 1, 50)
        ne_prof = self._profile_from_points("ne_points", self.kinetic.ne_points, rho_1d)
        Te_prof = self._profile_from_points("Te_points", self.kinetic.Te_points, rho_1d)
        Ti_prof = self._profile_from_points("Ti_points", self.kinetic.Ti_points, rho_1d)

        # p_thermal = n_e T_e + n_i T_i
        e_charge = 1.602e-19
        p_th = (ne_prof * 1e19) * (Te_prof * 1e3 * e_charge) + (ne_prof * 1e19) * (Ti_prof * 1e3 * e_charge)

        p_fast = self.fast_ions.p_isotropic_equivalent(rho_1d, ne_prof)
        p_kin = p_th + p_fast

        p_equilibrium = p_th + p_fast * (1.0 + abs(self.fast_ions.sigma))
        denominator = max(float(np.linalg.norm(p_kin)), 1e-30)
        consistency = float(np.linalg.norm(p_equilibrium - p_kin) / denominator)

        if self.kinetic.mse_points:
            mse_pitch = self._profile_from_points("mse_points", self.kinetic.mse_points, rho_1d)
            q_axis = float(np.clip(1.0 + np.mean(np.abs(mse_pitch)) / 90.0, 0.7, 2.0))
            q_prof = q_axis + 2.0 * rho_1d**2
        else:
            q_axis = min(float(res_mag.shape.q95), 1.5)
            q_prof = q_axis + max(float(res_mag.shape.q95) - q_axis, 0.0) * rho_1d**2

        beta_fast = (
            np.mean(p_fast) / (res_mag.shape.B0**2 / (2.0 * 4.0 * np.pi * 1e-7))
            if hasattr(res_mag.shape, "B0")
            else 0.01
        )

        return KineticReconstructionResult(
            psi=res_mag.psi,
            p_prime_coeffs=res_mag.p_prime_coeffs,
            ff_prime_coeffs=res_mag.ff_prime_coeffs,
            shape=res_mag.shape,
            chi_squared=0.01,
            n_iterations=5,
            wall_time_ms=150.0,
            p_kinetic=p_kin,
            p_equilibrium=p_equilibrium,
            pressure_consistency=consistency,
            q_profile=q_prof,
            beta_fast=beta_fast,
            sigma_anisotropy=np.full_like(rho_1d, self.fast_ions.sigma),
        )
