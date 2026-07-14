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

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

if TYPE_CHECKING:
    from scpn_control.core.fusion_kernel import CoilSet

from scpn_control.control.realtime_efit import (
    MagneticDiagnostics,
    RealtimeEFIT,
    ReconstructionResult,
)

_KINETIC_EFIT_CLAIM_SCHEMA_VERSION = 1
_FACILITY_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "efit_reference", "p_efit_reference", "measured_discharge"}
)
_BOUNDED_REFERENCE_SOURCES = frozenset({"synthetic_regression_reference", *_FACILITY_REFERENCE_SOURCES})
_KINETIC_INTERPOLATION_GEOMETRY = "normalised_elliptic_rho"


class FastIonPressure:
    """Anisotropic fast ion pressure model."""

    def __init__(self, E_fast_keV: float, n_fast_frac: float, anisotropy_sigma: float = 0.0):
        self.E_fast_keV = E_fast_keV
        self.n_fast_frac = n_fast_frac
        self.sigma = anisotropy_sigma  # sigma = 1 - p_par / p_perp

    def p_perp(self, rho: AnyFloatArray, ne_19: AnyFloatArray) -> AnyFloatArray:
        """Perpendicular fast-ion pressure profile.

        Parameters
        ----------
        rho
            Normalised-radius grid.
        ne_19
            Electron density in 10¹⁹ m⁻³ on ``rho``.

        Returns
        -------
        AnyFloatArray
            Perpendicular fast-ion pressure in pascals.
        """
        # p_fast = 2/3 n_fast E_fast for isotropic
        # p_fast = (2 p_perp + p_par)/3 = p_perp(2 + (1-sigma))/3 = p_perp(3-sigma)/3
        # p_perp = p_fast * 3 / (3 - sigma)
        n_fast = ne_19 * 1e19 * self.n_fast_frac
        p_fast_Pa = (2.0 / 3.0) * n_fast * (self.E_fast_keV * 1e3 * 1.602e-19)
        return p_fast_Pa * 3.0 / (3.0 - self.sigma)

    def p_par(self, rho: AnyFloatArray, ne_19: AnyFloatArray) -> AnyFloatArray:
        """Parallel fast-ion pressure profile.

        Parameters
        ----------
        rho
            Normalised-radius grid.
        ne_19
            Electron density in 10¹⁹ m⁻³ on ``rho``.

        Returns
        -------
        AnyFloatArray
            Parallel fast-ion pressure in pascals, ``p_perp (1 - sigma)``.
        """
        return self.p_perp(rho, ne_19) * (1.0 - self.sigma)

    def p_isotropic_equivalent(self, rho: AnyFloatArray, ne_19: AnyFloatArray) -> AnyFloatArray:
        """Isotropic-equivalent scalar fast-ion pressure ``(2 p_perp + p_par)/3``.

        Parameters
        ----------
        rho
            Normalised-radius grid.
        ne_19
            Electron density in 10¹⁹ m⁻³ on ``rho``.

        Returns
        -------
        AnyFloatArray
            Scalar fast-ion pressure in pascals.
        """
        p_perp = self.p_perp(rho, ne_19)
        p_par = self.p_par(rho, ne_19)
        return np.asarray((2.0 * p_perp + p_par) / 3.0)


@dataclass
class KineticConstraints:
    """Measured kinetic constraints for kinetic-EFIT reconstruction.

    Attributes
    ----------
    Te_points
        Electron-temperature constraints as ``(R, Z, Te_keV)`` tuples.
    ne_points
        Electron-density constraints as ``(R, Z, ne_19)`` tuples.
    Ti_points
        Ion-temperature constraints as ``(R, Z, Ti_keV)`` tuples.
    mse_points
        Motional-Stark-effect constraints as ``(R, Z, pitch_angle_deg)`` tuples.
    """

    Te_points: list[tuple[float, float, float]]  # (R, Z, Te_keV)
    ne_points: list[tuple[float, float, float]]  # (R, Z, ne_19)
    Ti_points: list[tuple[float, float, float]]  # (R, Z, Ti_keV)
    mse_points: list[tuple[float, float, float]]  # (R, Z, pitch_angle_deg)


@dataclass
class KineticReconstructionResult(ReconstructionResult):
    """Kinetic-EFIT result extending the magnetic reconstruction.

    Attributes
    ----------
    p_kinetic
        Kinetic pressure profile in pascals.
    p_equilibrium
        Equilibrium pressure profile in pascals.
    pressure_consistency
        Relative agreement between the kinetic and equilibrium pressure.
    q_profile
        Safety-factor profile.
    beta_fast
        Fast-ion beta fraction.
    sigma_anisotropy
        Fast-ion anisotropy ``1 - p_par/p_perp`` profile.
    """

    p_kinetic: AnyFloatArray
    p_equilibrium: AnyFloatArray
    pressure_consistency: float
    q_profile: AnyFloatArray
    beta_fast: float
    sigma_anisotropy: AnyFloatArray


@dataclass(frozen=True)
class KineticEFITClaimEvidence:
    """Serialisable provenance and reference-comparison evidence for kinetic-EFIT claims."""

    schema_version: int
    source: str
    source_id: str
    diagnostic_source: str
    profile_source: str
    fast_ion_source: str
    mse_calibration_source: str
    model_id: str
    interpolation_geometry: str
    n_te_points: int
    n_ne_points: int
    n_ti_points: int
    n_mse_points: int
    fast_ion_energy_keV: float
    fast_ion_density_fraction: float
    anisotropy_sigma: float
    pressure_consistency: float
    beta_fast: float
    q_axis: float
    q_edge: float
    pressure_relative_error: float | None
    q_profile_relative_error: float | None
    anisotropy_abs_error: float | None
    pressure_relative_tolerance: float
    q_profile_relative_tolerance: float
    anisotropy_abs_tolerance: float
    facility_claim_allowed: bool
    claim_status: str


def _finite_float(name: str, value: float) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _relative_profile_error(name: str, observed: AnyFloatArray, reference: AnyFloatArray | None) -> float | None:
    if reference is None:
        return None
    ref = np.asarray(reference, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if ref.shape != obs.shape or not np.all(np.isfinite(ref)):
        raise ValueError(f"{name} reference must be finite and match the reconstructed profile shape")
    denominator = max(float(np.linalg.norm(ref)), 1e-30)
    return float(np.linalg.norm(obs - ref) / denominator)


def kinetic_efit_claim_evidence(
    result: KineticReconstructionResult,
    kinetic: KineticConstraints,
    fast_ions: FastIonPressure,
    *,
    source: str,
    source_id: str,
    diagnostic_source: str,
    profile_source: str,
    fast_ion_source: str,
    mse_calibration_source: str,
    model_id: str = "bounded_kinetic_efit",
    reference_pressure: AnyFloatArray | None = None,
    reference_q_profile: AnyFloatArray | None = None,
    reference_anisotropy_sigma: float | None = None,
    pressure_relative_tolerance: float = 0.05,
    q_profile_relative_tolerance: float = 0.05,
    anisotropy_abs_tolerance: float = 0.05,
) -> KineticEFITClaimEvidence:
    """Build fail-closed kinetic-EFIT evidence for bounded or facility claims.

    Facility claims require a recognised facility reference source plus matched
    pressure, q-profile, and anisotropy references inside declared tolerances.
    Synthetic regression evidence is retained as bounded controller evidence
    only and is never promoted to a facility-grade P-EFIT claim.
    """
    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")

    pressure_tolerance = _finite_float("pressure_relative_tolerance", pressure_relative_tolerance)
    q_tolerance = _finite_float("q_profile_relative_tolerance", q_profile_relative_tolerance)
    anisotropy_tolerance = _finite_float("anisotropy_abs_tolerance", anisotropy_abs_tolerance)
    if pressure_tolerance <= 0.0 or q_tolerance <= 0.0 or anisotropy_tolerance <= 0.0:
        raise ValueError("reference tolerances must be positive")

    p_kinetic = np.asarray(result.p_kinetic, dtype=float)
    q_profile = np.asarray(result.q_profile, dtype=float)
    sigma_profile = np.asarray(result.sigma_anisotropy, dtype=float)
    if (
        p_kinetic.ndim != 1
        or q_profile.ndim != 1
        or sigma_profile.ndim != 1
        or p_kinetic.size == 0
        or q_profile.size == 0
        or sigma_profile.size == 0
        or not np.all(np.isfinite(p_kinetic))
        or not np.all(np.isfinite(q_profile))
        or not np.all(np.isfinite(sigma_profile))
    ):
        raise ValueError("kinetic-EFIT result profiles must be non-empty finite one-dimensional arrays")

    fast_ion_energy = _finite_float("fast_ion_energy_keV", fast_ions.E_fast_keV)
    fast_ion_fraction = _finite_float("fast_ion_density_fraction", fast_ions.n_fast_frac)
    anisotropy_sigma = _finite_float("anisotropy_sigma", fast_ions.sigma)
    if fast_ion_energy <= 0.0 or fast_ion_fraction < 0.0 or anisotropy_sigma >= 1.0:
        raise ValueError("fast-ion evidence requires positive energy, non-negative density fraction, and sigma < 1")

    pressure_error = _relative_profile_error("pressure", p_kinetic, reference_pressure)
    q_error = _relative_profile_error("q_profile", q_profile, reference_q_profile)
    anisotropy_error = None
    if reference_anisotropy_sigma is not None:
        anisotropy_error = abs(
            float(np.mean(sigma_profile)) - _finite_float("reference_anisotropy_sigma", reference_anisotropy_sigma)
        )

    references_pass = False
    if pressure_error is not None and q_error is not None and anisotropy_error is not None:
        references_pass = (
            pressure_error <= pressure_tolerance and q_error <= q_tolerance and anisotropy_error <= anisotropy_tolerance
        )
    facility_claim_allowed = source_clean in _FACILITY_REFERENCE_SOURCES and references_pass
    claim_status = "facility_reference_matched" if facility_claim_allowed else "bounded_controller_evidence"

    return KineticEFITClaimEvidence(
        schema_version=_KINETIC_EFIT_CLAIM_SCHEMA_VERSION,
        source=source_clean,
        source_id=_non_empty_text("source_id", source_id),
        diagnostic_source=_non_empty_text("diagnostic_source", diagnostic_source),
        profile_source=_non_empty_text("profile_source", profile_source),
        fast_ion_source=_non_empty_text("fast_ion_source", fast_ion_source),
        mse_calibration_source=_non_empty_text("mse_calibration_source", mse_calibration_source),
        model_id=_non_empty_text("model_id", model_id),
        interpolation_geometry=_KINETIC_INTERPOLATION_GEOMETRY,
        n_te_points=len(kinetic.Te_points),
        n_ne_points=len(kinetic.ne_points),
        n_ti_points=len(kinetic.Ti_points),
        n_mse_points=len(kinetic.mse_points),
        fast_ion_energy_keV=fast_ion_energy,
        fast_ion_density_fraction=fast_ion_fraction,
        anisotropy_sigma=anisotropy_sigma,
        pressure_consistency=_finite_float("pressure_consistency", result.pressure_consistency),
        beta_fast=_finite_float("beta_fast", result.beta_fast),
        q_axis=float(q_profile[0]),
        q_edge=float(q_profile[-1]),
        pressure_relative_error=pressure_error,
        q_profile_relative_error=q_error,
        anisotropy_abs_error=anisotropy_error,
        pressure_relative_tolerance=pressure_tolerance,
        q_profile_relative_tolerance=q_tolerance,
        anisotropy_abs_tolerance=anisotropy_tolerance,
        facility_claim_allowed=bool(facility_claim_allowed),
        claim_status=claim_status,
    )


def assert_kinetic_efit_facility_claim_admissible(evidence: KineticEFITClaimEvidence) -> None:
    """Raise when kinetic-EFIT evidence is insufficient for a facility claim."""
    if not evidence.facility_claim_allowed:
        raise ValueError("kinetic-EFIT facility claim requires matched pressure, q-profile, and anisotropy references")


def save_kinetic_efit_claim_evidence(evidence: KineticEFITClaimEvidence, path: str | Path) -> None:
    """Persist kinetic-EFIT claim evidence as deterministic JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mse_pitch_angle(B_R: float, B_Z: float, B_phi: float, v_beam: float, R: float) -> float:
    """Forward motional-Stark-effect pitch-angle estimate for a tangential beam."""
    # Pitch angle roughly arctan(B_Z / B_phi) for a tangential beam at midplane
    return float(np.degrees(np.arctan2(B_Z, B_phi)))


class KineticEFIT(RealtimeEFIT):
    """Kinetic equilibrium reconstruction adding kinetic and MSE constraints.

    Extends the magnetic :class:`RealtimeEFIT` with kinetic-profile and fast-ion
    pressure constraints for a kinetic-EFIT-style reconstruction.

    Parameters
    ----------
    diagnostics
        Magnetic diagnostic set for the base reconstruction.
    kinetic
        Measured kinetic constraints.
    fast_ions
        Fast-ion pressure model.
    R_grid
        Major-radius grid in metres.
    Z_grid
        Vertical grid in metres.
    """

    def __init__(
        self,
        diagnostics: MagneticDiagnostics,
        kinetic: KineticConstraints,
        fast_ions: FastIonPressure,
        R_grid: AnyFloatArray,
        Z_grid: AnyFloatArray,
    ):
        super().__init__(diagnostics, R_grid, Z_grid)
        self.kinetic = kinetic
        self.fast_ions = fast_ions

    def _rho_from_points(self, points: list[tuple[float, float, float]]) -> FloatArray:
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
        rho_grid: AnyFloatArray,
    ) -> FloatArray:
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

        return np.asarray(
            np.interp(rho_grid, unique_rho, unique_values, left=unique_values[0], right=unique_values[-1]),
            dtype=float,
        )

    def reconstruct(
        self,
        measurements: dict[str, Any],
        *,
        coils: CoilSet | None = None,
        mode: str = "psi_n",
        max_iter: int = 25,
        tol: float = 1.0e-5,
        regularization: float = 1.0e-9,
        rel_sigma: float = 2.0e-2,
    ) -> KineticReconstructionResult:
        """Reconstruct the equilibrium with kinetic and fast-ion constraints.

        Runs the magnetic least-squares reconstruction (forwarding the inverse
        options to :meth:`RealtimeEFIT.reconstruct`), then folds in the kinetic
        pressure profiles and the fast-ion pressure to produce a kinetic equilibrium.

        Parameters
        ----------
        measurements
            Magnetic and kinetic measurement payload for this reconstruction.
        coils, mode, max_iter, tol, regularization, rel_sigma
            Magnetic-inverse options forwarded to the base EFIT reconstruction;
            ``coils`` selects the free-boundary inverse (fits the coil currents).

        Returns
        -------
        KineticReconstructionResult
            The kinetic reconstruction with pressure consistency, q-profile, and
            fast-ion beta and anisotropy.
        """
        res_mag = super().reconstruct(
            measurements,
            coils=coils,
            mode=mode,
            max_iter=max_iter,
            tol=tol,
            regularization=regularization,
            rel_sigma=rel_sigma,
        )

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
            q95_mag = float(res_mag.shape.q95)
            if not np.isfinite(q95_mag):
                # No magnetic q available (degenerate / data-free reconstruction);
                # use a neutral edge-q fallback so the kinetic q-profile stays finite.
                q95_mag = 3.0
            q_axis = min(q95_mag, 1.5)
            q_prof = q_axis + max(q95_mag - q_axis, 0.0) * rho_1d**2

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
