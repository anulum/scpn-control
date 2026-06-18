# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Current-drive source models.
"""ECCD, NBI, and mixed current-drive source models for transport coupling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

# Current-drive profiles and grids are one-dimensional real-valued arrays on the
# normalised radial coordinate. Inputs are normalised with ``asarray(dtype=float)``
# internally, so the contract is any floating-precision array rather than strictly
# float64; this alias gives the annotations explicit generic arguments (satisfying
# ``disallow_any_generics``) without forcing callers to pre-cast their arrays.
FloatArray = npt.NDArray[np.floating[Any]]


class CurrentDriveMetricsDict(TypedDict, total=False):
    total_power_relative_error: float
    total_current_relative_error: float
    deposition_centroid_abs_error: float
    peak_current_density_relative_error: float
    nbi_slowing_down_relative_error: float


class CurrentDriveArtifactDict(TypedDict, total=False):
    source: str
    reference_dataset_id: str
    reference_artifact_sha256: str
    reference_case_count: int
    units: dict[str, str]
    metrics: CurrentDriveMetricsDict


# SI constants
E_CHARGE = 1.602176634e-19  # C
M_E = 9.1093837015e-31  # kg
M_P = 1.67262192369e-27  # kg
EPS_0 = 8.8541878128e-12  # F/m

# Deuterium beam defaults
_A_BEAM_D = 2.0  # dimensionless atomic mass
_Z_BEAM_D = 1.0  # charge state

# Coulomb logarithm for thermal-beam collisions; weakly sensitive to exact value
_LN_LAMBDA = 17.0  # dimensionless — Wesson, Tokamaks, 4th ed., §14.3

# LHCD efficiency range — Fisch 1978, PRL 41, 873; ITER design value ≈ 0.1–0.2
_ETA_LHCD_DEFAULT = 0.15  # A/W / (1e19 m^-3 · keV)^-1 normalised form

# ECCD baseline efficiency — Fisch & Boozer 1980, PRL 45, 720; typical range 0.02–0.05
_ETA_ECCD_DEFAULT = 0.03  # same normalisation as LHCD

# Stix 1972, Plasma Physics 14, 367, Eq. 8 — pre-factor for critical energy
_E_CRIT_PREFACTOR = 14.8  # keV · (A_b/A_i)^(2/3) per keV of T_e

# Stix 1972, Eq. 6 — pre-factor in the ion slowing-down time expression
_TAU_S_PREFACTOR = 3.0 * np.sqrt(2.0 * np.pi)  # = 3√(2π)
_CURRENT_DRIVE_CLAIM_SCHEMA_VERSION = 1
_FACILITY_CURRENT_DRIVE_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "ray_tracing_benchmark", "fokker_planck_benchmark", "measured_deposition_replay"}
)
_BOUNDED_CURRENT_DRIVE_REFERENCE_SOURCES = frozenset(
    {"repository_current_drive_regression", *_FACILITY_CURRENT_DRIVE_REFERENCE_SOURCES}
)


def _require_nonnegative_scalar(name: str, value: float) -> float:
    """Return a finite non-negative scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return scalar


def _require_positive_scalar(name: str, value: float) -> float:
    """Return a finite positive scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return scalar


def _require_unit_interval(name: str, value: float) -> float:
    """Return a finite scalar in [0, 1] or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or not (0.0 <= scalar <= 1.0):
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return scalar


def _require_positive_profile(
    name: str, values: float | FloatArray, shape: tuple[int, ...] | None = None
) -> FloatArray:
    """Return finite positive profile values with optional exact shape."""
    arr = np.asarray(values, dtype=float)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return arr


def _require_current_drive_profiles(
    rho: FloatArray,
    ne_19: FloatArray,
    Te_keV: FloatArray,
    Ti_keV: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None]:
    """Validate grid and kinetic profiles used by current-drive sources."""
    rho_arr = np.asarray(rho, dtype=float)
    if rho_arr.ndim != 1:
        raise ValueError("rho grid must be one-dimensional")
    if not np.all(np.isfinite(rho_arr)):
        raise ValueError("rho grid must be finite")
    if rho_arr.size > 1 and not np.all(np.diff(rho_arr) > 0.0):
        raise ValueError("rho grid must be strictly increasing")
    ne_arr = _require_positive_profile("ne_19", ne_19, rho_arr.shape)
    te_arr = _require_positive_profile("Te_keV", Te_keV, rho_arr.shape)
    ti_arr = None if Ti_keV is None else _require_positive_profile("Ti_keV", Ti_keV, rho_arr.shape)
    return rho_arr, ne_arr, te_arr, ti_arr


def _require_current_drive_grid(rho: FloatArray) -> FloatArray:
    """Validate a one-dimensional finite strictly increasing rho grid."""
    rho_arr = np.asarray(rho, dtype=float)
    if rho_arr.ndim != 1:
        raise ValueError("rho grid must be one-dimensional")
    if not np.all(np.isfinite(rho_arr)):
        raise ValueError("rho grid must be finite")
    if rho_arr.size > 1 and not np.all(np.diff(rho_arr) > 0.0):
        raise ValueError("rho grid must be strictly increasing")
    return rho_arr


def _trapezoid_integral(values: FloatArray, grid: FloatArray) -> float:
    """Integrate one-dimensional profiles across NumPy 1.x and 2.x runtimes."""
    values_arr = np.asarray(values, dtype=float)
    grid_arr = np.asarray(grid, dtype=float)
    if values_arr.ndim != 1 or grid_arr.ndim != 1:
        raise ValueError("trapezoidal integration expects one-dimensional arrays")
    if values_arr.shape[0] != grid_arr.shape[0]:
        raise ValueError("trapezoidal integration values and grid lengths must match")
    if grid_arr.size < 2:
        return 0.0
    widths = np.diff(grid_arr)
    return float(np.sum(0.5 * (values_arr[1:] + values_arr[:-1]) * widths))


def _normalised_radial_deposition(
    rho: FloatArray,
    total_power_w: float,
    rho_centre: float,
    width_rho: float,
) -> FloatArray:
    """Return a finite-width radial deposition kernel conserving total power on the supplied grid."""
    rho_arr = _require_current_drive_grid(rho)
    if rho_arr.size == 0:
        return np.asarray([], dtype=float)
    _require_unit_interval("rho_centre", rho_centre)
    if width_rho <= 0.0 or total_power_w <= 0.0:
        return np.zeros_like(rho_arr, dtype=float)
    _require_positive_scalar("width_rho", width_rho)
    _require_positive_scalar("total_power_w", total_power_w)

    kernel = np.exp(-((rho_arr - rho_centre) ** 2) / (2.0 * width_rho**2))
    norm = _trapezoid_integral(kernel, rho_arr) if rho_arr.size > 1 else float(kernel[0])
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("deposition kernel cannot be normalised on the supplied rho grid")
    return np.asarray(total_power_w * kernel / norm)


@dataclass(frozen=True)
class CurrentDriveClaimEvidence:
    """Serialisable evidence for bounded or externally validated current-drive claims."""

    schema_version: int
    source: str
    source_id: str
    model_id: str
    profile_points: int
    rho_min: float
    rho_max: float
    total_absorbed_power_W: float
    total_driven_current_A: float
    peak_current_density_A_m2: float
    eccd_power_MW: float
    lhcd_power_MW: float
    nbi_power_MW: float
    eccd_eta_cd: float
    lhcd_eta_cd: float
    nbi_beam_energy_keV: float | None
    nbi_slowing_down_time_s: float | None
    nbi_critical_energy_keV: float | None
    current_drive_efficiency_A_W: float
    grid_normalised_power: bool
    reference_source: str | None
    reference_dataset_id: str | None
    reference_artifact_sha256: str | None
    reference_case_count: int | None
    total_power_relative_error: float | None
    total_current_relative_error: float | None
    deposition_centroid_abs_error: float | None
    peak_current_density_relative_error: float | None
    nbi_slowing_down_relative_error: float | None
    total_power_relative_tolerance: float
    total_current_relative_tolerance: float
    deposition_centroid_abs_tolerance: float
    peak_current_density_relative_tolerance: float
    nbi_slowing_down_relative_tolerance: float
    external_claim_allowed: bool
    claim_status: str


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_reference_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and positive")
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _nonnegative_reference_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and non-negative")
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _sha256_text(name: str, value: object) -> str:
    text = _non_empty_text(name, str(value))
    if len(text) != 64 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return text


def _extract_current_drive_reference_artifact(
    reference_artifact: CurrentDriveArtifactDict | None,
) -> tuple[CurrentDriveArtifactDict | None, bool]:
    if reference_artifact is None:
        return None, False
    if not isinstance(reference_artifact, dict):
        raise ValueError("reference_artifact must be a dictionary")
    source = _non_empty_text("reference_artifact.source", str(reference_artifact.get("source", "")))
    if source not in _FACILITY_CURRENT_DRIVE_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_FACILITY_CURRENT_DRIVE_REFERENCE_SOURCES))
        raise ValueError(f"reference_artifact.source must be one of: {allowed}")
    units = reference_artifact.get("units")
    expected_units = {
        "power": "W",
        "current": "A",
        "current_density": "A/m^2",
        "density": "10^19 m^-3",
        "temperature": "keV",
        "rho": "1",
        "time": "s",
        "energy": "keV",
    }
    if not isinstance(units, dict) or any(units.get(key) != unit for key, unit in expected_units.items()):
        raise ValueError("reference_artifact.units must declare current-drive unit contracts")
    _sha256_text("reference_artifact.reference_artifact_sha256", reference_artifact.get("reference_artifact_sha256"))
    case_count = reference_artifact.get("reference_case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count <= 0:
        raise ValueError("reference_artifact.reference_case_count must be a positive integer")
    metrics = reference_artifact.get("metrics")
    tolerances = reference_artifact.get("tolerances")
    if not isinstance(metrics, dict) or not isinstance(tolerances, dict):
        raise ValueError("reference_artifact metrics and tolerances must be dictionaries")
    for metric in (
        "total_power_relative_error",
        "total_current_relative_error",
        "deposition_centroid_abs_error",
        "peak_current_density_relative_error",
        "nbi_slowing_down_relative_error",
    ):
        observed = _nonnegative_reference_scalar(f"reference_artifact.metrics.{metric}", metrics.get(metric))
        tolerance = _positive_reference_scalar(f"reference_artifact.tolerances.{metric}", tolerances.get(metric))
        if observed > tolerance:
            raise ValueError(f"reference_artifact metric {metric} exceeds declared tolerance")
    return reference_artifact, True


def eccd_efficiency(
    Te_keV: float | FloatArray,
    Z_eff: float,
    N_parallel: float,
    eta_0: float = _ETA_ECCD_DEFAULT,
) -> float | FloatArray:
    """
    ECCD figure of merit including T_e scaling and launch-angle factor.

    η_ECCD = η_0 · T_e / (5 + Z_eff) · ξ / (1 + ξ²)
    Prater 2004, Phys. Plasmas 11, 2349, Eq. 5.

    Parameters
    ----------
    Te_keV     : electron temperature [keV]
    Z_eff      : effective charge [dimensionless]
    N_parallel : parallel refractive index ξ [dimensionless]
    eta_0      : baseline coefficient [A/W per (1e19 m^-3 · keV) normalisation]
                 Fisch & Boozer 1980, PRL 45, 720

    Returns
    -------
    η_ECCD in same units as eta_0
    """
    Te_arr = _require_positive_profile("Te_keV", Te_keV)
    Z_eff = _require_positive_scalar("Z_eff", Z_eff)
    eta_0 = _require_nonnegative_scalar("eta_0", eta_0)
    if not np.isfinite(N_parallel):
        raise ValueError("N_parallel must be finite")
    xi = N_parallel
    angle_factor = xi / (1.0 + xi**2)
    result = eta_0 * Te_arr / (5.0 + Z_eff) * angle_factor
    return float(result) if np.ndim(result) == 0 else np.asarray(result)


def nbi_slowing_down_time(
    Te_keV: float | FloatArray,
    ne_19: float | FloatArray,
    A_beam: float = _A_BEAM_D,
    Z_eff: float = 1.5,
) -> float | FloatArray:
    """
    Beam-ion classical slowing-down time on electrons.

    τ_s = 3√(2π) m_i T_e^(3/2) / (4 √(m_e) n_e e^4 ln_Λ Z_eff)
    Stix 1972, Plasma Physics 14, 367, Eq. 6.

    Parameters
    ----------
    Te_keV  : electron temperature [keV]
    ne_19   : electron density [10^19 m^-3]
    A_beam  : beam ion mass number [dimensionless]
    Z_eff   : effective plasma charge [dimensionless]

    Returns
    -------
    τ_s [s]
    """
    Te_arr = _require_positive_profile("Te_keV", Te_keV)
    ne_arr = _require_positive_profile("ne_19", ne_19, Te_arr.shape)
    A_beam = _require_positive_scalar("A_beam", A_beam)
    Z_eff = _require_positive_scalar("Z_eff", Z_eff)
    m_beam = A_beam * M_P
    Te_J = Te_arr * 1e3 * E_CHARGE
    ne = ne_arr * 1e19

    numerator = _TAU_S_PREFACTOR * m_beam * Te_J**1.5
    denominator = 4.0 * np.sqrt(M_E) * ne * E_CHARGE**4 * _LN_LAMBDA * Z_eff
    result = numerator / denominator
    return float(result) if np.ndim(result) == 0 else np.asarray(result)


def nbi_critical_energy(
    Te_keV: float | FloatArray,
    A_beam: float = _A_BEAM_D,
    A_ion: float = 2.0,
) -> float | FloatArray:
    """
    Critical energy separating electron- from ion-dominated beam heating.

    E_crit = 14.8 · T_e · (A_b / A_i)^(2/3)
    Stix 1972, Plasma Physics 14, 367, Eq. 8.

    Parameters
    ----------
    Te_keV : electron temperature [keV]
    A_beam : beam mass number [dimensionless]
    A_ion  : majority-ion mass number [dimensionless]

    Returns
    -------
    E_crit [keV]
    """
    Te_arr = _require_positive_profile("Te_keV", Te_keV)
    A_beam = _require_positive_scalar("A_beam", A_beam)
    A_ion = _require_positive_scalar("A_ion", A_ion)
    result = _E_CRIT_PREFACTOR * Te_arr * (A_beam / A_ion) ** (2.0 / 3.0)
    return float(result) if np.ndim(result) == 0 else np.asarray(result)


class ECCDSource:
    """
    Electron Cyclotron Current Drive.

    Default η_cd: Fisch & Boozer 1980, PRL 45, 720 (theory);
    range 0.02–0.05 A/W depending on launch angle and T_e.
    For angle-resolved efficiency, use eccd_efficiency().
    """

    def __init__(
        self,
        P_ec_MW: float,
        rho_dep: float,
        sigma_rho: float,
        eta_cd: float = _ETA_ECCD_DEFAULT,
    ):
        self.P_ec_MW = _require_nonnegative_scalar("P_ec_MW", P_ec_MW)
        self.rho_dep = _require_unit_interval("rho_dep", rho_dep)
        self.sigma_rho = _require_nonnegative_scalar("sigma_rho", sigma_rho)
        self.eta_cd = _require_nonnegative_scalar("eta_cd", eta_cd)

    def P_absorbed(self, rho: FloatArray) -> FloatArray:
        """Absorbed power per unit rho [W] using a grid-normalised finite-width deposition kernel."""
        return _normalised_radial_deposition(rho, self.P_ec_MW * 1e6, self.rho_dep, self.sigma_rho)

    def j_cd(self, rho: FloatArray, ne_19: FloatArray, Te_keV: FloatArray) -> FloatArray:
        """
        Driven current density [A/m^2].

        j_cd = η_cd · P_abs / (n_e · T_e)
        Fisch & Boozer 1980, PRL 45, 720, normalised form.
        """
        rho_arr, ne_arr, te_arr, _ = _require_current_drive_profiles(rho, ne_19, Te_keV)
        p_abs = self.P_absorbed(rho_arr)
        denom = ne_arr * te_arr
        return np.asarray(self.eta_cd * p_abs / denom)


class NBISource:
    """
    Neutral Beam Injection.

    Current-drive efficiency formula: Ehst & Karney 1991, Nucl. Fusion 31, 1933.
    Slowing-down time: Stix 1972, Plasma Physics 14, 367, Eq. 6.
    """

    def __init__(
        self,
        P_nbi_MW: float,
        E_beam_keV: float,
        rho_tangency: float,
        sigma_rho: float = 0.15,
        A_beam: float = _A_BEAM_D,
        Z_beam: float = _Z_BEAM_D,
    ):
        self.P_nbi_MW = _require_nonnegative_scalar("P_nbi_MW", P_nbi_MW)
        self.E_beam_keV = _require_positive_scalar("E_beam_keV", E_beam_keV)
        self.rho_tangency = _require_unit_interval("rho_tangency", rho_tangency)
        self.sigma_rho = _require_nonnegative_scalar("sigma_rho", sigma_rho)
        self.A_beam = _require_positive_scalar("A_beam", A_beam)
        self.Z_beam = _require_positive_scalar("Z_beam", Z_beam)

    def P_heating(self, rho: FloatArray) -> FloatArray:
        """Beam heating power per unit rho [W] using a grid-normalised finite-width deposition kernel."""
        return _normalised_radial_deposition(rho, self.P_nbi_MW * 1e6, self.rho_tangency, self.sigma_rho)

    def j_cd(
        self,
        rho: FloatArray,
        ne_19: FloatArray,
        Te_keV: FloatArray,
        Ti_keV: FloatArray,
        Z_eff: float = 1.5,
    ) -> FloatArray:
        """
        Beam-driven current density [A/m^2].

        Fast-ion density from steady-state balance: n_fast = P_heat · τ_s / E_beam
        τ_s from Stix 1972, Eq. 6 (nbi_slowing_down_time).
        j_cd = e · n_fast · v_∥ / Z_beam
        Ehst & Karney 1991, Nucl. Fusion 31, 1933.

        Note: Ti_keV enters via Z_eff-dependent collisionality; here held constant.
        """
        rho_arr, ne_arr, te_arr, _ = _require_current_drive_profiles(rho, ne_19, Te_keV, Ti_keV)
        Z_eff = _require_positive_scalar("Z_eff", Z_eff)
        p_heat = self.P_heating(rho_arr)
        m_beam = self.A_beam * M_P
        E_beam_J = self.E_beam_keV * 1e3 * E_CHARGE
        v_parallel = np.sqrt(2.0 * E_beam_J / m_beam)

        j_prof = np.zeros_like(rho_arr, dtype=float)
        for i in range(len(rho_arr)):
            if p_heat[i] <= 0.0:
                continue
            tau_s = nbi_slowing_down_time(
                Te_keV=float(te_arr[i]),
                ne_19=float(ne_arr[i]),
                A_beam=self.A_beam,
                Z_eff=Z_eff,
            )
            n_fast = p_heat[i] * tau_s / E_beam_J
            j_prof[i] = E_CHARGE * n_fast * v_parallel / self.Z_beam

        return j_prof


class LHCDSource:
    """
    Lower Hybrid Current Drive.

    Default η_cd = 0.15 A/W (normalised):
    Fisch 1978, PRL 41, 873 (theory); ITER design range 0.1–0.2.
    """

    def __init__(
        self,
        P_lh_MW: float,
        rho_dep: float,
        sigma_rho: float,
        eta_cd: float = _ETA_LHCD_DEFAULT,
    ):
        self.P_lh_MW = _require_nonnegative_scalar("P_lh_MW", P_lh_MW)
        self.rho_dep = _require_unit_interval("rho_dep", rho_dep)
        self.sigma_rho = _require_nonnegative_scalar("sigma_rho", sigma_rho)
        self.eta_cd = _require_nonnegative_scalar("eta_cd", eta_cd)

    def P_absorbed(self, rho: FloatArray) -> FloatArray:
        return _normalised_radial_deposition(rho, self.P_lh_MW * 1e6, self.rho_dep, self.sigma_rho)

    def j_cd(self, rho: FloatArray, ne_19: FloatArray, Te_keV: FloatArray) -> FloatArray:
        """
        j_cd = η_cd · P_abs / (n_e · T_e)
        Fisch 1978, PRL 41, 873, normalised form.
        """
        rho_arr, ne_arr, te_arr, _ = _require_current_drive_profiles(rho, ne_19, Te_keV)
        p_abs = self.P_absorbed(rho_arr)
        denom = ne_arr * te_arr
        return np.asarray(self.eta_cd * p_abs / denom)


class CurrentDriveMix:
    """Superposition of multiple current-drive sources."""

    def __init__(self, a: float = 1.0):
        self.sources: list[ECCDSource | NBISource | LHCDSource] = []
        self.a = _require_positive_scalar("a", a)  # minor radius [m]

    def add_source(self, source: ECCDSource | NBISource | LHCDSource) -> None:
        self.sources.append(source)

    def total_j_cd(
        self,
        rho: FloatArray,
        ne: FloatArray,
        Te: FloatArray,
        Ti: FloatArray,
    ) -> FloatArray:
        rho_arr, ne_arr, te_arr, ti_arr = _require_current_drive_profiles(rho, ne, Te, Ti)
        assert ti_arr is not None
        j_tot = np.zeros_like(rho_arr, dtype=float)
        for src in self.sources:
            if isinstance(src, NBISource):
                j_tot += src.j_cd(rho_arr, ne_arr, te_arr, ti_arr)
            else:
                j_tot += src.j_cd(rho_arr, ne_arr, te_arr)
        return j_tot

    def total_heating_power(self, rho: FloatArray) -> FloatArray:
        rho_arr = _require_current_drive_grid(rho)
        p_tot = np.zeros_like(rho_arr, dtype=float)
        for src in self.sources:
            if isinstance(src, NBISource):
                p_tot += src.P_heating(rho_arr)
            else:
                p_tot += src.P_absorbed(rho_arr)
        return p_tot

    def total_driven_current(
        self,
        rho: FloatArray,
        ne: FloatArray,
        Te: FloatArray,
        Ti: FloatArray,
    ) -> float:
        """
        I_cd = ∫ j_cd · 2π r dr  [A]

        r = ρ · a,  dr = dρ · a  →  dA = 2π ρ a² dρ
        """
        rho_arr, ne_arr, te_arr, ti_arr = _require_current_drive_profiles(rho, ne, Te, Ti)
        assert ti_arr is not None
        j_tot = self.total_j_cd(rho_arr, ne_arr, te_arr, ti_arr)
        current_density_integrand = j_tot * 2.0 * np.pi * rho_arr * self.a**2
        return _trapezoid_integral(current_density_integrand, rho_arr) if rho_arr.size > 1 else 0.0


def current_drive_claim_evidence(
    mix: CurrentDriveMix,
    *,
    rho: FloatArray,
    ne_19: FloatArray,
    Te_keV: FloatArray,
    Ti_keV: FloatArray,
    source: str,
    source_id: str,
    model_id: str = "bounded_auxiliary_current_drive",
    reference_artifact: CurrentDriveArtifactDict | None = None,
    total_power_relative_tolerance: float = 0.03,
    total_current_relative_tolerance: float = 0.10,
    deposition_centroid_abs_tolerance: float = 0.05,
    peak_current_density_relative_tolerance: float = 0.15,
    nbi_slowing_down_relative_tolerance: float = 0.10,
) -> CurrentDriveClaimEvidence:
    """Build fail-closed evidence for auxiliary current-drive deposition claims."""

    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_CURRENT_DRIVE_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_CURRENT_DRIVE_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")
    rho_arr, ne_arr, te_arr, ti_arr = _require_current_drive_profiles(rho, ne_19, Te_keV, Ti_keV)
    assert ti_arr is not None
    p_tot = mix.total_heating_power(rho_arr)
    j_tot = mix.total_j_cd(rho_arr, ne_arr, te_arr, ti_arr)
    total_power = _trapezoid_integral(p_tot, rho_arr) if rho_arr.size > 1 else float(p_tot[0])
    total_current = mix.total_driven_current(rho_arr, ne_arr, te_arr, ti_arr)
    peak_j = float(np.max(j_tot)) if j_tot.size else 0.0
    eccd_power = sum(src.P_ec_MW for src in mix.sources if isinstance(src, ECCDSource))
    lhcd_power = sum(src.P_lh_MW for src in mix.sources if isinstance(src, LHCDSource))
    nbi_sources = [src for src in mix.sources if isinstance(src, NBISource)]
    nbi_power = sum(src.P_nbi_MW for src in nbi_sources)
    eccd_eta = max((src.eta_cd for src in mix.sources if isinstance(src, ECCDSource)), default=0.0)
    lhcd_eta = max((src.eta_cd for src in mix.sources if isinstance(src, LHCDSource)), default=0.0)
    nbi_energy = nbi_sources[0].E_beam_keV if nbi_sources else None
    te_mean = float(np.mean(te_arr))
    ne_mean = float(np.mean(ne_arr))
    nbi_tau = float(nbi_slowing_down_time(te_mean, ne_mean)) if nbi_sources else None
    nbi_ecrit = float(nbi_critical_energy(te_mean)) if nbi_sources else None
    requested_power = (eccd_power + lhcd_power + nbi_power) * 1.0e6
    grid_normalised = bool(np.isclose(total_power, requested_power, rtol=1.0e-10, atol=1.0e-6))
    artifact, artifact_passed = _extract_current_drive_reference_artifact(reference_artifact)
    external_claim_allowed = bool(source_clean in _FACILITY_CURRENT_DRIVE_REFERENCE_SOURCES and artifact_passed)
    claim_status = (
        "external_current_drive_reference_matched" if external_claim_allowed else "bounded_current_drive_evidence"
    )
    metrics = artifact.get("metrics", {}) if artifact else {}

    return CurrentDriveClaimEvidence(
        schema_version=_CURRENT_DRIVE_CLAIM_SCHEMA_VERSION,
        source=source_clean,
        source_id=_non_empty_text("source_id", source_id),
        model_id=_non_empty_text("model_id", model_id),
        profile_points=int(rho_arr.size),
        rho_min=float(rho_arr[0]),
        rho_max=float(rho_arr[-1]),
        total_absorbed_power_W=total_power,
        total_driven_current_A=float(total_current),
        peak_current_density_A_m2=peak_j,
        eccd_power_MW=float(eccd_power),
        lhcd_power_MW=float(lhcd_power),
        nbi_power_MW=float(nbi_power),
        eccd_eta_cd=float(eccd_eta),
        lhcd_eta_cd=float(lhcd_eta),
        nbi_beam_energy_keV=None if nbi_energy is None else float(nbi_energy),
        nbi_slowing_down_time_s=nbi_tau,
        nbi_critical_energy_keV=nbi_ecrit,
        current_drive_efficiency_A_W=float(total_current / max(total_power, 1.0e-30)),
        grid_normalised_power=grid_normalised,
        reference_source=None if artifact is None else str(artifact["source"]),
        reference_dataset_id=None if artifact is None else str(artifact["reference_dataset_id"]),
        reference_artifact_sha256=None if artifact is None else str(artifact["reference_artifact_sha256"]),
        reference_case_count=None if artifact is None else int(artifact["reference_case_count"]),
        total_power_relative_error=None if artifact is None else float(metrics["total_power_relative_error"]),
        total_current_relative_error=None if artifact is None else float(metrics["total_current_relative_error"]),
        deposition_centroid_abs_error=None if artifact is None else float(metrics["deposition_centroid_abs_error"]),
        peak_current_density_relative_error=None
        if artifact is None
        else float(metrics["peak_current_density_relative_error"]),
        nbi_slowing_down_relative_error=None if artifact is None else float(metrics["nbi_slowing_down_relative_error"]),
        total_power_relative_tolerance=_positive_reference_scalar(
            "total_power_relative_tolerance", total_power_relative_tolerance
        ),
        total_current_relative_tolerance=_positive_reference_scalar(
            "total_current_relative_tolerance", total_current_relative_tolerance
        ),
        deposition_centroid_abs_tolerance=_positive_reference_scalar(
            "deposition_centroid_abs_tolerance", deposition_centroid_abs_tolerance
        ),
        peak_current_density_relative_tolerance=_positive_reference_scalar(
            "peak_current_density_relative_tolerance", peak_current_density_relative_tolerance
        ),
        nbi_slowing_down_relative_tolerance=_positive_reference_scalar(
            "nbi_slowing_down_relative_tolerance", nbi_slowing_down_relative_tolerance
        ),
        external_claim_allowed=external_claim_allowed,
        claim_status=claim_status,
    )


def assert_current_drive_external_claim_admissible(evidence: CurrentDriveClaimEvidence) -> CurrentDriveClaimEvidence:
    """Raise when current-drive evidence is insufficient for external deposition claims."""

    if not isinstance(evidence, CurrentDriveClaimEvidence):
        raise ValueError("evidence must be CurrentDriveClaimEvidence")
    if not evidence.external_claim_allowed:
        raise ValueError(
            "current-drive claim requires matched ray-tracing, Fokker-Planck, or measured deposition evidence"
        )
    return evidence


def save_current_drive_claim_evidence(evidence: CurrentDriveClaimEvidence, path: str | Path) -> None:
    """Persist current-drive claim evidence as deterministic JSON."""

    if not isinstance(evidence, CurrentDriveClaimEvidence):
        raise ValueError("evidence must be CurrentDriveClaimEvidence")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "CurrentDriveClaimEvidence",
    "CurrentDriveMix",
    "ECCDSource",
    "EPS_0",
    "E_CHARGE",
    "LHCDSource",
    "M_E",
    "M_P",
    "NBISource",
    "assert_current_drive_external_claim_admissible",
    "current_drive_claim_evidence",
    "eccd_efficiency",
    "nbi_critical_energy",
    "nbi_slowing_down_time",
    "save_current_drive_claim_evidence",
]
