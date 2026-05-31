# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — VMEC-Lite Reduced Equilibrium Solver
"""
Simplified spectral 3D MHD equilibrium solver (VMEC-lite, fixed-boundary).

References
----------
Hirshman, S. P. & Whitson, J. C., Phys. Fluids 26 (1983) 3553.
    Variational moment method for 3D MHD equilibria; spectral representation
    R(s,θ,ζ) = Σ R_mn cos(mθ − nζ),  Z = Σ Z_mn sin(mθ − nζ)  [Eqs. 1–2].
Freidberg, J. P., "Ideal MHD" (Cambridge, 2014), Ch. 3.
    Force balance ∇p = J × B in 3D geometry.
Wesson, J., "Tokamaks" 4th ed. (Oxford, 2011), Ch. 3.
    Toroidal field B_φ ∝ 1/R; poloidal field from rotational transform.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_VMEC_LITE_CLAIM_SCHEMA_VERSION = 1
_FULL_VMEC_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "vmec_reference", "external_mhd_reference", "measured_stellarator"}
)
_BOUNDED_VMEC_REFERENCE_SOURCES = frozenset({"synthetic_regression_reference", *_FULL_VMEC_REFERENCE_SOURCES})


def _positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _profile_array(name: str, values: np.ndarray, *, nonnegative: bool = False, positive: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a one-dimensional profile with at least two points")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative everywhere")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive everywhere")
    return arr


@dataclass
class VMECResult:
    R_mn: np.ndarray
    Z_mn: np.ndarray
    B_mn: np.ndarray
    force_residual: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class VMECLiteClaimEvidence:
    """Serialisable provenance and reference-comparison evidence for VMEC-lite claims."""

    schema_version: int
    source: str
    source_id: str
    geometry_source: str
    profile_source: str
    current_assumption: str
    model_id: str
    n_s: int
    m_pol: int
    n_tor: int
    n_fp: int
    n_modes: int
    force_residual: float
    iterations: int
    converged: bool
    min_major_radius: float
    max_abs_z: float
    pressure_axis: float
    pressure_edge: float
    iota_min: float
    iota_max: float
    q_min: float
    q_max: float
    r_mn_relative_error: float | None
    z_mn_relative_error: float | None
    iota_relative_error: float | None
    residual_tolerance: float
    spectral_relative_tolerance: float
    iota_relative_tolerance: float
    full_vmec_claim_allowed: bool
    claim_status: str


def _non_empty_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _relative_array_error(name: str, observed: np.ndarray, reference: np.ndarray | None) -> float | None:
    if reference is None:
        return None
    obs = np.asarray(observed, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if obs.shape != ref.shape or not np.all(np.isfinite(ref)):
        raise ValueError(f"{name} reference must be finite and match the observed array shape")
    denominator = max(float(np.linalg.norm(ref)), 1e-30)
    return float(np.linalg.norm(obs - ref) / denominator)


def _minimum_sampled_major_radius(result: VMECResult, basis: SpectralBasis) -> tuple[float, float]:
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    min_r = float("inf")
    max_abs_z = 0.0
    for surface_idx in range(result.R_mn.shape[0]):
        r_surface = basis.evaluate(result.R_mn[surface_idx], theta_grid.ravel(), zeta_grid.ravel(), is_sin=False)
        z_surface = basis.evaluate(result.Z_mn[surface_idx], theta_grid.ravel(), zeta_grid.ravel(), is_sin=True)
        min_r = min(min_r, float(np.min(r_surface)))
        max_abs_z = max(max_abs_z, float(np.max(np.abs(z_surface))))
    return min_r, max_abs_z


def vmec_lite_claim_evidence(
    solver: VMECLiteSolver,
    result: VMECResult,
    *,
    source: str,
    source_id: str,
    geometry_source: str,
    profile_source: str,
    current_assumption: str,
    model_id: str = "bounded_vmec_lite",
    reference_R_mn: np.ndarray | None = None,
    reference_Z_mn: np.ndarray | None = None,
    reference_iota: np.ndarray | None = None,
    residual_tolerance: float = 1e-3,
    spectral_relative_tolerance: float = 0.05,
    iota_relative_tolerance: float = 0.02,
) -> VMECLiteClaimEvidence:
    """Build fail-closed evidence for bounded VMEC-lite or full VMEC claims."""

    source_clean = _non_empty_text("source", source)
    if source_clean not in _BOUNDED_VMEC_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(_BOUNDED_VMEC_REFERENCE_SOURCES))
        raise ValueError(f"source must be one of: {allowed}")

    residual_tol = _finite_float("residual_tolerance", residual_tolerance)
    spectral_tol = _finite_float("spectral_relative_tolerance", spectral_relative_tolerance)
    iota_tol = _finite_float("iota_relative_tolerance", iota_relative_tolerance)
    if residual_tol <= 0.0 or spectral_tol <= 0.0 or iota_tol <= 0.0:
        raise ValueError("VMEC-lite claim tolerances must be positive")

    r_mn = np.asarray(result.R_mn, dtype=float)
    z_mn = np.asarray(result.Z_mn, dtype=float)
    b_mn = np.asarray(result.B_mn, dtype=float)
    if (
        r_mn.shape != (solver.n_s, solver.basis.n_modes)
        or z_mn.shape != r_mn.shape
        or b_mn.shape != r_mn.shape
        or not np.all(np.isfinite(r_mn))
        or not np.all(np.isfinite(z_mn))
        or not np.all(np.isfinite(b_mn))
    ):
        raise ValueError("VMEC-lite result arrays must be finite and match the solver spectral grid")

    force_residual = _finite_float("force_residual", result.force_residual)
    if result.iterations < 1:
        raise ValueError("VMEC-lite result iterations must be positive")
    min_major_radius, max_abs_z = _minimum_sampled_major_radius(result, solver.basis)
    if min_major_radius <= 0.0:
        raise ValueError("VMEC-lite claim evidence requires positive sampled major radius")

    pressure = _profile_array("solver.pressure", solver.pressure, nonnegative=True)
    iota = _profile_array("solver.iota", solver.iota, positive=True)
    r_error = _relative_array_error("R_mn", r_mn, reference_R_mn)
    z_error = _relative_array_error("Z_mn", z_mn, reference_Z_mn)
    iota_error = _relative_array_error("iota", iota, reference_iota)
    references_pass = False
    if r_error is not None and z_error is not None and iota_error is not None:
        references_pass = (
            r_error <= spectral_tol
            and z_error <= spectral_tol
            and iota_error <= iota_tol
            and force_residual <= residual_tol
            and result.converged
        )
    full_vmec_claim_allowed = source_clean in _FULL_VMEC_REFERENCE_SOURCES and references_pass
    claim_status = "full_vmec_reference_matched" if full_vmec_claim_allowed else "bounded_vmec_lite_evidence"

    q_profile = 1.0 / np.asarray(iota, dtype=float)
    return VMECLiteClaimEvidence(
        schema_version=_VMEC_LITE_CLAIM_SCHEMA_VERSION,
        source=source_clean,
        source_id=_non_empty_text("source_id", source_id),
        geometry_source=_non_empty_text("geometry_source", geometry_source),
        profile_source=_non_empty_text("profile_source", profile_source),
        current_assumption=_non_empty_text("current_assumption", current_assumption),
        model_id=_non_empty_text("model_id", model_id),
        n_s=solver.n_s,
        m_pol=solver.basis.m_pol,
        n_tor=solver.basis.n_tor,
        n_fp=solver.basis.n_fp,
        n_modes=solver.basis.n_modes,
        force_residual=force_residual,
        iterations=int(result.iterations),
        converged=bool(result.converged),
        min_major_radius=float(min_major_radius),
        max_abs_z=float(max_abs_z),
        pressure_axis=float(pressure[0]),
        pressure_edge=float(pressure[-1]),
        iota_min=float(np.min(iota)),
        iota_max=float(np.max(iota)),
        q_min=float(np.min(q_profile)),
        q_max=float(np.max(q_profile)),
        r_mn_relative_error=r_error,
        z_mn_relative_error=z_error,
        iota_relative_error=iota_error,
        residual_tolerance=residual_tol,
        spectral_relative_tolerance=spectral_tol,
        iota_relative_tolerance=iota_tol,
        full_vmec_claim_allowed=bool(full_vmec_claim_allowed),
        claim_status=claim_status,
    )


def assert_vmec_lite_full_vmec_claim_admissible(evidence: VMECLiteClaimEvidence) -> None:
    """Raise when VMEC-lite evidence is insufficient for a full VMEC claim."""

    if not evidence.full_vmec_claim_allowed:
        raise ValueError("full VMEC claim requires matched R_mn, Z_mn, iota, convergence, and residual evidence")


def save_vmec_lite_claim_evidence(evidence: VMECLiteClaimEvidence, path: str | Path) -> None:
    """Persist VMEC-lite claim evidence as deterministic JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SpectralBasis:
    """Fourier basis for VMEC spectral representation.

    Hirshman & Whitson 1983, Phys. Fluids 26, 3553, Eqs. 1–2:
        R(s,θ,ζ) = Σ_{m,n} R_mn(s) cos(mθ − n N_fp ζ)
        Z(s,θ,ζ) = Σ_{m,n} Z_mn(s) sin(mθ − n N_fp ζ)

    The basis requires non-negative poloidal and toroidal truncation orders and
    a positive integer number of field periods.
    """

    def __init__(self, m_pol: int, n_tor: int, n_fp: int):
        self.m_pol = _positive_int("m_pol", m_pol, minimum=0)
        self.n_tor = _positive_int("n_tor", n_tor, minimum=0)
        self.n_fp = _positive_int("n_fp", n_fp)

        self.mn_modes: list[tuple[int, int]] = []
        for m in range(self.m_pol + 1):
            n_min = -self.n_tor if m > 0 else 0
            for n in range(n_min, self.n_tor + 1):
                self.mn_modes.append((m, n))

        self.n_modes = len(self.mn_modes)

    def evaluate(self, coeffs_mn: np.ndarray, theta: np.ndarray, zeta: np.ndarray, is_sin: bool = False) -> np.ndarray:
        """Evaluate spectral expansion at (θ, ζ) grid points.

        Hirshman & Whitson 1983, Eq. 1–2:
            Σ C_mn cos(mθ − n N_fp ζ)  or  Σ C_mn sin(mθ − n N_fp ζ)
        """
        coeffs_mn = np.asarray(coeffs_mn, dtype=float)
        theta = np.asarray(theta, dtype=float)
        zeta = np.asarray(zeta, dtype=float)
        if coeffs_mn.shape != (self.n_modes,):
            raise ValueError(f"coeffs_mn must have shape ({self.n_modes},)")
        if theta.shape != zeta.shape:
            raise ValueError("theta and zeta must have the same shape")
        if not np.all(np.isfinite(coeffs_mn)):
            raise ValueError("coeffs_mn must contain only finite values")
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(zeta)):
            raise ValueError("theta and zeta must contain only finite values")
        val = np.zeros_like(theta)
        for i, (m, n) in enumerate(self.mn_modes):
            arg = m * theta - n * self.n_fp * zeta
            basis = np.sin(arg) if is_sin else np.cos(arg)
            val += coeffs_mn[i] * basis
        return val


class VMECLiteSolver:
    """Spectral 3D equilibrium solver using steepest descent on the MHD energy.

    Hirshman & Whitson 1983, Phys. Fluids 26, 3553 — variational moment method.
    Force balance ∇p = J × B (Freidberg 2014, Ch. 3) drives the Shafranov shift
    on the R₀₀ mode; radial tension (finite-difference Laplacian) regularises
    the spectral coefficients.

    ``n_s`` must provide at least axis, interior, and boundary surfaces. Profile
    inputs are finite one-dimensional arrays; pressure is non-negative and
    rotational transform is positive for the reduced q-profile calculation.
    """

    def __init__(self, n_s: int = 21, m_pol: int = 3, n_tor: int = 2, n_fp: int = 1):
        self.n_s = _positive_int("n_s", n_s, minimum=3)
        self.basis = SpectralBasis(m_pol, n_tor, n_fp)

        self.R_mn = np.zeros((n_s, self.basis.n_modes))
        self.Z_mn = np.zeros((n_s, self.basis.n_modes))

        self.pressure = np.zeros(n_s)
        self.iota = np.zeros(n_s)

        self.s_grid = np.linspace(0.0, 1.0, n_s)

    def set_boundary(self, R_bound: dict[tuple[int, int], float], Z_bound: dict[tuple[int, int], float]) -> None:
        """Fix finite boundary Fourier coefficients at s = 1."""
        for i, (m, n) in enumerate(self.basis.mn_modes):
            if (m, n) in R_bound:
                self.R_mn[-1, i] = _finite_float(f"R_bound[{(m, n)}]", R_bound[(m, n)])
            if (m, n) in Z_bound:
                self.Z_mn[-1, i] = _finite_float(f"Z_bound[{(m, n)}]", Z_bound[(m, n)])

    def set_profiles(self, pressure: np.ndarray, iota: np.ndarray) -> None:
        """Interpolate finite pressure and rotational-transform profiles onto the solver grid."""
        pressure = _profile_array("pressure", pressure, nonnegative=True)
        iota = _profile_array("iota", iota, positive=True)
        self.pressure = np.interp(self.s_grid, np.linspace(0, 1, len(pressure)), pressure)
        self.iota = np.interp(self.s_grid, np.linspace(0, 1, len(iota)), iota)

    def _initial_guess(self) -> None:
        # Linear radial interpolation from axis to boundary
        # Hirshman & Whitson 1983: s^(m/2) scaling for poloidal harmonics
        idx_00 = self.basis.mn_modes.index((0, 0)) if (0, 0) in self.basis.mn_modes else -1
        R00_bound = self.R_mn[-1, idx_00] if idx_00 >= 0 else 0.0

        for i, (m, n) in enumerate(self.basis.mn_modes):
            if m == 0 and n == 0:
                self.R_mn[:, i] = R00_bound
                self.Z_mn[:, i] = 0.0
            else:
                self.R_mn[:, i] = self.s_grid ** (m / 2.0) * self.R_mn[-1, i]
                self.Z_mn[:, i] = self.s_grid ** (m / 2.0) * self.Z_mn[-1, i]

    def solve(self, max_iter: int = 100, tol: float = 1e-4) -> VMECResult:
        """Steepest-descent equilibrium iteration.

        Hirshman & Whitson 1983, Phys. Fluids 26, 3553.

        Force balance ∇p = J × B (Freidberg 2014, Ch. 3) drives the
        pressure-gradient Shafranov shift on R₀₀ via q²(dp/ds)/R₀.
        Radial tension (d²/ds² finite difference) regularises spectral modes.

        ``max_iter`` must be a positive integer and ``tol`` must be positive and
        finite; invalid controls are rejected before iteration.
        """
        max_iter = _positive_int("max_iter", max_iter)
        tol = _finite_float("tol", tol)
        if tol <= 0.0:
            raise ValueError("tol must be positive")

        self._initial_guess()

        converged = False
        residual = float("inf")
        lr = 0.1

        idx_00 = next((i for i, mn in enumerate(self.basis.mn_modes) if mn == (0, 0)), -1)
        if idx_00 < 0 or self.R_mn[-1, idx_00] <= 0.0:
            raise ValueError("R00 boundary mode must be positive before VMEC-lite solve")
        dp_ds = np.gradient(self.pressure, self.s_grid)
        q_profile = 1.0 / np.maximum(np.abs(self.iota), 0.01)
        R_00_bound = max(abs(self.R_mn[-1, idx_00]), 1e-3) if idx_00 >= 0 else 1.0

        for it in range(max_iter):
            F_R = np.zeros_like(self.R_mn)
            F_Z = np.zeros_like(self.Z_mn)

            for i in range(1, self.n_s - 1):
                F_R[i] = (self.R_mn[i + 1] - 2 * self.R_mn[i] + self.R_mn[i - 1]) * 2.0
                F_Z[i] = (self.Z_mn[i + 1] - 2 * self.Z_mn[i] + self.Z_mn[i - 1]) * 2.0

            # Pressure-driven Shafranov shift: F ~ −q²(dp/ds)/R₀
            # Freidberg 2014, Ideal MHD, Ch. 3 — radial force balance
            if idx_00 >= 0:
                for i in range(1, self.n_s - 1):
                    F_R[i, idx_00] -= q_profile[i] ** 2 * dp_ds[i] / R_00_bound * 1e-6

            residual = float(np.max(np.abs(F_R)) + np.max(np.abs(F_Z)))

            if residual < tol:
                converged = True
                break

            self.R_mn[1:-1] += lr * F_R[1:-1]
            self.Z_mn[1:-1] += lr * F_Z[1:-1]

        # B field from rotational transform ι and geometry
        # B_φ ∝ 1/R (Wesson 2011, Ch. 3), B_θ ∝ ι
        B_mn = np.zeros_like(self.R_mn)
        for s in range(self.n_s):
            R_00 = max(abs(self.R_mn[s, idx_00]), 1e-6) if idx_00 >= 0 else 1.0
            iota_s = self.iota[s]
            if idx_00 >= 0:
                B_mn[s, idx_00] = 1.0
            for k, (m, _n) in enumerate(self.basis.mn_modes):
                if k == idx_00:
                    continue
                B_mn[s, k] = -self.R_mn[s, k] / R_00
                if m == 1:
                    B_mn[s, k] += iota_s * abs(self.Z_mn[s, k]) / R_00

        return VMECResult(self.R_mn.copy(), self.Z_mn.copy(), B_mn, float(residual), it + 1, converged)


class AxisymmetricTokamakBoundary:
    @staticmethod
    def from_parameters(
        R0: float, a: float, kappa: float, delta: float
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        """Low-order Fourier approximation of shaped tokamak boundary.

        R = R₀ + a cos(θ + δ sin θ),  Z = a κ sin θ
        Hirshman & Whitson 1983, Phys. Fluids 26, 3553, Eqs. 1–2.
        """
        R0 = _finite_float("R0", R0)
        a = _finite_float("a", a)
        kappa = _finite_float("kappa", kappa)
        delta = _finite_float("delta", delta)
        if R0 <= 0.0:
            raise ValueError("R0 must be positive")
        if a <= 0.0:
            raise ValueError("a must be positive")
        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if not -1.0 < delta < 1.0:
            raise ValueError("delta must lie in (-1, 1)")
        if R0 <= a * (1.0 + abs(delta)):
            raise ValueError("R0 must keep the axisymmetric boundary major radius positive")
        b_R: dict[tuple[int, int], float] = {
            (0, 0): R0,
            (1, 0): a,
            (2, 0): -0.5 * a * delta,
        }
        b_Z: dict[tuple[int, int], float] = {(1, 0): a * kappa}
        return b_R, b_Z


class StellaratorBoundary:
    @staticmethod
    def w7x_standard() -> tuple[dict, dict]:
        """W7-X standard configuration boundary.

        Grieger et al., Phys. Fluids B 4 (1992) 2081 — modular coil geometry.
        N_fp = 5.
        """
        b_R = {(0, 0): 5.5, (1, 0): 0.5, (1, 1): 0.1, (0, 1): 0.2}
        b_Z = {(1, 0): 0.6, (1, 1): -0.1}
        return b_R, b_Z
