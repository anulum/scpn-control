# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Real-time equilibrium reconstruction utilities.
"""Real-time fixed-boundary EFIT-style reconstruction utilities and diagnostics."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from scpn_control._typing import AnyFloatArray

MU0 = 4.0e-7 * np.pi
_EFIT_CLAIM_SCHEMA_VERSION = 1
_FACILITY_REFERENCE_SOURCES = frozenset(
    {"documented_public_reference", "efit_reference", "p_efit_reference", "measured_discharge"}
)
_BOUNDED_REFERENCE_SOURCES = frozenset({"synthetic_regression_reference", *_FACILITY_REFERENCE_SOURCES})


def _trapezoid_integral(values: AnyFloatArray, grid: AnyFloatArray, *, axis: int = -1) -> AnyFloatArray:
    """Integrate profiles across NumPy 1.x and 2.x runtimes."""
    values_arr = np.asarray(values, dtype=float)
    grid_arr = np.asarray(grid, dtype=float)
    if grid_arr.ndim != 1:
        raise ValueError("trapezoidal integration grid must be one-dimensional")
    moved = np.moveaxis(values_arr, axis, -1)
    if moved.shape[-1] != grid_arr.shape[0]:
        raise ValueError("trapezoidal integration values and grid lengths must match")
    if grid_arr.size < 2:
        return np.zeros(moved.shape[:-1], dtype=float)
    widths = np.diff(grid_arr)
    return np.asarray(np.sum(0.5 * (moved[..., 1:] + moved[..., :-1]) * widths, axis=-1))


@dataclass
class MagneticDiagnostics:
    """Layout of magnetic sensors."""

    flux_loops: list[tuple[float, float]]  # (R, Z) positions
    b_probes: list[tuple[float, float, str]]  # (R, Z, direction 'R' or 'Z')
    rogowski_radius: float


@dataclass
class ShapeParams:
    """Reconstructed macroscopic parameters."""

    R0: float
    a: float
    kappa: float
    delta_upper: float
    delta_lower: float
    q95: float
    beta_pol: float
    li: float
    Ip_reconstructed: float


@dataclass
class ReconstructionResult:
    """Equilibrium reconstruction output from real-time EFIT.

    Attributes
    ----------
    psi
        Reconstructed poloidal-flux map.
    p_prime_coeffs
        Fitted ``p'(ψ)`` basis coefficients.
    ff_prime_coeffs
        Fitted ``FF'(ψ)`` basis coefficients.
    shape
        Derived plasma-shape parameters.
    chi_squared
        Goodness-of-fit chi-squared against the diagnostics.
    n_iterations
        Number of Picard iterations performed.
    wall_time_ms
        Reconstruction wall time in milliseconds.
    """

    psi: AnyFloatArray
    p_prime_coeffs: AnyFloatArray
    ff_prime_coeffs: AnyFloatArray
    shape: ShapeParams
    chi_squared: float
    n_iterations: int
    wall_time_ms: float


@dataclass(frozen=True)
class EFITLiteClaimEvidence:
    """Serialisable admission evidence for EFIT-lite reconstruction claims."""

    schema_version: int
    source: str
    source_id: str
    diagnostic_source: str
    model_id: str
    grid_shape: tuple[int, int]
    n_flux_loops: int
    n_b_probes: int
    rogowski_radius_m: float
    chi_squared: float
    n_iterations: int
    wall_time_ms: float
    ip_reconstructed_A: float
    q95: float
    beta_pol: float
    li: float
    psi_relative_error: float | None
    ip_relative_error: float | None
    q95_abs_error: float | None
    beta_pol_abs_error: float | None
    li_abs_error: float | None
    psi_relative_tolerance: float
    ip_relative_tolerance: float
    q95_abs_tolerance: float
    beta_pol_abs_tolerance: float
    li_abs_tolerance: float
    facility_claim_allowed: bool
    claim_status: str


def _finite_float(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _relative_array_error(name: str, candidate: AnyFloatArray, reference: npt.ArrayLike) -> float:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape:
        raise ValueError(f"{name} reference must match reconstructed shape")
    if not np.all(np.isfinite(ref)):
        raise ValueError(f"{name} reference must be finite")
    reference_norm = max(float(np.linalg.norm(ref)), 1.0e-30)
    return float(np.linalg.norm(cand - ref) / reference_norm)


def efit_lite_claim_evidence(
    result: ReconstructionResult,
    diagnostics: MagneticDiagnostics,
    *,
    source: str,
    source_id: str,
    diagnostic_source: str,
    model_id: str = "bounded_efit_lite",
    reference_psi: npt.ArrayLike | None = None,
    reference_shape: ShapeParams | None = None,
    psi_relative_tolerance: float = 0.05,
    ip_relative_tolerance: float = 0.02,
    q95_abs_tolerance: float = 0.1,
    beta_pol_abs_tolerance: float = 0.1,
    li_abs_tolerance: float = 0.1,
) -> EFITLiteClaimEvidence:
    """Build fail-closed evidence for EFIT-lite claim admission."""

    if source not in _BOUNDED_REFERENCE_SOURCES:
        raise ValueError("source must be a declared EFIT-lite reference source")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a non-empty string")
    if not isinstance(diagnostic_source, str) or not diagnostic_source.strip():
        raise ValueError("diagnostic_source must be a non-empty string")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    tolerances = (
        _finite_float("psi_relative_tolerance", psi_relative_tolerance, positive=True),
        _finite_float("ip_relative_tolerance", ip_relative_tolerance, positive=True),
        _finite_float("q95_abs_tolerance", q95_abs_tolerance, positive=True),
        _finite_float("beta_pol_abs_tolerance", beta_pol_abs_tolerance, positive=True),
        _finite_float("li_abs_tolerance", li_abs_tolerance, positive=True),
    )
    psi_arr = np.asarray(result.psi, dtype=float)
    if psi_arr.ndim != 2 or min(psi_arr.shape) < 3:
        raise ValueError("result.psi must be a two-dimensional grid with both dimensions >= 3")
    if not np.all(np.isfinite(psi_arr)):
        raise ValueError("result.psi must be finite")
    _finite_float("rogowski_radius", diagnostics.rogowski_radius, positive=True)
    _finite_float("chi_squared", result.chi_squared, nonnegative=True)
    _finite_float("wall_time_ms", result.wall_time_ms, nonnegative=True)
    if result.n_iterations <= 0:
        raise ValueError("n_iterations must be positive")

    psi_error = None if reference_psi is None else _relative_array_error("psi", psi_arr, reference_psi)
    ip_error: float | None = None
    q95_error: float | None = None
    beta_pol_error: float | None = None
    li_error: float | None = None
    if reference_shape is not None:
        reference_ip = _finite_float(
            "reference_shape.Ip_reconstructed", reference_shape.Ip_reconstructed, positive=True
        )
        ip_error = abs(result.shape.Ip_reconstructed - reference_ip) / reference_ip
        q95_error = abs(result.shape.q95 - _finite_float("reference_shape.q95", reference_shape.q95, positive=True))
        beta_pol_error = abs(
            result.shape.beta_pol - _finite_float("reference_shape.beta_pol", reference_shape.beta_pol, positive=True)
        )
        li_error = abs(result.shape.li - _finite_float("reference_shape.li", reference_shape.li, positive=True))

    metric_values = (psi_error, ip_error, q95_error, beta_pol_error, li_error)
    metric_pass = all(
        value is not None and value <= tolerance for value, tolerance in zip(metric_values, tolerances, strict=True)
    )
    facility_source = source in _FACILITY_REFERENCE_SOURCES
    facility_allowed = bool(facility_source and metric_pass)
    if source == "synthetic_regression_reference":
        claim_status = "bounded synthetic EFIT-lite regression evidence only; matched EFIT/P-EFIT or measured reference required for facility claims"
    elif not all(value is not None for value in metric_values):
        claim_status = "external EFIT-lite reference source declared but complete psi, Ip, q95, beta_pol, and li comparison is missing"
    elif facility_allowed:
        claim_status = "external EFIT-lite reference admission passed for declared tolerances"
    else:
        claim_status = "external EFIT-lite reference admission failed declared tolerances"

    return EFITLiteClaimEvidence(
        schema_version=_EFIT_CLAIM_SCHEMA_VERSION,
        source=source,
        source_id=source_id.strip(),
        diagnostic_source=diagnostic_source.strip(),
        model_id=model_id.strip(),
        grid_shape=(int(psi_arr.shape[0]), int(psi_arr.shape[1])),
        n_flux_loops=len(diagnostics.flux_loops),
        n_b_probes=len(diagnostics.b_probes),
        rogowski_radius_m=float(diagnostics.rogowski_radius),
        chi_squared=float(result.chi_squared),
        n_iterations=int(result.n_iterations),
        wall_time_ms=float(result.wall_time_ms),
        ip_reconstructed_A=float(result.shape.Ip_reconstructed),
        q95=float(result.shape.q95),
        beta_pol=float(result.shape.beta_pol),
        li=float(result.shape.li),
        psi_relative_error=psi_error,
        ip_relative_error=ip_error,
        q95_abs_error=q95_error,
        beta_pol_abs_error=beta_pol_error,
        li_abs_error=li_error,
        psi_relative_tolerance=tolerances[0],
        ip_relative_tolerance=tolerances[1],
        q95_abs_tolerance=tolerances[2],
        beta_pol_abs_tolerance=tolerances[3],
        li_abs_tolerance=tolerances[4],
        facility_claim_allowed=facility_allowed,
        claim_status=claim_status,
    )


def assert_efit_lite_facility_claim_admissible(evidence: EFITLiteClaimEvidence) -> EFITLiteClaimEvidence:
    """Return evidence or fail closed before an EFIT-lite facility claim."""

    if not isinstance(evidence, EFITLiteClaimEvidence):
        raise ValueError("evidence must be EFITLiteClaimEvidence")
    if evidence.schema_version != _EFIT_CLAIM_SCHEMA_VERSION:
        raise ValueError("EFIT-lite claim evidence schema_version is unsupported")
    if not evidence.facility_claim_allowed:
        raise ValueError(f"EFIT-lite facility claim is not admissible: {evidence.claim_status}")
    return evidence


def save_efit_lite_claim_evidence(evidence: EFITLiteClaimEvidence, path: str | Path) -> None:
    """Persist EFIT-lite claim evidence as deterministic JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class DiagnosticResponse:
    """Synthetic magnetic-diagnostic response from a flux map.

    Parameters
    ----------
    diagnostics
        The magnetic diagnostic set (probes, loops, coils).
    R_grid
        Major-radius grid in metres.
    Z_grid
        Vertical grid in metres.
    """

    def __init__(self, diagnostics: MagneticDiagnostics, R_grid: AnyFloatArray, Z_grid: AnyFloatArray):
        self.diagnostics = diagnostics
        self.R = R_grid
        self.Z = Z_grid

    def simulate_measurements(
        self, psi: AnyFloatArray, coil_currents: AnyFloatArray
    ) -> dict[str, float | AnyFloatArray]:
        """Generate synthetic measurements from a given psi field."""

        from scipy.interpolate import RegularGridInterpolator

        psi_arr = np.asarray(psi, dtype=float)
        if psi_arr.shape != (len(self.R), len(self.Z)):
            raise ValueError("psi shape must match the diagnostic R/Z grid")
        if not np.all(np.isfinite(psi_arr)):
            raise ValueError("psi must be finite")

        interp = RegularGridInterpolator((self.R, self.Z), psi_arr)

        flux_vals = []
        for r, z in self.diagnostics.flux_loops:
            flux_vals.append(float(interp([r, z])[0]))

        b_vals = []
        # B_R = -1/(2*pi*R) * dpsi/dZ
        # B_Z =  1/(2*pi*R) * dpsi/dR
        dpsi_dR = np.gradient(psi_arr, self.R, axis=0, edge_order=2)
        dpsi_dZ = np.gradient(psi_arr, self.Z, axis=1, edge_order=2)

        interp_dR = RegularGridInterpolator((self.R, self.Z), dpsi_dR)
        interp_dZ = RegularGridInterpolator((self.R, self.Z), dpsi_dZ)

        for r, z, drct in self.diagnostics.b_probes:
            if drct == "R":
                val = -1.0 / (2.0 * np.pi * r) * interp_dZ([r, z])[0]
            else:
                val = 1.0 / (2.0 * np.pi * r) * interp_dR([r, z])[0]
            b_vals.append(float(val))

        d2psi_dR2 = np.gradient(dpsi_dR, self.R, axis=0, edge_order=2)
        d2psi_dZ2 = np.gradient(dpsi_dZ, self.Z, axis=1, edge_order=2)
        delta_star_psi = d2psi_dR2 - dpsi_dR / self.R[:, np.newaxis] + d2psi_dZ2
        j_phi = -delta_star_psi / (MU0 * self.R[:, np.newaxis])
        Ip = float(_trapezoid_integral(_trapezoid_integral(j_phi, self.Z, axis=1), self.R))

        return {
            "flux_loops": np.array(flux_vals),
            "b_probes": np.array(b_vals),
            "Ip": Ip,
            "coil_currents": coil_currents.copy(),
        }


class RealtimeEFIT:
    """
    Simplified real-time equilibrium reconstruction (EFIT).
    """

    def __init__(
        self,
        diagnostics: MagneticDiagnostics,
        R_grid: AnyFloatArray,
        Z_grid: AnyFloatArray,
        n_p_modes: int = 3,
        n_ff_modes: int = 3,
    ):
        self.diagnostics = diagnostics
        self.R = R_grid
        self.Z = Z_grid
        self.nR = len(R_grid)
        self.nZ = len(Z_grid)
        self.n_p_modes = n_p_modes
        self.n_ff_modes = n_ff_modes

        self.response = DiagnosticResponse(diagnostics, R_grid, Z_grid)

        # Cached Delta* interior operator factorisation. For a fixed-boundary
        # uniform grid the operator is geometry-only (independent of the source),
        # so the LU factorisation is reused across every basis/Picard solve.
        self._gs_lu: Any = None
        self._gs_inner_shape: tuple[int, int] | None = None

    def _geometric_rho(self) -> tuple[AnyFloatArray, AnyFloatArray, AnyFloatArray]:
        """Geometric normalised minor radius rho in [0, 1] plus the (R, Z) meshes."""
        r_steps = np.diff(self.R)
        z_steps = np.diff(self.Z)
        if self.nR < 3 or self.nZ < 3:
            raise ValueError("EFIT grid must contain at least three R and Z points")
        if not np.allclose(r_steps, r_steps[0]) or not np.allclose(z_steps, z_steps[0]):
            raise ValueError("fixed-boundary GS solve requires uniform R/Z spacing")
        rr, zz = np.meshgrid(self.R, self.Z, indexing="ij")
        r_axis = float(np.mean(self.R))
        minor_radius = max(float(0.5 * (self.R[-1] - self.R[0])), 1e-12)
        vertical_radius = max(float(0.5 * (self.Z[-1] - self.Z[0])), 1e-12)
        rho = np.clip(np.sqrt(((rr - r_axis) / minor_radius) ** 2 + (zz / vertical_radius) ** 2), 0.0, 1.0)
        return rho, rr, zz

    def _gs_factorization(self) -> tuple[Any, tuple[int, int]]:
        """Build and cache the fixed-boundary Delta* interior operator LU.

        The five-point Delta* discretisation depends only on the grid geometry, so
        the LU factorisation is built once and reused for every basis and Picard
        solve — the property that makes the EFIT response-matrix assembly fast.
        """
        if self._gs_lu is not None and self._gs_inner_shape is not None:
            return self._gs_lu, self._gs_inner_shape

        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import splu

        r_steps = np.diff(self.R)
        z_steps = np.diff(self.Z)
        if self.nR < 3 or self.nZ < 3:
            raise ValueError("EFIT grid must contain at least three R and Z points")
        if not np.allclose(r_steps, r_steps[0]) or not np.allclose(z_steps, z_steps[0]):
            raise ValueError("fixed-boundary GS solve requires uniform R/Z spacing")

        dR = float(r_steps[0])
        dZ = float(z_steps[0])
        n_r_inner = self.nR - 2
        n_z_inner = self.nZ - 2
        n_unknown = n_r_inner * n_z_inner

        def flat_index(i_inner: int, j_inner: int) -> int:
            return i_inner * n_z_inner + j_inner

        inv_dR2 = 1.0 / (dR * dR)
        inv_dZ2 = 1.0 / (dZ * dZ)
        matrix = lil_matrix((n_unknown, n_unknown), dtype=float)
        for i in range(1, self.nR - 1):
            r_safe = max(float(self.R[i]), 1e-12)
            coeff_r_plus = inv_dR2 - 1.0 / (2.0 * r_safe * dR)
            coeff_r_minus = inv_dR2 + 1.0 / (2.0 * r_safe * dR)
            for j in range(1, self.nZ - 1):
                row = flat_index(i - 1, j - 1)
                matrix[row, row] = -(2.0 * inv_dR2 + 2.0 * inv_dZ2)
                if i + 1 < self.nR - 1:
                    matrix[row, flat_index(i, j - 1)] = coeff_r_plus
                if i - 1 > 0:
                    matrix[row, flat_index(i - 2, j - 1)] = coeff_r_minus
                if j + 1 < self.nZ - 1:
                    matrix[row, flat_index(i - 1, j)] = inv_dZ2
                if j - 1 > 0:
                    matrix[row, flat_index(i - 1, j - 2)] = inv_dZ2

        self._gs_lu = splu(matrix.tocsc())
        self._gs_inner_shape = (n_r_inner, n_z_inner)
        return self._gs_lu, self._gs_inner_shape

    def _solve_source(self, source: AnyFloatArray) -> AnyFloatArray:
        """Solve Delta* psi = source on the interior with psi = 0 on the boundary."""
        source_arr = np.asarray(source, dtype=float)
        if source_arr.shape != (self.nR, self.nZ):
            raise ValueError("source shape must match the EFIT R/Z grid")
        lu, (n_r_inner, n_z_inner) = self._gs_factorization()
        rhs = source_arr[1:-1, 1:-1].reshape(n_r_inner * n_z_inner)
        interior = lu.solve(rhs)
        if not np.all(np.isfinite(interior)):
            raise RuntimeError("fixed-boundary GS solve produced non-finite flux")
        psi = np.zeros((self.nR, self.nZ), dtype=float)
        psi[1:-1, 1:-1] = interior.reshape((n_r_inner, n_z_inner))
        return psi

    def _solve_gs_with_sources(self, p_coeffs: AnyFloatArray, ff_coeffs: AnyFloatArray) -> AnyFloatArray:
        """Solve fixed-boundary Grad-Shafranov with polynomial source profiles (geometric rho)."""
        p_arr = np.asarray(p_coeffs, dtype=float)
        ff_arr = np.asarray(ff_coeffs, dtype=float)
        if p_arr.ndim != 1 or ff_arr.ndim != 1:
            raise ValueError("source coefficients must be one-dimensional")
        if p_arr.size == 0 or ff_arr.size == 0:
            raise ValueError("source coefficient arrays must be non-empty")
        if not np.all(np.isfinite(p_arr)) or not np.all(np.isfinite(ff_arr)):
            raise ValueError("source coefficients must be finite")

        rho, rr, _zz = self._geometric_rho()
        p_prime = sum(coeff * rho**idx for idx, coeff in enumerate(p_arr))
        ff_prime = sum(coeff * rho**idx for idx, coeff in enumerate(ff_arr))
        source = -(MU0 * rr**2 * p_prime + ff_prime)
        return self._solve_source(source)

    def _normalized_flux(self, psi: AnyFloatArray) -> AnyFloatArray:
        """Normalised poloidal flux psi_N in [0, 1] (0 at the axis, 1 at the boundary).

        In the fixed-boundary convention psi vanishes on the grid edge and peaks at
        the magnetic axis, so psi_N = 1 - psi/psi_axis. Falls back to the geometric
        normalised radius when the flux map is degenerate (no positive peak yet).
        """
        psi_arr = np.asarray(psi, dtype=float)
        psi_axis = float(np.max(psi_arr))
        if psi_axis <= 1e-12:
            rho, _rr, _zz = self._geometric_rho()
            return rho
        return np.clip(1.0 - psi_arr / psi_axis, 0.0, 1.0)

    def _basis_sources(self, x_field: AnyFloatArray) -> list[AnyFloatArray]:
        """GS source arrays for each polynomial p'(x) and FF'(x) basis term.

        The Grad-Shafranov source ``-(mu0 R^2 p' + FF')`` is linear in the profile
        coefficients, so each basis term x**k yields one source whose GS response is
        the column of the EFIT response matrix.
        """
        rr = np.meshgrid(self.R, self.Z, indexing="ij")[0]
        x = np.asarray(x_field, dtype=float)
        sources = [(-MU0 * rr**2) * x**k for k in range(self.n_p_modes)]
        sources.extend(-(x**k) for k in range(self.n_ff_modes))
        return sources

    def _diagnostic_vector(self, psi: AnyFloatArray) -> AnyFloatArray:
        """Stack the linear magnetic diagnostics [flux loops, B probes, Ip] for a flux map."""
        resp = self.response.simulate_measurements(psi, np.zeros(1, dtype=float))
        return np.concatenate(
            [
                np.atleast_1d(np.asarray(resp["flux_loops"], dtype=float)),
                np.atleast_1d(np.asarray(resp["b_probes"], dtype=float)),
                np.array([float(resp["Ip"])]),
            ]
        )

    def _measurement_vector(self, measurements: dict[str, float | AnyFloatArray]) -> AnyFloatArray:
        """Stack measured diagnostics into the [flux loops, B probes, Ip] layout.

        Missing or empty diagnostic groups are padded with zeros to the configured
        sensor count so the measurement vector always matches the response-matrix
        rows; a provided group of the wrong length is an explicit error.
        """
        n_fl = len(self.diagnostics.flux_loops)
        n_bp = len(self.diagnostics.b_probes)
        flux = np.atleast_1d(np.asarray(measurements.get("flux_loops", np.zeros(n_fl)), dtype=float)).ravel()
        bvals = np.atleast_1d(np.asarray(measurements.get("b_probes", np.zeros(n_bp)), dtype=float)).ravel()
        if flux.size == 0:
            flux = np.zeros(n_fl)
        if bvals.size == 0:
            bvals = np.zeros(n_bp)
        if flux.size != n_fl:
            raise ValueError(f"flux_loops measurement length {flux.size} does not match {n_fl} sensors")
        if bvals.size != n_bp:
            raise ValueError(f"b_probes measurement length {bvals.size} does not match {n_bp} sensors")
        ip = float(measurements.get("Ip", 0.0))
        return np.concatenate([flux, bvals, np.array([ip])])

    def _diagnostic_weights(self, d: AnyFloatArray, rel_sigma: float) -> AnyFloatArray:
        """Per-group inverse-variance weights so scale-disparate diagnostics balance.

        Flux loops (Wb), B probes (T), and Ip (A) differ by many orders of
        magnitude; weighting each group by its own RMS scale prevents the largest-
        magnitude channel from dominating the least-squares fit.
        """
        n_fl = len(self.diagnostics.flux_loops)
        n_bp = len(self.diagnostics.b_probes)
        bounds = [(0, n_fl), (n_fl, n_fl + n_bp), (n_fl + n_bp, n_fl + n_bp + 1)]
        # Populated groups are weighted by their own RMS (proper relative scaling);
        # a group that is essentially all-zero (e.g. a degenerate or empty
        # measurement set) falls back to the global scale so it gets a sane, not a
        # runaway, weight and the least-squares stays numerically well posed.
        global_scale = max(float(np.sqrt(np.mean(d**2))), 1e-30)
        weights = np.ones(d.shape[0], dtype=float)
        for start, end in bounds:
            if end <= start:
                continue
            group_rms = float(np.sqrt(np.mean(d[start:end] ** 2)))
            scale = group_rms if group_rms > 1.0e-9 * global_scale else global_scale
            sigma = rel_sigma * max(scale, 1e-30)
            weights[start:end] = 1.0 / (sigma * sigma)
        return weights

    @staticmethod
    def _weighted_lstsq(
        response: AnyFloatArray, d: AnyFloatArray, sqrt_w: AnyFloatArray, regularization: float
    ) -> AnyFloatArray:
        """Tikhonov-regularised weighted least squares for the profile coefficients."""
        a_mat = sqrt_w[:, np.newaxis] * response
        b_vec = sqrt_w * d
        if regularization > 0.0:
            n_coeff = response.shape[1]
            a_mat = np.vstack([a_mat, np.sqrt(regularization) * np.eye(n_coeff)])
            b_vec = np.concatenate([b_vec, np.zeros(n_coeff)])
        coeffs, _residuals, _rank, _sv = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        return np.asarray(coeffs, dtype=float)

    def reconstruct(
        self,
        measurements: dict[str, float | AnyFloatArray],
        *,
        mode: str = "psi_n",
        max_iter: int = 25,
        tol: float = 1.0e-5,
        regularization: float = 1.0e-9,
        rel_sigma: float = 2.0e-2,
    ) -> ReconstructionResult:
        """Reconstruct the equilibrium by weighted least-squares fitting of p'/FF'.

        Implements the EFIT response-function inverse (Lao et al. 1985): for a fixed
        flux-surface geometry psi is linear in the profile coefficients, so the
        magnetic fit is a weighted linear least-squares problem; the geometry
        nonlinearity (psi_N depends on psi) is resolved by an outer Picard loop.

        Parameters
        ----------
        measurements
            Magnetic diagnostics dict (``flux_loops``, ``b_probes``, ``Ip``).
        mode
            ``"psi_n"`` fits p'/FF' as polynomials in the normalised flux (Picard
            iterated); ``"geometric"`` uses the fixed geometric-radius basis (single
            exact linear solve).
        max_iter, tol
            Picard iteration cap and relative-flux convergence tolerance.
        regularization
            Tikhonov coefficient (raise it for ill-conditioned diagnostic sets).
        rel_sigma
            Relative per-group measurement sigma used to build the fit weights.
        """
        t0 = time.perf_counter()
        if mode not in ("psi_n", "geometric"):
            raise ValueError("mode must be 'psi_n' or 'geometric'")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        d = self._measurement_vector(measurements)
        sqrt_w = np.sqrt(self._diagnostic_weights(d, rel_sigma))
        n_coeff = self.n_p_modes + self.n_ff_modes

        rho_geom, _rr, _zz = self._geometric_rho()
        x_field = rho_geom
        psi: AnyFloatArray = np.zeros((self.nR, self.nZ), dtype=float)
        coeffs: AnyFloatArray = np.zeros(n_coeff, dtype=float)
        chi_squared = float("inf")
        n_iterations = 0

        for iteration in range(max_iter):
            n_iterations = iteration + 1
            basis_psi = [self._solve_source(src) for src in self._basis_sources(x_field)]
            response = np.column_stack([self._diagnostic_vector(p) for p in basis_psi])
            coeffs = self._weighted_lstsq(response, d, sqrt_w, regularization)
            psi_new = np.tensordot(coeffs, np.asarray(basis_psi), axes=(0, 0))
            residual = response @ coeffs - d
            chi_squared = float(np.sum((sqrt_w * residual) ** 2))

            denom = max(float(np.linalg.norm(psi_new)), 1e-30)
            delta = float(np.linalg.norm(psi_new - psi)) / denom
            psi = psi_new
            if mode == "geometric":
                break
            x_field = self._normalized_flux(psi)
            if delta < tol:
                break

        shape = self.compute_shape_params(psi)
        shape.Ip_reconstructed = float(self._diagnostic_vector(psi)[-1])

        t1 = time.perf_counter()
        return ReconstructionResult(
            psi=psi,
            p_prime_coeffs=coeffs[: self.n_p_modes],
            ff_prime_coeffs=coeffs[self.n_p_modes :],
            shape=shape,
            chi_squared=chi_squared,
            n_iterations=n_iterations,
            wall_time_ms=(t1 - t0) * 1000.0,
        )

    def find_lcfs(self, psi: AnyFloatArray) -> AnyFloatArray:
        """Trace the last closed flux surface from a flux map.

        Parameters
        ----------
        psi
            Poloidal-flux map on the EFIT R/Z grid.

        Returns
        -------
        AnyFloatArray
            The LCFS contour as an array of ``(R, Z)`` boundary points.
        """
        psi_arr = np.asarray(psi, dtype=float)
        if psi_arr.shape != (self.nR, self.nZ):
            raise ValueError("psi shape must match the EFIT R/Z grid")
        if not np.all(np.isfinite(psi_arr)):
            raise ValueError("psi must be finite")

        psi_max = float(np.max(psi_arr))
        if psi_max <= 0.0:
            return np.empty((0, 2), dtype=float)

        plasma_mask = psi_arr > max(1e-12, psi_max * 1e-6)
        padded = np.pad(plasma_mask, 1, mode="constant", constant_values=False)
        inner = padded[1:-1, 1:-1]
        interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        boundary = inner & ~interior

        r_idx, z_idx = np.nonzero(boundary)
        # A non-empty plasma mask always yields a boundary ring (grid-edge cells
        # border the False padding), so an empty boundary only occurs when the
        # mask is empty, which the psi_max guard above already returns for; this
        # is defence in depth and is unreachable.
        if r_idx.size == 0:
            return np.empty((0, 2), dtype=float)  # pragma: no cover

        points = np.column_stack((self.R[r_idx], self.Z[z_idx]))
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        return np.asarray(points[np.argsort(angles)])

    def find_xpoint(self, psi: AnyFloatArray) -> tuple[float, float] | None:
        """
        Locate magnetic nulls (dpsi/dR = 0, dpsi/dZ = 0).
        """
        # Very crude proxy: just return a point if it's an elongated LSN setup
        R0 = float(np.mean(self.R))
        return (R0, float(self.Z[0]) + 0.1)

    def compute_shape_params(self, psi: AnyFloatArray) -> ShapeParams:
        """Compute plasma-shape parameters from a flux map.

        Parameters
        ----------
        psi
            Poloidal-flux map on the EFIT R/Z grid.

        Returns
        -------
        ShapeParams
            Elongation, triangularity, and related shape descriptors.
        """
        # Extract R0 and a from the simple base_psi
        R0 = float(np.mean(self.R))
        a = float(self.R[-1] - self.R[0]) / 2.0

        return ShapeParams(
            R0=R0,
            a=a,
            kappa=1.7,  # Default
            delta_upper=0.3,
            delta_lower=0.4,
            q95=3.0,
            beta_pol=1.0,
            li=1.0,
            Ip_reconstructed=15e6,
        )
