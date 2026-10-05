# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-time equilibrium reconstruction utilities.

"""Inverse reconstruction and shape extraction for realtime EFIT."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from scpn_control._typing import AnyFloatArray
from scpn_control.control.realtime_efit_contracts import ReconstructionResult, ShapeParams
from scpn_control.control.realtime_efit_solver import _EFITGridSolver
from scpn_control.control.realtime_efit_topology import find_magnetic_xpoint
from scpn_control.core.equilibrium_shape import compute_equilibrium_shape
from scpn_control.core.fusion_kernel import CoilSet


class RealtimeEFIT(_EFITGridSolver):
    """Reconstruct equilibrium from diagnostic measurements."""

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
            variance = sigma * sigma
            if not np.isfinite(variance) or variance <= 0.0:
                raise ValueError("diagnostic weights must be finite and positive")
            weights[start:end] = 1.0 / variance
        return weights

    @staticmethod
    def _weighted_lstsq(
        response: AnyFloatArray, d: AnyFloatArray, sqrt_w: AnyFloatArray, regularization: float
    ) -> AnyFloatArray:
        """Tikhonov-regularised weighted least squares for the fit coefficients."""
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
        coils: CoilSet | None = None,
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
        coils
            Optional external :class:`CoilSet`. When supplied, the reconstruction is
            free-boundary: the plasma basis uses the von Hagenow free-space boundary
            condition, the coil currents become additional unknowns (toroidal
            Green's-function columns), and the total flux is ``psi_plasma +
            psi_coil``. When ``None`` the fixed-boundary inverse is used.
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
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if isinstance(tol, bool) or not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be finite and positive")
        if isinstance(regularization, bool) or not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be finite and non-negative")
        if isinstance(rel_sigma, bool) or not np.isfinite(rel_sigma) or rel_sigma <= 0.0:
            raise ValueError("rel_sigma must be finite and positive")

        d = self._measurement_vector(measurements)
        if not np.all(np.isfinite(d)):
            raise ValueError("measurements must be finite")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            weights = self._diagnostic_weights(d, rel_sigma)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("diagnostic weights must be finite and positive")
        sqrt_w = np.sqrt(weights)
        n_coeff = self.n_p_modes + self.n_ff_modes

        free_boundary = coils is not None
        # Coil flux columns are geometry-only, so they are assembled once and reused
        # across every Picard iteration. The diagnostic columns are scaled by a
        # reference current (the plasma Ip) so the fitted coil coefficients are
        # order-1 like the p'/FF' coefficients, keeping the shared Tikhonov penalty
        # scale-fair (coil currents are ~MA, profile coefficients are ~1).
        coil_cols = self._coil_flux_columns(coils) if coils is not None else []
        i_scale = max(abs(float(measurements.get("Ip", 0.0))), 1.0)
        coil_diag = i_scale * np.column_stack([self._diagnostic_vector(g) for g in coil_cols]) if coil_cols else None

        rho_geom, _rr, _zz = self._geometric_rho()
        x_field = rho_geom
        psi: AnyFloatArray = np.zeros((self.nR, self.nZ), dtype=float)
        coeffs: AnyFloatArray = np.zeros(n_coeff, dtype=float)
        coil_currents: AnyFloatArray | None = None
        chi_squared = float("inf")
        n_iterations = 0
        iteration_status: Literal["geometric_solve", "picard_converged", "picard_limit"] = "picard_limit"
        final_relative_change: float | None = None

        for iteration in range(max_iter):
            n_iterations = iteration + 1
            sources = self._basis_sources(x_field)
            if free_boundary:
                basis_psi = [self._solve_source_freespace(src) for src in sources]
            else:
                basis_psi = [self._solve_source(src) for src in sources]
            plasma_diag = np.column_stack([self._diagnostic_vector(p) for p in basis_psi])
            response = np.column_stack([plasma_diag, coil_diag]) if coil_diag is not None else plasma_diag

            params = self._weighted_lstsq(response, d, sqrt_w, regularization)
            coeffs = params[:n_coeff]
            psi_plasma = np.tensordot(coeffs, np.asarray(basis_psi), axes=(0, 0))
            psi_new: AnyFloatArray
            if free_boundary:
                coil_currents = params[n_coeff:] * i_scale
                psi_new = np.asarray(
                    psi_plasma + np.tensordot(coil_currents, np.asarray(coil_cols), axes=(0, 0)), dtype=float
                )
            else:
                psi_new = psi_plasma
            residual = response @ params - d
            chi_squared = float(np.sum((sqrt_w * residual) ** 2))

            denom = max(float(np.linalg.norm(psi_new)), 1e-30)
            delta = float(np.linalg.norm(psi_new - psi)) / denom
            psi = psi_new
            if mode == "geometric":
                iteration_status = "geometric_solve"
                break
            final_relative_change = delta
            x_field = self._normalized_flux(psi)
            if delta < tol:
                iteration_status = "picard_converged"
                break

        shape = self.compute_shape_params(psi, coeffs[: self.n_p_modes], coeffs[self.n_p_modes :])

        t1 = time.perf_counter()
        return ReconstructionResult(
            psi=psi,
            p_prime_coeffs=coeffs[: self.n_p_modes],
            ff_prime_coeffs=coeffs[self.n_p_modes :],
            shape=shape,
            chi_squared=chi_squared,
            n_iterations=n_iterations,
            wall_time_ms=(t1 - t0) * 1000.0,
            coil_currents=coil_currents,
            iteration_status=iteration_status,
            final_relative_change=final_relative_change,
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
        points = np.column_stack((self.R[r_idx], self.Z[z_idx]))
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        return np.asarray(points[np.argsort(angles)])

    def find_xpoint(self, psi: AnyFloatArray) -> tuple[float, float] | None:
        """Estimate an interior magnetic saddle from the flux geometry."""
        return find_magnetic_xpoint(psi, self.R, self.Z)

    def compute_shape_params(
        self,
        psi: AnyFloatArray,
        p_coeffs: AnyFloatArray | None = None,
        ff_coeffs: AnyFloatArray | None = None,
    ) -> ShapeParams:
        """Compute plasma-shape parameters from a reconstructed flux map.

        Delegates the macroscopic descriptors to the reusable
        :func:`scpn_control.core.equilibrium_shape.compute_equilibrium_shape`:
        R0/minor radius/elongation/triangularity from the boundary contour,
        ``li(3)`` from the poloidal-field volume integral, poloidal beta from the
        fitted pressure profile, and q95 from the toroidal flux function. The
        plasma current is taken from the flux map; a degenerate (no-plasma) map
        returns geometric defaults.

        Parameters
        ----------
        psi
            Poloidal-flux map on the EFIT R/Z grid.
        p_coeffs, ff_coeffs
            Fitted ``p'(psi_N)`` / ``FF'(psi_N)`` coefficients (needed for beta_pol
            and q95); default to zeros when called without a reconstruction.
        """
        psi_arr = np.asarray(psi, dtype=float)
        ip = float(self._diagnostic_vector(psi_arr)[-1])
        p_arr = np.zeros(self.n_p_modes) if p_coeffs is None else np.asarray(p_coeffs, dtype=float)
        ff_arr = np.zeros(self.n_ff_modes) if ff_coeffs is None else np.asarray(ff_coeffs, dtype=float)

        shape = compute_equilibrium_shape(psi_arr, self.R, self.Z, p_arr, ff_arr, ip, self.vacuum_rb_phi)
        if shape is None:
            return ShapeParams(
                R0=float(np.mean(self.R)),
                a=float(self.R[-1] - self.R[0]) / 2.0,
                kappa=1.0,
                delta_upper=0.0,
                delta_lower=0.0,
                q95=float("nan"),
                beta_pol=0.0,
                li=0.0,
                Ip_reconstructed=ip,
            )
        return ShapeParams(
            R0=shape.R0,
            a=shape.a,
            kappa=shape.kappa,
            delta_upper=shape.delta_upper,
            delta_lower=shape.delta_lower,
            q95=shape.q95,
            beta_pol=shape.beta_pol,
            li=shape.li,
            Ip_reconstructed=ip,
        )
