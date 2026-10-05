# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Real-time equilibrium reconstruction utilities.

"""Geometry and Grad-Shafranov operators for realtime EFIT."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scpn_control._typing import AnyFloatArray
from scpn_control.control.realtime_efit_contracts import MU0, MagneticDiagnostics
from scpn_control.control.realtime_efit_diagnostics import DiagnosticResponse

if TYPE_CHECKING:
    from scpn_control.core.fusion_kernel import CoilSet


class _EFITGridSolver:
    """Simplified real-time equilibrium reconstruction (EFIT)."""

    def __init__(
        self,
        diagnostics: MagneticDiagnostics,
        R_grid: AnyFloatArray,
        Z_grid: AnyFloatArray,
        n_p_modes: int = 3,
        n_ff_modes: int = 3,
        vacuum_rb_phi: float = 33.0,
    ):
        self.diagnostics = diagnostics
        self.R = R_grid
        self.Z = Z_grid
        self.nR = len(R_grid)
        self.nZ = len(Z_grid)
        self.n_p_modes = n_p_modes
        self.n_ff_modes = n_ff_modes
        # Vacuum toroidal-field flux function F_edge = R0 * B_phi0 [T m]; the
        # magnetic inverse fixes the poloidal field but not the toroidal field, so
        # the (externally known) vacuum TF is required to evaluate q and B_phi.
        self.vacuum_rb_phi = float(vacuum_rb_phi)

        self.response = DiagnosticResponse(diagnostics, R_grid, Z_grid)

        # Cached Delta* interior operator factorisation. For a fixed-boundary
        # uniform grid the operator is geometry-only (independent of the source),
        # so the LU factorisation is reused across every basis/Picard solve.
        self._gs_lu: Any = None
        self._gs_inner_shape: tuple[int, int] | None = None
        # Cached von Hagenow source-to-boundary-flux operator (free-boundary only),
        # geometry-only like the LU, so built once on first free-boundary solve.
        self._freespace_op: tuple[AnyFloatArray, tuple[AnyFloatArray, AnyFloatArray]] | None = None

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

    def _coil_flux_columns(self, coils: CoilSet) -> list[AnyFloatArray]:
        """Per-coil vacuum poloidal-flux maps on the EFIT grid (unit current).

        Each column is ``turns_j`` times the axisymmetric toroidal Green's function
        of coil ``j`` evaluated over the grid, reusing
        :func:`FusionKernel._green_function_array`. These are the free-boundary
        response columns for the coil currents — geometry-only and independent of
        the plasma source, so they are assembled once per reconstruction.
        """
        from scpn_control.core.fusion_kernel import FusionKernel

        positions = list(coils.positions)
        turns = list(coils.turns) if len(coils.turns) else [1] * len(positions)
        if len(turns) != len(positions):
            raise ValueError("CoilSet turns length must match positions")
        rr, zz = np.meshgrid(self.R, self.Z, indexing="ij")
        columns: list[AnyFloatArray] = []
        for (r_c, z_c), n_turn in zip(positions, turns, strict=True):
            green = FusionKernel._green_function_array(float(r_c), float(z_c), rr, zz)
            columns.append(np.asarray(float(n_turn) * np.asarray(green, dtype=float), dtype=float))
        return columns

    def _boundary_node_indices(self) -> tuple[AnyFloatArray, AnyFloatArray]:
        """``(i, j)`` index arrays of the grid-perimeter nodes (top/bottom rows, then side columns)."""
        i_idx: list[int] = []
        j_idx: list[int] = []
        for i in range(self.nR):
            i_idx.extend((i, i))
            j_idx.extend((0, self.nZ - 1))
        for j in range(1, self.nZ - 1):
            i_idx.extend((0, self.nR - 1))
            j_idx.extend((j, j))
        return np.asarray(i_idx, dtype=int), np.asarray(j_idx, dtype=int)

    def _freespace_boundary_operator(self) -> tuple[AnyFloatArray, tuple[AnyFloatArray, AnyFloatArray]]:
        """Return the cached source-to-boundary-flux operator for the von Hagenow free-space BC.

        Returns ``(G_eff, (bi, bj))`` such that the free-space poloidal flux the
        plasma current produces on the grid boundary is ``G_eff @ source.ravel()``,
        where ``G_eff[b, c] = green(node_b, cell_c) * (-dR dZ / (mu0 R_c))`` maps the
        Grad-Shafranov source ``source = Delta* psi = -mu0 R j_phi`` (so ``j_phi =
        -source/(mu0 R)``) through the toroidal Green's function. Geometry-only.
        """
        if self._freespace_op is not None:
            return self._freespace_op
        from scpn_control.core.fusion_kernel import FusionKernel

        r_steps = np.diff(self.R)
        z_steps = np.diff(self.Z)
        if not np.allclose(r_steps, r_steps[0]) or not np.allclose(z_steps, z_steps[0]):
            raise ValueError("free-boundary GS solve requires uniform R/Z spacing")
        rr, zz = np.meshgrid(self.R, self.Z, indexing="ij")
        r_flat = rr.ravel()
        z_flat = zz.ravel()
        d_r = float(r_steps[0])
        d_z = float(z_steps[0])
        cell_factor = -d_r * d_z / (MU0 * np.maximum(r_flat, 1e-12))
        bi, bj = self._boundary_node_indices()
        g_eff = np.empty((bi.size, r_flat.size), dtype=float)
        for k in range(bi.size):
            green = FusionKernel._green_function_array(
                float(self.R[int(bi[k])]), float(self.Z[int(bj[k])]), r_flat, z_flat
            )
            g_eff[k] = np.asarray(green, dtype=float) * cell_factor
        self._freespace_op = (g_eff, (bi, bj))
        return self._freespace_op

    def _solve_source_with_bc(self, source: AnyFloatArray, psi_bc: AnyFloatArray) -> AnyFloatArray:
        """Solve ``Delta* psi = source`` on the interior with Dirichlet ``psi = psi_bc`` on the boundary.

        Reuses the cached fixed-boundary interior LU; the non-zero boundary values
        enter the interior right-hand side through the stencil links the operator
        omits at the boundary (so ``psi_bc = 0`` reproduces :meth:`_solve_source`).
        """
        source_arr = np.asarray(source, dtype=float)
        bc_arr = np.asarray(psi_bc, dtype=float)
        if source_arr.shape != (self.nR, self.nZ) or bc_arr.shape != (self.nR, self.nZ):
            raise ValueError("source and psi_bc shapes must match the EFIT R/Z grid")
        lu, (n_r_inner, n_z_inner) = self._gs_factorization()
        d_r = float(self.R[1] - self.R[0])
        d_z = float(self.Z[1] - self.Z[0])
        inv_dr2 = 1.0 / (d_r * d_r)
        inv_dz2 = 1.0 / (d_z * d_z)
        r_inner = np.maximum(np.asarray(self.R[1:-1], dtype=float), 1e-12)
        coeff_r_minus = inv_dr2 + 1.0 / (2.0 * r_inner * d_r)  # link to the (i-1) neighbour
        coeff_r_plus = inv_dr2 - 1.0 / (2.0 * r_inner * d_r)  # link to the (i+1) neighbour

        rhs = source_arr[1:-1, 1:-1].copy()
        rhs[0, :] -= coeff_r_minus[0] * bc_arr[0, 1:-1]
        rhs[-1, :] -= coeff_r_plus[-1] * bc_arr[-1, 1:-1]
        rhs[:, 0] -= inv_dz2 * bc_arr[1:-1, 0]
        rhs[:, -1] -= inv_dz2 * bc_arr[1:-1, -1]

        interior = lu.solve(rhs.reshape(n_r_inner * n_z_inner))
        if not np.all(np.isfinite(interior)):
            raise RuntimeError("free-boundary GS solve produced non-finite flux")
        psi = np.zeros((self.nR, self.nZ), dtype=float)
        psi[0, :] = bc_arr[0, :]
        psi[-1, :] = bc_arr[-1, :]
        psi[:, 0] = bc_arr[:, 0]
        psi[:, -1] = bc_arr[:, -1]
        psi[1:-1, 1:-1] = interior.reshape((n_r_inner, n_z_inner))
        return psi

    def _solve_source_freespace(self, source: AnyFloatArray) -> AnyFloatArray:
        """Free-boundary GS solve: ``Delta* psi = source`` with the von Hagenow free-space BC.

        The boundary flux is the free-space flux the plasma current produces (not
        pinned to zero), so the plasma flux decays correctly — the free-boundary
        counterpart of :meth:`_solve_source`.
        """
        source_arr = np.asarray(source, dtype=float)
        if source_arr.shape != (self.nR, self.nZ):
            raise ValueError("source shape must match the EFIT R/Z grid")
        g_eff, (bi, bj) = self._freespace_boundary_operator()
        boundary_vals = g_eff @ source_arr.ravel()
        psi_bc = np.zeros((self.nR, self.nZ), dtype=float)
        psi_bc[bi, bj] = boundary_vals
        return self._solve_source_with_bc(source_arr, psi_bc)

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
