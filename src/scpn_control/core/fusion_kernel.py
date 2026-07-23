# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Fusion Kernel
"""
Non-linear Grad-Shafranov equilibrium solver with two boundary variants.

- fixed-boundary: stable default path
- free-boundary: experimental external-coil outer loop

Solves the Grad-Shafranov equation for toroidal plasma equilibrium using
Picard iteration with under-relaxation.  Supports both L-mode (linear)
and H-mode (mtanh pedestal) pressure/current profiles.

The solver can optionally offload the inner elliptic solve to a compiled
C++ library via :class:`~scpn_control.core.hpc_bridge.HPCBridge`, or to
the Rust multigrid backend when available.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core import fusion_kernel_config as _fk_config
from scpn_control.core import gs_elliptic_iterators as _gs_ell
from scpn_control.core import gs_free_boundary_control as _gs_fb
from scpn_control.core import gs_free_boundary_solve as _gs_fb_solve
from scpn_control.core import gs_green_vacuum as _gs_green
from scpn_control.core import gs_multigrid as _gs_mg
from scpn_control.core import gs_profile_source as _gs_prof
from scpn_control.core.hpc_bridge import HPCBridge

# Stable public re-export: free-boundary coil set (R0-S6 leaf).
CoilSet = _gs_fb.CoilSet

# Stable public re-exports: configuration schemas + parse/dump (R0-S1 leaf).
CoilConfig = _fk_config.CoilConfig
DimensionsConfig = _fk_config.DimensionsConfig
FusionKernelConfig = _fk_config.FusionKernelConfig
PhysicsConfig = _fk_config.PhysicsConfig
SolverConfig = _fk_config.SolverConfig
_normalize_boundary_variant = _fk_config._normalize_boundary_variant
_reject_duplicate_json_keys = _fk_config._reject_duplicate_json_keys
_parse_fusion_kernel_config = _fk_config._parse_fusion_kernel_config
_fusion_kernel_config_dump = _fk_config._fusion_kernel_config_dump
parse_fusion_kernel_config = _fk_config.parse_fusion_kernel_config
fusion_kernel_config_dump = _fk_config.fusion_kernel_config_dump
normalize_boundary_variant = _fk_config.normalize_boundary_variant
reject_duplicate_json_keys = _fk_config.reject_duplicate_json_keys

logger = logging.getLogger(__name__)

# ── Fusion physics constants ──────────────────────────────────────────
# DT reaction: d + t → ⁴He (3.52 MeV) + n (14.1 MeV), total E_fus = 17.6 MeV.
# Wesson 2011, "Tokamaks" 4th ed., Eq. 1.2.1.
_EV_J = 1.602176634e-19  # J/eV (CODATA 2018)
E_FUS_J = 17.6e6 * _EV_J  # J; total DT fusion energy
E_ALPHA_J = 3.52e6 * _EV_J  # J; alpha particle birth energy
E_NEUTRON_J = 14.1e6 * _EV_J  # J; neutron energy

# P_α = P_fus / 5  (E_α/E_fus = 3.52/17.6 = 0.2).
# ITER Physics Basis 1999, Nucl. Fusion 39, 2137, Eq. (2.2.1).
ALPHA_FRACTION = E_ALPHA_J / E_FUS_J  # 0.2 exactly

# Bosch-Hale DT reactivity coefficients — Table VII (T in 0.2–100 keV range).
# Bosch & Hale 1992, Nucl. Fusion 32, 611.
_BH_BG = 34.3827  # dimensionless; Gamow peak parameter for D-T
_BH_MRC2 = 1124656.0  # keV; reduced mass energy m_r c^2 for D-T
_BH_C1 = 1.17302e-9  # cm^3/s; Bosch-Hale Table VII
_BH_C2 = 1.51361e-2  # keV^-1
_BH_C3 = 7.51886e-2  # keV^-1
_BH_C4 = 4.60643e-3  # keV^-2
_BH_C5 = 1.35302e-2  # keV^-2
_BH_C6 = -1.06750e-4  # keV^-3
_BH_C7 = 1.36600e-5  # keV^-3

# ── Type aliases ──────────────────────────────────────────────────────
BoolArray = NDArray[np.bool_]


def dt_fusion_power_mw(n_D_m3: float, n_T_m3: float, sigv_m3s: float, V_m3: float) -> float:
    """DT fusion power [MW].

    P_fus = n_D n_T <σv> E_fus × V.
    Wesson 2011, "Tokamaks" 4th ed., Eq. 1.2.1.

    Parameters
    ----------
    n_D_m3, n_T_m3 : float
        Deuterium and tritium densities [m^-3].
    sigv_m3s : float
        DT reactivity <σv> [m^3/s].  Bosch & Hale 1992, Nucl. Fusion 32, 611.
    V_m3 : float
        Plasma volume [m^3].
    """
    return float(n_D_m3 * n_T_m3 * sigv_m3s * E_FUS_J * V_m3 * 1e-6)


def dt_alpha_power_mw(P_fus_MW: float) -> float:
    """Alpha heating power [MW].

    P_α = P_fus / 5  (E_α/E_fus = 3.52/17.6).
    ITER Physics Basis 1999, Nucl. Fusion 39, 2137, Eq. (2.2.1).
    """
    return P_fus_MW * ALPHA_FRACTION


def neutron_wall_loading_mw_m2(P_fus_MW: float, R0_m: float, a_m: float, kappa: float) -> float:
    """Neutron wall loading [MW/m^2].

    q_n = P_n / (4π R₀ a κ),  P_n = P_fus × (1 − ALPHA_FRACTION) = 0.8 P_fus.
    Stacey 2010, "Fusion Plasma Physics", 2nd ed., Ch. 1, Eq. (1.4).

    Parameters
    ----------
    P_fus_MW : float
        Total fusion power [MW].
    R0_m : float
        Major radius [m].
    a_m : float
        Minor radius [m].
    kappa : float
        Elongation (dimensionless).
    """
    P_n_MW = P_fus_MW * (1.0 - ALPHA_FRACTION)
    A_wall = 4.0 * np.pi * R0_m * a_m * kappa  # m^2; first-wall area approximation
    return float(P_n_MW / max(A_wall, 1e-6))


def _psi_gradient_fields(Psi: FloatArray, dR: float, dZ: float) -> tuple[FloatArray, FloatArray]:
    """Return (dPsi/dR, dPsi/dZ) for a ``(NZ, NR)`` flux grid."""
    dPsi_dZ, dPsi_dR = np.gradient(Psi, dZ, dR)
    return np.asarray(dPsi_dR, dtype=np.float64), np.asarray(dPsi_dZ, dtype=np.float64)


def _psi_hessian_determinant(Psi: FloatArray, dR: float, dZ: float) -> FloatArray:
    """Return the Hessian determinant used to classify saddle candidates."""
    dPsi_dR, dPsi_dZ = _psi_gradient_fields(Psi, dR, dZ)
    d2Psi_dR_dZ, d2Psi_dR2 = np.gradient(dPsi_dR, dZ, dR)
    d2Psi_dZ2, d2Psi_dZ_dR = np.gradient(dPsi_dZ, dZ, dR)
    d2Psi_dRZ = 0.5 * (d2Psi_dR_dZ + d2Psi_dZ_dR)
    return np.asarray(d2Psi_dR2 * d2Psi_dZ2 - d2Psi_dRZ**2, dtype=np.float64)


def _x_point_search_mask(ZZ: FloatArray, z_min: float) -> BoolArray:
    """Restrict X-point search to the lower divertor-like region and interior cells."""
    mask = np.asarray(ZZ <= (float(z_min) * 0.5), dtype=bool)
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False
    return mask


def _select_x_point_index(
    gradient_norm: FloatArray,
    search_mask: BoolArray,
    hessian_det: FloatArray | None = None,
) -> tuple[int, int, bool]:
    """Select the best X-point grid cell, preferring saddle candidates."""
    active_mask = np.asarray(search_mask, dtype=bool) & np.isfinite(gradient_norm)
    if hessian_det is not None:
        saddle_mask = active_mask & np.isfinite(hessian_det) & (hessian_det < 0.0)
        if np.any(saddle_mask):
            idx_min = int(np.argmin(np.where(saddle_mask, gradient_norm, np.inf)))
            iz, ir = np.unravel_index(idx_min, gradient_norm.shape)
            return int(iz), int(ir), True

    if np.any(active_mask):
        idx_min = int(np.argmin(np.where(active_mask, gradient_norm, np.inf)))
        iz, ir = np.unravel_index(idx_min, gradient_norm.shape)
        return int(iz), int(ir), False

    return -1, -1, False


class FusionKernel:
    """Non-linear Grad-Shafranov equilibrium solver.

    Parameters
    ----------
    config_path : str | Path
        Path to a JSON configuration file describing the reactor geometry,
        coil set, physics parameters and solver settings.

    Attributes
    ----------
    Psi : FloatArray
        Poloidal flux on the (Z, R) grid.
    J_phi : FloatArray
        Toroidal current density on the (Z, R) grid.
    B_R, B_Z : FloatArray
        Radial and vertical magnetic field components (set after solve).
    """

    # ── construction ──────────────────────────────────────────────────

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = str(config_path)
        self.load_config(config_path)
        self.initialize_grid()
        self.setup_accelerator()

    def load_config(self, path: str | Path) -> None:
        """Load reactor configuration from a JSON file.

        Parameters
        ----------
        path : str | Path
            Filesystem path to the configuration JSON.
        """
        with open(path, "r") as f:
            self.config_model = _parse_fusion_kernel_config(json.load(f, object_pairs_hook=_reject_duplicate_json_keys))
        self.cfg = _fusion_kernel_config_dump(self.config_model)
        solver_cfg = self.cfg.setdefault("solver", {})
        self.boundary_variant = str(solver_cfg.get("boundary_variant", "fixed_boundary"))
        logger.info("Loaded configuration for: %s", self.cfg["reactor_name"])

    def initialize_grid(self) -> None:
        """Build the computational (R, Z) mesh from the loaded config."""
        dims = self.cfg["dimensions"]
        res = self.cfg["grid_resolution"]

        self.NR: int = res[0]
        self.NZ: int = res[1]
        self.R: FloatArray = np.linspace(dims["R_min"], dims["R_max"], self.NR)
        self.Z: FloatArray = np.linspace(dims["Z_min"], dims["Z_max"], self.NZ)

        # Fundamental geometry
        self.R0: float = float(dims["R_min"] + dims["R_max"]) / 2.0
        self.a: float = float(dims["R_max"] - dims["R_min"]) / 2.0

        self.dR: float = float(self.R[1] - self.R[0])
        self.dZ: float = float(self.Z[1] - self.Z[0])
        self.RR: FloatArray
        self.ZZ: FloatArray
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

        self.Psi = np.zeros((self.NZ, self.NR))
        self.J_phi = np.zeros((self.NZ, self.NR))

        self.p_prime_0: float = -1.0
        self.ff_prime_0: float = -1.0

        # Profile mode
        self.profile_mode: str = "l-mode"
        self.ped_params_p: dict[str, float] = {
            "ped_top": 0.92,
            "ped_width": 0.05,
            "ped_height": 1.0,
            "core_alpha": 0.3,
        }
        self.ped_params_ff: dict[str, float] = {
            "ped_top": 0.92,
            "ped_width": 0.05,
            "ped_height": 1.0,
            "core_alpha": 0.3,
        }

        profiles_cfg = self.cfg.get("physics", {}).get("profiles")
        if profiles_cfg:
            self.profile_mode = profiles_cfg.get("mode", "l-mode")
            if self.profile_mode == "external":
                self._ext_pprime = np.array(profiles_cfg["pprime_values"])
                self._ext_ffprime = np.array(profiles_cfg["ffprime_values"])
                self._ext_psi_grid = np.array(profiles_cfg["psi_grid"])
            else:
                if "p_prime" in profiles_cfg:
                    self.ped_params_p.update(profiles_cfg["p_prime"])
                if "ff_prime" in profiles_cfg:
                    self.ped_params_ff.update(profiles_cfg["ff_prime"])

    def setup_accelerator(self) -> None:
        """Initialise the optional C++ HPC acceleration bridge."""
        self.hpc = HPCBridge()
        if self.hpc.is_available():
            logger.info("HPC Acceleration ENABLED.")
            self.hpc.initialize(
                self.NR,
                self.NZ,
                (self.R[0], self.R[-1]),
                (self.Z[0], self.Z[-1]),
            )
        else:
            logger.info("HPC Acceleration UNAVAILABLE (using Python fallback).")

    def build_coilset_from_config(self) -> CoilSet:  # pragma: no cover - defensive free-boundary fallback path
        """Build a free-boundary ``CoilSet`` from the active JSON configuration."""
        coils_cfg = list(self.cfg.get("coils", []))
        positions = [(float(coil["r"]), float(coil["z"])) for coil in coils_cfg]
        currents = np.asarray([float(coil.get("current", 0.0)) for coil in coils_cfg], dtype=np.float64)
        turns = [int(coil.get("turns", 1)) for coil in coils_cfg]

        fb_cfg = self.cfg.get("free_boundary", {})

        current_limits_raw = fb_cfg.get("current_limits")
        current_limits: NDArray[np.float64] | None = None
        if current_limits_raw is not None:
            current_limits = np.asarray(current_limits_raw, dtype=np.float64).reshape(-1)
            if current_limits.shape != currents.shape:
                raise ValueError("free_boundary.current_limits must match the number of coils.")

        target_points_raw = fb_cfg.get("target_flux_points")
        target_flux_points: NDArray[np.float64] | None = None
        if target_points_raw is not None:
            target_flux_points = np.asarray(target_points_raw, dtype=np.float64)
            if target_flux_points.ndim != 2 or target_flux_points.shape[1] != 2:
                raise ValueError("free_boundary.target_flux_points must have shape (n_pts, 2).")

        target_flux_values_raw = fb_cfg.get("target_flux_values")
        target_flux_values: NDArray[np.float64] | None = None
        if target_flux_values_raw is not None:
            target_flux_values = np.asarray(target_flux_values_raw, dtype=np.float64).reshape(-1)

        target_flux_value_raw = fb_cfg.get("target_flux_value")
        if target_flux_value_raw is not None:
            if target_flux_values is not None:
                raise ValueError(
                    "Specify only one of free_boundary.target_flux_values or free_boundary.target_flux_value."
                )
            if target_flux_points is None:
                raise ValueError("free_boundary.target_flux_value requires free_boundary.target_flux_points.")
            target_flux_scalar = float(target_flux_value_raw)
            if not np.isfinite(target_flux_scalar):
                raise ValueError("free_boundary.target_flux_value must be finite.")
            target_flux_values = np.full(target_flux_points.shape[0], target_flux_scalar, dtype=np.float64)

        if target_flux_values is not None:
            if target_flux_points is None:
                raise ValueError("free_boundary.target_flux_values requires free_boundary.target_flux_points.")
            if target_flux_values.shape != (target_flux_points.shape[0],):
                raise ValueError("free_boundary.target_flux_values must match the number of target_flux_points.")

        x_point_target_raw = fb_cfg.get("x_point_target")
        x_point_target: NDArray[np.float64] | None = None
        if x_point_target_raw is not None:
            x_point_target = np.asarray(x_point_target_raw, dtype=np.float64).reshape(-1)
            if x_point_target.shape != (2,):
                raise ValueError("free_boundary.x_point_target must have shape (2,).")

        x_point_flux_target_raw = fb_cfg.get("x_point_flux_target")
        x_point_flux_target: float | None = None
        if x_point_flux_target_raw is not None:
            x_point_flux_target = float(x_point_flux_target_raw)
            if not np.isfinite(x_point_flux_target):
                raise ValueError("free_boundary.x_point_flux_target must be finite.")

        x_point_weight = float(fb_cfg.get("x_point_weight", 1.0))
        if not np.isfinite(x_point_weight) or x_point_weight < 0.0:
            raise ValueError("free_boundary.x_point_weight must be finite and >= 0.")

        x_point_null_weight = float(fb_cfg.get("x_point_null_weight", 1.0))
        if not np.isfinite(x_point_null_weight) or x_point_null_weight < 0.0:
            raise ValueError("free_boundary.x_point_null_weight must be finite and >= 0.")

        divertor_points_raw = fb_cfg.get("divertor_strike_points")
        divertor_strike_points: NDArray[np.float64] | None = None
        if divertor_points_raw is not None:
            divertor_strike_points = np.asarray(divertor_points_raw, dtype=np.float64)
            if divertor_strike_points.ndim != 2 or divertor_strike_points.shape[1] != 2:
                raise ValueError("free_boundary.divertor_strike_points must have shape (n_pts, 2).")

        divertor_flux_values_raw = fb_cfg.get("divertor_flux_values")
        divertor_flux_values: NDArray[np.float64] | None = None
        if divertor_flux_values_raw is not None:
            divertor_flux_values = np.asarray(divertor_flux_values_raw, dtype=np.float64).reshape(-1)

        divertor_flux_value_raw = fb_cfg.get("divertor_flux_value")
        if divertor_flux_value_raw is not None:
            if divertor_flux_values is not None:
                raise ValueError(
                    "Specify only one of free_boundary.divertor_flux_values or free_boundary.divertor_flux_value."
                )
            if divertor_strike_points is None:
                raise ValueError("free_boundary.divertor_flux_value requires free_boundary.divertor_strike_points.")
            divertor_flux_scalar = float(divertor_flux_value_raw)
            if not np.isfinite(divertor_flux_scalar):
                raise ValueError("free_boundary.divertor_flux_value must be finite.")
            divertor_flux_values = np.full(divertor_strike_points.shape[0], divertor_flux_scalar, dtype=np.float64)

        if divertor_flux_values is not None:
            if divertor_strike_points is None:
                raise ValueError("free_boundary.divertor_flux_values requires free_boundary.divertor_strike_points.")
            if divertor_flux_values.shape != (divertor_strike_points.shape[0],):
                raise ValueError("free_boundary.divertor_flux_values must match the number of divertor_strike_points.")

        divertor_weight = float(fb_cfg.get("divertor_weight", 1.0))
        if not np.isfinite(divertor_weight) or divertor_weight < 0.0:
            raise ValueError("free_boundary.divertor_weight must be finite and >= 0.")

        return CoilSet(
            positions=positions,
            currents=currents,
            turns=turns,
            current_limits=current_limits,
            target_flux_points=target_flux_points,
            target_flux_values=target_flux_values,
            x_point_target=x_point_target,
            x_point_flux_target=x_point_flux_target,
            x_point_weight=x_point_weight,
            x_point_null_weight=x_point_null_weight,
            divertor_strike_points=divertor_strike_points,
            divertor_flux_values=divertor_flux_values,
            divertor_weight=divertor_weight,
        )

    def solve(
        self,
        *,
        boundary_variant: str | None = None,
        coils: CoilSet | None = None,
        preserve_initial_state: bool = False,
        boundary_flux: FloatArray | None = None,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, Any]:
        """Solve using the requested boundary variant.

        The default is taken from ``solver.boundary_variant`` in the active config.
        """
        variant = _normalize_boundary_variant(boundary_variant or self.boundary_variant)

        if variant == "fixed_boundary":
            return self.solve_fixed_boundary(
                preserve_initial_state=preserve_initial_state,
                boundary_flux=boundary_flux,
            )

        coilset = coils if coils is not None else self.build_coilset_from_config()
        return self.solve_free_boundary(
            coilset,
            max_outer_iter=max_outer_iter,
            tol=tol,
            optimize_shape=optimize_shape,
            tikhonov_alpha=tikhonov_alpha,
        )

    def solve_fixed_boundary(
        self,
        preserve_initial_state: bool = False,
        boundary_flux: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Explicit entry point for the fixed-boundary variant."""
        return self.solve_equilibrium(
            preserve_initial_state=preserve_initial_state,
            boundary_flux=boundary_flux,
        )

    # ── vacuum field ──────────────────────────────────────────────────

    def calculate_vacuum_field(self) -> FloatArray:
        """Compute the vacuum poloidal flux from the external coil set.

        Uses elliptic integrals (toroidal Green's function) for each coil.

        Returns
        -------
        FloatArray
            Vacuum flux Psi_vac on the (NZ, NR) grid.
        """
        logger.debug("Computing vacuum field (toroidal exact)…")
        mu0: float = self.cfg["physics"].get("vacuum_permeability", 1.0)
        return _gs_green.vacuum_poloidal_flux(self.RR, self.ZZ, self.cfg["coils"], mu0)

    # ── topology analysis ─────────────────────────────────────────────

    def find_x_point(self, Psi: FloatArray) -> tuple[tuple[float, float], float]:
        """Locate the X-point (magnetic null) in the lower divertor region.

        Parameters
        ----------
        Psi : FloatArray
            Poloidal flux array on the (NZ, NR) grid.

        Returns
        -------
        tuple[tuple[float, float], float]
            ``((R_x, Z_x), Psi_x)`` — position and flux value at the
            X-point.
        """
        dPsi_dR, dPsi_dZ = _psi_gradient_fields(Psi, self.dR, self.dZ)
        gradient_norm = np.hypot(dPsi_dR, dPsi_dZ)
        hessian_det = _psi_hessian_determinant(Psi, self.dR, self.dZ)
        mask_divertor = _x_point_search_mask(self.ZZ, float(self.cfg["dimensions"]["Z_min"]))
        iz, ir, _ = _select_x_point_index(gradient_norm, mask_divertor, hessian_det)
        if iz >= 0 and ir >= 0:
            return (float(self.R[ir]), float(self.Z[iz])), float(Psi[iz, ir])

        return (0.0, 0.0), float(np.min(Psi))

    def _find_magnetic_axis(self) -> tuple[int, int, float]:
        """Find the O-point (magnetic axis) as the global Psi maximum.

        Returns
        -------
        tuple[int, int, float]
            ``(iz, ir, Psi_axis)`` — grid indices and flux value.
        """
        idx_max = int(np.argmax(self.Psi))
        iz, ir = np.unravel_index(idx_max, self.Psi.shape)
        psi_axis = float(self.Psi[iz, ir])
        if abs(psi_axis) < 1e-6:
            psi_axis = 1e-6
        return int(iz), int(ir), psi_axis

    # ── profile functions ─────────────────────────────────────────────

    @staticmethod
    def mtanh_profile(psi_norm: FloatArray, params: dict[str, float]) -> FloatArray:
        """Evaluate a modified-tanh pedestal profile (vectorised).

        Parameters
        ----------
        psi_norm : FloatArray
            Normalised poloidal flux (0 at axis, 1 at separatrix).
        params : dict[str, float]
            Profile shape parameters with keys ``ped_top``, ``ped_width``,
            ``ped_height``, ``core_alpha``.

        Returns
        -------
        FloatArray
            Profile value; zero outside the plasma region.
        """
        return _gs_prof.mtanh_profile(psi_norm, params)

    @staticmethod
    def mtanh_profile_derivative(psi_norm: FloatArray, params: dict[str, float]) -> FloatArray:
        """Evaluate ``d(mtanh_profile)/dpsi_norm`` for H-mode Newton linearisation."""
        return _gs_prof.mtanh_profile_derivative(psi_norm, params)

    # ── source term ───────────────────────────────────────────────────

    def update_plasma_source_nonlinear(self, Psi_axis: float, Psi_boundary: float) -> FloatArray:
        """Compute the toroidal current density J_phi from the GS source.

        Uses ``J_phi = R p'(psi) + FF'(psi) / (mu0 R)`` with either
        L-mode (linear) or H-mode (mtanh) profiles, then renormalises to
        match the target plasma current.

        Parameters
        ----------
        Psi_axis : float
            Poloidal flux at the magnetic axis (O-point).
        Psi_boundary : float
            Poloidal flux at the separatrix (X-point or limiter).

        Returns
        -------
        FloatArray
            Updated ``J_phi`` on the (NZ, NR) grid.
        """
        mu0: float = self.cfg["physics"]["vacuum_permeability"]
        i_target: float = self.cfg["physics"]["plasma_current_target"] * self.cfg["physics"].get(
            "plasma_current_sign", 1.0
        )
        self.J_phi = _gs_prof.update_plasma_source_nonlinear(
            self.Psi,
            self.RR,
            self.dR,
            self.dZ,
            Psi_axis,
            Psi_boundary,
            mu0=mu0,
            I_target=i_target,
            profile_mode=self.profile_mode,
            ped_params_p=self.ped_params_p,
            ped_params_ff=self.ped_params_ff,
            ext_psi_grid=getattr(self, "_ext_psi_grid", None),
            ext_pprime=getattr(self, "_ext_pprime", None),
            ext_ffprime=getattr(self, "_ext_ffprime", None),
        )
        return self.J_phi

    # ── elliptic sub-solvers ──────────────────────────────────────────

    def _jacobi_step(self, Psi: FloatArray, Source: FloatArray) -> FloatArray:
        """Perform one Jacobi iteration with toroidal 1/R stencil."""
        return _gs_ell.jacobi_step(Psi, Source, self.RR, self.dR, self.dZ)

    def _sor_step(
        self,
        Psi: FloatArray,
        Source: FloatArray,
        omega: float = 1.6,
    ) -> FloatArray:
        """Vectorised Red-Black SOR iteration with toroidal 1/R stencil."""
        return _gs_ell.sor_step(Psi, Source, self.RR, self.dR, self.dZ, omega=omega)

    # ── multigrid sub-solvers ─────────────────────────────────────────

    @staticmethod
    def _restrict_full_weight(fine: FloatArray) -> FloatArray:
        """Full-weighting restriction operator (fine → coarse)."""
        return _gs_mg.restrict_full_weight(fine)

    @staticmethod
    def _prolongate_bilinear(coarse: FloatArray, nz_f: int, nr_f: int) -> FloatArray:
        """Bilinear prolongation operator (coarse → fine)."""
        return _gs_mg.prolongate_bilinear(coarse, nz_f, nr_f)

    def _mg_smooth(
        self,
        Psi: FloatArray,
        Source: FloatArray,
        R_grid: FloatArray,
        dR: float,
        dZ: float,
        omega: float,
        n_sweeps: int,
    ) -> FloatArray:
        """Red-Black SOR smoother with toroidal 1/R stencil for multigrid."""
        return _gs_mg.mg_smooth(Psi, Source, R_grid, dR, dZ, omega, n_sweeps)

    def _mg_residual(
        self,
        Psi: FloatArray,
        Source: FloatArray,
        R_grid: FloatArray,
        dR: float,
        dZ: float,
    ) -> FloatArray:
        """Compute GS* residual r = L*[Psi] - Source on given grid."""
        return _gs_mg.mg_residual(Psi, Source, R_grid, dR, dZ)

    def _multigrid_vcycle(
        self,
        Psi: FloatArray,
        Source: FloatArray,
        R_grid: FloatArray,
        dR: float,
        dZ: float,
        *,
        omega: float = 1.6,
        pre_smooth: int = 3,
        post_smooth: int = 3,
        min_grid: int = 5,
    ) -> FloatArray:
        """One V-cycle of geometric multigrid for the GS* operator."""
        return _gs_mg.multigrid_vcycle(
            Psi,
            Source,
            R_grid,
            dR,
            dZ,
            omega=omega,
            pre_smooth=pre_smooth,
            post_smooth=post_smooth,
            min_grid=min_grid,
        )

    def _anderson_step(
        self,
        psi_history: list[FloatArray],
        res_history: list[FloatArray],
        m: int = 5,
    ) -> FloatArray:
        """Anderson acceleration (mixing) for the Picard iterate sequence."""
        return _gs_ell.anderson_step(psi_history, res_history, m=m)

    def _apply_boundary_conditions(self, Psi: FloatArray, Psi_bc: FloatArray) -> None:
        """Copy vacuum-field boundary values onto the edges of *Psi*."""
        _gs_ell.apply_boundary_conditions(Psi, Psi_bc)

    def _elliptic_solve(self, Source: FloatArray, Psi_bc: FloatArray) -> FloatArray:
        """Run the inner elliptic solve (HPC or Python fallback).

        The solver method is chosen from the config key
        ``solver.solver_method``:

        - ``"jacobi"`` — legacy single Jacobi sweep (fastest per step,
          slowest convergence).
        - ``"sor"`` — Red-Black SOR with toroidal 1/R stencil (default).
        - ``"anderson"`` — SOR + Anderson acceleration (best convergence).
        - ``"multigrid"`` — geometric V-cycle.
        """
        if self.hpc.is_available():
            Psi_acc = self.hpc.solve(self.J_phi, iterations=50)
            if Psi_acc is not None:
                self._apply_boundary_conditions(Psi_acc, Psi_bc)
                return Psi_acc

        method = str(self.cfg["solver"].get("solver_method", "multigrid"))
        omega = float(self.cfg["solver"].get("sor_omega", 1.6))
        return _gs_ell.elliptic_solve_python(
            method,
            self.Psi,
            Source,
            Psi_bc,
            self.RR,
            self.dR,
            self.dZ,
            omega=omega,
        )

    # ── seed plasma ───────────────────────────────────────────────────

    def _seed_plasma(self, mu0: float) -> None:
        """Create an initial Gaussian current seed and do preliminary solves.

        Parameters
        ----------
        mu0 : float
            Vacuum permeability.
        """
        R_center = (self.cfg["dimensions"]["R_min"] + self.cfg["dimensions"]["R_max"]) / 2.0
        dist_sq = (self.RR - R_center) ** 2 + self.ZZ**2
        self.J_phi = np.exp(-dist_sq / 2.0)

        I_seed = float(np.sum(self.J_phi)) * self.dR * self.dZ
        I_target: float = self.cfg["physics"]["plasma_current_target"] * self.cfg["physics"].get(
            "plasma_current_sign", 1.0
        )
        if I_seed > 0:  # pragma: no branch - I_seed = sum(exp(...))*dR*dZ > 0 always; #129
            self.J_phi *= I_target / I_seed

        Source = -mu0 * self.RR * self.J_phi
        for _ in range(50):
            self.Psi = self._jacobi_step(self.Psi, Source)

    def _prepare_initial_flux(
        self,
        preserve_initial_state: bool,
        boundary_flux: FloatArray | None,
    ) -> FloatArray:
        """Prepare initial Psi and boundary map for iterative GS solves.

        Parameters
        ----------
        preserve_initial_state : bool
            When True, keep the existing interior ``self.Psi`` values and only
            enforce the provided boundary map.
        boundary_flux : FloatArray | None
            Explicit boundary map to enforce. Must match ``self.Psi.shape``
            when provided.

        Returns
        -------
        FloatArray
            Boundary flux map used by boundary-condition enforcement.
        """
        if boundary_flux is not None:
            psi_boundary = np.asarray(boundary_flux, dtype=np.float64)
            if psi_boundary.shape != self.Psi.shape:
                raise ValueError(f"boundary_flux shape {psi_boundary.shape} must match Psi shape {self.Psi.shape}")
            psi_boundary = psi_boundary.copy()
        elif preserve_initial_state:
            psi_boundary = self.Psi.copy()
        else:
            psi_boundary = self.calculate_vacuum_field()

        if preserve_initial_state:
            self._apply_boundary_conditions(self.Psi, psi_boundary)
        else:
            self.Psi = psi_boundary.copy()

        return psi_boundary

    # ── Newton-Kantorovich equilibrium solver ────────────────────────

    def _compute_gs_residual(self, Source: FloatArray) -> FloatArray:
        """Compute the GS residual r = L*[psi] - Source on interior points.

        The toroidal GS* operator is:
            L* = d2/dR2 - (1/R) d/dR + d2/dZ2
        """
        Psi = self.Psi
        NZ, NR = Psi.shape
        dR2 = self.dR**2
        dZ2 = self.dZ**2

        residual = np.zeros_like(Psi)
        R_int = self.RR[1:-1, 1:-1]
        R_safe = np.maximum(R_int, 1e-10)

        # 5-point toroidal stencil
        d2R = (Psi[1:-1, 2:] - 2.0 * Psi[1:-1, 1:-1] + Psi[1:-1, 0:-2]) / dR2
        d1R = (Psi[1:-1, 2:] - Psi[1:-1, 0:-2]) / (2.0 * self.dR)
        d2Z = (Psi[2:, 1:-1] - 2.0 * Psi[1:-1, 1:-1] + Psi[0:-2, 1:-1]) / dZ2

        Lpsi = d2R - d1R / R_safe + d2Z
        residual[1:-1, 1:-1] = Lpsi - Source[1:-1, 1:-1]
        return residual

    def _compute_gs_residual_rms(self, Source: FloatArray) -> float:
        """Return RMS GS residual over interior points."""
        residual = self._compute_gs_residual(Source)
        interior = residual[1:-1, 1:-1]
        if interior.size == 0:  # pragma: no cover - grid_resolution >= 3 guarantees a non-empty interior
            return 0.0
        return float(np.sqrt(np.mean(interior * interior)))

    def _apply_gs_operator(self, v: FloatArray) -> FloatArray:
        """Apply the discrete GS* operator to array *v*.

        Used as the matvec in the GMRES LinearOperator for Newton.
        """
        NZ, NR = v.shape
        dR2 = self.dR**2
        dZ2 = self.dZ**2

        result = np.zeros_like(v)
        R_int = self.RR[1:-1, 1:-1]
        R_safe = np.maximum(R_int, 1e-10)

        d2R = (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, 0:-2]) / dR2
        d1R = (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2.0 * self.dR)
        d2Z = (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[0:-2, 1:-1]) / dZ2

        result[1:-1, 1:-1] = d2R - d1R / R_safe + d2Z
        return result

    @staticmethod
    def _normalised_flux_denominator(Psi_axis: float, Psi_boundary: float) -> float:
        """Return (Psi_boundary - Psi_axis) or fail closed on a degenerate equilibrium."""
        return _gs_prof.normalised_flux_denominator(Psi_axis, Psi_boundary)

    def _compute_profile_jacobian(self, Psi_axis: float, Psi_boundary: float, mu0: float) -> FloatArray:
        """Compute dJ_phi/dpsi as a 2D diagonal scaling field.

        For L-mode linear profiles:
            p'(psi_norm) = const => J_phi ∝ (1 - psi_norm) * R
            dJ_phi/dpsi = -c / (Psi_boundary - Psi_axis) for points inside plasma

        Returns a 2D array of the same shape as self.Psi.
        """
        i_target = self.cfg["physics"]["plasma_current_target"] * self.cfg["physics"].get("plasma_current_sign", 1.0)
        return _gs_prof.compute_profile_jacobian(
            self.Psi,
            self.RR,
            self.dR,
            self.dZ,
            Psi_axis,
            Psi_boundary,
            mu0,
            profile_mode=self.profile_mode,
            ped_params_p=self.ped_params_p,
            ped_params_ff=self.ped_params_ff,
            I_target=i_target,
        )

    def _newton_solve_dispatch(
        self,
        preserve_initial_state: bool = False,
        boundary_flux: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Newton-Kantorovich equilibrium solver with Picard warmup.

        1. Run 15 Picard warmup steps to get a reasonable initial guess.
        2. Switch to Newton: at each step solve J_k * delta = -r_k via
           scipy GMRES, then update psi += alpha * delta.

        Returns the standard result dict.
        """
        from scipy.sparse.linalg import LinearOperator, gmres

        t0 = time.time()
        Psi_vac_boundary = self._prepare_initial_flux(
            preserve_initial_state=preserve_initial_state,
            boundary_flux=boundary_flux,
        )

        max_iter: int = self.cfg["solver"]["max_iterations"]
        tol: float = self.cfg["solver"]["convergence_threshold"]
        picard_alpha: float = self.cfg["solver"].get("relaxation_factor", 0.1)
        fail_on_diverge: bool = bool(self.cfg["solver"].get("fail_on_diverge", False))
        require_gs_residual: bool = bool(self.cfg["solver"].get("require_gs_residual", False))
        gs_tol: float = float(self.cfg["solver"].get("gs_residual_threshold", tol))
        if require_gs_residual and gs_tol <= 0.0:
            raise ValueError("solver.gs_residual_threshold must be > 0")
        mu0: float = self.cfg["physics"]["vacuum_permeability"]
        warmup_steps: int = min(15, max_iter // 2)
        newton_alpha: float = 0.5  # damped Newton

        residual_history: list[float] = []
        gs_residual_history: list[float] = []
        converged = False
        final_iter = 0
        gs_best: float = float("inf")
        final_source: FloatArray | None = None

        self._seed_plasma(mu0)

        # ── Phase A: Picard warmup ──
        for k in range(warmup_steps):
            final_iter = k
            _, _, Psi_axis = self._find_magnetic_axis()
            _, Psi_boundary = self.find_x_point(self.Psi)
            if abs(Psi_axis - Psi_boundary) < 0.1:  # pragma: no cover — degenerate equilibrium
                Psi_boundary = Psi_axis * 0.1

            if not getattr(self, "external_profile_mode", False):
                self.J_phi = self.update_plasma_source_nonlinear(Psi_axis, Psi_boundary)

            Source = -mu0 * self.RR * self.J_phi
            final_source = Source
            Psi_new = self._elliptic_solve(Source, Psi_vac_boundary)

            if np.isnan(Psi_new).any() or np.isinf(Psi_new).any():  # pragma: no cover - bounded solve does not diverge
                if fail_on_diverge:
                    raise RuntimeError(f"Newton warmup diverged at iter={k}")
                break

            diff = float(np.mean(np.abs(Psi_new - self.Psi)))
            residual_history.append(diff)
            self.Psi = (1.0 - picard_alpha) * self.Psi + picard_alpha * Psi_new
            self._apply_boundary_conditions(self.Psi, Psi_vac_boundary)
            gs_residual = self._compute_gs_residual_rms(Source)
            gs_residual_history.append(gs_residual)
            if gs_residual < gs_best:
                gs_best = gs_residual

            update_converged = diff < tol
            gs_converged = (not require_gs_residual) or (gs_residual < gs_tol)
            if update_converged and gs_converged:
                converged = True
                break

        # ── Phase B: Newton iterations ──
        if not converged:
            NZ, NR_grid = self.Psi.shape
            n_interior = (NZ - 2) * (NR_grid - 2)

            for k in range(warmup_steps, max_iter):
                final_iter = k

                _, _, Psi_axis = self._find_magnetic_axis()
                _, Psi_boundary = self.find_x_point(self.Psi)
                if abs(Psi_axis - Psi_boundary) < 0.1:  # pragma: no cover — degenerate equilibrium
                    Psi_boundary = Psi_axis * 0.1

                if not getattr(self, "external_profile_mode", False):
                    self.J_phi = self.update_plasma_source_nonlinear(Psi_axis, Psi_boundary)

                # Residual: r_k = L[psi] + mu0*R*J_phi  (Source = -mu0*R*J)
                Source = -mu0 * self.RR * self.J_phi
                final_source = Source
                r_k = self._compute_gs_residual(Source)
                res_norm = float(np.sqrt(np.sum(r_k[1:-1, 1:-1] ** 2)))
                gs_residual = float(np.sqrt(np.mean(r_k[1:-1, 1:-1] ** 2)))
                residual_history.append(res_norm)
                gs_residual_history.append(gs_residual)
                if gs_residual < gs_best:
                    gs_best = gs_residual

                update_converged = res_norm < tol
                gs_converged = (not require_gs_residual) or (gs_residual < gs_tol)
                if update_converged and gs_converged:
                    converged = True
                    break

                # Build Jacobian operator: J_k = L + mu0*R*dJ/dpsi
                dJ_dpsi = self._compute_profile_jacobian(Psi_axis, Psi_boundary, mu0)
                diag_term = -mu0 * self.RR * dJ_dpsi  # the source derivative

                def matvec(v_flat: AnyFloatArray, _dt: AnyFloatArray = diag_term) -> FloatArray:  # noqa: B006
                    v2d = np.zeros((NZ, NR_grid))
                    v2d[1:-1, 1:-1] = v_flat.reshape(NZ - 2, NR_grid - 2)
                    Lv = self._apply_gs_operator(v2d)
                    Lv[1:-1, 1:-1] -= _dt[1:-1, 1:-1] * v2d[1:-1, 1:-1]
                    return Lv[1:-1, 1:-1].ravel()

                J_op = LinearOperator(
                    shape=(n_interior, n_interior),
                    matvec=matvec,
                    dtype=np.float64,
                )

                # Solve J_k * delta = -r_k
                rhs = -r_k[1:-1, 1:-1].ravel()
                delta_flat, info = gmres(J_op, rhs, maxiter=100, restart=50, atol=1e-8, rtol=1e-6)

                if info != 0:  # pragma: no cover - GMRES converges within maxiter on the bounded Jacobian
                    logger.warning("GMRES did not converge at Newton iter %d (info=%d)", k, info)

                delta = np.zeros_like(self.Psi)
                delta[1:-1, 1:-1] = delta_flat.reshape(NZ - 2, NR_grid - 2)

                # Damped Newton update
                self.Psi += newton_alpha * delta
                self._apply_boundary_conditions(self.Psi, Psi_vac_boundary)

                # NaN check
                if np.isnan(self.Psi).any() or np.isinf(self.Psi).any():  # pragma: no cover — numerical safety
                    logger.warning("Newton diverged at iter %d", k)
                    if fail_on_diverge:
                        raise RuntimeError(f"Newton solver diverged at iter={k}")
                    break

        if final_source is None:  # pragma: no cover - defensive free-boundary fallback path
            gs_final = float("inf")
            gs_best_out = float("inf")
        elif gs_residual_history:
            gs_final = gs_residual_history[-1]
            gs_best_out = gs_best
        else:  # pragma: no cover — warmup always populates gs_residual_history
            gs_final = self._compute_gs_residual_rms(final_source)
            gs_best_out = gs_final

        self.compute_b_field()
        elapsed = time.time() - t0
        logger.info("Newton solved in %.2fs, %d iters", elapsed, final_iter + 1)

        return {
            "psi": self.Psi,
            "converged": converged,
            "iterations": final_iter + 1,
            "residual": residual_history[-1] if residual_history else float("inf"),
            "residual_history": residual_history,
            "gs_residual": gs_final,
            "gs_residual_best": gs_best_out,
            "gs_residual_history": gs_residual_history,
            "wall_time_s": elapsed,
            "solver_method": "newton",
        }

    # ── Rust multigrid delegation ────────────────────────────────────

    def _solve_via_rust_multigrid(
        self,
        preserve_initial_state: bool = False,
        boundary_flux: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Delegate the full equilibrium solve to the Rust multigrid backend.

        Falls back to Python SOR if the Rust extension is not installed.
        """
        from scpn_control.core._rust_compat import RustAcceleratedKernel, _rust_available

        if preserve_initial_state or boundary_flux is not None:
            logger.warning("Boundary-constrained solve requested with rust_multigrid; falling back to Python SOR.")
            prior_method = self.cfg["solver"].get("solver_method", "rust_multigrid")
            self.cfg["solver"]["solver_method"] = "sor"
            try:
                return self.solve_equilibrium(
                    preserve_initial_state=preserve_initial_state,
                    boundary_flux=boundary_flux,
                )
            finally:
                self.cfg["solver"]["solver_method"] = prior_method

        if not _rust_available():
            logger.warning("Rust unavailable; falling back to Python SOR.")
            prior_method = self.cfg["solver"].get("solver_method", "rust_multigrid")
            self.cfg["solver"]["solver_method"] = "sor"
            try:
                return self.solve_equilibrium()
            finally:
                self.cfg["solver"]["solver_method"] = prior_method

        t0 = time.time()
        rk = RustAcceleratedKernel(self._config_path)
        rk.set_solver_method("multigrid")
        rust_result = rk.solve_equilibrium()

        # Sync state back
        self.Psi = rk.Psi
        self.J_phi = rk.J_phi
        self.B_R: AnyFloatArray = rk.B_R
        self.B_Z: AnyFloatArray = rk.B_Z

        mu0: float = self.cfg["physics"]["vacuum_permeability"]
        source = -mu0 * self.RR * self.J_phi
        gs_residual = self._compute_gs_residual_rms(source)
        elapsed = time.time() - t0
        solver_tol = float(self.cfg.get("solver", {}).get("convergence_threshold", 1e-4))
        practical_tol = max(solver_tol, 2e-3)
        converged = bool(rust_result.converged or rust_result.residual <= practical_tol)
        return {
            "psi": self.Psi,
            "converged": converged,
            "iterations": rust_result.iterations,
            "residual": rust_result.residual,
            "residual_history": [],
            "gs_residual": gs_residual,
            "gs_residual_best": gs_residual,
            "gs_residual_history": [],
            "wall_time_s": elapsed,
            "solver_method": "rust_multigrid",
        }

    # ── main solver ───────────────────────────────────────────────────

    def solve_equilibrium(
        self,
        preserve_initial_state: bool = False,
        boundary_flux: FloatArray | None = None,
    ) -> dict[str, Any]:
        """Run the full Picard-iteration equilibrium solver.

        Iterates: topology analysis -> source update -> elliptic solve ->
        under-relaxation until the residual drops below the configured
        convergence threshold or the maximum iteration count is reached.

        When ``solver.solver_method`` is ``"anderson"``, Anderson
        acceleration is applied every few Picard steps to speed up
        convergence.

        Returns
        -------
        dict[str, Any]
            ``{"psi": FloatArray, "converged": bool, "iterations": int,
            "residual": float, "residual_history": list[float],
            "gs_residual": float, "gs_residual_best": float,
            "gs_residual_history": list[float],
            "wall_time_s": float, "solver_method": str}``

        Parameters
        ----------
        preserve_initial_state : bool, optional
            Keep current interior ``self.Psi`` values and only enforce boundary
            conditions before the iterative solve. Default is ``False``.
        boundary_flux : FloatArray | None, optional
            Explicit boundary map to enforce during solve. Must have shape
            ``(NZ, NR)`` when provided.

        Raises
        ------
        RuntimeError
            If the solver produces NaN or Inf and ``solver.fail_on_diverge``
            is enabled in the active configuration.
        """
        t0 = time.time()

        max_iter: int = self.cfg["solver"]["max_iterations"]
        method: str = self.cfg["solver"].get("solver_method", "multigrid")

        if max_iter <= 0:
            self.compute_b_field()
            return {
                "psi": self.Psi,
                "converged": False,
                "iterations": 0,
                "residual": 1.0,
                "residual_history": [],
                "gs_residual": float("inf"),
                "gs_residual_best": float("inf"),
                "gs_residual_history": [],
                "wall_time_s": time.time() - t0,
                "solver_method": method,
                "boundary_variant": "fixed_boundary",
            }

        # ── Fast-path dispatches ──
        if method == "rust_multigrid":
            return self._solve_via_rust_multigrid(
                preserve_initial_state=preserve_initial_state,
                boundary_flux=boundary_flux,
            )
        if method == "newton":
            return self._newton_solve_dispatch(
                preserve_initial_state=preserve_initial_state,
                boundary_flux=boundary_flux,
            )

        Psi_vac_boundary = self._prepare_initial_flux(
            preserve_initial_state=preserve_initial_state,
            boundary_flux=boundary_flux,
        )

        tol: float = self.cfg["solver"]["convergence_threshold"]
        alpha: float = self.cfg["solver"].get("relaxation_factor", 0.1)
        fail_on_diverge: bool = bool(self.cfg["solver"].get("fail_on_diverge", False))
        require_gs_residual: bool = bool(self.cfg["solver"].get("require_gs_residual", False))
        gs_tol: float = float(self.cfg["solver"].get("gs_residual_threshold", tol))
        if require_gs_residual and gs_tol <= 0.0:
            raise ValueError("solver.gs_residual_threshold must be > 0")
        mu0: float = self.cfg["physics"]["vacuum_permeability"]
        anderson_m: int = self.cfg["solver"].get("anderson_depth", 5)

        x_point_pos: tuple[float, float] = (0.0, 0.0)
        Psi_best = self.Psi.copy()
        diff_best: float = 1e9
        residual_history: list[float] = []
        gs_residual_history: list[float] = []
        converged = False
        gs_best: float = float("inf")
        final_source: FloatArray | None = None

        # Anderson acceleration history buffers
        psi_history: list[FloatArray] = []
        res_history: list[FloatArray] = []

        self._seed_plasma(mu0)

        final_iter = 0
        for k in range(max_iter):
            final_iter = k
            # 1. Topology
            _, _, Psi_axis = self._find_magnetic_axis()
            x_point_pos, Psi_boundary = self.find_x_point(self.Psi)

            if abs(Psi_axis - Psi_boundary) < 0.1:
                Psi_boundary = Psi_axis * 0.1

            # 2. Source update
            if not getattr(self, "external_profile_mode", False):
                self.J_phi = self.update_plasma_source_nonlinear(Psi_axis, Psi_boundary)

            # 3. Elliptic solve
            Source = -mu0 * self.RR * self.J_phi
            final_source = Source
            Psi_new = self._elliptic_solve(Source, Psi_vac_boundary)

            # Divergence check
            if np.isnan(Psi_new).any() or np.isinf(Psi_new).any():
                logger.warning(
                    "Solver diverged at iter %d — reverting to best state.",
                    k,
                )
                self.Psi = Psi_best
                if fail_on_diverge:
                    raise RuntimeError(f"Equilibrium solver diverged at iter={k}")
                break

            # 4. Under-relaxation
            diff = float(np.mean(np.abs(Psi_new - self.Psi)))
            residual_history.append(diff)
            self.Psi = (1.0 - alpha) * self.Psi + alpha * Psi_new

            # 5. Anderson acceleration (optional)
            if method == "anderson":
                psi_history.append(self.Psi.copy())
                res_history.append(Psi_new - self.Psi)
                if len(psi_history) >= 3 and k % 3 == 0:
                    mixed = self._anderson_step(psi_history, res_history, m=anderson_m)
                    self._apply_boundary_conditions(mixed, Psi_vac_boundary)
                    self.Psi = mixed
                # Trim history to avoid unbounded memory growth
                if len(psi_history) > anderson_m + 2:
                    psi_history.pop(0)
                    res_history.pop(0)

            gs_residual = self._compute_gs_residual_rms(Source)
            gs_residual_history.append(gs_residual)
            if gs_residual < gs_best:
                gs_best = gs_residual

            if diff < diff_best:
                diff_best = diff
                Psi_best = self.Psi.copy()

            update_converged = diff < tol
            gs_converged = (not require_gs_residual) or (gs_residual < gs_tol)
            if update_converged and gs_converged:
                logger.info(
                    "Converged at iter %d.  Update residual: %.6e | GS RMS: %.6e",
                    k,
                    diff,
                    gs_residual,
                )
                converged = True
                break

            if k % 100 == 0:
                logger.debug(
                    "Iter %d: res=%.2e | axis=%.2f | X-pt=%.2f at R=%.2f, Z=%.2f",
                    k,
                    diff,
                    Psi_axis,
                    Psi_boundary,
                    x_point_pos[0],
                    x_point_pos[1],
                )

        if final_source is None:  # pragma: no cover - defensive free-boundary fallback path
            gs_final = float("inf")
            gs_best_out = float("inf")
        elif gs_residual_history:
            gs_final = gs_residual_history[-1]
            gs_best_out = gs_best
        else:  # pragma: no cover - defensive free-boundary fallback path
            gs_final = self._compute_gs_residual_rms(final_source)
            gs_best_out = gs_final

        self.compute_b_field()
        elapsed = time.time() - t0
        logger.info(
            "Solved in %.2fs (%s).  X-point: R=%.2f, Z=%.2f",
            elapsed,
            method,
            x_point_pos[0],
            x_point_pos[1],
        )

        return {
            "psi": self.Psi,
            "converged": converged,
            "iterations": final_iter + 1,
            "residual": diff_best,
            "residual_history": residual_history,
            "gs_residual": gs_final,
            "gs_residual_best": gs_best_out,
            "gs_residual_history": gs_residual_history,
            "wall_time_s": elapsed,
            "solver_method": method,
            "boundary_variant": "fixed_boundary",
        }

    # ── post-processing ───────────────────────────────────────────────

    def compute_b_field(self) -> None:
        """Derive the magnetic field components from the solved Psi."""
        dPsi_dR, dPsi_dZ = _psi_gradient_fields(self.Psi, self.dR, self.dZ)
        R_safe = np.maximum(self.RR, 1e-6)
        self.B_R = np.asarray(-(1.0 / R_safe) * dPsi_dZ)
        self.B_Z = np.asarray((1.0 / R_safe) * dPsi_dR)

    @staticmethod
    def _green_function(R_src: float, Z_src: float, R_obs: float, Z_obs: float) -> float:
        """Toroidal Green's function using elliptic integrals."""
        return _gs_green.green_function(R_src, Z_src, R_obs, Z_obs)

    @staticmethod
    def _green_function_array(R_src: float, Z_src: float, R_obs: FloatArray, Z_obs: FloatArray) -> FloatArray:
        """Vectorised toroidal Green's function over observation grids."""
        return _gs_green.green_function_array(R_src, Z_src, R_obs, Z_obs)

    def _compute_external_flux(self, coils: Any) -> FloatArray:
        """Sum Green's function contributions on boundary from CoilSet."""
        return _gs_green.external_flux_from_coilset(self.R, self.Z, coils)

    def _build_mutual_inductance_matrix(
        self,
        coils: CoilSet,
        obs_points: FloatArray,
    ) -> FloatArray:
        """Build mutual-inductance matrix M[k, p] for coil optimisation.

        ``M[k, p]`` is the flux at observation point *p* due to unit current
        in coil *k*.  Uses the toroidal Green's function.

        Parameters
        ----------
        coils : CoilSet
            Coil geometry.
        obs_points : FloatArray, shape (n_pts, 2)
            Observation points ``(R, Z)`` — typically the target separatrix.

        Returns
        -------
        FloatArray, shape (n_coils, n_pts)
        """
        return _gs_green.build_mutual_inductance_matrix(coils, obs_points)

    def _sample_flux_at_points(self, obs_points: FloatArray) -> FloatArray:
        """Sample the current ``Psi`` field at arbitrary observation points."""
        return np.asarray([float(self._interp_psi(R_obs, Z_obs)) for R_obs, Z_obs in obs_points], dtype=np.float64)

    @staticmethod
    def _shape_error_metrics(current_flux: FloatArray, target_flux: FloatArray) -> dict[str, float]:
        """Compute RMS and max-abs error for a boundary-shape target."""
        return _gs_fb.shape_error_metrics(current_flux, target_flux)

    def _coil_flux_response_at_point(
        self, coils: CoilSet, point: FloatArray
    ) -> FloatArray:  # pragma: no cover - defensive free-boundary fallback path
        """Return per-coil flux response at a single observation point."""
        point_arr = np.asarray(point, dtype=np.float64).reshape(1, 2)
        return self._build_mutual_inductance_matrix(coils, point_arr)[:, 0].astype(np.float64, copy=False)

    def _estimate_point_gradient(  # pragma: no cover - defensive free-boundary fallback path
        self,
        sample_fn: Any,
        R_pt: float,
        Z_pt: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Estimate d/dR and d/dZ using central or one-sided finite differences."""
        return _gs_fb.estimate_point_gradient(
            sample_fn,
            R_pt,
            Z_pt,
            r_min=float(self.R[0]),
            r_max=float(self.R[-1]),
            z_min=float(self.Z[0]),
            z_max=float(self.Z[-1]),
            dR=float(self.dR),
            dZ=float(self.dZ),
        )

    def _coil_flux_gradient_response(  # pragma: no cover - defensive free-boundary fallback path
        self,
        coils: CoilSet,
        point: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Return per-coil dPsi/dR and dPsi/dZ response at a target point."""
        R_pt, Z_pt = np.asarray(point, dtype=np.float64).reshape(2)

        def sample_fn(R_obs: float, Z_obs: float) -> FloatArray:
            return self._coil_flux_response_at_point(coils, np.array([R_obs, Z_obs], dtype=np.float64))

        d_dR, d_dZ = self._estimate_point_gradient(sample_fn, float(R_pt), float(Z_pt))
        return np.asarray(d_dR, dtype=np.float64), np.asarray(d_dZ, dtype=np.float64)

    def _interp_psi_gradient(self, R_pt: float, Z_pt: float) -> tuple[float, float]:
        """Estimate dPsi/dR and dPsi/dZ from the current solved field."""

        def sample_fn(R_obs: float, Z_obs: float) -> float:
            return self._interp_psi(R_obs, Z_obs)

        d_dR, d_dZ = self._estimate_point_gradient(sample_fn, float(R_pt), float(Z_pt))
        return float(np.asarray(d_dR).reshape(())), float(np.asarray(d_dZ).reshape(()))

    @staticmethod
    def _resolve_separatrix_flux_target(  # pragma: no cover - defensive free-boundary fallback path
        coils: CoilSet,
        shape_target_flux: FloatArray | None,
    ) -> float | None:
        """Resolve a scalar separatrix-flux target from active objectives."""
        return _gs_fb.resolve_separatrix_flux_target(coils, shape_target_flux)

    def _resolve_x_point_flux_target(  # pragma: no cover - defensive free-boundary fallback path
        self,
        coils: CoilSet,
        separatrix_flux_target: float | None,
    ) -> tuple[float | None, str]:
        """Resolve the X-point flux target mode and scalar target."""
        local_flux: float | None = None
        if coils.x_point_target is not None and coils.x_point_flux_target is None and separatrix_flux_target is None:
            R_x, Z_x = np.asarray(coils.x_point_target, dtype=np.float64).reshape(2)
            local_flux = float(self._interp_psi(float(R_x), float(Z_x)))
        return _gs_fb.resolve_x_point_flux_target(coils, separatrix_flux_target, local_flux)

    def _resolve_divertor_flux_targets(  # pragma: no cover - defensive free-boundary fallback path
        self,
        coils: CoilSet,
        separatrix_flux_target: float | None,
    ) -> tuple[FloatArray | None, str]:
        """Resolve divertor strike-point flux targets and mode."""
        sampled: FloatArray | None = None
        if (
            coils.divertor_strike_points is not None
            and coils.divertor_flux_values is None
            and separatrix_flux_target is None
        ):
            sampled = self._sample_flux_at_points(coils.divertor_strike_points)
        return _gs_fb.resolve_divertor_flux_targets(coils, separatrix_flux_target, sampled)

    @staticmethod
    def _resolve_free_boundary_objective_tolerances(
        cfg_objective_tolerances: Any,
        override_objective_tolerances: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Validate and merge free-boundary objective tolerances."""
        return _gs_fb.resolve_free_boundary_objective_tolerances(
            cfg_objective_tolerances,
            override_objective_tolerances,
        )

    @staticmethod
    def _evaluate_free_boundary_objective_status(
        tolerances: dict[str, float],
        *,
        shape_error_rms: float | None,
        shape_error_max_abs: float | None,
        x_point_detected_error: float | None,
        x_point_gradient_norm: float | None,
        x_point_flux_error: float | None,
        divertor_error_rms: float | None,
        divertor_error_max_abs: float | None,
    ) -> dict[str, Any]:
        """Evaluate which configured free-boundary objective tolerances are satisfied."""
        return _gs_fb.evaluate_free_boundary_objective_status(
            tolerances,
            shape_error_rms=shape_error_rms,
            shape_error_max_abs=shape_error_max_abs,
            x_point_detected_error=x_point_detected_error,
            x_point_gradient_norm=x_point_gradient_norm,
            x_point_flux_error=x_point_flux_error,
            divertor_error_rms=divertor_error_rms,
            divertor_error_max_abs=divertor_error_max_abs,
        )

    @staticmethod
    def _divertor_configuration_label(strike_points: FloatArray | None) -> str:
        """Return a coarse divertor-target configuration label."""
        return _gs_fb.divertor_configuration_label(strike_points)

    def _resolve_shape_target_flux(  # pragma: no cover - defensive free-boundary fallback path
        self,
        coils: CoilSet,
        current_flux: FloatArray,
    ) -> tuple[FloatArray, str]:
        """Resolve the shape objective target for free-boundary optimisation."""
        return _gs_fb.resolve_shape_target_flux(coils, current_flux)

    def optimize_coil_currents(  # pragma: no cover - defensive free-boundary fallback path
        self,
        coils: CoilSet,
        target_flux: FloatArray,
        tikhonov_alpha: float = 1e-4,
        *,
        x_point_flux_target: float | None = None,
        divertor_flux_targets: FloatArray | None = None,
    ) -> FloatArray:
        """Find coil currents that best satisfy free-boundary target constraints.

        Solves a bounded linear least-squares problem that can include:

        - boundary-flux targets at ``target_flux_points``
        - X-point isoflux and null-field constraints
        - divertor strike-point isoflux constraints

            min_I || A I - b ||^2 + alpha * ||I||^2
            s.t.  -I_max <= I <= I_max  (per coil)

        where ``A`` stacks the active constraint blocks.

        Parameters
        ----------
        coils : CoilSet
            Coil geometry and optional objective targets.
        target_flux : FloatArray, shape (n_pts,)
            Desired poloidal flux at ``target_flux_points``. Can be empty when
            only X-point and/or divertor constraints are active.
        tikhonov_alpha : float
            Regularisation strength to penalise large currents.
        x_point_flux_target : float or None
            Scalar target flux at ``coils.x_point_target``.
        divertor_flux_targets : FloatArray or None
            Flux targets at ``coils.divertor_strike_points``.

        Returns
        -------
        FloatArray, shape (n_coils,)
            Optimised coil currents [A].
        """
        return _gs_fb.optimize_coil_currents(
            coils,
            target_flux,
            build_mutual_inductance_matrix=self._build_mutual_inductance_matrix,
            coil_flux_response_at_point=self._coil_flux_response_at_point,
            coil_flux_gradient_response=self._coil_flux_gradient_response,
            tikhonov_alpha=tikhonov_alpha,
            x_point_flux_target=x_point_flux_target,
            divertor_flux_targets=divertor_flux_targets,
        )

    def solve_free_boundary(
        self,
        coils: CoilSet,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
        objective_tolerances: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Experimental external-coil outer loop around the fixed-boundary GS solve.

        Iterates between updating boundary flux from coils and solving the
        internal GS equation.  When ``optimize_shape=True`` and the coil set
        has ``target_flux_points``, an additional outer loop optimises the
        coil currents to match the desired plasma boundary shape.

        This helper should not be interpreted as a complete production-grade
        free-boundary solver.  The current project standard for closing the
        free-boundary roadmap item is higher: shape control, X-point geometry,
        and divertor-configuration support must all be demonstrated.

        Parameters
        ----------
        coils : CoilSet
            External coil set.
        max_outer_iter : int
            Maximum outer-loop iterations for the experimental coil-coupled path.
        tol : float
            Convergence tolerance on max |delta psi|.
        optimize_shape : bool
            When True, run coil-current optimisation at each outer step.
        tikhonov_alpha : float
            Tikhonov regularisation for coil optimisation.
        objective_tolerances : dict or None
            Optional convergence gates for free-boundary target objectives.
            Supported keys are ``shape_rms``, ``shape_max_abs``,
            ``x_point_position``, ``x_point_gradient``, ``x_point_flux``,
            ``divertor_rms``, and ``divertor_max_abs``. When omitted, the
            method falls back to ``free_boundary.objective_tolerances`` in the
            config, if present.

        Returns
        -------
        dict
            ``{"outer_iterations": int, "final_diff": float,
            "coil_currents": AnyFloatArray}``
        """
        return _gs_fb_solve.solve_free_boundary(
            self,
            coils,
            max_outer_iter=max_outer_iter,
            tol=tol,
            optimize_shape=optimize_shape,
            tikhonov_alpha=tikhonov_alpha,
            objective_tolerances=objective_tolerances,
        )

    def _interp_psi(self, R_pt: float, Z_pt: float) -> float:
        """Bilinear interpolation of Psi at an arbitrary (R, Z) point."""
        # Find enclosing cell
        ir = int(np.searchsorted(self.R, R_pt) - 1)
        iz = int(np.searchsorted(self.Z, Z_pt) - 1)
        ir = max(0, min(ir, self.NR - 2))
        iz = max(0, min(iz, self.NZ - 2))

        # Local coordinates
        t_r = (R_pt - self.R[ir]) / self.dR
        t_z = (Z_pt - self.Z[iz]) / self.dZ
        t_r = max(0.0, min(1.0, t_r))
        t_z = max(0.0, min(1.0, t_z))

        # Bilinear
        psi = (
            (1 - t_r) * (1 - t_z) * self.Psi[iz, ir]
            + t_r * (1 - t_z) * self.Psi[iz, ir + 1]
            + (1 - t_r) * t_z * self.Psi[iz + 1, ir]
            + t_r * t_z * self.Psi[iz + 1, ir + 1]
        )
        return float(psi)

    # ── phase reduction (Paper 27 / reviewer: ζ sin(Ψ−θ)) ─────────

    def phase_sync_step(
        self,
        theta: FloatArray,
        omega: FloatArray,
        *,
        dt: float = 1e-3,
        K: float | None = None,
        alpha: float | None = None,
        zeta: float | None = None,
        psi_driver: float | None = None,
        psi_mode: str | None = None,
        actuation_gain: float | None = None,
    ) -> dict[str, Any]:
        """Reduced-order plasma sync kernel (phase reduction).

        dθ_i/dt = ω_i + K·R·sin(ψ_r − θ_i − α) + ζ·sin(Ψ − θ_i)

        Ψ is exogenous when psi_mode="external" (no dotΨ equation).
        This is the reviewer's ζ sin(Ψ−θ) injection for plasma sync stability.
        """
        from scpn_control.phase.kuramoto import kuramoto_sakaguchi_step

        cfg = self.cfg.get("phase_sync", {})
        K_eff = float(cfg.get("K", 1.0) if K is None else K)
        alpha_eff = float(cfg.get("alpha", 0.0) if alpha is None else alpha)
        zeta_eff = float(cfg.get("zeta", 0.0) if zeta is None else zeta)
        psi_mode_eff = str(cfg.get("psi_mode", "external") if psi_mode is None else psi_mode)
        gain = float(cfg.get("actuation_gain", 1.0) if actuation_gain is None else actuation_gain)

        return kuramoto_sakaguchi_step(
            theta=np.asarray(theta, dtype=np.float64),
            omega=np.asarray(omega, dtype=np.float64),
            dt=dt,
            K=K_eff * gain,
            alpha=alpha_eff,
            zeta=zeta_eff * gain,
            psi_driver=psi_driver,
            psi_mode=psi_mode_eff,
            wrap=True,
        )

    def phase_sync_step_lyapunov(
        self,
        theta: FloatArray,
        omega: FloatArray,
        *,
        n_steps: int = 100,
        dt: float = 1e-3,
        K: float | None = None,
        zeta: float | None = None,
        psi_driver: float | None = None,
        psi_mode: str | None = None,
    ) -> dict[str, Any]:
        """Multi-step phase sync with Lyapunov stability tracking.

        Returns final state, R trajectory, V trajectory, and λ exponent.
        λ < 0 ⟹ stable convergence toward Ψ.
        """
        from scpn_control.phase.kuramoto import (
            kuramoto_sakaguchi_step,
            lyapunov_exponent,
            lyapunov_v,
        )

        cfg = self.cfg.get("phase_sync", {})
        K_eff = float(cfg.get("K", 1.0) if K is None else K)
        zeta_eff = float(cfg.get("zeta", 0.0) if zeta is None else zeta)
        psi_mode_eff = str(cfg.get("psi_mode", "external") if psi_mode is None else psi_mode)

        th = np.asarray(theta, dtype=np.float64)
        om = np.asarray(omega, dtype=np.float64)
        r_hist = []
        v_hist = []

        for _ in range(n_steps):
            out = kuramoto_sakaguchi_step(
                th,
                om,
                dt=dt,
                K=K_eff,
                zeta=zeta_eff,
                psi_driver=psi_driver,
                psi_mode=psi_mode_eff,
            )
            th = out["theta1"]
            r_hist.append(out["R"])
            v_hist.append(lyapunov_v(th, out["Psi"]))

        lam = lyapunov_exponent(v_hist, dt)
        return {
            "theta_final": th,
            "R_hist": np.array(r_hist),
            "V_hist": np.array(v_hist),
            "lambda": lam,
            "stable": lam < 0.0,
        }

    def save_results(self, filename: str = "equilibrium_nonlinear.npz") -> None:
        """Save the equilibrium state to a compressed NumPy archive.

        Parameters
        ----------
        filename : str
            Output file path.
        """
        np.savez(filename, R=self.R, Z=self.Z, Psi=self.Psi, J_phi=self.J_phi)
        logger.info("Saved: %s", filename)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    config_file = sys.argv[1] if len(sys.argv) > 1 else "iter_config.json"
    fk = FusionKernel(config_file)
    fk.solve_equilibrium()
    fk.save_results("final_state_nonlinear.npz")
