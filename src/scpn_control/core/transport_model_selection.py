# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Transport model selection

"""Bounded coefficient and fallback selection behind the control solver facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.anomalous_transport import gk_flux_surface_transport, gyro_bohm_chi_profile
from scpn_control.core.momentum_transport import MomentumTransportSolver
from scpn_control.core.transport_geometry import estimate_plasma_surface_area_m2
from scpn_control.core.transport_neoclassical import (
    _finite_scalar,
    _load_gyro_bohm_coefficient,
    _validate_tokamak_geometry,
    calculate_sauter_bootstrap_current_full,
    chang_hinton_chi_profile,
)

if TYPE_CHECKING:
    from scpn_control.core.integrated_transport_solver import TransportSolver


def set_neoclassical_impl(
    self: TransportSolver,
    R0: float,
    a: float,
    B0: float,
    A_ion: float = 2.0,
    Z_eff: float = 1.5,
    q0: float = 1.0,
    q_edge: float = 3.0,
) -> None:
    """Configure Chang-Hinton neoclassical transport model.

    When set, update_transport_model uses the Chang-Hinton formula instead
    of the constant chi_base = 0.5.
    """
    R0, a, B0 = _validate_tokamak_geometry(R0, a, B0)
    A_ion = _finite_scalar("A_ion", A_ion, positive=True)
    Z_eff = _finite_scalar("Z_eff", Z_eff, positive=True)
    q0 = _finite_scalar("q0", q0, positive=True)
    q_edge = _finite_scalar("q_edge", q_edge, positive=True)
    if q_edge < q0:
        raise ValueError("q_edge must be greater than or equal to q0")
    q_profile = q0 + (q_edge - q0) * self.rho**2
    self.neoclassical_params = {
        "R0": R0,
        "a": a,
        "B0": B0,
        "A_ion": A_ion,
        "Z_eff": Z_eff,
        "q_profile": q_profile,
    }
    self._momentum_solver = MomentumTransportSolver(self.rho, R0, a, B0)


def chang_hinton_chi_profile_impl(self: TransportSolver) -> FloatArray:
    """Backward-compatible Chang-Hinton profile helper.

    Older parity tests call this no-arg method on a partially-initialized
    transport object. Keep the method as a thin adapter over the module
    function so those tests remain stable.
    """
    rho = np.asarray(self.rho, dtype=np.float64)

    t_i_raw = getattr(self, "t_i", None)
    if t_i_raw is None:
        t_i_raw = self.Ti
    t_i = np.asarray(t_i_raw, dtype=np.float64)

    n_e_raw = getattr(self, "n_e", None)
    if n_e_raw is None:
        n_e_raw = self.ne
    n_e = np.asarray(n_e_raw, dtype=np.float64)
    q_profile = np.asarray(
        getattr(self, "q_profile", np.linspace(1.0, 3.0, len(rho))),
        dtype=np.float64,
    )

    params = getattr(self, "neoclassical_params", None)
    if not isinstance(params, dict):
        params = {}
    R0 = float(params.get("R0", 6.2))
    a = float(params.get("a", 2.0))
    B0 = float(params.get("B0", 5.3))
    A_ion = float(params.get("A_ion", 2.0))
    Z_eff = float(params.get("Z_eff", 1.5))

    if q_profile.shape != rho.shape:
        q_profile = np.linspace(1.0, 3.0, len(rho), dtype=np.float64)

    return chang_hinton_chi_profile(rho, t_i, n_e, q_profile, R0, a, B0, A_ion=A_ion, Z_eff=Z_eff)


def inject_impurities_impl(self: TransportSolver, flux_from_wall_per_sec: float, dt: float) -> None:
    """
    Models impurity influx from PWI erosion.

    Simple diffusion model: Source at edge, diffuses inward.
    """
    # Source at edge (last grid point)
    # Flux is total particles. Volume of edge shell approx 20 m3.
    # Delta_n = Flux * dt / Vol_edge
    # Scaling factor adjusted for simulation stability
    d_n_edge = (flux_from_wall_per_sec * dt) / 20.0 * 1e-18

    # Add to edge
    self.n_impurity[-1] += d_n_edge

    # Diffuse inward (Explicit step)
    D_imp = 1.0  # m2/s
    new_imp = self.n_impurity.copy()

    grad = np.gradient(self.n_impurity, self.drho)
    flux = -D_imp * grad
    div = np.gradient(flux, self.drho) / (self.rho + 1e-6)

    new_imp += (-div) * dt

    # Boundary
    new_imp[0] = new_imp[1]  # Axis symmetry

    np.maximum(0, new_imp, out=self.n_impurity)


def _legacy_bootstrap_current_approx_impl(self: TransportSolver, R0: float, B_pol: AnyFloatArray) -> FloatArray:
    """Legacy approximate bootstrap-current closure (compatibility mode only)."""
    # Legacy reduced-order closure retained only for compatibility mode.
    dims = self.cfg["dimensions"]
    R0 = 0.5 * (dims["R_max"] + dims["R_min"]) if R0 == 0.0 else R0
    neo = self.neoclassical_params or {}
    a = neo.get("a", 0.5 * (dims["R_max"] - dims["R_min"]))
    f_trapped = 1.46 * np.sqrt(self.rho * a / (2 * R0))

    P = (self.ion_density * self.Ti + self.ne * self.Te) * 1e19 * 1.602176634e-16  # Pa
    dP_drho = np.gradient(P, self.drho)

    # Wesson, "Tokamaks" 4th ed., Eq. 4.8.1: J_bs ~ -f_t / B_p * dp/dr
    # Factor 1.2: Z_eff≈1.5 correction (Hirshman & Sigmar, NF 21, 1079, 1981)
    B_pol = np.maximum(B_pol, 0.1)
    BOOTSTRAP_ZEFF_CORRECTION = 1.2
    J_bs = BOOTSTRAP_ZEFF_CORRECTION * (f_trapped / B_pol) * dP_drho / a

    J_bs[0] = 0
    J_bs[-1] = 0

    return np.asarray(J_bs)


def calculate_bootstrap_current_impl(self: TransportSolver, R0: float, B_pol: AnyFloatArray) -> FloatArray:
    """Calculate bootstrap current using full Sauter closure by default.

    If neoclassical configuration is missing, this method fails closed unless
    ``allow_simplified_bootstrap_fallback=True`` was explicitly enabled.
    """
    if hasattr(self, "neoclassical_params") and self.neoclassical_params is not None:
        return calculate_sauter_bootstrap_current_full(
            self.rho,
            self.Te,
            self.Ti,
            self.ne,
            self.neoclassical_params.get("q_profile", np.linspace(1, 4, len(self.rho))),
            R0,
            self.neoclassical_params.get("a", 2.0),
            self.neoclassical_params.get("B0", 5.3),
            self.neoclassical_params.get("Z_eff", 1.5),
        )
    if not (self.allow_simplified_bootstrap_fallback and self.allow_legacy_approximations):
        raise RuntimeError(
            "neoclassical transport configuration is required for bootstrap current; "
            "set neoclassical parameters or explicitly enable "
            "allow_simplified_bootstrap_fallback + allow_legacy_approximations for legacy behaviour"
        )
    return self._legacy_bootstrap_current_approx(R0, B_pol)


def _gyro_bohm_chi_impl(self: TransportSolver) -> FloatArray:
    """Gyro-Bohm anomalous transport diffusivity [m^2/s].

    chi_gB = c_gB * rho_s^2 * c_s / (a * q * R)

    where rho_s = sqrt(T_i m_i) / (e B), c_s = sqrt(T_e / m_i).

    The calibration coefficient c_gB is loaded from
    ``validation/reference_data/itpa/gyro_bohm_coefficients.json``
    if available (calibrated against the ITPA H-mode confinement
    database by ``tools/calibrate_gyro_bohm.py``).  Falls back to
    the value in ``neoclassical_params['c_gB']`` if explicitly set,
    or to the module-level default (0.1) otherwise.
    """
    if self.neoclassical_params is None:
        if self.allow_constant_transport_fallback and self.allow_legacy_approximations:
            return np.full_like(self.rho, 0.5)
        raise RuntimeError(
            "neoclassical transport configuration is required for gyro-Bohm transport; "
            "set neoclassical parameters or explicitly enable "
            "allow_constant_transport_fallback + allow_legacy_approximations for legacy behaviour"
        )

    p = self.neoclassical_params

    # Load c_gB: explicit param > JSON file > default
    c_gB = p["c_gB"] if "c_gB" in p else _load_gyro_bohm_coefficient()

    return gyro_bohm_chi_profile(
        self.rho,
        self.Ti,
        self.Te,
        p["q_profile"],
        p["R0"],
        p["a"],
        p["B0"],
        p.get("A_ion", 2.0),
        c_gB,
    )


def _external_gk_transport_impl(self: TransportSolver, p: dict[str, Any]) -> FloatArray:
    """Run an external GK solver at each flux surface, return chi_i profile.

    Also updates self.chi_e and self.D_n directly. By default, fails closed
    if the external solver is unavailable, throws, or returns unconverged
    output. Legacy gyro-Bohm fallback can be explicitly enabled by setting
    ``external_gk_allow_gyrobohm_fallback=True`` on construction.
    """
    from scpn_control.core.gk_interface import GKSolverBase

    solver: GKSolverBase | None = getattr(self, "_gk_solver", None)
    if solver is None:
        from scpn_control.core.gk_tglf import TGLFSolver

        solver = TGLFSolver()
        self._gk_solver = solver

    chi_i_out, chi_e_out, D_e_out = gk_flux_surface_transport(
        solver=solver,
        rho=self.rho,
        Te=self.Te,
        Ti=self.Ti,
        ne=self.ne,
        params=p,
        solver_label="external_gk",
        catch_execution_errors=True,
        allow_gyrobohm_fallback=self.external_gk_allow_gyrobohm_fallback and self.allow_legacy_approximations,
        gyro_bohm_fallback=self._gyro_bohm_chi,
    )
    self.chi_e = chi_e_out
    self.D_n = D_e_out
    return chi_i_out


def _tglf_native_transport_impl(self: TransportSolver, p: dict[str, Any]) -> FloatArray:
    """Run the native TGLF-equivalent solver at each flux surface.

    Uses SAT1 spectral saturation with E×B shear quench.  Also
    updates self.chi_e and self.D_n. By default, fails closed when
    native-GK transport is unconverged. Legacy gyro-Bohm fallback can
    be explicitly enabled with ``tglf_native_allow_gyrobohm_fallback=True``.
    """
    from scpn_control.core.gk_tglf_native import TGLFNativeConfig, TGLFNativeSolver

    solver: TGLFNativeSolver | None = getattr(self, "_tglf_native_solver", None)
    if solver is None:
        solver = TGLFNativeSolver(TGLFNativeConfig(sat_model="SAT1", n_ky_ion=12, n_theta=32))
        self._tglf_native_solver = solver

    chi_i_out, chi_e_out, D_e_out = gk_flux_surface_transport(
        solver=solver,
        rho=self.rho,
        Te=self.Te,
        Ti=self.Ti,
        ne=self.ne,
        params=p,
        solver_label="tglf_native",
        catch_execution_errors=False,
        allow_gyrobohm_fallback=self.tglf_native_allow_gyrobohm_fallback and self.allow_legacy_approximations,
        gyro_bohm_fallback=self._gyro_bohm_chi,
    )
    self.chi_e = chi_e_out
    self.D_n = D_e_out
    return chi_i_out


def update_transport_model_impl(self: TransportSolver, P_aux: float) -> None:
    """
    Gyro-Bohm + neoclassical transport model with EPED-like pedestal.

    When neoclassical params are set, uses:
    - Chang-Hinton neoclassical chi as additive floor
    - Gyro-Bohm anomalous transport (calibrated c_gB)
    - EPED-like pedestal model for H-mode boundary condition

    Gyrokinetic, native TGLF and external GK modes retain their separately
    computed electron diffusivity and D_n profile. Their ion diffusivity
    receives the Chang-Hinton contribution; the legacy critical-gradient
    estimate is not added on top of those turbulent channels. Other modes
    retain the shared ion/electron estimate and D_n = 0.1*chi_e closure.
    For native/external modes, an explicitly permitted cellwise gyro-Bohm
    fallback defines chi_gB = max(raw_gyro_bohm, 0.01) per non-core
    cell, then follows the three-channel composition: chi_i = chi_gB
    + chi_nc, chi_e = chi_gB, D_n = 0.1*chi_gB, with no additional
    critical-gradient term. Core/vacuum cells retain the driver floors
    (0.01 in each returned channel before ion chi_nc is added). Both the
    mode-specific fallback flag and allow_legacy_approximations are required.
    External execution errors enter that fallback decision; native solver
    exceptions propagate, while invalid/unconverged native results can
    fall back. Selecting a mode alone does not prove backend success.

    D_n is an electron-particle coefficient, possibly a fallback estimate;
    the separate D_species scalar or radial profile controls D/T/He.
    This method does not infer a
    multispecies particle or heat-convection closure from D_n.

    Fails closed when neoclassical parameters are not configured, unless
    ``allow_constant_transport_fallback=True`` is explicitly enabled.
    """
    self._ensure_valid_radial_grid()
    separate_channels = False

    # 1. Critical Gradient Model
    grad_T = np.gradient(self.Ti, self.drho)
    threshold = 2.0

    # Base level transport from configured neoclassical + selected anomalous model.
    if self.neoclassical_params is not None:
        p = self.neoclassical_params
        chi_nc = chang_hinton_chi_profile(
            self.rho, self.Ti, self.ne, p["q_profile"], p["R0"], p["a"], p["B0"], p["A_ion"], p["Z_eff"]
        )
        transport_mode = getattr(self, "transport_model", "gyro_bohm")
        separate_channels = transport_mode in {"gyrokinetic", "tglf_native", "external_gk"}
        if transport_mode == "gyrokinetic":
            from scpn_control.core.gyrokinetic_transport import GyrokineticTransportModel

            gk_model = GyrokineticTransportModel()
            Te_gk = np.maximum(self.Te, 1e-6)
            Ti_gk = np.maximum(self.Ti, 1e-6)
            ne_gk = np.maximum(self.ne, 1e-6)
            dTe_dr = np.gradient(Te_gk, self.rho * p["a"])
            dTi_dr = np.gradient(Ti_gk, self.rho * p["a"])
            dne_dr = np.gradient(ne_gk, self.rho * p["a"])
            profiles_dict = {
                "R0": p["R0"],
                "a": p["a"],
                "B0": p["B0"],
                "q": p["q_profile"],
                "Te": Te_gk,
                "Ti": Ti_gk,
                "ne": ne_gk,
                "dTe_dr": dTe_dr,
                "dTi_dr": dTi_dr,
                "dne_dr": dne_dr,
                "Z_eff": p.get("Z_eff", 1.5),
            }
            chi_i_gk, chi_e_gk, D_e_gk = gk_model.evaluate_profile(self.rho, profiles_dict)
            chi_gB = chi_i_gk  # Use for base ion chi
            self.chi_e = chi_e_gk  # Update electron chi directly
            self.D_n = D_e_gk  # Update particle diffusivity directly
        elif transport_mode == "tglf_native":
            chi_gB = self._tglf_native_transport(p)
        elif transport_mode == "external_gk":
            chi_gB = self._external_gk_transport(p)
        else:
            chi_gB = self._gyro_bohm_chi()
        chi_base = chi_nc + chi_gB
    else:
        if not (self.allow_constant_transport_fallback and self.allow_legacy_approximations):
            raise RuntimeError(
                "neoclassical transport configuration is required; "
                "set neoclassical parameters or explicitly enable "
                "allow_constant_transport_fallback + allow_legacy_approximations for legacy behaviour"
            )
        chi_base = np.full_like(self.rho, 0.5)

    # Critical-gradient turbulent transport: chi_turb ~ c * max(0, |∇T| - ∇T_crit)
    # c_turb=5.0 m²/s per keV/m: phenomenological fit to TGLF/GKW predictions
    # for ITG-dominated transport (Dimits et al., Phys. Plasmas 7, 969, 2000)
    C_TURB = 5.0  # m²/s per keV/m
    chi_turb = C_TURB * np.maximum(0, -grad_T - threshold)

    # H-mode detection from Martin et al. (2008) with low-density branch
    # correction (Ryter et al. 2014), evaluated from the active discharge
    # state and machine geometry.
    is_H_mode = False
    if self.neoclassical_params is not None:
        from scpn_control.core.lh_transition import MartinThreshold

        p = self.neoclassical_params
        ne_19 = float(np.clip(np.nanmean(self.ne), 0.0, 1e4))
        b_t = float(p.get("B0", 0.0))
        a_m = float(p.get("a", 0.0))
        r0_m = float(p.get("R0", 0.0))
        kappa = float(p.get("kappa", 1.7))
        s_m2 = float(
            p.get(
                "surface_area_m2",
                estimate_plasma_surface_area_m2(r0_m, a_m, kappa),
            )
        )
        i_p_ma = float(
            p.get(
                "Ip_MA",
                self.cfg.get("physics", {}).get("plasma_current_target", 0.0),
            )
        )
        p_lh_mw = MartinThreshold.power_threshold_with_low_density_branch_MW(
            ne_19=ne_19,
            B_T=b_t,
            S_m2=s_m2,
            I_p_MA=i_p_ma,
            a_m=a_m,
        )
        is_H_mode = bool(P_aux > p_lh_mw)

    if is_H_mode and self.neoclassical_params is not None:
        try:
            from scpn_control.core.eped_pedestal import EpedPedestalModel

            p = self.neoclassical_params
            eped = EpedPedestalModel(
                R0=p["R0"],
                a=p["a"],
                B0=p["B0"],
                Ip_MA=p.get("Ip_MA", 15.0),
                kappa=p.get("kappa", 1.7),
                A_ion=p.get("A_ion", 2.0),
                Z_eff=p.get("Z_eff", 1.5),
            )
            # Use current edge density for pedestal prediction
            n_ped = max(float(self.ne[-5]), 1.0)
            ped = eped.predict(n_ped)

            # Apply pedestal: suppress transport inside pedestal region
            ped_start = 1.0 - ped.Delta_ped
            edge_mask = self.rho > ped_start
            chi_turb[edge_mask] *= 0.05  # Strong transport barrier

            # Set pedestal boundary conditions on profiles
            ped_idx = np.searchsorted(self.rho, ped_start)
            if ped_idx < len(self.Te):  # pragma: no branch - searchsorted<nr==len(Te) always; see #129
                self.Te[ped_idx:] = np.minimum(
                    self.Te[ped_idx:], ped.T_ped_keV * np.linspace(1.0, 0.1, len(self.Te[ped_idx:]))
                )
                self.Ti[ped_idx:] = np.minimum(
                    self.Ti[ped_idx:], ped.T_ped_keV * np.linspace(1.0, 0.1, len(self.Ti[ped_idx:]))
                )
        except (ImportError, ValueError, IndexError, AttributeError):
            edge_mask = self.rho > 0.9
            chi_turb[edge_mask] *= 0.1

    if separate_channels:
        self.chi_i = chi_base
    else:
        self.chi_e = chi_base + chi_turb
        self.chi_i = chi_base + chi_turb
        self.D_n[:] = 0.1 * self.chi_e
