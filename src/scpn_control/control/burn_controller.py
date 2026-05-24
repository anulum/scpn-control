# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Burn controller
"""D-T burn-control, alpha-heating, Lawson, and auxiliary-heating utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_control.core.uncertainty import bosch_hale_reactivity

# ── Fusion physics constants ──────────────────────────────────────────
# E_alpha = 3.52 MeV: DT alpha particle birth energy.
# ITER Physics Basis 1999, Nucl. Fusion 39, 2137, Ch. 2.
E_ALPHA_J = 3.52e6 * 1.602176634e-19  # J

# E_fus = 17.6 MeV: total DT reaction energy release.
# Wesson 2011, "Tokamaks" 4th ed., Eq. 1.2.1.
E_FUS_J = 17.6e6 * 1.602176634e-19  # J

# Lawson criterion (ignition): n τ_E T > 3×10^21 m^-3 s keV.
# Lawson 1957, Proc. Phys. Soc. B 70, 6.
LAWSON_TRIPLE_PRODUCT = 3.0e21  # m^-3 s keV

# Burn fraction reference: f_b ≈ a² n_DT <σv> / (4 v_th).
# Wesson 2011, Eq. 1.7.3.


def _require_positive_scalar(name: str, value: float) -> float:
    """Return a finite positive scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return scalar


def _require_nonnegative_scalar(name: str, value: float) -> float:
    """Return a finite non-negative scalar or fail closed."""
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return scalar


def _require_nonnegative_profile(name: str, values: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Return a finite non-negative profile with an optional exact shape."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must contain non-negative values")
    return arr


def _require_normalised_rho(rho: np.ndarray) -> np.ndarray:
    rho_arr = _require_nonnegative_profile("rho", rho)
    if np.any(rho_arr > 1.0):
        raise ValueError("rho must stay within the normalised interval [0, 1]")
    if rho_arr.size > 1 and np.any(np.diff(rho_arr) <= 0.0):
        raise ValueError("rho must be strictly increasing")
    return rho_arr


class AlphaHeating:
    """Alpha heating power from D-T fusion.

    P_α = n_D n_T <σv> E_α × V.
    ITER Physics Basis 1999, Nucl. Fusion 39, 2137.
    Reactivity <σv>_DT: Bosch & Hale 1992, Nucl. Fusion 32, 611.
    """

    def __init__(self, R0: float, a: float, kappa: float = 1.0):
        self.R0 = _require_positive_scalar("R0", R0)
        self.a = _require_positive_scalar("a", a)
        if self.a >= self.R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")
        self.kappa = _require_positive_scalar("kappa", kappa)
        self.E_alpha_J = E_ALPHA_J

    def power_density(self, ne_20: np.ndarray, Te_keV: np.ndarray, Ti_keV: np.ndarray) -> np.ndarray:
        """Alpha power density [MW/m^3] for 50:50 D-T mixture.

        p_α = n_D n_T <σv>(T_i) E_α.
        ITER Physics Basis 1999, Nucl. Fusion 39, 2137, Eq. (2.2.1).
        """
        ne_arr = _require_nonnegative_profile("ne_20", ne_20)
        te_arr = _require_nonnegative_profile("Te_keV", Te_keV, ne_arr.shape)
        ti_arr = _require_nonnegative_profile("Ti_keV", Ti_keV, ne_arr.shape)

        ne_m3 = ne_arr * 1e20
        nD = ne_m3 / 2.0
        nT = ne_m3 / 2.0

        # <σv>_DT: Bosch & Hale 1992, Nucl. Fusion 32, 611, Table IV
        positive_ti = ti_arr > 0.0
        sigv = np.zeros_like(ti_arr, dtype=float)
        if np.any(positive_ti):
            sigv[positive_ti] = np.asarray(bosch_hale_reactivity(ti_arr[positive_ti]), dtype=float)

        p_alpha_W = nD * nT * sigv * self.E_alpha_J
        return np.asarray(p_alpha_W / 1e6, dtype=float)

    def power(self, ne_20: np.ndarray, Te_keV: np.ndarray, Ti_keV: np.ndarray, rho: np.ndarray) -> float:
        """P_alpha [MW] integrated over plasma volume.

        dV = 4π² R₀ a² κ ρ dρ (torus shell element in normalised radius).
        """
        rho_arr = _require_normalised_rho(rho)
        ne_arr = _require_nonnegative_profile("ne_20", ne_20, rho_arr.shape)
        te_arr = _require_nonnegative_profile("Te_keV", Te_keV, rho_arr.shape)
        ti_arr = _require_nonnegative_profile("Ti_keV", Ti_keV, rho_arr.shape)
        p_dens = self.power_density(ne_arr, te_arr, ti_arr)

        dV = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.kappa * rho_arr

        _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        P_tot = _trapz(p_dens * dV, rho_arr)
        return float(P_tot)

    def Q(self, P_alpha_MW: float, P_aux_MW: float) -> float:
        """Fusion energy gain Q = P_fus / P_aux = 5 P_α / P_aux.

        P_fus = 5 P_α because α carries 1/5 of total DT energy (E_α/E_fus = 3.52/17.6).
        Delayed feedback for alpha-heating runaway:
        Mitarai & Muraoka 1999, Nucl. Fusion 39, 725.
        """
        P_alpha_MW = _require_nonnegative_scalar("P_alpha_MW", P_alpha_MW)
        P_aux_MW = _require_nonnegative_scalar("P_aux_MW", P_aux_MW)
        if P_aux_MW <= 0.0:
            return float("inf") if P_alpha_MW > 0 else 0.0
        return 5.0 * P_alpha_MW / P_aux_MW


def lawson_triple_product(ne_m3: float, tau_E_s: float, T_keV: float) -> float:
    """Return n τ_E T [m^-3 s keV].

    Ignition requires n τ_E T > 3×10^21 m^-3 s keV.
    Lawson 1957, Proc. Phys. Soc. B 70, 6.
    """
    ne_m3 = _require_nonnegative_scalar("ne_m3", ne_m3)
    tau_E_s = _require_positive_scalar("tau_E_s", tau_E_s)
    T_keV = _require_nonnegative_scalar("T_keV", T_keV)
    return ne_m3 * tau_E_s * T_keV


def burn_fraction(n_dt_m3: float, sigv: float, v_th_ms: float, a_m: float) -> float:
    """Approximate DT burn fraction.

    f_b ≈ a² n_DT <σv> / (4 v_th).
    Wesson 2011, "Tokamaks" 4th ed., Eq. 1.7.3.
    """
    n_dt_m3 = _require_nonnegative_scalar("n_dt_m3", n_dt_m3)
    sigv = _require_nonnegative_scalar("sigv", sigv)
    v_th_ms = _require_positive_scalar("v_th_ms", v_th_ms)
    a_m = _require_positive_scalar("a_m", a_m)
    return (a_m**2 * n_dt_m3 * sigv) / (4.0 * v_th_ms)


class BurnStabilityAnalysis:
    """Thermal burn stability based on reactivity exponent.

    Delayed alpha-heating feedback control:
    Mitarai & Muraoka 1999, Nucl. Fusion 39, 725.
    """

    def __init__(self, alpha_heating: AlphaHeating):
        self.alpha_heating = alpha_heating

    def reactivity_exponent(self, Ti_keV: float) -> float:
        """d(ln <σv>) / d(ln T), evaluated via finite difference.

        Stability requires this exponent < 2.
        Mitarai & Muraoka 1999, Nucl. Fusion 39, 725, Eq. (5).
        """
        Ti_keV = _require_nonnegative_scalar("Ti_keV", Ti_keV)
        if Ti_keV <= 0.1:
            return 10.0

        dT = 0.01 * Ti_keV
        sv_arr_plus = np.asarray(bosch_hale_reactivity(np.array([Ti_keV + dT])))
        sv_arr_minus = np.asarray(bosch_hale_reactivity(np.array([Ti_keV - dT])))
        sv_plus = sv_arr_plus[0]
        sv_minus = sv_arr_minus[0]

        if sv_minus <= 0 or sv_plus <= 0:
            return 10.0

        d_ln_sv = np.log(sv_plus) - np.log(sv_minus)
        d_ln_T = np.log(Ti_keV + dT) - np.log(Ti_keV - dT)

        return float(d_ln_sv / d_ln_T)

    def is_thermally_stable(self, Ti_keV: float) -> bool:
        """True if d(ln <σv>)/d(ln T) < 2 (Mitarai & Muraoka 1999)."""
        return self.reactivity_exponent(Ti_keV) < 2.0

    def stability_boundary_keV(self) -> float:
        """T where d(ln <σv>)/d(ln T) = 2 (bisection, 5–30 keV)."""
        T_low = 5.0
        T_high = 30.0

        for _ in range(20):
            T_mid = (T_low + T_high) / 2.0
            if self.reactivity_exponent(T_mid) > 2.0:
                T_low = T_mid
            else:
                T_high = T_mid

        return T_high


class BurnController:
    """PI burn controller targeting Q and T_i.

    Delayed feedback prevents alpha-heating runaway.
    Mitarai & Muraoka 1999, Nucl. Fusion 39, 725.
    P_aux_max = 73 MW matches ITER heating system capacity.
    ITER Physics Basis 1999, Nucl. Fusion 39, 2137, Table I.
    """

    def __init__(self, Q_target: float = 10.0, T_target_keV: float = 20.0, P_aux_max_MW: float = 73.0):
        self.Q_target = _require_positive_scalar("Q_target", Q_target)
        self.T_target = _require_positive_scalar("T_target_keV", T_target_keV)
        self.P_aux_max = _require_positive_scalar("P_aux_max_MW", P_aux_max_MW)

        self.integral_T = 0.0

        self.K_T_p = -5.0  # MW/keV; proportional gain
        self.K_T_i = -1.0  # MW/(keV·s); integral gain

        self.last_P_aux = P_aux_max_MW / 2.0

    def step(self, Q_meas: float, T_meas_keV: float, P_alpha_MW: float, dt: float) -> float:
        _require_nonnegative_scalar("Q_meas", Q_meas)
        T_meas_keV = _require_nonnegative_scalar("T_meas_keV", T_meas_keV)
        _require_nonnegative_scalar("P_alpha_MW", P_alpha_MW)
        dt = _require_positive_scalar("dt", dt)
        # > 30 keV: suppress heating to prevent alpha runaway (Mitarai & Muraoka 1999)
        if T_meas_keV > 30.0:
            self.last_P_aux = 0.0
            return 0.0

        e_T = T_meas_keV - self.T_target
        self.integral_T += e_T * dt

        # P_aux = P_ff + P_fb
        # If we just want to stabilize T:
        P_fb = self.K_T_p * e_T + self.K_T_i * self.integral_T

        # Base power to sustain target
        # Simplified: rely on integral to find it
        P_cmd = self.P_aux_max / 2.0 + P_fb

        P_cmd = np.clip(P_cmd, 0.0, self.P_aux_max)
        self.last_P_aux = P_cmd

        return float(P_cmd)


@dataclass
class BurnPoint:
    Te_keV: float
    P_alpha_MW: float
    P_loss_MW: float
    Q: float
    stable: bool


class SubignitedBurnPoint:
    def __init__(self, alpha_heating: AlphaHeating):
        self.alpha = alpha_heating
        self.stability = BurnStabilityAnalysis(alpha_heating)

    def find_operating_point(self, ne_20: float, P_aux_MW: float, tau_E_s: float) -> list[BurnPoint]:
        """Scan T to find P_α(T) + P_aux = P_loss(T) intersections.

        P_loss = 3 n T V / τ_E (energy balance, 0-D model).
        Lawson criterion: n τ_E T > 3×10^21 m^-3 s keV for ignition.
        Lawson 1957, Proc. Phys. Soc. B 70, 6.
        """
        ne_20 = _require_positive_scalar("ne_20", ne_20)
        P_aux_MW = _require_nonnegative_scalar("P_aux_MW", P_aux_MW)
        tau_E_s = _require_positive_scalar("tau_E_s", tau_E_s)
        T_scan = np.linspace(1.0, 40.0, 400)
        points = []

        V = 2.0 * np.pi**2 * self.alpha.R0 * self.alpha.a**2 * self.alpha.kappa
        e_charge = 1.602e-19

        P_alphas = np.zeros_like(T_scan)
        P_losses = np.zeros_like(T_scan)

        for i, T in enumerate(T_scan):
            ne_arr = np.array([ne_20])
            T_arr = np.array([T])
            p_dens = self.alpha.power_density(ne_arr, T_arr, T_arr)[0]
            P_alphas[i] = p_dens * V

            # P_loss = 3 n T V / τ_E (Lawson 1957, Proc. Phys. Soc. B 70, 6)
            W_J = 3.0 * (ne_20 * 1e20) * (T * 1e3 * e_charge) * V
            P_losses[i] = (W_J / tau_E_s) / 1e6

        P_net = P_alphas + P_aux_MW - P_losses

        # Find zero crossings
        crossings = np.where(np.diff(np.sign(P_net)))[0]

        for idx in crossings:
            T_cross = T_scan[idx]
            P_a = P_alphas[idx]
            P_l = P_losses[idx]
            Q = self.alpha.Q(P_a, P_aux_MW)

            # Stable if dP_net/dT < 0 (loss grows faster than source)
            dP_net_dT = P_net[idx + 1] - P_net[idx]
            stable = dP_net_dT < 0

            points.append(
                BurnPoint(
                    Te_keV=float(T_cross), P_alpha_MW=float(P_a), P_loss_MW=float(P_l), Q=float(Q), stable=bool(stable)
                )
            )

        return points
