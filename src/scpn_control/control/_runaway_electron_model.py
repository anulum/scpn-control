# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Halo Runaway Physics
"""Runaway electron dynamics for bounded disruption simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scpn_control.core._validators import require_finite_float, require_non_negative_float, require_positive_float

_E_CHARGE = 1.602176634e-19  # C
_M_ELECTRON = 9.1093837015e-31  # kg
_C_LIGHT = 299792458.0  # m/s
_MU0 = 4.0 * np.pi * 1e-7  # H/m, vacuum permeability
_EPSILON0 = 8.8541878128e-12  # F/m, vacuum permittivity
_LN_LAMBDA = 15.0  # Coulomb logarithm, Wesson Ch. 14


@dataclass
class RunawayElectronResult:
    """Time-resolved runaway electron simulation output."""

    time_ms: list[float]
    runaway_current_ma: list[float]
    dreicer_rate_per_s: list[float]
    avalanche_rate_per_s: list[float]
    electric_field_v_m: list[float]
    peak_re_current_ma: float
    final_re_current_ma: float
    avalanche_gain: float


class RunawayElectronModel:
    r"""Rosenbluth-Putvinski runaway electron avalanche model.

    Parameters
    ----------
    n_e : float
        Electron density (m^{-3}).
    T_e_keV : float
        Electron temperature (keV).
    z_eff : float
        Effective ion charge.
    major_radius_m : float
        Major radius for loop voltage → E-field.
    """

    def __init__(
        self,
        n_e: float = 1e20,
        T_e_keV: float = 20.0,
        z_eff: float = 1.0,
        major_radius_m: float = 6.2,
        magnetic_field_t: float = 5.3,
        enable_relativistic_losses: bool = True,
        neon_mol: float = 0.0,
    ) -> None:
        self.n_e_free = require_positive_float("n_e", n_e)
        self.T_e0 = require_positive_float("T_e_keV", T_e_keV)
        self.R0 = require_positive_float("major_radius_m", major_radius_m)
        self.B_t = require_positive_float("magnetic_field_t", magnetic_field_t)
        self.enable_relativistic_losses = bool(enable_relativistic_losses)

        # Collision time: based on free electrons
        v_th = np.sqrt(2.0 * self.T_e0 * 1e3 * _E_CHARGE / _M_ELECTRON)
        self.tau_coll = (
            6.0 * np.pi**2 * _EPSILON0**2 * _M_ELECTRON**2 * v_th**3 / (self.n_e_free * _E_CHARGE**4 * _LN_LAMBDA)
        )

        # Dreicer field E_D depends on FREE electron density
        T_e_joules = self.T_e0 * 1e3 * _E_CHARGE
        self.E_D = self.n_e_free * _E_CHARGE**3 * _LN_LAMBDA / (4.0 * np.pi * _EPSILON0**2 * T_e_joules)

        self.neon_mol = 0.0
        self.n_e_tot = self.n_e_free
        self.Z_eff = 1.0
        self.E_c = 0.0
        self.tau_av = 0.0
        self._update_impurity_state(neon_mol=neon_mol, z_eff=z_eff)

    def _update_impurity_state(self, *, neon_mol: float, z_eff: float) -> None:
        """Update impurity-dependent fields used by avalanche dynamics."""
        neon_mol = require_non_negative_float("neon_mol", neon_mol)
        z_eff = require_finite_float("z_eff", z_eff)
        if z_eff < 1.0:
            raise ValueError(f"z_eff must be >= 1.0, got {z_eff!r}")

        # Account for total electron density (free + bound).
        # In a real 800 m^3 vessel (ITER), 0.5 mol is ~3e23 atoms.
        n_neon = neon_mol * 5.0e21
        self.neon_mol = neon_mol
        self.n_e_tot = self.n_e_free + n_neon
        self.Z_eff = z_eff

        # Critical field E_c depends on TOTAL electron density (free + bound).
        self.E_c = self.n_e_tot * _E_CHARGE**3 * _LN_LAMBDA / (4.0 * np.pi * _EPSILON0**2 * _M_ELECTRON * _C_LIGHT**2)

        # Avalanche time constant: slows with Z_eff and density.
        self.tau_av = (_M_ELECTRON * _C_LIGHT / (_E_CHARGE * max(self.E_c, 1e-6)) * _LN_LAMBDA) * (
            1.0 + 1.5 * (self.Z_eff - 1.0)
        )

    def _dreicer_rate(self, E: float, T_e_keV: float) -> float:
        """Primary (Dreicer) runaway generation rate (s^{-1} m^{-3}).

        Based on Connor-Hastie (1975) asymptotic formula.
        """
        if not np.isfinite(E) or not np.isfinite(T_e_keV):
            return 0.0
        if E <= 0 or T_e_keV <= 0.01:
            return 0.0

        T_joules = T_e_keV * 1e3 * _E_CHARGE
        E_D = self.n_e_free * _E_CHARGE**3 * _LN_LAMBDA / (4.0 * np.pi * _EPSILON0**2 * T_joules)

        ratio = E_D / max(E, 1e-6)
        if not np.isfinite(ratio) or ratio <= 0.0:
            return 0.0
        if ratio > 200.0:  # negligible generation
            return 0.0

        # Connor-Hastie: h(Z_eff) = 3(Z_eff+1)/16
        h_z = 3.0 * (self.Z_eff + 1.0) / 16.0
        # nu_eff = sqrt((1+Z_eff) * E_D / (2 * E))
        nu_eff = float(np.sqrt(max((1.0 + self.Z_eff) * ratio / 2.0, 0.0)))

        C_D = 0.35  # Connor-Hastie (1975) Eq. 2.12, 0.2-0.5 range
        ratio_term = float(np.exp(-h_z * np.log(max(ratio, 1e-20))))
        exp_arg = float(np.clip(-ratio / 4.0 - nu_eff, -700.0, 0.0))
        rate = (self.n_e_free / max(self.tau_coll, 1e-20)) * C_D * ratio_term * np.exp(exp_arg)
        if not np.isfinite(rate):
            return 0.0
        return max(float(rate), 0.0)

    def _avalanche_rate(self, E: float, n_re: float) -> float:
        """Secondary (avalanche) runaway multiplication rate (s^{-1} m^{-3}).

        Based on Rosenbluth-Putvinski (1997).
        """
        if not np.isfinite(E) or not np.isfinite(n_re):
            return 0.0
        if E <= self.E_c or n_re <= 0:
            return 0.0

        # Heuristic RMP-induced deconfinement: above Ne > 0.3 mol, stochastic field
        # destroys flux surfaces → avalanche suppressed. Factor 0.001 is empirical;
        # cf. Paz-Soldan et al., Nucl. Fusion 59 (2019) 066025 for SPI/RMP data.
        deconfinement_factor = 1.0
        if self.neon_mol > 0.3:
            deconfinement_factor = 0.001

        growth = n_re * (E / self.E_c - 1.0) / (max(self.tau_av, 1e-20) * _LN_LAMBDA)
        if not np.isfinite(growth):
            return 0.0
        return max(float(growth * deconfinement_factor), 0.0)

    def _momentum_space_growth(self, E: float, n_re: float) -> float:
        """Phenomenological momentum-space diffusion growth (not a true FP solver).

        Empirical: dn/dt ~ n_re (E/E_c - 1)^1.5 / (5 tau_av).
        """
        if not np.isfinite(E) or not np.isfinite(n_re):
            return 0.0
        if E <= self.E_c or n_re <= 0.0:
            return 0.0

        # Effective diffusion coefficient in momentum space
        # Highly sensitive to E-field over critical field
        e_ratio = E / self.E_c

        # Empirical FP-like growth term
        try:
            fp_rate = n_re * max(e_ratio - 1.0, 0.0) ** 1.5 / max(self.tau_av * 5.0, 1e-20)
        except OverflowError:
            return 0.0
        if not np.isfinite(fp_rate):
            return 0.0
        return float(max(fp_rate, 0.0))

    def _relativistic_loss_rate(self, *, E: float, n_re: float) -> float:
        """Synchrotron + bremsstrahlung damping rate [s^-1 m^-3], classical regime."""
        if not self.enable_relativistic_losses:
            return 0.0
        if not np.isfinite(E) or not np.isfinite(n_re):
            return 0.0
        if n_re <= 0.0:
            return 0.0

        e_ratio = max(E / max(self.E_c, 1e-9), 0.0)
        gamma_eff = 1.0 + 4.0 * max(e_ratio - 1.0, 0.0)

        # Empirical loss time scales: order-of-magnitude fits to relativistic
        # synchrotron/bremsstrahlung cooling in ITER-like plasmas.
        # 0.08, 0.12 [s·T²] calibrated to match Martín-Solís et al., NF 57 (2017) 066025.
        tau_sync = 0.08 / max(self.B_t * self.B_t * gamma_eff, 1e-12)
        tau_brem = 0.12 / max((1.0 + 0.08 * self.Z_eff) * (self.n_e_tot / 1e20) * gamma_eff, 1e-12)
        tau_rel = max(min(tau_sync, tau_brem), 1e-6)
        loss = n_re / tau_rel
        if not np.isfinite(loss):
            return 0.0
        return float(max(loss, 0.0))

    def simulate(
        self,
        plasma_current_ma: float = 15.0,
        tau_cq_s: float = 0.01,
        T_e_quench_keV: float = 0.5,
        neon_z_eff: float = 3.0,
        neon_mol: float | None = None,
        duration_s: float = 0.05,
        dt_s: float = 1e-5,
        seed_re_fraction: float = 1e-8,
    ) -> RunawayElectronResult:
        """Run the RE generation model during a current quench.

        The toroidal electric field E is derived from Faraday's law:
            E = (L_p / (2*pi*R0)) * dI_p/dt

        The final integration step is shortened to remain inside
        ``duration_s`` when the duration is not divisible by ``dt_s``.
        """
        plasma_current_ma = require_positive_float("plasma_current_ma", plasma_current_ma)
        tau_cq_s = require_positive_float("tau_cq_s", tau_cq_s)
        T_e = require_positive_float("T_e_quench_keV", T_e_quench_keV)
        duration_s = require_positive_float("duration_s", duration_s)
        dt = require_positive_float("dt_s", dt_s)
        if dt > duration_s:
            raise ValueError(f"dt_s ({dt}) must be <= duration_s ({duration_s})")

        seed_re_fraction = require_finite_float("seed_re_fraction", seed_re_fraction)
        if not (0.0 < seed_re_fraction <= 1.0):
            raise ValueError(f"seed_re_fraction must be in (0, 1], got {seed_re_fraction!r}")
        neon_mol_eff = self.neon_mol if neon_mol is None else require_non_negative_float("neon_mol", neon_mol)
        self._update_impurity_state(neon_mol=neon_mol_eff, z_eff=neon_z_eff)

        n_steps = math.ceil(duration_s / dt)

        Ip = plasma_current_ma * 1e6  # A
        Ip0 = Ip
        L_p = _MU0 * self.R0 * (np.log(8.0 * self.R0 / 2.0) - 2.0 + 0.5)

        # Seed runaway population
        n_re = self.n_e_free * seed_re_fraction

        time_ms: list[float] = []
        re_current_ma: list[float] = []
        dreicer_rates: list[float] = []
        avalanche_rates: list[float] = []
        fp_rates: list[float] = []
        e_field_list: list[float] = []

        for step in range(n_steps):
            t = step * dt
            step_dt = min(dt, duration_s - t)
            time_ms.append(t * 1e3)

            # 1. Plasma current decay (Ohmic component)
            # In a real quench with REs, the total current Ip = I_ohmic + I_re.
            # Only the Ohmic part produces the E-field that drives the quenches.
            I_ohmic = max(Ip - (re_current_ma[-1] * 1e6 if re_current_ma else 0.0), 0.0)
            dI_ohmic_dt = -I_ohmic / tau_cq_s

            # 2. Toroidal electric field (Back-EMF included)
            # E_tor is driven by dI_ohmic/dt. As I_ohmic -> 0 (replaced by I_re), E_tor -> 0.
            E_tor = L_p * abs(dI_ohmic_dt) / (2.0 * np.pi * self.R0)
            e_field_list.append(E_tor)

            # 3. Generation rates
            gamma_D = self._dreicer_rate(E_tor, T_e)
            dreicer_rates.append(gamma_D)
            gamma_av = self._avalanche_rate(E_tor, n_re)
            avalanche_rates.append(gamma_av)
            gamma_FP = self._momentum_space_growth(E_tor, n_re)
            fp_rates.append(gamma_FP)
            relativistic_loss = self._relativistic_loss_rate(E=E_tor, n_re=n_re)

            # 4. Collisional loss
            loss_rate = n_re / max(self.tau_av * 5.0, 1e-12) if E_tor < self.E_c else 0.0
            if not np.isfinite(loss_rate):
                loss_rate = 0.0

            # 5. Evolution
            dn_re = (gamma_D + gamma_av + gamma_FP - loss_rate - relativistic_loss) * step_dt
            if not np.isfinite(dn_re):
                dn_re = 0.0
            n_re = max(n_re + dn_re, 0.0)
            if not np.isfinite(n_re):
                n_re = 0.0

            # 6. Current conversion
            I_re_val = _E_CHARGE * n_re * _C_LIGHT * np.pi * 2.0**2
            # BACK-EMF Limit: RE current cannot exceed total plasma current
            if not np.isfinite(I_re_val):  # pragma: no cover — numerical safety
                I_re_val = Ip0
            I_re_val = min(I_re_val, Ip0)
            re_current_ma.append(I_re_val / 1e6)

            # Total plasma current evolution (slowed by RE conversion)
            Ip += dI_ohmic_dt * step_dt
            Ip = max(Ip, 0.0)

        peak_re = max(re_current_ma) if re_current_ma else 0.0
        final_re = re_current_ma[-1] if re_current_ma else 0.0
        avalanche_gain = n_re / max(self.n_e_free * seed_re_fraction, 1e-30) if n_re > 0 else 1.0

        return RunawayElectronResult(
            time_ms=time_ms,
            runaway_current_ma=re_current_ma,
            dreicer_rate_per_s=dreicer_rates,
            avalanche_rate_per_s=avalanche_rates,
            electric_field_v_m=e_field_list,
            peak_re_current_ma=peak_re,
            final_re_current_ma=final_re,
            avalanche_gain=avalanche_gain,
        )
