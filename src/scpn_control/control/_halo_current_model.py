# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Halo Runaway Physics
"""Halo circuit dynamics for bounded disruption simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scpn_control.core._validators import require_finite_float, require_positive_float

_MU0 = 4.0 * np.pi * 1e-7  # H/m, vacuum permeability


@dataclass
class HaloCurrentResult:
    """Time-resolved halo current simulation output."""

    time_ms: list[float]
    halo_current_ma: list[float]
    plasma_current_ma: list[float]
    tpf_x_ihalo_over_ip: list[float]
    peak_halo_ma: float
    peak_tpf_product: float
    wall_force_mn_m: float


class HaloCurrentModel:
    r"""Fitzpatrick-style L/R circuit halo current model.

    Parameters
    ----------
    plasma_current_ma : float
        Pre-disruption plasma current (MA).
    minor_radius_m : float
        Plasma minor radius (m).
    major_radius_m : float
        Plasma major radius (m).
    wall_resistivity_ohm_m : float
        Wall resistivity (Ohm·m). Default: stainless steel ~7e-7.
    wall_thickness_m : float
        Wall thickness (m).
    tpf : float
        Toroidal peaking factor (1.0 = uniform, up to ~2.5 in severe VDEs).
    contact_fraction : float
        Fraction of plasma cross-section in wall contact (0–1).
    """

    def __init__(
        self,
        plasma_current_ma: float = 15.0,
        minor_radius_m: float = 2.0,
        major_radius_m: float = 6.2,
        wall_resistivity_ohm_m: float = 7e-7,
        wall_thickness_m: float = 0.06,
        tpf: float = 2.0,
        contact_fraction: float = 0.3,
    ) -> None:
        plasma_current_ma = require_positive_float("plasma_current_ma", plasma_current_ma)
        minor_radius_m = require_positive_float("minor_radius_m", minor_radius_m)
        major_radius_m = require_positive_float("major_radius_m", major_radius_m)
        wall_resistivity_ohm_m = require_positive_float("wall_resistivity_ohm_m", wall_resistivity_ohm_m)
        wall_thickness_m = require_positive_float("wall_thickness_m", wall_thickness_m)
        tpf = require_positive_float("tpf", tpf)
        if minor_radius_m >= major_radius_m:
            raise ValueError("minor_radius_m must be smaller than major_radius_m")
        contact_fraction = require_finite_float("contact_fraction", contact_fraction)
        if not (0.0 < contact_fraction <= 1.0):
            raise ValueError(f"contact_fraction must be in (0, 1], got {contact_fraction!r}")

        self.Ip0 = plasma_current_ma * 1e6  # A
        self.a = minor_radius_m
        self.R0 = major_radius_m
        self.eta_wall = wall_resistivity_ohm_m
        self.d_wall = wall_thickness_m
        self.tpf = tpf
        self.f_contact = contact_fraction

        # Derived circuit parameters
        # Halo resistance: R_h = eta * 2*pi*R0 / (d_wall * a * f_contact)
        self.R_h = self.eta_wall * 2.0 * np.pi * self.R0 / (self.d_wall * self.a * max(self.f_contact, 0.01))
        # Halo inductance: L_h = mu0 R0 (ln(8R0/a) - 2 + li/2), li=1 → -1.5
        self.L_h = _MU0 * self.R0 * (np.log(8.0 * self.R0 / self.a) - 1.5)
        # Mutual inductance: M ~ k * sqrt(L_p * L_h), k ~ f_contact
        L_p = _MU0 * self.R0 * (np.log(8.0 * self.R0 / self.a) - 2.0 + 0.5)
        self.M = self.f_contact * np.sqrt(L_p * self.L_h)
        # Halo L/R time constant
        self.tau_h = self.L_h / max(self.R_h, 1e-12)

    def simulate(
        self,
        tau_cq_s: float = 0.01,
        duration_s: float = 0.05,
        dt_s: float = 1e-5,
    ) -> HaloCurrentResult:
        """Run the L/R circuit halo current model.

        Parameters
        ----------
        tau_cq_s : float
            Current quench time constant (s).
        duration_s : float
            Simulation duration (s); the final integration step is shortened
            when the duration is not divisible by ``dt_s``.
        dt_s : float
            Time step (s).
        """
        tau_cq_s = require_positive_float("tau_cq_s", tau_cq_s)
        duration_s = require_positive_float("duration_s", duration_s)
        dt = require_positive_float("dt_s", dt_s)
        if dt > duration_s:
            raise ValueError(f"dt_s ({dt}) must be <= duration_s ({duration_s})")

        n_steps = math.ceil(duration_s / dt)

        Ip = self.Ip0
        Ih = 0.0
        time_ms: list[float] = []
        halo_ma: list[float] = []
        plasma_ma: list[float] = []
        tpf_product: list[float] = []

        for step in range(n_steps):
            t = step * dt
            step_dt = min(dt, duration_s - t)
            time_ms.append(t * 1e3)

            # Plasma current decay (exponential + linear tail)
            dIp_dt = -Ip / tau_cq_s
            Ip += dIp_dt * step_dt
            Ip = max(Ip, 0.0)

            # L/R circuit: L_h dI_h/dt + R_h I_h = M |dI_p/dt|
            driving_emf = self.M * abs(dIp_dt)
            dIh_dt = (driving_emf - self.R_h * Ih) / max(self.L_h, 1e-12)
            Ih += dIh_dt * step_dt
            Ih = max(Ih, 0.0)

            halo_ma.append(Ih / 1e6)
            plasma_ma.append(Ip / 1e6)

            # TPF × (I_halo / I_p0) — ITER limit is 0.75
            # Per ITER DDD convention, denominator is pre-disruption Ip (Ip0),
            # not instantaneous decaying Ip (which would diverge as Ip → 0).
            ratio = self.tpf * Ih / self.Ip0
            tpf_product.append(ratio)

        peak_halo = max(halo_ma)
        peak_tpf = max(tpf_product) if tpf_product else 0.0

        # Electromagnetic wall force: F ~ mu0 * I_halo * I_p / (2*pi*a)
        peak_Ih = peak_halo * 1e6
        wall_force = _MU0 * peak_Ih * self.Ip0 / (2.0 * np.pi * self.a)
        wall_force_mn_m = wall_force / 1e6  # N/m -> MN/m

        return HaloCurrentResult(
            time_ms=time_ms,
            halo_current_ma=halo_ma,
            plasma_current_ma=plasma_ma,
            tpf_x_ihalo_over_ip=tpf_product,
            peak_halo_ma=peak_halo,
            peak_tpf_product=peak_tpf,
            wall_force_mn_m=wall_force_mn_m,
        )
