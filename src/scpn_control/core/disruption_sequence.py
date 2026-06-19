# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Disruption Sequence Model
"""Thermal-quench, current-quench, runaway-electron, and halo-current disruption sequence model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scpn_control._typing import FloatArray
from scpn_control.core.runaway_electrons import RunawayEvolution, RunawayParams


@dataclass
class DisruptionConfig:
    """Pre-disruption machine and plasma state for a disruption simulation.

    Attributes
    ----------
    R0
        Major radius in metres.
    a
        Minor radius in metres (must be smaller than ``R0``).
    B0
        Toroidal field on axis in tesla.
    kappa
        Plasma elongation (dimensionless).
    Ip_MA
        Pre-disruption plasma current in MA.
    W_th_MJ
        Thermal stored energy in MJ.
    Te_pre_keV
        Pre-disruption electron temperature in keV.
    ne_pre_20
        Pre-disruption electron density in 10²⁰ m⁻³.
    dBr_over_B_trigger
        Radial magnetic-perturbation fraction δB_r/B that triggers the quench.
    """

    R0: float
    a: float
    B0: float
    kappa: float
    Ip_MA: float
    W_th_MJ: float
    Te_pre_keV: float
    ne_pre_20: float
    dBr_over_B_trigger: float

    def __post_init__(self) -> None:
        self.R0 = _finite_scalar("R0", self.R0, positive=True)
        self.a = _finite_scalar("a", self.a, positive=True)
        self.B0 = _finite_scalar("B0", self.B0, positive=True)
        self.kappa = _finite_scalar("kappa", self.kappa, positive=True)
        self.Ip_MA = _finite_scalar("Ip_MA", self.Ip_MA, positive=True)
        self.W_th_MJ = _finite_scalar("W_th_MJ", self.W_th_MJ, positive=True)
        self.Te_pre_keV = _finite_scalar("Te_pre_keV", self.Te_pre_keV, positive=True)
        self.ne_pre_20 = _finite_scalar("ne_pre_20", self.ne_pre_20, positive=True)
        self.dBr_over_B_trigger = _finite_scalar("dBr_over_B_trigger", self.dBr_over_B_trigger, nonnegative=True)
        if self.a >= self.R0:
            raise ValueError("a must be smaller than R0 for tokamak ordering")


@dataclass
class TQResult:
    """Thermal-quench phase outputs.

    Attributes
    ----------
    tau_tq_ms
        Thermal-quench timescale in milliseconds.
    post_tq_Te_eV
        Residual electron temperature after the quench in eV.
    heat_load_MJ_m2
        Peak first-wall heat load in MJ/m².
    """

    tau_tq_ms: float
    post_tq_Te_eV: float
    heat_load_MJ_m2: float


@dataclass
class CQResult:
    """Current-quench phase outputs.

    Attributes
    ----------
    cq_duration_ms
        Current-quench timescale (L/R) in milliseconds.
    Ip_trace
        Plasma-current trace in MA over the simulated steps.
    E_par_trace
        Induced parallel electric-field trace in V/m.
    """

    cq_duration_ms: float
    Ip_trace: FloatArray
    E_par_trace: FloatArray


@dataclass
class REResult:
    """Runaway-electron beam phase outputs.

    Attributes
    ----------
    I_RE_MA
        Runaway-electron beam current in MA.
    W_RE_MJ
        Runaway-electron beam energy in MJ.
    wall_heat_load_MJ_m2
        Localised termination heat load on the wall in MJ/m².
    """

    I_RE_MA: float
    W_RE_MJ: float
    wall_heat_load_MJ_m2: float


@dataclass
class HaloResult:
    """Halo-current phase outputs.

    Attributes
    ----------
    f_halo
        Halo-current fraction of the initial plasma current.
    tpf
        Toroidal peaking factor of the halo current.
    F_z_MN
        Vertical halo force on the vessel in MN.
    F_sideways_MN
        Sideways halo force on the vessel in MN.
    within_iter_limits
        ``True`` if ``f_halo × TPF`` stays below the ITER 0.75 limit.
    """

    f_halo: float
    tpf: float
    F_z_MN: float
    F_sideways_MN: float
    within_iter_limits: bool


@dataclass
class DisruptionResult:
    """Aggregate outputs of a full four-phase disruption sequence.

    Attributes
    ----------
    tq_result
        Thermal-quench phase result.
    cq_result
        Current-quench phase result.
    re_result
        Runaway-electron phase result.
    halo_result
        Halo-current phase result.
    total_duration_ms
        Combined thermal- plus current-quench duration in milliseconds.
    wall_heat_load_MJ_m2
        Combined thermal-quench and runaway termination wall load in MJ/m².
    vessel_force_MN
        Vertical vessel force in MN.
    """

    tq_result: TQResult
    cq_result: CQResult
    re_result: REResult
    halo_result: HaloResult
    total_duration_ms: float
    wall_heat_load_MJ_m2: float
    vessel_force_MN: float


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


class ThermalQuench:
    """Stochastic thermal-quench model (Rechester-Rosenbluth transport).

    Parameters
    ----------
    W_th_MJ
        Thermal stored energy in MJ.
    a
        Minor radius in metres.
    R0
        Major radius in metres.
    q
        Edge safety factor (dimensionless).
    B0
        Toroidal field on axis in tesla.
    """

    def __init__(self, W_th_MJ: float, a: float, R0: float, q: float, B0: float):
        self.W_th_MJ = W_th_MJ
        self.a = a
        self.R0 = R0
        self.q = q
        self.B0 = B0

    def rechester_rosenbluth_chi(self, dBr_over_B: float, v_e: float) -> float:
        """
        Stochastic heat transport chi [m^2/s].
        chi_stoch = v_e * pi * q * R0 * (delta B_r / B)^2
        """
        return v_e * math.pi * self.q * self.R0 * (dBr_over_B**2)

    def quench_timescale(self, dBr_over_B: float, Te_pre_keV: float) -> float:
        """tau_TQ [s]"""
        if dBr_over_B <= 0.0:
            return float("inf")

        # v_e = sqrt(2 * Te / m_e)
        e_charge = 1.602e-19
        m_e = 9.109e-31
        Te_J = Te_pre_keV * 1e3 * e_charge
        v_e = math.sqrt(2.0 * Te_J / m_e)

        chi = self.rechester_rosenbluth_chi(dBr_over_B, v_e)

        # tau ~ a^2 / chi
        return self.a**2 / chi

    def heat_deposition(self, W_th_MJ: float, A_wall_m2: float, peaking_factor: float = 3.0) -> float:
        """Peak heat flux [MJ/m^2]"""
        if A_wall_m2 <= 0.0:
            return float("inf")
        return (W_th_MJ / A_wall_m2) * peaking_factor

    def post_tq_temperature(self, Te_pre_keV: float, tau_tq_ms: float, tau_radiation_ms: float = 0.5) -> float:
        """Residual electron temperature [eV] after stochastic transport and radiation cooling."""
        if Te_pre_keV <= 0.0:
            raise ValueError("Te_pre_keV must be positive")
        if tau_tq_ms < 0.0:
            raise ValueError("tau_tq_ms must be non-negative")
        if tau_radiation_ms <= 0.0:
            raise ValueError("tau_radiation_ms must be positive")

        initial_eV = Te_pre_keV * 1e3
        cold_floor_eV = 5.0
        reference_tq_ms = 0.1
        reference_radiation_ms = 0.5
        stochastic_exposure = 8.0 * math.sqrt(reference_tq_ms / max(tau_tq_ms, 1e-9))
        radiative_exposure = reference_radiation_ms / tau_radiation_ms
        total_exposure = stochastic_exposure + radiative_exposure
        return float(cold_floor_eV + (initial_eV - cold_floor_eV) * math.exp(-total_exposure))


class CurrentQuench:
    """L/R current-quench model with Spitzer post-quench resistivity.

    Parameters
    ----------
    Ip_MA
        Pre-quench plasma current in MA.
    L_plasma_uH
        Plasma self-inductance in microhenries.
    R0
        Major radius in metres.
    a
        Minor radius in metres.
    kappa
        Plasma elongation (dimensionless).
    """

    def __init__(self, Ip_MA: float, L_plasma_uH: float, R0: float, a: float, kappa: float = 1.0):
        self.Ip_MA = Ip_MA
        self.L_plasma_H = L_plasma_uH * 1e-6
        self.R0 = R0
        self.a = a
        self.kappa = kappa

    def resistivity_post_tq(self, Te_eV: float, Z_eff: float) -> float:
        """Spitzer resistivity [Ohm m]"""
        ln_Lambda = 10.0  # Typically lower in cold plasma
        # eta = 1.65e-9 * Z_eff * ln_Lambda / (Te_keV^1.5)
        Te_keV = max(Te_eV / 1000.0, 1e-6)
        return float(1.65e-9 * Z_eff * ln_Lambda / (Te_keV**1.5))

    def cq_timescale(self, Te_eV: float, Z_eff: float) -> float:
        """tau_CQ = L / R_p [ms]"""
        eta = self.resistivity_post_tq(Te_eV, Z_eff)
        # R_p = eta * 2 R0 / (a^2 kappa) — toroidal resistance
        R_p = eta * 2.0 * self.R0 / (self.a**2 * self.kappa)

        tau_s = self.L_plasma_H / R_p
        return tau_s * 1000.0

    def induced_electric_field(self, dIp_dt: float) -> float:
        """E_par = L / (2 pi R0) * |dIp/dt| [V/m] where Ip is in Amps."""
        return self.L_plasma_H / (2.0 * math.pi * self.R0) * abs(dIp_dt)

    def evolve(self, Te_post_tq_eV: float, Z_eff: float, dt: float, n_steps: int) -> CQResult:
        """Integrate the exponential current decay and the induced field.

        Parameters
        ----------
        Te_post_tq_eV
            Post-thermal-quench electron temperature in eV.
        Z_eff
            Effective charge of the post-quench plasma.
        dt
            Time step in seconds.
        n_steps
            Number of time steps.

        Returns
        -------
        CQResult
            The quench duration and the current and electric-field traces.
        """
        tau_cq_ms = self.cq_timescale(Te_post_tq_eV, Z_eff)
        tau_cq_s = tau_cq_ms / 1000.0

        t_arr = np.linspace(0, n_steps * dt, n_steps)
        Ip_trace = self.Ip_MA * 1e6 * np.exp(-t_arr / tau_cq_s)

        dIp_dt = -(self.Ip_MA * 1e6) / tau_cq_s * np.exp(-t_arr / tau_cq_s)
        E_par_trace = self.L_plasma_H / (2.0 * math.pi * self.R0) * np.abs(dIp_dt)

        return CQResult(
            cq_duration_ms=tau_cq_ms,
            Ip_trace=np.asarray(Ip_trace / 1e6),
            E_par_trace=np.asarray(E_par_trace),
        )


class REBeamPhase:
    """Runaway-electron beam current, energy, and termination heat load.

    Parameters
    ----------
    re_evolution
        The runaway-electron avalanche/seed evolution model.
    """

    def __init__(self, re_evolution: RunawayEvolution):
        self.re = re_evolution

    def beam_current(self, n_RE: float, v_par: float, A_beam: float) -> float:
        """I_RE [MA]"""
        e_charge = 1.602e-19
        I_A = n_RE * e_charge * v_par * A_beam
        return I_A / 1e6

    def beam_energy(self, n_RE: float, E_max_MeV: float, V_beam: float) -> float:
        """W_RE [MJ]"""
        e_charge = 1.602e-19
        E_avg_J = (E_max_MeV / 2.0) * 1e6 * e_charge
        W_J = n_RE * V_beam * E_avg_J
        return W_J / 1e6

    def termination_heat_load(self, W_RE_MJ: float, A_deposition_m2: float) -> float:
        """Localised wall heat load when the runaway beam terminates.

        Parameters
        ----------
        W_RE_MJ
            Runaway beam energy in MJ.
        A_deposition_m2
            Deposition footprint area in m².

        Returns
        -------
        float
            Heat load in MJ/m² (``inf`` for a non-positive area).
        """
        if A_deposition_m2 <= 0.0:
            return float("inf")
        return W_RE_MJ / A_deposition_m2


class HaloCurrentModel:
    """Halo-current fractions and vessel forces during a vertical displacement.

    Parameters
    ----------
    Ip_MA
        Pre-disruption plasma current in MA.
    R0
        Major radius in metres.
    B0
        Toroidal field on axis in tesla.
    kappa
        Plasma elongation (dimensionless).
    """

    def __init__(self, Ip_MA: float, R0: float, B0: float, kappa: float):
        self.Ip_MA = Ip_MA
        self.R0 = R0
        self.B0 = B0
        self.kappa = kappa

    def halo_fraction(self, dZ_dt: float, tau_cq_ms: float) -> float:
        """Fraction of initial Ip."""
        # Empirical scaling: faster VDE -> higher f_halo
        # f_halo scales roughly as ~ 1 / tau_cq
        # Usually between 0.1 and 0.5.
        f_halo = 0.5 * math.exp(-tau_cq_ms / 100.0) + 0.1 * abs(dZ_dt)
        return min(max(f_halo, 0.1), 0.6)

    def toroidal_peaking_factor(self, n_mode: int = 1) -> float:
        """TPF for n=1 asymmetry"""
        # Usually 1.2 to 2.5
        return 1.5

    def vertical_force(self, f_halo: float, tpf: float) -> float:
        """Engineering halo-load convention F_z [MN] = I_halo * B_tor * 2πR0 * TPF."""
        I_halo = f_halo * self.Ip_MA * 1e6
        F_N = I_halo * self.B0 * (2.0 * math.pi * self.R0) * tpf
        return F_N / 1e6

    def sideways_force(self, f_halo: float, tpf: float) -> float:
        """F_sideways [MN]"""
        # Roughly comparable to vertical force for n=1
        return self.vertical_force(f_halo, tpf) * 0.5

    def iter_limit_check(self, f_halo: float, tpf: float) -> bool:
        """f_halo * TPF < 0.75"""
        return (f_halo * tpf) < 0.75


class DisruptionSequence:
    """Four-phase disruption simulation: TQ, CQ, runaway beam, and halo forces.

    Parameters
    ----------
    config
        The pre-disruption machine and plasma state.
    """

    def __init__(self, config: DisruptionConfig):
        self.config = config
        self.V_plasma = 2.0 * math.pi**2 * config.R0 * config.a**2 * config.kappa
        self.A_wall = 4.0 * math.pi**2 * config.R0 * config.a * config.kappa  # approx

    def run(self, spi_density_target: float | None = None) -> DisruptionResult:
        """Simulate the full disruption sequence end to end.

        Runs the thermal quench, current quench, runaway-electron evolution
        driven by the induced field, and halo-current forces in turn.

        Parameters
        ----------
        spi_density_target
            Optional shattered-pellet-injection target density in 10²⁰ m⁻³; when
            given it sets the post-quench density and dilutes ``Z_eff`` to 1.0.

        Returns
        -------
        DisruptionResult
            The combined per-phase results and the aggregate wall load,
            duration, and vessel force.
        """
        # Phase 1: TQ
        tq = ThermalQuench(self.config.W_th_MJ, self.config.a, self.config.R0, 3.0, self.config.B0)
        tau_tq_s = tq.quench_timescale(self.config.dBr_over_B_trigger, self.config.Te_pre_keV)
        tau_tq_ms = tau_tq_s * 1000.0

        post_T = tq.post_tq_temperature(self.config.Te_pre_keV, tau_tq_ms)
        tq_load = tq.heat_deposition(self.config.W_th_MJ, self.A_wall, 3.0)

        tq_res = TQResult(tau_tq_ms, post_T, tq_load)

        # Mitigation effect on parameters
        if spi_density_target is not None:
            ne_20 = spi_density_target
            Z_eff = 1.0  # High density SPI (e.g. D2) dilution, or high Z if Ne/Ar
            # For simplicity, just use the target density
        else:
            ne_20 = self.config.ne_pre_20
            Z_eff = 1.5

        # Phase 2: CQ
        cq = CurrentQuench(self.config.Ip_MA, 10.0, self.config.R0, self.config.a, self.config.kappa)

        # We want to simulate ~150 ms
        dt = 1e-3
        n_steps = 150
        cq_res = cq.evolve(post_T, Z_eff, dt, n_steps)

        # Phase 3: RE
        re_params = RunawayParams(ne_20, post_T / 1000.0, 0.0, Z_eff, self.config.B0, self.config.R0)
        re_model = RunawayEvolution(re_params)

        # Initial seed from hot tail
        seed = 1e10  # Fallback
        try:
            from scpn_control.core.runaway_electrons import hot_tail_seed

            seed = hot_tail_seed(self.config.Te_pre_keV, post_T / 1000.0, ne_20, tau_tq_ms)
        except ImportError:
            pass

        n_RE = seed

        # Evolve REs using E_par from CQ
        for E_p in cq_res.E_par_trace:
            n_RE = re_model.step(dt, n_RE, E_p)

        re_phase = REBeamPhase(re_model)
        A_beam = math.pi * (self.config.a / 2) ** 2 * self.config.kappa  # beam cross section
        V_beam = 2.0 * math.pi * self.config.R0 * A_beam
        I_RE = re_phase.beam_current(n_RE, 3e8, A_beam)
        W_RE = re_phase.beam_energy(n_RE, 25.0, V_beam)

        # Conservation clamp
        I_RE = min(I_RE, self.config.Ip_MA)

        re_load = re_phase.termination_heat_load(W_RE, 1.0)  # Localized termination

        re_res = REResult(I_RE, W_RE, re_load)

        # Phase 4: Halo
        halo = HaloCurrentModel(self.config.Ip_MA, self.config.R0, self.config.B0, self.config.kappa)
        f_halo = halo.halo_fraction(0.1, cq_res.cq_duration_ms)
        tpf = halo.toroidal_peaking_factor()

        F_z = halo.vertical_force(f_halo, tpf)
        F_side = halo.sideways_force(f_halo, tpf)
        within = halo.iter_limit_check(f_halo, tpf)

        halo_res = HaloResult(f_halo, tpf, F_z, F_side, within)

        tot_dur = tau_tq_ms + cq_res.cq_duration_ms
        tot_heat = tq_load + re_load

        return DisruptionResult(tq_res, cq_res, re_res, halo_res, tot_dur, tot_heat, F_z)

    def with_mitigation(self, spi_density_target: float) -> DisruptionResult:
        """Run the sequence with shattered-pellet-injection mitigation.

        Parameters
        ----------
        spi_density_target
            SPI target density in 10²⁰ m⁻³.

        Returns
        -------
        DisruptionResult
            The mitigated disruption result.
        """
        return self.run(spi_density_target)
