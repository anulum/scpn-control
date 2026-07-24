# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Integrated scenario configuration presets

"""Scenario configuration dataclass, validation helpers, and facility presets.

This leaf owns :class:`ScenarioConfig`, finite-field validators used by the
integrated shot simulator, facility presets (ITER baseline/hybrid, NSTX-U),
and config payload hashing (CTL-G07 R5-S1). Coupling audit and the
:class:`~scpn_control.core.integrated_scenario.IntegratedScenarioSimulator`
orchestration remain on the owner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray

JsonScalar = float | int | str | bool


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _profile_array(
    name: str,
    values: AnyFloatArray,
    shape: tuple[int, ...],
    *,
    nonnegative: bool = False,
) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must match the simulator rho-grid shape")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative everywhere")
    return arr


def _validate_config(config: ScenarioConfig) -> ScenarioConfig:
    R0 = _finite_scalar("R0", config.R0, positive=True)
    a = _finite_scalar("a", config.a, positive=True)
    if a >= R0:
        raise ValueError("a must be smaller than R0 for tokamak ordering")
    _finite_scalar("B0", config.B0, positive=True)
    _finite_scalar("kappa", config.kappa, positive=True)
    delta = _finite_scalar("delta", config.delta)
    if abs(delta) >= 1.0:
        raise ValueError("delta must remain inside the physical triangularity interval (-1, 1)")
    _finite_scalar("Ip_MA", config.Ip_MA, positive=True)
    _finite_scalar("P_aux_MW", config.P_aux_MW, nonnegative=True)
    _finite_scalar("P_eccd_MW", config.P_eccd_MW, nonnegative=True)
    rho_eccd = _finite_scalar("rho_eccd", config.rho_eccd, nonnegative=True)
    if rho_eccd > 1.0:
        raise ValueError("rho_eccd must stay within [0, 1]")
    _finite_scalar("P_nbi_MW", config.P_nbi_MW, nonnegative=True)
    _finite_scalar("E_nbi_keV", config.E_nbi_keV, positive=True)
    _finite_scalar("t_start", config.t_start)
    t_end = _finite_scalar("t_end", config.t_end)
    dt = _finite_scalar("dt", config.dt, positive=True)
    if t_end <= config.t_start:
        raise ValueError("t_end must be greater than t_start")
    if dt > (t_end - config.t_start):
        raise ValueError("dt must not exceed the scenario duration")
    return config


@dataclass
class ScenarioConfig:
    """Configuration of an integrated tokamak scenario simulation.

    Attributes
    ----------
    R0
        Major radius in metres.
    a
        Minor radius in metres.
    B0
        Toroidal field on axis in tesla.
    kappa
        Plasma elongation (dimensionless).
    delta
        Plasma triangularity (dimensionless).
    Ip_MA
        Plasma current in MA.
    P_aux_MW
        Auxiliary heating power in MW.
    P_eccd_MW
        Electron-cyclotron current-drive power in MW.
    rho_eccd
        Normalised radius of the ECCD deposition.
    P_nbi_MW
        Neutral-beam-injection power in MW.
    E_nbi_keV
        Neutral-beam energy in keV.
    t_start
        Scenario start time in seconds.
    t_end
        Scenario end time in seconds.
    dt
        Time step in seconds.
    transport_model
        Name of the transport model used for the profiles.
    include_sawteeth
        Enable the sawtooth-crash model.
    include_ntm
        Enable the neoclassical-tearing-mode model.
    include_sol
        Enable the scrape-off-layer/detachment model.
    include_elm
        Enable the edge-localised-mode model.
    include_stability
        Enable the MHD stability checks.
    include_phase_bridge
        Enable the Kuramoto phase-bridge coupling.
    use_transport_solver
        Use the full transport solver instead of the analytic profiles.
    """

    # Geometry
    R0: float
    a: float
    B0: float
    kappa: float
    delta: float

    # Actuators
    Ip_MA: float
    P_aux_MW: float

    # CD parameters
    P_eccd_MW: float = 0.0
    rho_eccd: float = 0.5
    P_nbi_MW: float = 0.0
    E_nbi_keV: float = 100.0

    # Duration
    t_start: float = 0.0
    t_end: float = 10.0
    dt: float = 0.1

    transport_model: str = "gyro_bohm"

    # Flags
    include_sawteeth: bool = True
    include_ntm: bool = True
    include_sol: bool = True
    include_elm: bool = True
    include_stability: bool = True
    include_phase_bridge: bool = False
    use_transport_solver: bool = False


def iter_baseline_scenario() -> ScenarioConfig:
    """Return the ITER baseline (15 MA, full heating) scenario configuration."""
    return ScenarioConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        delta=0.33,
        Ip_MA=15.0,
        P_aux_MW=50.0,
        P_eccd_MW=17.0,
        rho_eccd=0.5,
        P_nbi_MW=33.0,
        E_nbi_keV=1000.0,
        t_end=100.0,
        dt=1.0,
    )


def iter_hybrid_scenario() -> ScenarioConfig:
    """Return the ITER hybrid (12 MA) scenario configuration."""
    return ScenarioConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        kappa=1.7,
        delta=0.33,
        Ip_MA=12.0,
        P_aux_MW=50.0,
        t_end=100.0,
        dt=1.0,
    )


def nstx_u_scenario() -> ScenarioConfig:
    """Return the NSTX-U spherical-tokamak (low aspect ratio) scenario."""
    return ScenarioConfig(
        R0=0.93,
        a=0.58,
        B0=1.0,
        kappa=2.0,
        delta=0.4,
        Ip_MA=1.0,
        P_aux_MW=10.0,
        t_end=2.0,
        dt=0.01,
    )


def scenario_config_payload(config: ScenarioConfig) -> dict[str, JsonScalar]:
    """Return a stable JSON payload for scenario replay hashing."""
    return {key: value for key, value in asdict(config).items()}


def scenario_config_sha256(config: ScenarioConfig) -> str:
    """Hash the exact scenario configuration used by a replay artefact."""
    payload = json.dumps(scenario_config_payload(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enabled_scenario_modules(config: ScenarioConfig) -> tuple[str, ...]:
    """List physics/control modules participating in the scenario exchange."""
    modules = ["transport", "current_diffusion", "bootstrap_current"]
    if config.P_aux_MW > 0.0:
        modules.append("auxiliary_heating")
    if config.P_eccd_MW > 0.0 or config.P_nbi_MW > 0.0:
        modules.append("current_drive")
    if config.include_sawteeth:
        modules.append("sawtooth")
    if config.include_ntm:
        modules.append("ntm")
    if config.include_sol:
        modules.append("sol")
    if config.include_elm:
        modules.append("elm")
    if config.include_stability:
        modules.append("mhd_stability")
    if config.include_phase_bridge:
        modules.append("phase_bridge")
    return tuple(modules)
