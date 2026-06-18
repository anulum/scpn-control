# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851  Contact: protoscience@anulum.li
"""
Top-level time-dependent tokamak shot simulator.

Advances the coupled (psi, Te, Ti, ne) state on a 1D rho grid using
Strang operator splitting (Strang 1968, SIAM J. Numer. Anal. 5, 506):

    a) half-step transport diffusion
    b) full-step sources  (heating, CD, radiation)
    c) full-step MHD events (sawtooth, NTM)
    d) half-step transport diffusion

Transport equation (Wesson 2011, "Tokamaks" 4th ed., Ch. 14):

    (3/2) n d(T)/dt = (1/r) d/dr [ r (chi_neo + chi_anom) n dT/dr ] + S_heat

Current diffusion uses full Sauter bootstrap (Sauter et al. 1999,
Phys. Plasmas 6, 2834).  NTM island width follows the Modified Rutherford
Equation (La Haye 2006, Phys. Plasmas 13, 055501).

Key references
--------------
Strang 1968    : G. Strang, SIAM J. Numer. Anal. 5, 506 (1968).
Spitzer 1962   : L. Spitzer, "Physics of Fully Ionized Gases", Interscience (1962).
Wesson 2011    : J. Wesson, "Tokamaks", 4th ed., Oxford (2011), Ch. 14.
Jardin 2010    : S. Jardin, "Computational Methods in Plasma Physics",
                  CRC Press (2010).
Sauter 1999    : O. Sauter et al., Phys. Plasmas 6, 2834 (1999).
"""

from __future__ import annotations

from typing import Any

import json
import hashlib
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scipy.integrate import trapezoid

from scpn_control.core.current_diffusion import (
    CurrentDiffusionSolver,
    psi_from_q,
    q_from_psi,
)
from scpn_control.core.current_drive import CurrentDriveMix, ECCDSource, NBISource
from scpn_control.core.elm_model import ELMCrashModel, ELMCycler, PeelingBallooningBoundary
from scpn_control.core.integrated_transport_solver import (
    TransportSolver,
    chang_hinton_chi_profile,
)
from scpn_control.core.lh_transition import MartinThreshold
from scpn_control.core.neoclassical import sauter_bootstrap
from scpn_control.core.ntm_dynamics import NTMController, NTMIslandDynamics, find_rational_surfaces
from scpn_control.core.sawtooth import SawtoothCycler
from scpn_control.core.sol_model import TwoPointSOL
from scpn_control.core.stability_mhd import (
    QProfile,
    ballooning_stability,
    troyon_beta_limit,
)
from scpn_control.core.tearing_mode_coupling import SawtoothNTMSeeding
from scpn_control.core.gk_interface import GKOutput
from scpn_control.phase.gk_upde_bridge import adaptive_knm, gk_natural_frequencies

# ── Physical constants (CODATA 2018) ─────────────────────────────────────────
_E_CHARGE: float = 1.602176634e-19  # C
_LN_LAMBDA: float = 17.0  # Coulomb logarithm, Wesson 2011, Ch. 14
_MU_0: float = 4.0 * np.pi * 1e-7  # H/m

# Spitzer 1962, "Physics of Fully Ionized Gases" — resistivity coefficient
# η = SPITZER_COEFF * Z_eff * ln_Λ / T_e^1.5   [Ω·m],  T_e in keV
_SPITZER_COEFF: float = 1.65e-9

# Gyro-Bohm anomalous diffusivity coefficient, ITPA Transport DB,
# Nucl. Fusion 39, 2175 (1999), Table II — c_gB = 0.1
_C_GB: float = 0.1

# Minimum diffusivity floor [m^2/s]: prevents singular matrices
_CHI_FLOOR: float = 0.01

# NTM seeding threshold [m]: islands below this do not grow, La Haye 2006, Eq. 6
_W_SEED: float = 5e-3

# Island width above which local T/n flattening is applied [m]
_W_FLAT: float = 1e-2

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


@dataclass
class ScenarioState:
    time: float
    rho: AnyFloatArray
    Te: AnyFloatArray
    Ti: AnyFloatArray
    ne: AnyFloatArray
    q: AnyFloatArray
    psi: AnyFloatArray
    j_total: AnyFloatArray
    j_bs: AnyFloatArray
    j_cd: AnyFloatArray

    Ip_MA: float
    beta_N: float
    tau_E: float
    P_loss: float
    W_thermal: float
    li: float

    ballooning_stable: bool
    troyon_stable: bool
    ntm_island_widths: dict[str, float]

    T_target: float
    q_peak: float
    detached: bool

    last_crash_time: float
    n_crashes: int


@dataclass(frozen=True)
class ScenarioModuleExchange:
    """Per-module state exchange record for replayable scenario artefacts."""

    time_s: float
    module: str
    inputs: dict[str, JsonScalar]
    outputs: dict[str, JsonScalar]
    diagnostics: dict[str, JsonScalar]


@dataclass(frozen=True)
class ScenarioCouplingMetadata:
    """Replay metadata for a completed integrated-scenario trajectory."""

    schema_version: str
    scenario_name: str
    config_sha256: str
    n_steps: int
    dt_s: float
    t_start_s: float
    t_end_s: float
    enabled_modules: tuple[str, ...]
    max_abs_current_deviation_MA: float
    max_relative_thermal_energy_step: float
    all_profiles_finite: bool
    strictly_monotonic_time: bool
    exchange_count: int
    claim_status: str


@dataclass(frozen=True)
class ScenarioCouplingAudit:
    """Fail-closed audit result for an integrated-scenario replay."""

    passed: bool
    metadata: ScenarioCouplingMetadata
    module_exchanges: tuple[ScenarioModuleExchange, ...]
    violations: tuple[str, ...]


def iter_baseline_scenario() -> ScenarioConfig:
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


def _finite_profile_set(state: ScenarioState) -> bool:
    profiles = (
        state.rho,
        state.Te,
        state.Ti,
        state.ne,
        state.q,
        state.psi,
        state.j_total,
        state.j_bs,
        state.j_cd,
    )
    return all(np.all(np.isfinite(np.asarray(profile, dtype=float))) for profile in profiles)


def _relative_energy_steps(states: list[ScenarioState]) -> FloatArray:
    energies = np.asarray([state.W_thermal for state in states], dtype=float)
    if len(energies) < 2:
        return np.zeros(0, dtype=float)
    scale = np.maximum(np.abs(energies[:-1]), 1.0)
    return np.asarray(np.abs(np.diff(energies)) / scale, dtype=float)


def _exchange_for_module(
    module: str,
    state: ScenarioState,
    config: ScenarioConfig,
) -> ScenarioModuleExchange:
    common_inputs: dict[str, JsonScalar] = {
        "dt_s": float(config.dt),
        "time_s": float(state.time),
        "rho_points": int(len(state.rho)),
    }
    outputs: dict[str, JsonScalar]
    diagnostics: dict[str, JsonScalar]
    if module == "transport":
        outputs = {
            "Te_axis_keV": float(state.Te[0]),
            "Ti_axis_keV": float(state.Ti[0]),
            "ne_axis_1e19_m3": float(state.ne[0]),
        }
        diagnostics = {
            "Te_min_keV": float(np.min(state.Te)),
            "Ti_min_keV": float(np.min(state.Ti)),
            "ne_min_1e19_m3": float(np.min(state.ne)),
        }
    elif module == "current_diffusion":
        outputs = {
            "Ip_MA": float(state.Ip_MA),
            "q_axis": float(state.q[0]),
            "q_edge": float(state.q[-1]),
        }
        diagnostics = {
            "q_min": float(np.min(state.q)),
            "q_max": float(np.max(state.q)),
            "li": float(state.li),
        }
    elif module == "bootstrap_current":
        outputs = {
            "j_bs_axis_A_m2": float(state.j_bs[0]),
            "j_bs_volume_proxy_A_m2": float(trapezoid(state.j_bs * state.rho, state.rho) * 2.0),
        }
        diagnostics = {"j_bs_finite": bool(np.all(np.isfinite(state.j_bs)))}
    elif module == "current_drive":
        outputs = {
            "j_cd_axis_A_m2": float(state.j_cd[0]),
            "j_cd_volume_proxy_A_m2": float(trapezoid(state.j_cd * state.rho, state.rho) * 2.0),
        }
        diagnostics = {"j_cd_finite": bool(np.all(np.isfinite(state.j_cd)))}
    elif module == "sol":
        outputs = {
            "T_target_eV": float(state.T_target),
            "q_parallel_MW_m2": float(state.q_peak),
            "detached": bool(state.detached),
        }
        diagnostics = {"sol_enabled": bool(config.include_sol)}
    elif module == "mhd_stability":
        outputs = {
            "beta_N": float(state.beta_N),
            "ballooning_stable": bool(state.ballooning_stable),
            "troyon_stable": bool(state.troyon_stable),
        }
        diagnostics = {"tau_E_s": float(state.tau_E)}
    elif module == "sawtooth":
        outputs = {
            "n_crashes": int(state.n_crashes),
            "last_crash_time_s": float(state.last_crash_time),
        }
        diagnostics = {"sawtooth_enabled": bool(config.include_sawteeth)}
    elif module == "ntm":
        outputs = {"ntm_island_count": int(len(state.ntm_island_widths))}
        diagnostics = {
            "max_ntm_width_m": float(max(state.ntm_island_widths.values(), default=0.0)),
        }
    elif module == "elm":
        outputs = {"P_loss_W": float(state.P_loss), "W_thermal_J": float(state.W_thermal)}
        diagnostics = {"elm_enabled": bool(config.include_elm)}
    elif module == "auxiliary_heating":
        outputs = {"P_aux_MW": float(config.P_aux_MW), "W_thermal_J": float(state.W_thermal)}
        diagnostics = {"power_balance_loss_W": float(state.P_loss)}
    elif module == "phase_bridge":
        outputs = {"phase_bridge_enabled": bool(config.include_phase_bridge)}
        diagnostics = {"claim": "adaptive K_nm coupling metadata only"}
    else:
        outputs = {"module_enabled": True}
        diagnostics = {"module": module}
    return ScenarioModuleExchange(
        time_s=float(state.time),
        module=module,
        inputs=common_inputs,
        outputs=outputs,
        diagnostics=diagnostics,
    )


def audit_scenario_coupling(
    states: list[ScenarioState],
    config: ScenarioConfig,
    *,
    scenario_name: str = "integrated_scenario",
    current_relative_tolerance: float = 1e-9,
    energy_relative_step_limit: float = 5.0,
) -> ScenarioCouplingAudit:
    """Audit a scenario replay for finite, bounded, traceable module coupling.

    The audit is deliberately a bounded controller-facing contract. It proves
    replay consistency, finite state exchange, declared timestep semantics, and
    conservation-bound diagnostics for a completed trajectory. It is not a
    measured-discharge or external integrated-modelling validation claim.
    """

    config = _validate_config(config)
    if not states:
        metadata = ScenarioCouplingMetadata(
            schema_version="scenario-coupling-audit-v1",
            scenario_name=scenario_name,
            config_sha256=scenario_config_sha256(config),
            n_steps=0,
            dt_s=float(config.dt),
            t_start_s=float(config.t_start),
            t_end_s=float(config.t_start),
            enabled_modules=enabled_scenario_modules(config),
            max_abs_current_deviation_MA=float("inf"),
            max_relative_thermal_energy_step=float("inf"),
            all_profiles_finite=False,
            strictly_monotonic_time=False,
            exchange_count=0,
            claim_status="bounded replay audit only; external trajectory validation required",
        )
        return ScenarioCouplingAudit(False, metadata, tuple(), ("states must not be empty",))

    modules = enabled_scenario_modules(config)
    times = np.asarray([state.time for state in states], dtype=float)
    current_deviation = np.asarray([abs(state.Ip_MA - config.Ip_MA) for state in states], dtype=float)
    energy_steps = _relative_energy_steps(states)
    finite_profiles = all(_finite_profile_set(state) for state in states)
    finite_scalars = all(
        np.isfinite(
            [
                state.time,
                state.Ip_MA,
                state.beta_N,
                state.tau_E,
                state.P_loss,
                state.W_thermal,
                state.li,
                state.T_target,
                state.q_peak,
            ]
        ).all()
        for state in states
    )
    if len(times) > 1:
        monotonic_time = bool(np.all(np.diff(times) > 0.0))
        timestep_consistent = bool(np.allclose(np.diff(times), config.dt, rtol=1e-8, atol=max(1e-10, config.dt * 1e-8)))
    else:
        monotonic_time = bool(np.isfinite(times[0]))
        timestep_consistent = True

    exchanges = tuple(_exchange_for_module(module, state, config) for state in states for module in modules)
    max_current_deviation = float(np.max(current_deviation))
    max_energy_step = float(np.max(energy_steps)) if len(energy_steps) else 0.0
    violations: list[str] = []
    if not finite_profiles or not finite_scalars:
        violations.append("all exchanged profiles and scalar diagnostics must be finite")
    if not monotonic_time:
        violations.append("scenario replay time must be strictly monotonic")
    if not timestep_consistent:
        violations.append("scenario replay timestep must match ScenarioConfig.dt")
    if max_current_deviation > abs(config.Ip_MA) * current_relative_tolerance:
        violations.append("scenario current deviates beyond declared Ip tolerance")
    if any(state.W_thermal <= 0.0 for state in states):
        violations.append("thermal energy must remain positive")
    if max_energy_step > energy_relative_step_limit:
        violations.append("thermal-energy step change exceeds replay audit bound")
    if any(np.any(state.q <= 0.0) for state in states):
        violations.append("safety factor q must remain positive")
    if any(np.any(state.Te <= 0.0) or np.any(state.Ti <= 0.0) or np.any(state.ne <= 0.0) for state in states):
        violations.append("temperature and density profiles must remain positive")

    metadata = ScenarioCouplingMetadata(
        schema_version="scenario-coupling-audit-v1",
        scenario_name=scenario_name,
        config_sha256=scenario_config_sha256(config),
        n_steps=len(states),
        dt_s=float(config.dt),
        t_start_s=float(times[0]),
        t_end_s=float(times[-1]),
        enabled_modules=modules,
        max_abs_current_deviation_MA=max_current_deviation,
        max_relative_thermal_energy_step=max_energy_step,
        all_profiles_finite=bool(finite_profiles and finite_scalars),
        strictly_monotonic_time=monotonic_time,
        exchange_count=len(exchanges),
        claim_status="bounded replay audit only; external trajectory validation required",
    )
    return ScenarioCouplingAudit(not violations, metadata, exchanges, tuple(violations))


def scenario_coupling_audit_to_dict(audit: ScenarioCouplingAudit) -> dict[str, object]:
    """Serialise a scenario-coupling audit to a JSON-safe dictionary."""

    return asdict(audit)


def save_scenario_coupling_report(audit: ScenarioCouplingAudit, path: Path) -> None:
    """Persist a scenario-coupling audit report with deterministic JSON layout."""

    data = scenario_coupling_audit_to_dict(audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _spitzer_resistivity(Te_keV: AnyFloatArray, Z_eff: float) -> FloatArray:
    """Spitzer parallel resistivity profile.

    η = SPITZER_COEFF * Z_eff * ln_Λ / T_e^1.5

    Spitzer 1962, "Physics of Fully Ionized Gases", Interscience, Ch. 5.
    Returns Ω·m; T_e in keV.
    """
    Te_keV = np.asarray(Te_keV, dtype=float)
    if not np.all(np.isfinite(Te_keV)) or np.any(Te_keV < 0.0):
        raise ValueError("Te_keV must contain only finite non-negative values")
    Z_eff = _finite_scalar("Z_eff", Z_eff, positive=True)
    return np.asarray(_SPITZER_COEFF * Z_eff * _LN_LAMBDA / np.maximum(Te_keV, 0.01) ** 1.5)


def _gyro_bohm_chi(
    rho: AnyFloatArray,
    Te: AnyFloatArray,
    Ti: AnyFloatArray,
    ne: AnyFloatArray,
    q: AnyFloatArray,
    a: float,
    B0: float,
) -> FloatArray:
    """Gyro-Bohm anomalous electron/ion thermal diffusivity.

    chi_gB = c_gB * rho_i * v_ti * (rho_i / a)^2 * q^2

    ITPA Transport DB, Nucl. Fusion 39, 2175 (1999), Eq. 1.
    c_gB = 0.1 calibrated to L-mode database.
    """
    rho = _profile_array("rho", rho, np.shape(rho))
    Te = _profile_array("Te", Te, rho.shape, nonnegative=True)
    Ti = _profile_array("Ti", Ti, rho.shape, nonnegative=True)
    ne = _profile_array("ne", ne, rho.shape, nonnegative=True)
    q = _profile_array("q", q, rho.shape)
    if np.any(q <= 0.0):
        raise ValueError("q must be positive everywhere")
    a = _finite_scalar("a", a, positive=True)
    B0 = _finite_scalar("B0", B0, positive=True)
    m_p: float = 1.67262192369e-27  # kg, CODATA 2018
    m_i = 2.0 * m_p  # deuterium
    T_i_J = np.maximum(Ti, 0.01) * 1.602176634e-16  # keV → J
    v_ti = np.sqrt(2.0 * T_i_J / m_i)
    rho_i = m_i * v_ti / (_E_CHARGE * B0)

    # Wesson 2011, Ch. 7 — chi_gB proportional to rho_i * v_ti * (rho_i/a)^2
    chi = _C_GB * rho_i * v_ti * (rho_i / a) ** 2 * np.maximum(q, 0.5) ** 2
    return np.asarray(np.maximum(chi, _CHI_FLOOR))


def _diffusion_step(
    T: AnyFloatArray,
    rho: AnyFloatArray,
    chi: AnyFloatArray,
    ne: AnyFloatArray,
    S: AnyFloatArray,
    dt: float,
    a: float,
) -> FloatArray:
    """Explicit cylindrical thermal diffusion step.

    Solves one half-step of:

        (3/2) n dT/dt = (1/r) d/dr [ r chi n dT/dr ] + S

    where r = rho * a.  Wesson 2011, Ch. 14, Eq. (14.5.1).
    Jardin 2010, "Computational Methods in Plasma Physics", Ch. 7.

    Uses forward Euler for simplicity; dt must satisfy the parabolic CFL:
        dt <= drho^2 / (2 * chi_max)
    The caller (step()) enforces sub-stepping when needed.
    """
    rho = _profile_array("rho", rho, np.shape(rho))
    if rho.ndim != 1 or rho.size < 2 or np.any(np.diff(rho) <= 0.0):
        raise ValueError("rho must be a strictly increasing one-dimensional grid")
    T = _profile_array("T", T, rho.shape, nonnegative=True)
    chi = _profile_array("chi", chi, rho.shape, nonnegative=True)
    ne = _profile_array("ne", ne, rho.shape, nonnegative=True)
    S = _profile_array("S", S, rho.shape)
    dt = _finite_scalar("dt", dt, nonnegative=True)
    a = _finite_scalar("a", a, positive=True)
    nr = len(rho)
    drho = rho[1] - rho[0]
    dr = drho * a

    dT_dt = np.zeros(nr)
    n_si = np.maximum(ne, 0.01) * 1e19  # 10^19 m^-3 → m^-3

    for i in range(1, nr - 1):
        r = rho[i] * a
        # Wesson 2011, Ch. 14 — flux: F = -chi * n * dT/dr
        # (1/r) d/dr(r F) expanded in finite differences:
        r_p = 0.5 * (rho[i] + rho[i + 1]) * a
        r_m = 0.5 * (rho[i] + rho[i - 1]) * a
        chi_p = 0.5 * (chi[i] + chi[i + 1])
        chi_m = 0.5 * (chi[i] + chi[i - 1])
        n_p = 0.5 * (n_si[i] + n_si[i + 1])
        n_m = 0.5 * (n_si[i] + n_si[i - 1])

        flux_p = r_p * chi_p * n_p * (T[i + 1] - T[i]) / dr
        flux_m = r_m * chi_m * n_m * (T[i] - T[i - 1]) / dr

        dT_dt[i] = (flux_p - flux_m) / (r * dr * n_si[i]) + S[i] / n_si[i]

    # Axis: symmetry → dT/dr = 0 at r = 0
    dT_dt[0] = dT_dt[1]
    # Edge: fixed-temperature BC (edge radiates away, T stays low)
    dT_dt[-1] = 0.0

    T_new = T + (2.0 / 3.0) * dt * dT_dt
    # Clamp to physical floor: 0.01 keV
    return np.maximum(T_new, 0.01)


class IntegratedScenarioSimulator:
    """Time-dependent tokamak shot simulator.

    Operator splitting sequence per step() (Strang 1968):
        a) half-step transport diffusion
        b) full-step sources
        c) full-step MHD events
        d) half-step transport diffusion

    State variables carried on 1D rho grid:  Te, Ti, ne, psi (→ q, j).
    """

    def __init__(self, config: ScenarioConfig):
        self.config = _validate_config(config)
        self.time = config.t_start

        self.nr = 50
        self.rho = np.linspace(0, 1, self.nr)

        # Current drive sources
        self.cd_mix = CurrentDriveMix(a=self.config.a)
        self.eccd: ECCDSource | None = None
        if self.config.P_eccd_MW > 0:
            self.eccd = ECCDSource(self.config.P_eccd_MW, self.config.rho_eccd, 0.05)
            self.cd_mix.add_source(self.eccd)

        if self.config.P_nbi_MW > 0:
            self.nbi = NBISource(self.config.P_nbi_MW, self.config.E_nbi_keV, 0.2)
            self.cd_mix.add_source(self.nbi)

        self.cd_solver = CurrentDiffusionSolver(self.rho, self.config.R0, self.config.a, self.config.B0)

        self.sawtooth = SawtoothCycler(self.rho, self.config.R0, self.config.a)
        self.n_crashes = 0
        self.last_crash_time = 0.0

        self.sol = TwoPointSOL(
            self.config.R0,
            self.config.a,
            q95=3.0,
            B_pol=0.5,
            kappa=self.config.kappa,
        )

        # NTM state: rational surface key → (NTMIslandDynamics, island_width)
        self.ntm_widths: dict[str, float] = {}
        self._ntm_islands: dict[str, NTMIslandDynamics] = {}
        self.ntm_controller = NTMController()
        self.ntm_seeding = SawtoothNTMSeeding(self.sawtooth)

        # ELM cycler
        pb_boundary = PeelingBallooningBoundary(
            q95=3.0,
            kappa=self.config.kappa,
            delta=self.config.delta,
            a=self.config.a,
            R0=self.config.R0,
        )
        self.elm_cycler = ELMCycler(pb_boundary, ELMCrashModel())

        # L-H transition threshold
        self.lh_threshold = MartinThreshold()

        # Phase bridge state (populated when include_phase_bridge=True)
        self._phase_K_nm: FloatArray | None = None
        self._phase_omega: FloatArray | None = None
        self._last_gk_output: GKOutput | None = None

    # ── initialisation ────────────────────────────────────────────────────────

    def _setup_transport_solver(self) -> TransportSolver:
        cfg_dict = {
            "reactor_name": "IntegratedScenario",
            "grid_resolution": [33, 33],
            "tokamak": {
                "R0": self.config.R0,
                "a": self.config.a,
                "B0": self.config.B0,
            },
            "dimensions": {
                "R_min": max(self.config.R0 - self.config.a - 0.5, 1e-6),
                "R_max": self.config.R0 + self.config.a + 0.5,
                "Z_min": -self.config.a * self.config.kappa - 0.5,
                "Z_max": self.config.a * self.config.kappa + 0.5,
                "nR": 33,
                "nZ": 33,
            },
            "coils": [],
            "grid": {"nr": self.nr},
            "physics": {
                "plasma_current_target": self.config.Ip_MA,
                "transport_model": "gyro_bohm",
            },
            "control": {"P_aux": self.config.P_aux_MW},
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(cfg_dict, f)
            temp_path = f.name

        solver = TransportSolver(temp_path, nr=self.nr, transport_model=self.config.transport_model)
        solver.Te = np.ones(self.nr) * 1.0  # keV
        solver.Ti = np.ones(self.nr) * 1.0
        solver.ne = np.ones(self.nr) * 5.0  # 10^19 m^-3
        return solver

    def initialize(self, profiles: dict[str, Any] | None = None) -> ScenarioState:
        self.ts_solver = self._setup_transport_solver()
        if profiles:
            if "Te" in profiles:
                self.ts_solver.Te = _profile_array("Te", profiles["Te"], self.rho.shape, nonnegative=True)
            if "Ti" in profiles:
                self.ts_solver.Ti = _profile_array("Ti", profiles["Ti"], self.rho.shape, nonnegative=True)
            if "ne" in profiles:
                self.ts_solver.ne = _profile_array("ne", profiles["ne"], self.rho.shape, nonnegative=True)
            if "psi" in profiles:
                self.cd_solver.psi = _profile_array("psi", profiles["psi"], self.rho.shape)

        if self.config.use_transport_solver:
            q_prof = q_from_psi(
                self.rho,
                self.cd_solver.psi,
                self.config.R0,
                self.config.a,
                self.config.B0,
            )
            self.ts_solver.set_neoclassical(
                R0=self.config.R0,
                a=self.config.a,
                B0=self.config.B0,
                q0=float(q_prof[0]),
                q_edge=float(q_prof[-1]),
            )
            if self.ts_solver.neoclassical_params is not None:
                self.ts_solver.neoclassical_params.update(
                    {
                        "Ip_MA": self.config.Ip_MA,
                        "kappa": self.config.kappa,
                        "surface_area_m2": 4.0
                        * np.pi**2
                        * self.config.R0
                        * self.config.a
                        * np.sqrt((1.0 + self.config.kappa**2) / 2.0),
                    }
                )

        return self._build_state()

    # ── internal physics helpers ──────────────────────────────────────────────

    def _chi_total(self, q: AnyFloatArray) -> FloatArray:
        """Combined neoclassical + gyro-Bohm anomalous diffusivity.

        Wesson 2011, Ch. 14 — total chi = chi_neo + chi_anom.
        """
        chi_neo = chang_hinton_chi_profile(
            self.rho,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q,
            self.config.R0,
            self.config.a,
            self.config.B0,
        )
        chi_anom = _gyro_bohm_chi(
            self.rho,
            self.ts_solver.Te,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q,
            self.config.a,
            self.config.B0,
        )
        return np.asarray(chi_neo + chi_anom)

    def _source_density(self, S_heat_W_m3: AnyFloatArray) -> FloatArray:
        """Convert volumetric power density [W/m^3] to keV/s / (10^19 m^-3)."""
        n_si = np.maximum(self.ts_solver.ne, 0.01) * 1e19
        return np.asarray(S_heat_W_m3 / (_E_CHARGE * 1e3 * n_si))

    def _ohmic_power_density(self, q_prof: AnyFloatArray) -> FloatArray:
        """Ohmic heating power density.

        P_ohm = η_Spitzer * j_ohm^2  where  j_ohm = E_loop / (η * (1 - f_t))
        and E_loop is set by j_ohm = j_total - j_bs - j_cd.

        For each cell we use η_Spitzer (Spitzer 1962) and reconstruct the
        ohmic contribution from the total current profile.
        Wesson 2011, Ch. 14, Eq. (14.2.6).
        """
        eta = _spitzer_resistivity(self.ts_solver.Te, Z_eff=1.5)
        # Trapped-fraction correction for conductivity: σ_∥ → σ_∥ * (1 - f_t)
        # Sauter, Angioni & Lin-Liu 1999, Phys. Plasmas 6, 2834, Eq. 4.
        f_t = np.zeros(self.nr)
        for i in range(self.nr):
            eps = self.rho[i] * self.config.a / self.config.R0
            if eps > 1e-6:
                f_t[i] = 1.0 - (1.0 - eps) ** 2 / (np.sqrt(1.0 - eps**2) * (1.0 + 1.46 * np.sqrt(eps)))

        # E_loop ≈ μ_0 * R0 * V_loop / (2π), approximate with η·j_total
        # j_total reconstructed from psi gradient (Jardin 2010, Ch. 7)
        j_total = self._j_total_from_psi(q_prof)
        P_ohm = eta * j_total**2 * (1.0 - f_t)
        return np.asarray(np.maximum(P_ohm, 0.0))

    def _j_total_from_psi(self, q_prof: AnyFloatArray) -> FloatArray:
        """Approximate toroidal current density from q profile.

        j_tor ≈ B0 * a / (μ_0 * R0) * d/dr(1/q) * (1/r)
        Wesson 2011, Ch. 3, Eq. (3.5.5).
        """
        inv_q = 1.0 / np.maximum(q_prof, 0.5)
        # d(1/q)/dr in physical units: r = rho * a
        dinvq_drho = np.gradient(inv_q, self.rho)
        r = np.maximum(self.rho, 1e-4) * self.config.a
        j_tor = (self.config.B0 * self.config.a / (_MU_0 * self.config.R0)) * dinvq_drho / r
        return np.asarray(j_tor)

    def _aux_source_profile(self, q_prof: AnyFloatArray) -> FloatArray:
        """Gaussian auxiliary heating profile [W/m^3].

        Peaked at rho = 0.3 with sigma = 0.15 for NBI-like deposition.
        Jardin 2010, Ch. 9 — auxiliary power modelling.
        """
        if self.config.P_aux_MW <= 0:
            return np.zeros(self.nr)

        # Deposition profile: Gaussian centred at rho=0.3
        rho_aux: float = 0.3
        sigma_aux: float = 0.15
        profile = np.exp(-((self.rho - rho_aux) ** 2) / (2.0 * sigma_aux**2))
        profile /= max(trapezoid(profile, self.rho), 1e-12)

        # Normalise to total power: P = ∫ S dV,  dV ≈ 2π² R0 a² κ * 2ρ dρ
        # We work per unit normalised volume: S_tot [W] = P_aux * 1e6
        vol_norm = 2.0 * np.pi**2 * self.config.R0 * self.config.a**2 * self.config.kappa
        return np.asarray(
            profile * self.config.P_aux_MW * 1e6 / (vol_norm * trapezoid(2.0 * self.rho * profile, self.rho) + 1e-12)
        )

    def _transport_step(self, dt: float, q_prof: AnyFloatArray) -> None:
        """Advance Te and Ti by one transport half-step.

        When ``config.use_transport_solver`` is True, delegates to the full
        Crank-Nicolson solver (``TransportSolver.evolve_profiles``).
        Otherwise falls back to the explicit Euler sub-stepped diffusion.

        Wesson 2011, "Tokamaks" 4th ed., Ch. 14, Eq. (14.5.1).
        """
        if self.config.use_transport_solver:
            self.ts_solver.update_transport_model(self.config.P_aux_MW)
            self.ts_solver.evolve_profiles(dt, self.config.P_aux_MW)
            return

        chi = self._chi_total(q_prof)

        # CFL sub-stepping, Jardin 2010, Ch. 7
        drho = self.rho[1] - self.rho[0]
        chi_max = float(np.max(chi))
        dt_cfl = 0.5 * (drho * self.config.a) ** 2 / max(chi_max, 1e-10)
        n_sub = max(1, int(np.ceil(dt / dt_cfl)))
        dt_sub = dt / n_sub

        S_heat = self._aux_source_profile(q_prof)
        S_ohm = self._ohmic_power_density(q_prof)
        S_total = S_heat + S_ohm  # W/m^3

        # Source in keV/s units for the diffusion step
        S_Te = self._source_density(S_total * 0.5)  # split equally e/i
        S_Ti = self._source_density(S_total * 0.5)

        for _ in range(n_sub):
            self.ts_solver.Te = _diffusion_step(
                self.ts_solver.Te,
                self.rho,
                chi,
                self.ts_solver.ne,
                S_Te,
                dt_sub,
                self.config.a,
            )
            self.ts_solver.Ti = _diffusion_step(
                self.ts_solver.Ti,
                self.rho,
                chi,
                self.ts_solver.ne,
                S_Ti,
                dt_sub,
                self.config.a,
            )

    def _ntm_step(self, dt: float, q_prof: AnyFloatArray) -> None:
        """NTM island dynamics via the Modified Rutherford Equation.

        La Haye 2006, Phys. Plasmas 13, 055501 — full MRE.
        Rutherford 1973, Phys. Fluids 16, 1903 — original equation.

        For each rational surface with q = m/n:
        1. Compute local j_bs from Sauter model.
        2. Evolve island width w via NTMIslandDynamics.evolve().
        3. If w > _W_FLAT, flatten Te/Ti within ±w of the surface.
        """
        surfaces = find_rational_surfaces(q_prof, self.rho, self.config.a)
        j_bs_profile = sauter_bootstrap(
            self.rho,
            self.ts_solver.Te,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q_prof,
            self.config.R0,
            self.config.a,
            self.config.B0,
        )
        j_cd_profile = self.cd_mix.total_j_cd(self.rho, self.ts_solver.ne, self.ts_solver.Te, self.ts_solver.Ti)

        for surf in surfaces:
            key = f"{surf.m}/{surf.n}"
            w0 = self.ntm_widths.get(key, _W_SEED)

            eta_s = float(
                _spitzer_resistivity(
                    np.array([self.ts_solver.Te[min(int(surf.rho * self.nr), self.nr - 1)]]),
                    Z_eff=1.5,
                )[0]
            )

            # Interpolate j_bs and j_cd to the rational surface
            i_s = int(np.clip(surf.rho * (self.nr - 1), 0, self.nr - 2))
            j_bs_s = float(j_bs_profile[i_s])
            j_cd_s = float(j_cd_profile[i_s])
            j_tor_s = float(abs(self._j_total_from_psi(q_prof)[i_s]))
            j_phi = max(j_tor_s, 1e3)  # A/m^2

            # Pressure gradient at surface: dp/dr ≈ (d/dr)(n_e * (T_e + T_i)) [Pa/m]
            n_si = self.ts_solver.ne * 1e19
            T_total_J = (self.ts_solver.Te + self.ts_solver.Ti) * 1.602176634e-16
            p = n_si * T_total_J
            dp_dr = float(np.gradient(p, self.rho * self.config.a)[i_s])
            B_pol = self.config.B0 * (surf.rho * self.config.a / self.config.R0) / max(q_prof[i_s], 0.1)

            # Bootstrap controller NTM power request
            self.ntm_controller.step(w0, surf.rho)

            if key not in self._ntm_islands:
                self._ntm_islands[key] = NTMIslandDynamics(
                    r_s=surf.r_s,
                    m=surf.m,
                    n=surf.n,
                    a=self.config.a,
                    R0=self.config.R0,
                    B0=self.config.B0,
                    s_hat=surf.shear,
                    q_s=surf.q,
                )

            island = self._ntm_islands[key]
            _, w_arr = island.evolve(
                w0=w0,
                t_span=(0.0, dt),
                dt=dt,
                j_bs=j_bs_s,
                j_phi=j_phi,
                j_cd=j_cd_s,
                eta=eta_s,
                pressure_gradient=dp_dr,
                B_pol=float(B_pol),
            )
            w_new = float(w_arr[-1])
            self.ntm_widths[key] = w_new

            # Flatten Te/Ti in the island if w > threshold
            # Wesson 2011, Ch. 7 — NTM islands suppress temperature gradients
            if w_new > _W_FLAT:
                rho_flat_lo = surf.rho - w_new / self.config.a
                rho_flat_hi = surf.rho + w_new / self.config.a
                mask = (self.rho >= rho_flat_lo) & (self.rho <= rho_flat_hi)
                if mask.sum() > 1:
                    T_mean_e = float(np.mean(self.ts_solver.Te[mask]))
                    T_mean_i = float(np.mean(self.ts_solver.Ti[mask]))
                    self.ts_solver.Te[mask] = T_mean_e
                    self.ts_solver.Ti[mask] = T_mean_i

    def _compute_W_thermal(self) -> float:
        """Total stored thermal energy.

        W = (3/2) ∫ (n_e T_e + n_i T_i) dV
        Wesson 2011, Ch. 1, Eq. (1.1.1).

        Units: n [m^-3], T [J] → W [J].
        """
        vol = 2.0 * np.pi**2 * self.config.R0 * self.config.a**2 * self.config.kappa
        n_si = self.ts_solver.ne * 1e19
        T_e_J = self.ts_solver.Te * 1.602176634e-16
        T_i_J = self.ts_solver.Ti * 1.602176634e-16
        energy_dens = 1.5 * n_si * (T_e_J + T_i_J)
        # Volume-integral over normalised radius: dV/drho = 2π² R0 a² κ * 2ρ
        return float(trapezoid(energy_dens * self.rho, self.rho) * vol * 2.0)

    def _build_state(self) -> ScenarioState:
        q_prof = q_from_psi(self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0)

        W_th = self._compute_W_thermal()
        P_aux_W = self.config.P_aux_MW * 1e6

        j_bs = sauter_bootstrap(
            self.rho,
            self.ts_solver.Te,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q_prof,
            self.config.R0,
            self.config.a,
            self.config.B0,
        )
        j_cd = self.cd_mix.total_j_cd(self.rho, self.ts_solver.ne, self.ts_solver.Te, self.ts_solver.Ti)
        eta_prof = _spitzer_resistivity(self.ts_solver.Te, Z_eff=1.5)

        # Trapped-fraction correction for ohmic current
        f_t = np.zeros(self.nr)
        for i in range(self.nr):
            eps = self.rho[i] * self.config.a / self.config.R0
            if eps > 1e-6:
                f_t[i] = 1.0 - (1.0 - eps) ** 2 / (np.sqrt(1.0 - eps**2) * (1.0 + 1.46 * np.sqrt(eps)))

        # j_ohmic = E_loop / (η * (1 - f_t))
        # Spitzer 1962; Wesson 2011, Ch. 3 — approximate E_loop from η·j_tot
        j_tor = self._j_total_from_psi(q_prof)
        j_ohmic = j_tor - j_bs - j_cd

        # Power balance: dW/dt = P_aux + P_ohm - P_loss
        # P_ohm = ∫ η j_ohm^2 dV
        vol = 2.0 * np.pi**2 * self.config.R0 * self.config.a**2 * self.config.kappa
        P_ohm = float(trapezoid(eta_prof * j_ohmic**2 * self.rho, self.rho) * vol * 2.0)

        P_loss = max(P_aux_W + P_ohm, 1.0)
        tau_E = (W_th / 1e6) / max(P_loss / 1e6, 1.0)

        # INT-7: compute beta_N from profiles — Troyon et al. 1984
        # beta_t = 2 mu_0 <p> / B0^2
        n_si_avg = self.ts_solver.ne * 1e19
        p_avg = n_si_avg * (self.ts_solver.Te + self.ts_solver.Ti) * _E_CHARGE * 1e3
        p_vol = float(trapezoid(p_avg * self.rho, self.rho) * 2.0 / max(trapezoid(self.rho, self.rho), 1e-12))
        beta_t = 2.0 * _MU_0 * p_vol / self.config.B0**2
        I_N = self.config.Ip_MA / max(self.config.a * self.config.B0, 1e-10)
        beta_N = 100.0 * beta_t / max(I_N, 1e-10)

        # INT-7: compute li from current profile — Wesson 2011 Ch. 3
        # li = <B_pol^2> / B_pol(a)^2 ≈ 2 ∫ j^2 rho drho / j_avg^2
        j_tor_sq = j_tor**2
        j2_vol = trapezoid(j_tor_sq * self.rho, self.rho)
        j_vol = trapezoid(np.abs(j_tor) * self.rho, self.rho)
        li = float(j2_vol / max(j_vol**2, 1e-20)) if j_vol > 1e-10 else 1.0

        # INT-5: run stability checks
        ballooning_ok = True
        troyon_ok = True
        if self.config.include_stability:
            shear_prof = np.gradient(q_prof, self.rho) * (self.rho / np.maximum(q_prof, 1e-3))
            # alpha_MHD ≈ -q² R0 dp/dr / (B0²/(2 mu_0))
            p_prof_Pa = n_si_avg * (self.ts_solver.Te + self.ts_solver.Ti) * _E_CHARGE * 1e3
            dp_drho = np.gradient(p_prof_Pa, self.rho)
            alpha_mhd = -(q_prof**2) * self.config.R0 * dp_drho / (self.config.B0**2 / (2.0 * _MU_0))
            alpha_mhd = np.maximum(alpha_mhd, 0.0)

            qp = QProfile(
                rho=self.rho.astype(np.float64),
                q=q_prof.astype(np.float64),
                shear=shear_prof.astype(np.float64),
                alpha_mhd=alpha_mhd.astype(np.float64),
                q_min=float(np.min(q_prof)),
                q_min_rho=float(self.rho[int(np.argmin(q_prof))]),
                q_edge=float(q_prof[-1]),
            )
            ball_res = ballooning_stability(qp)
            ballooning_ok = bool(np.all(ball_res.stable))

            troyon_res = troyon_beta_limit(beta_t, self.config.Ip_MA, self.config.a, self.config.B0)
            troyon_ok = troyon_res.stable_nowall

        T_t, q_peak, detached = 0.0, 0.0, False
        if self.config.include_sol:
            n_u = self.ts_solver.ne[-1]
            sol_power_mw = max(P_loss / 1e6, 1.0e-6)
            sol_res = self.sol.solve(sol_power_mw, n_u, f_rad=0.3)
            T_t = sol_res.T_target_eV
            q_peak = sol_res.q_parallel_MW_m2
            detached = T_t < 5.0

        return ScenarioState(
            time=self.time,
            rho=self.rho.copy(),
            Te=self.ts_solver.Te.copy(),
            Ti=self.ts_solver.Ti.copy(),
            ne=self.ts_solver.ne.copy(),
            q=q_prof.copy(),
            psi=self.cd_solver.psi.copy(),
            j_total=j_bs + j_cd + j_ohmic,
            j_bs=j_bs,
            j_cd=j_cd,
            Ip_MA=self.config.Ip_MA,
            beta_N=beta_N,
            tau_E=tau_E,
            P_loss=P_loss,
            W_thermal=W_th,
            li=li,
            ballooning_stable=ballooning_ok,
            troyon_stable=troyon_ok,
            ntm_island_widths=dict(self.ntm_widths),
            T_target=T_t,
            q_peak=q_peak,
            detached=detached,
            last_crash_time=self.last_crash_time,
            n_crashes=self.n_crashes,
        )

    # ── public step interface ─────────────────────────────────────────────────

    def step(self) -> ScenarioState:
        """Advance one time step using Strang operator splitting.

        Strang 1968, SIAM J. Numer. Anal. 5, 506:
            a) half-step transport
            b) full-step sources
            c) full-step MHD events
            d) half-step transport
        """
        dt = _finite_scalar("dt", self.config.dt, positive=True)
        dt_half = 0.5 * dt

        q_prof = q_from_psi(self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0)

        # ── (a) half-step transport ───────────────────────────────────────────
        self._transport_step(dt_half, q_prof)

        # ── phase bridge: GK output → K_nm modulation ────────────────────────
        if self.config.include_phase_bridge and self._last_gk_output is not None:
            from scpn_control.phase.plasma_knm import build_knm_plasma

            if self._phase_K_nm is None:
                knm_spec = build_knm_plasma()
                self._phase_K_nm = knm_spec.K.copy()
                self._phase_omega = np.zeros(knm_spec.K.shape[0])

            chi_i_profile = self._chi_total(q_prof)
            self._phase_K_nm = adaptive_knm(
                self._phase_K_nm,
                self._last_gk_output,
                chi_i_profile=chi_i_profile,
            )
            if self._phase_omega is not None:
                self._phase_omega = gk_natural_frequencies(
                    self._phase_omega,
                    self._last_gk_output,
                )

        # ── (b) full-step current drive + bootstrap + current diffusion ───────
        j_bs = sauter_bootstrap(
            self.rho,
            self.ts_solver.Te,
            self.ts_solver.Ti,
            self.ts_solver.ne,
            q_prof,
            self.config.R0,
            self.config.a,
            self.config.B0,
        )
        j_cd = self.cd_mix.total_j_cd(self.rho, self.ts_solver.ne, self.ts_solver.Te, self.ts_solver.Ti)

        self.cd_solver.step(dt, self.ts_solver.Te, self.ts_solver.ne, 1.5, j_bs, j_cd)
        q_prof = q_from_psi(self.rho, self.cd_solver.psi, self.config.R0, self.config.a, self.config.B0)

        # ── (c) full-step MHD events ──────────────────────────────────────────
        if self.config.include_sawteeth:
            shear = np.gradient(q_prof, self.rho) * (self.rho / np.maximum(q_prof, 1e-3))
            event = self.sawtooth.step(dt, q_prof, shear, self.ts_solver.Te, self.ts_solver.ne)
            if event:
                self.n_crashes += 1
                self.last_crash_time = event.crash_time

                # INT-1: write crashed q back to psi so current diffusion is consistent
                self.cd_solver.psi = psi_from_q(self.rho, q_prof, self.config.R0, self.config.a, self.config.B0)

                # Also flatten Ti inside the mixing radius (sawtooth only crashes Te+ne)
                if event.rho_mix > 0:
                    idx_mix = int(np.searchsorted(self.rho, event.rho_mix))
                    if idx_mix > 1:
                        rho_inner = self.rho[:idx_mix]
                        vol_int = trapezoid(self.ts_solver.Ti[:idx_mix] * rho_inner, rho_inner)
                        vol_tot = trapezoid(rho_inner, rho_inner)
                        Ti_mix = vol_int / vol_tot if vol_tot > 0 else self.ts_solver.Ti[0]
                        self.ts_solver.Ti[:idx_mix] = Ti_mix

                # INT-6: seed NTM from sawtooth crash energy
                for key, island in self._ntm_islands.items():
                    w_seed = self.ntm_seeding.seed_amplitude(event.seed_energy / 1e6, island.r_s)
                    if w_seed > self.ntm_widths.get(key, 0.0):
                        self.ntm_widths[key] = w_seed

                q_prof = q_from_psi(
                    self.rho,
                    self.cd_solver.psi,
                    self.config.R0,
                    self.config.a,
                    self.config.B0,
                )

        if self.config.include_ntm:
            self._ntm_step(dt, q_prof)
            q_prof = q_from_psi(
                self.rho,
                self.cd_solver.psi,
                self.config.R0,
                self.config.a,
                self.config.B0,
            )

        # ── L-H transition check ──────────────────────────────────────────────
        # Martin et al. 2008, J. Phys.: Conf. Ser. 123, 012033
        S_plasma = 4.0 * np.pi**2 * self.config.R0 * self.config.a * np.sqrt((1.0 + self.config.kappa**2) / 2.0)
        ne_avg = float(np.mean(self.ts_solver.ne))
        P_lh = self.lh_threshold.power_threshold_MW(ne_avg, self.config.B0, S_plasma)
        is_H_mode = self.config.P_aux_MW > P_lh

        # ── ELM check (H-mode only) ──────────────────────────────────────────
        if self.config.include_elm and is_H_mode:
            # Pedestal values at rho ~ 0.95
            i_ped = min(int(0.95 * self.nr), self.nr - 2)
            T_ped = float(self.ts_solver.Te[i_ped])
            n_ped = float(self.ts_solver.ne[i_ped])
            W_ped = self._compute_W_thermal() * 0.3  # pedestal fraction ~30%

            s_edge = float(np.gradient(q_prof, self.rho)[i_ped] * self.rho[i_ped] / max(q_prof[i_ped], 1e-3))
            n_si_edge = n_ped * 1e19
            T_J_edge = T_ped * _E_CHARGE * 1e3
            dp_dr_edge = float(np.gradient(n_si_edge * T_J_edge * np.ones(self.nr), self.rho * self.config.a)[i_ped])
            alpha_edge = float(
                -(q_prof[i_ped] ** 2) * self.config.R0 * dp_dr_edge / (self.config.B0**2 / (2.0 * _MU_0))
            )
            j_edge = float(abs(self._j_total_from_psi(q_prof)[i_ped]))

            elm_event = self.elm_cycler.step(dt, alpha_edge, j_edge, s_edge, T_ped, n_ped, W_ped)
            if elm_event:
                # Apply ELM crash: energy fraction lost from pedestal
                frac_lost = min(elm_event.delta_W_MJ * 1e6 / max(W_ped, 1.0), 0.5)
                drop = np.sqrt(max(1.0 - frac_lost, 0.25))
                ped_mask = self.rho >= 0.9
                self.ts_solver.Te[ped_mask] *= drop
                self.ts_solver.Ti[ped_mask] *= drop
                self.ts_solver.ne[ped_mask] *= drop

        # ── (d) half-step transport ───────────────────────────────────────────
        self._transport_step(dt_half, q_prof)

        self.time += dt
        return self._build_state()

    def run(self) -> list[ScenarioState]:
        if not hasattr(self, "ts_solver"):
            self.initialize()
        n_steps = int((self.config.t_end - self.config.t_start) / self.config.dt)
        states = []
        for _ in range(n_steps):
            states.append(self.step())
        return states

    def to_json(self, path: Path) -> None:
        state = self._build_state()
        data = {
            "time": state.time,
            "rho": state.rho.tolist(),
            "Te": state.Te.tolist(),
            "Ti": state.Ti.tolist(),
            "ne": state.ne.tolist(),
            "q": state.q.tolist(),
            "psi": state.psi.tolist(),
            "j_total": state.j_total.tolist(),
            "j_bs": state.j_bs.tolist(),
            "j_cd": state.j_cd.tolist(),
            "Ip_MA": state.Ip_MA,
            "beta_N": state.beta_N,
            "tau_E": state.tau_E,
            "P_loss": state.P_loss,
            "W_thermal": state.W_thermal,
            "li": state.li,
            "ballooning_stable": state.ballooning_stable,
            "troyon_stable": state.troyon_stable,
            "ntm_island_widths": state.ntm_island_widths,
            "T_target": state.T_target,
            "q_peak": state.q_peak,
            "detached": state.detached,
            "coupling_audit": scenario_coupling_audit_to_dict(
                audit_scenario_coupling([state], self.config, scenario_name="instantaneous_state_export")
            ),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
