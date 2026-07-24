# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Integrated scenario coupling audit and report I/O

"""Replay coupling audit, module exchange records, and report persistence.

This leaf owns :class:`ScenarioModuleExchange`, :class:`ScenarioCouplingMetadata`,
:class:`ScenarioCouplingAudit`, the fail-closed :func:`audit_scenario_coupling`
contract, and JSON report serialisation (CTL-G07 R5-S2). Transport micro-physics
and :class:`~scpn_control.core.integrated_scenario.IntegratedScenarioSimulator`
orchestration remain on the owner. Scenario state snapshots stay on the owner;
this leaf accepts duck-typed state views with the declared attribute surface.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.integrate import trapezoid

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.core.integrated_scenario_presets import (
    JsonScalar,
    ScenarioConfig,
    _validate_config,
    enabled_scenario_modules,
    scenario_config_sha256,
)


@runtime_checkable
class ScenarioStateLike(Protocol):
    """Attribute surface required by the scenario-coupling audit contract."""

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


def _finite_profile_set(state: ScenarioStateLike) -> bool:
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


def _relative_energy_steps(states: Sequence[ScenarioStateLike]) -> FloatArray:
    energies = np.asarray([state.W_thermal for state in states], dtype=float)
    if len(energies) < 2:
        return np.zeros(0, dtype=float)
    scale = np.maximum(np.abs(energies[:-1]), 1.0)
    return np.asarray(np.abs(np.diff(energies)) / scale, dtype=float)


def _exchange_for_module(
    module: str,
    state: ScenarioStateLike,
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
    states: Sequence[ScenarioStateLike],
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
