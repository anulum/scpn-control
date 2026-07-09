# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Integrated Scenario Closed-Loop Wiring

"""Bounded closed-loop controller wiring for integrated scenario demos."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import numpy as np

from scpn_control._typing import AnyFloatArray, FloatArray
from scpn_control.control.scenario_scheduler import FeedforwardController, ScenarioSchedule, ScenarioWaveform
from scpn_control.core.integrated_scenario import (
    IntegratedScenarioSimulator,
    ScenarioConfig,
    ScenarioCouplingAudit,
    audit_scenario_coupling,
    nstx_u_scenario,
    scenario_coupling_audit_to_dict,
)


@dataclass(frozen=True)
class ClosedLoopScenarioStep:
    """One controller-to-plant exchange in the bounded scenario loop.

    Attributes
    ----------
    step_index
        Zero-based controller step index.
    time_s
        Plant time after the integrated-scenario step.
    measured_w_thermal_j
        Thermal energy observed before the controller command.
    target_w_thermal_j
        Thermal-energy target used by the feedback trim.
    thermal_error_fraction
        Normalised ``(target - measured) / target`` error.
    commanded_p_aux_mw
        Auxiliary-heating command emitted by feedforward plus feedback.
    applied_p_aux_mw
        Auxiliary-heating command after actuator bounds.
    ip_ref_ma
        Plasma-current reference from the feedforward schedule.
    beta_n
        Normalised beta after the plant step.
    q_min
        Minimum q value after the plant step.
    w_thermal_j
        Thermal energy after the plant step.
    """

    step_index: int
    time_s: float
    measured_w_thermal_j: float
    target_w_thermal_j: float
    thermal_error_fraction: float
    commanded_p_aux_mw: float
    applied_p_aux_mw: float
    ip_ref_ma: float
    beta_n: float
    q_min: float
    w_thermal_j: float


@dataclass(frozen=True)
class ClosedLoopScenarioResult:
    """Completed bounded integrated-scenario closed-loop result.

    Attributes
    ----------
    schema_version
        Stable result schema identifier for downstream JSON consumers.
    claim_status
        Human-readable claim boundary for the local wiring evidence.
    steps
        Ordered controller-to-plant exchanges captured during the run.
    coupling_audit
        Integrated-scenario replay audit for the produced plant states.
    initial_w_thermal_j
        Thermal energy at plant initialisation.
    target_w_thermal_j
        Thermal-energy target used by the feedback trim.
    final_w_thermal_j
        Thermal energy after the final plant step.
    final_abs_error_fraction
        Absolute final thermal-energy error normalised by the target.
    p_aux_min_mw
        Lower auxiliary-heating actuator bound in megawatts.
    p_aux_max_mw
        Upper auxiliary-heating actuator bound in megawatts.
    """

    schema_version: str
    claim_status: str
    steps: tuple[ClosedLoopScenarioStep, ...]
    coupling_audit: ScenarioCouplingAudit
    initial_w_thermal_j: float
    target_w_thermal_j: float
    final_w_thermal_j: float
    final_abs_error_fraction: float
    p_aux_min_mw: float
    p_aux_max_mw: float


def _finite_scalar(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _bounded_auxiliary_power(command_mw: float, bounds_mw: tuple[float, float]) -> float:
    lower = _finite_scalar("p_aux_bounds_mw[0]", bounds_mw[0], nonnegative=True)
    upper = _finite_scalar("p_aux_bounds_mw[1]", bounds_mw[1], nonnegative=True)
    if upper < lower:
        raise ValueError("p_aux_bounds_mw upper bound must be >= lower bound")
    return float(np.clip(command_mw, lower, upper))


def _constant_schedule(config: ScenarioConfig) -> ScenarioSchedule:
    times = np.array([float(config.t_start), float(config.t_end)], dtype=float)
    return ScenarioSchedule(
        {
            "P_aux": ScenarioWaveform("P_aux", times, np.array([config.P_aux_MW, config.P_aux_MW], dtype=float)),
            "Ip": ScenarioWaveform("Ip", times, np.array([config.Ip_MA, config.Ip_MA], dtype=float)),
        }
    )


def _thermal_feedback(
    target_w_thermal_j: float, feedback_gain_mw: float
) -> Callable[[AnyFloatArray, AnyFloatArray, float, float], FloatArray]:
    target = _finite_scalar("target_w_thermal_j", target_w_thermal_j, positive=True)
    gain = _finite_scalar("feedback_gain_mw", feedback_gain_mw, nonnegative=True)

    def feedback(x: AnyFloatArray, _x_ref: AnyFloatArray, _t: float, _dt: float) -> FloatArray:
        measured = float(np.asarray(x, dtype=float)[0])
        error_fraction = (target - measured) / max(abs(target), 1.0)
        return np.array([gain * error_fraction, 0.0, 0.0], dtype=float)

    return feedback


def _default_closed_loop_config(max_steps: int) -> ScenarioConfig:
    base = nstx_u_scenario()
    steps = max(1, int(max_steps))
    return replace(
        base,
        P_aux_MW=0.5,
        P_eccd_MW=0.0,
        P_nbi_MW=0.0,
        t_start=0.0,
        t_end=float(base.dt * steps),
        include_sawteeth=False,
        include_ntm=False,
        include_sol=False,
        include_elm=False,
        include_stability=True,
        include_phase_bridge=False,
        use_transport_solver=False,
    )


def run_integrated_scenario_closed_loop(
    config: ScenarioConfig | None = None,
    *,
    target_w_thermal_j: float | None = None,
    feedback_gain_mw: float = 25.0,
    p_aux_bounds_mw: tuple[float, float] = (0.0, 20.0),
    max_steps: int | None = None,
) -> ClosedLoopScenarioResult:
    """Run a bounded controller-to-integrated-scenario closed loop.

    The loop uses :class:`~scpn_control.control.scenario_scheduler.FeedforwardController`
    to combine a constant scenario schedule with a thermal-energy feedback trim.
    The command is applied to ``ScenarioConfig.P_aux_MW`` before each
    :class:`~scpn_control.core.integrated_scenario.IntegratedScenarioSimulator`
    plant step, then the completed replay is audited with
    :func:`~scpn_control.core.integrated_scenario.audit_scenario_coupling`.

    This is a deterministic repository wiring contract, not measured-discharge
    validation or facility-control evidence.
    """

    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be >= 1 when provided")
    loop_config = replace(config) if config is not None else _default_closed_loop_config(max_steps or 5)
    simulator = IntegratedScenarioSimulator(loop_config)
    initial_state = simulator.initialize()
    target = (
        _finite_scalar("target_w_thermal_j", target_w_thermal_j, positive=True)
        if target_w_thermal_j is not None
        else float(initial_state.W_thermal * 1.02)
    )
    controller = FeedforwardController(_constant_schedule(loop_config), _thermal_feedback(target, feedback_gain_mw))

    planned_steps = int(np.ceil((loop_config.t_end - loop_config.t_start) / loop_config.dt))
    if max_steps is not None:
        planned_steps = min(planned_steps, int(max_steps))
    steps: list[ClosedLoopScenarioStep] = []
    states = []
    current_state = initial_state
    for step_index in range(planned_steps):
        measurement = np.array([current_state.W_thermal, current_state.beta_N], dtype=float)
        command = controller.step(measurement, current_state.time, loop_config.dt)
        commanded_p_aux = float(command[0])
        applied_p_aux = _bounded_auxiliary_power(commanded_p_aux, p_aux_bounds_mw)
        simulator.config.P_aux_MW = applied_p_aux
        current_state = simulator.step()
        states.append(current_state)
        error_fraction = (target - float(measurement[0])) / max(abs(target), 1.0)
        steps.append(
            ClosedLoopScenarioStep(
                step_index=step_index,
                time_s=float(current_state.time),
                measured_w_thermal_j=float(measurement[0]),
                target_w_thermal_j=target,
                thermal_error_fraction=float(error_fraction),
                commanded_p_aux_mw=commanded_p_aux,
                applied_p_aux_mw=applied_p_aux,
                ip_ref_ma=float(command[1]),
                beta_n=float(current_state.beta_N),
                q_min=float(np.min(current_state.q)),
                w_thermal_j=float(current_state.W_thermal),
            )
        )

    audit = audit_scenario_coupling(
        states,
        simulator.config,
        scenario_name="closed_loop_integrated_scenario",
    )
    final_w = float(states[-1].W_thermal) if states else float(initial_state.W_thermal)
    final_error = abs(target - final_w) / max(abs(target), 1.0)
    return ClosedLoopScenarioResult(
        schema_version="closed-loop-integrated-scenario-v1",
        claim_status="bounded controller-to-plant wiring audit only; external trajectory validation required",
        steps=tuple(steps),
        coupling_audit=audit,
        initial_w_thermal_j=float(initial_state.W_thermal),
        target_w_thermal_j=target,
        final_w_thermal_j=final_w,
        final_abs_error_fraction=float(final_error),
        p_aux_min_mw=float(p_aux_bounds_mw[0]),
        p_aux_max_mw=float(p_aux_bounds_mw[1]),
    )


def closed_loop_scenario_result_to_dict(result: ClosedLoopScenarioResult) -> dict[str, object]:
    """Serialise a closed-loop scenario result to JSON-safe data."""

    return {
        "schema_version": result.schema_version,
        "claim_status": result.claim_status,
        "initial_w_thermal_j": result.initial_w_thermal_j,
        "target_w_thermal_j": result.target_w_thermal_j,
        "final_w_thermal_j": result.final_w_thermal_j,
        "final_abs_error_fraction": result.final_abs_error_fraction,
        "p_aux_min_mw": result.p_aux_min_mw,
        "p_aux_max_mw": result.p_aux_max_mw,
        "steps": [asdict(step) for step in result.steps],
        "coupling_audit": scenario_coupling_audit_to_dict(result.coupling_audit),
    }
