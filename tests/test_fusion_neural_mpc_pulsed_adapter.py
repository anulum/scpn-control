# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Pulsed-shot MPC adapter tests.
from __future__ import annotations

import numpy as np
import pytest

from scpn_control.control.capacitor_bank_state import CapacitorBank, CapacitorBankSpec, PulseSpec
from scpn_control.control.fusion_neural_mpc import (
    ModelPredictiveController,
    NeuralSurrogate,
    PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION,
    PulsedShotMPCAdapter,
)
from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    CapacitorBankTelemetry,
    PulsedPlasmaTelemetry,
    PulsedScenarioAction,
    PulsedScenarioCommand,
    PulsedScenarioScheduler,
    PulsedScenarioSpec,
    PulsedScenarioState,
)


def _mpc() -> ModelPredictiveController:
    surrogate = NeuralSurrogate(n_coils=2, n_state=2, verbose=False)
    surrogate.B = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64)
    return ModelPredictiveController(surrogate, np.array([6.0, 0.0], dtype=np.float64))


def _scheduler(state: PulsedScenarioState = PulsedScenarioState.BURN) -> PulsedScenarioScheduler:
    scheduler = PulsedScenarioScheduler(
        PulsedScenarioSpec(
            min_precharge_energy_J=5.0,
            ramp_current_A=10.0,
            phase_tolerance_rad=0.1,
            spatial_tolerance_m=0.01,
            burn_temperature_eV=100.0,
            min_fusion_power_W=50.0,
            expansion_velocity_m_s=5.0,
            dump_energy_floor_J=0.5,
            recharge_voltage_fraction=0.8,
            cooldown_temperature_eV=10.0,
            cooldown_current_A=1.0,
            min_burn_duration_s=0.0,
        )
    )
    scheduler.state = state
    return scheduler


def _bank(initial_voltage_V: float = 10.0, resistance_ohm: float = 0.05) -> CapacitorBank:
    return CapacitorBank(
        CapacitorBankSpec(
            capacitance_F=1.0,
            inductance_H=1.0,
            series_resistance_ohm=resistance_ohm,
            voltage_max_V=20.0,
            recharge_power_kW=1.0,
        ),
        initial_voltage_V=initial_voltage_V,
    )


@pytest.mark.parametrize("state", [s for s in PulsedScenarioState if s is not PulsedScenarioState.BURN])
def test_non_burn_states_mask_burn_components(state: PulsedScenarioState) -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(state),
        _bank(),
        burn_action_mask=np.array([True, False]),
    )

    action = adapter.step(np.array([5.0, 1.0]), np.array([6.0, 0.0]))
    decision = adapter.explain_last_decision()

    assert action[0] == 0.0
    assert action[1] < 0.0
    assert decision["scheduler_state"] == state.value
    assert decision["burn_components_masked"] is True
    assert decision["safe_action_applied"] is False


def test_infeasible_bank_prevents_burn_command() -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.BURN),
        _bank(initial_voltage_V=0.0, resistance_ohm=1.0),
        burn_action_mask=np.array([True, True]),
        pulse_duration_s=1.0,
    )

    action = adapter.step(np.array([5.0, 1.0]), np.array([6.0, 0.0]))
    decision = adapter.explain_last_decision()

    assert np.array_equal(action, np.zeros(2))
    assert decision["bank_feasible"] is False
    assert decision["safe_action_applied"] is True
    assert decision["constraint_slack"] < 0.0


def test_explicit_feasible_pulse_is_admitted_during_burn() -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.BURN),
        _bank(initial_voltage_V=10.0, resistance_ohm=0.01),
        burn_action_mask=np.array([True, True]),
    )

    action = adapter.step(
        np.array([5.0, 1.0]),
        np.array([6.0, 0.0]),
        pulse=PulseSpec(peak_current_A=0.5, duration_s=0.001),
    )
    decision = adapter.explain_last_decision()

    assert action[0] > 0.0
    assert action[1] < 0.0
    assert decision["bank_feasible"] is True
    assert decision["safe_action_applied"] is False
    assert decision["mpc_objective"] >= 0.0
    assert decision["evidence_schema_version"] == PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION
    assert isinstance(decision["admission_digest"], str)
    assert len(decision["admission_digest"]) == 64
    assert len(str(decision["action_sha256"])) == 64
    assert len(str(decision["safe_action_sha256"])) == 64
    assert len(str(decision["burn_action_mask_sha256"])) == 64


def test_step_without_ref_tracks_the_mpc_target() -> None:
    """Omitting ``ref`` keeps the controller's own target instead of overriding it."""
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.RAMP_UP),
        _bank(),
        burn_action_mask=np.array([True, False]),
    )

    action = adapter.step(np.array([5.0, 1.0]))
    decision = adapter.explain_last_decision()

    assert action.shape == (2,)
    assert np.all(np.isfinite(action))
    assert decision["scheduler_state"] == PulsedScenarioState.RAMP_UP.value


def test_invalid_adapter_shapes_fail_closed() -> None:
    with pytest.raises(ValueError, match="burn_action_mask"):
        PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank(), burn_action_mask=np.array([True]))
    with pytest.raises(ValueError, match="safe_action"):
        PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank(), safe_action=np.array([0.0]))
    with pytest.raises(ValueError, match="pulse_duration_s"):
        PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank(), pulse_duration_s=0.0)


def test_explain_requires_prior_decision() -> None:
    adapter = PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank())

    with pytest.raises(RuntimeError, match="no pulsed MPC decision"):
        adapter.explain_last_decision()


def test_decision_digest_changes_when_admission_state_changes() -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.BURN),
        _bank(initial_voltage_V=10.0, resistance_ohm=0.01),
        burn_action_mask=np.array([True, False]),
    )

    adapter.step(
        np.array([5.0, 1.0]),
        np.array([6.0, 0.0]),
        context=PulsedScenarioState.BURN,
        pulse=PulseSpec(peak_current_A=0.5, duration_s=0.001),
    )
    burn_decision = adapter.explain_last_decision()

    adapter.step(
        np.array([5.0, 1.0]),
        np.array([6.0, 0.0]),
        context=PulsedScenarioState.FLAT_TOP,
    )
    masked_decision = adapter.explain_last_decision()

    assert burn_decision["evidence_schema_version"] == PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION
    assert masked_decision["evidence_schema_version"] == PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION
    assert burn_decision["admission_digest"] != masked_decision["admission_digest"]
    assert burn_decision["action_sha256"] != masked_decision["action_sha256"]
    assert masked_decision["burn_components_masked"] is True


def test_ten_tick_campaign_yields_expected_scheduler_and_action_pairs() -> None:
    scheduler = _scheduler(PulsedScenarioState.IDLE)
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        scheduler,
        _bank(initial_voltage_V=10.0, resistance_ohm=0.01),
        burn_action_mask=np.array([True, True]),
    )
    state = np.array([5.0, 1.0], dtype=np.float64)
    ref = np.array([6.0, 0.0], dtype=np.float64)
    expected = [
        (PulsedScenarioState.RAMP_UP, PulsedScenarioAction.RAMP_FIELD, False),
        (PulsedScenarioState.FLAT_TOP, PulsedScenarioAction.HOLD_FLAT_TOP, False),
        (PulsedScenarioState.BURN, PulsedScenarioAction.FIRE_COMPRESSION, True),
        (PulsedScenarioState.EXPANSION, PulsedScenarioAction.RECOVER_ENERGY, False),
        (PulsedScenarioState.DUMP, PulsedScenarioAction.DUMP_RESIDUAL, False),
        (PulsedScenarioState.RECHARGE, PulsedScenarioAction.RECHARGE_BANK, False),
        (PulsedScenarioState.COOL_DOWN, PulsedScenarioAction.COOL_DOWN, False),
        (PulsedScenarioState.IDLE, PulsedScenarioAction.ARM_PRECHARGE, False),
        (PulsedScenarioState.RAMP_UP, PulsedScenarioAction.RAMP_FIELD, False),
        (PulsedScenarioState.FLAT_TOP, PulsedScenarioAction.HOLD_FLAT_TOP, False),
    ]
    observed: list[tuple[PulsedScenarioState, PulsedScenarioAction, bool]] = []

    for tick, (expected_state, expected_action, _) in enumerate(expected):
        command = _advance_scheduler_for_expected_state(scheduler, expected_state, tick)
        action = adapter.step(
            state,
            ref,
            context=command,
            pulse=PulseSpec(peak_current_A=0.5, duration_s=0.001),
        )
        decision = adapter.explain_last_decision()
        burn_admitted = bool(np.any(action != 0.0) and not decision["burn_components_masked"])

        assert command.action is expected_action
        assert command.state is expected_state
        assert decision["scheduler_state"] == expected_state.value
        if expected_state is PulsedScenarioState.BURN:
            assert decision["bank_feasible"] is True
            assert decision["safe_action_applied"] is False
        else:
            assert np.array_equal(action, np.zeros(2))
            assert decision["burn_components_masked"] is True
        observed.append((command.state, command.action, burn_admitted))

    assert observed == expected


def test_all_false_burn_action_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one burn action component"):
        PulsedShotMPCAdapter(
            _mpc(),
            _scheduler(),
            _bank(),
            burn_action_mask=np.array([False, False]),
        )


def test_invalid_pulse_waveform_is_rejected() -> None:
    with pytest.raises(ValueError, match="pulse_waveform must be one of"):
        PulsedShotMPCAdapter(
            _mpc(),
            _scheduler(),
            _bank(),
            pulse_waveform="triangle",  # type: ignore[arg-type]
        )


def test_step_rejects_misshaped_state() -> None:
    adapter = PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank())
    with pytest.raises(ValueError, match="state must be finite and match MPC target"):
        adapter.step(np.array([5.0, 1.0, 0.0]))


def test_step_rejects_nonfinite_state() -> None:
    adapter = PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank())
    with pytest.raises(ValueError, match="state must be finite and match MPC target"):
        adapter.step(np.array([np.nan, 1.0]))


def test_step_rejects_misshaped_ref() -> None:
    adapter = PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank())
    with pytest.raises(ValueError, match="ref must be finite and match MPC target"):
        adapter.step(np.array([5.0, 1.0]), np.array([6.0, 0.0, 0.0]))


def test_step_rejects_nonfinite_planned_action(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = PulsedShotMPCAdapter(_mpc(), _scheduler(PulsedScenarioState.BURN), _bank())
    monkeypatch.setattr(
        adapter.nmpc,
        "plan_trajectory",
        lambda _state: np.array([np.nan, 0.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="MPC action must be finite and match safe_action"):
        adapter.step(np.array([5.0, 1.0]), np.array([6.0, 0.0]))


def test_burn_with_zero_demand_reports_no_burn_demand() -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.BURN),
        _bank(initial_voltage_V=10.0, resistance_ohm=0.01),
        burn_action_mask=np.array([True, True]),
    )
    # state == ref drives the planned action to zero, so no pulse is demanded.
    action = adapter.step(np.array([6.0, 0.0]), np.array([6.0, 0.0]))
    decision = adapter.explain_last_decision()

    assert np.array_equal(action, np.zeros(2))
    assert decision["bank_feasibility"] == "no burn demand"
    assert decision["bank_feasible"] is True
    assert decision["safe_action_applied"] is False
    # constraint_slack falls back to the full bank energy when no pulse is demanded.
    assert decision["constraint_slack"] == pytest.approx(adapter.bank.state.energy_J)


def test_burn_with_bank_guard_disabled_admits_without_feasibility_check() -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.BURN),
        _bank(initial_voltage_V=0.0, resistance_ohm=1.0),
        burn_action_mask=np.array([True, True]),
        refuse_burn_when_uncharged=False,
    )
    action = adapter.step(np.array([5.0, 1.0]), np.array([6.0, 0.0]))
    decision = adapter.explain_last_decision()

    # The bank is fully discharged, yet the action is admitted because the guard is off.
    assert not np.array_equal(action, np.zeros(2))
    assert decision["bank_feasibility"] == "bank guard disabled by policy"
    assert decision["safe_action_applied"] is False
    assert decision["bank_feasible"] is True


@pytest.mark.parametrize(
    "context",
    [
        "burn",
        {"state": "burn"},
    ],
)
def test_scheduler_state_resolves_string_and_mapping_context(context: object) -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.IDLE),
        _bank(initial_voltage_V=10.0, resistance_ohm=0.01),
        burn_action_mask=np.array([True, True]),
    )
    adapter.step(
        np.array([5.0, 1.0]),
        np.array([6.0, 0.0]),
        context=context,
        pulse=PulseSpec(peak_current_A=0.5, duration_s=0.001),
    )
    decision = adapter.explain_last_decision()
    assert decision["scheduler_state"] == PulsedScenarioState.BURN.value


def test_scheduler_state_falls_back_to_scheduler_for_stateless_context() -> None:
    adapter = PulsedShotMPCAdapter(
        _mpc(),
        _scheduler(PulsedScenarioState.FLAT_TOP),
        _bank(),
        burn_action_mask=np.array([True, False]),
    )
    # An opaque object without a usable ``state`` attribute falls back to scheduler.state.
    adapter.step(np.array([5.0, 1.0]), np.array([6.0, 0.0]), context=object())
    decision = adapter.explain_last_decision()
    assert decision["scheduler_state"] == PulsedScenarioState.FLAT_TOP.value
    assert decision["burn_components_masked"] is True


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("mpc_objective", float("inf"), "mpc_objective must be finite"),
        ("constraint_slack", float("nan"), "constraint_slack must be finite"),
        ("peak_current_A", -1.0, "peak_current_A must be finite and non-negative"),
    ],
)
def test_decision_evidence_rejects_nonfinite_fields(field: str, value: float, match: str) -> None:
    adapter = PulsedShotMPCAdapter(_mpc(), _scheduler(), _bank(), burn_action_mask=np.array([True, True]))
    kwargs: dict[str, object] = dict(
        action=np.zeros(2, dtype=np.float64),
        safe_action=np.zeros(2, dtype=np.float64),
        burn_action_mask=np.array([True, True]),
        mpc_objective=1.0,
        constraint_slack=2.0,
        scheduler_state="burn",
        bank_feasibility="ok",
        reason="test",
        bank_feasible=True,
        safe_action_applied=False,
        burn_components_masked=False,
        peak_current_A=0.5,
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        adapter._decision_evidence(**kwargs)  # type: ignore[arg-type]


def _advance_scheduler_for_expected_state(
    scheduler: PulsedScenarioScheduler,
    expected_state: PulsedScenarioState,
    tick: int,
) -> PulsedScenarioCommand:
    bank_energy = 10.0
    bank_voltage = 10.0
    temperature = 150.0
    coil_current = 10.0
    fusion_power = 60.0
    radial_velocity = 6.0
    if expected_state in (PulsedScenarioState.RECHARGE, PulsedScenarioState.COOL_DOWN):
        bank_energy = 0.0
    if expected_state is PulsedScenarioState.COOL_DOWN:
        bank_voltage = 16.0
    if expected_state is PulsedScenarioState.IDLE:
        temperature = 5.0
        coil_current = 0.0

    return scheduler.step(
        t_s=float(tick + 1),
        plasma=PulsedPlasmaTelemetry(
            coil_current_A=coil_current,
            temperature_eV=temperature,
            phase_lock_error_rad=0.0,
            reference_error_m=0.0,
            fusion_power_W=fusion_power,
            radial_velocity_m_s=radial_velocity,
        ),
        bank=CapacitorBankTelemetry(
            voltage_V=bank_voltage,
            voltage_max_V=20.0,
            energy_J=bank_energy,
        ),
    )
