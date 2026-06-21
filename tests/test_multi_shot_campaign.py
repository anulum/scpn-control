# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-shot campaign orchestrator tests.
from __future__ import annotations

from typing import Any

import pytest

from scpn_control.control.capacitor_bank_state import CapacitorBankSpec
from scpn_control.control.multi_shot_campaign import (
    MULTI_SHOT_CAMPAIGN_SCHEMA_VERSION,
    PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION,
    CampaignShotPlan,
    CampaignShotResult,
    CampaignShotSample,
    MultiShotCampaignOrchestrator,
)
from scpn_control.control.pulsed_scenario_scheduler_v2 import (
    CapacitorBankTelemetry,
    PulsedPlasmaTelemetry,
    PulsedScenarioSpec,
)


def _scheduler_spec() -> PulsedScenarioSpec:
    return PulsedScenarioSpec(
        min_precharge_energy_J=100.0,
        ramp_current_A=2.0e6,
        phase_tolerance_rad=0.01,
        spatial_tolerance_m=0.002,
        burn_temperature_eV=1.0e3,
        min_fusion_power_W=2.0e6,
        expansion_velocity_m_s=1.0e3,
        dump_energy_floor_J=40.0,
        recharge_voltage_fraction=0.95,
        cooldown_temperature_eV=20.0,
        cooldown_current_A=1.0e3,
        min_burn_duration_s=0.0,
    )


def _bank_spec() -> CapacitorBankSpec:
    return CapacitorBankSpec(
        capacitance_F=100e-6,
        inductance_H=100e-6,
        series_resistance_ohm=0.5,
        voltage_max_V=10_000.0,
        recharge_power_kW=20.0,
    )


def _plasma(
    coil_current_A: float,
    temperature_eV: float,
    phase_lock_error_rad: float,
    reference_error_m: float,
    fusion_power_W: float,
    radial_velocity_m_s: float,
) -> PulsedPlasmaTelemetry:
    return PulsedPlasmaTelemetry(
        coil_current_A=coil_current_A,
        temperature_eV=temperature_eV,
        phase_lock_error_rad=phase_lock_error_rad,
        reference_error_m=reference_error_m,
        fusion_power_W=fusion_power_W,
        radial_velocity_m_s=radial_velocity_m_s,
    )


def _bank(voltage_V: float, energy_J: float) -> CapacitorBankTelemetry:
    return CapacitorBankTelemetry(voltage_V=voltage_V, voltage_max_V=10_000.0, energy_J=energy_J)


def _admission_digest(label: str) -> str:
    return f"{label:0<64}"[:64]


def _complete_shot(shot_id: str) -> CampaignShotPlan:
    samples = (
        CampaignShotSample(0.0, _plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0), _bank(9800.0, 200.0)),
        CampaignShotSample(1.0e-3, _plasma(2.5e6, 10.0, 0.02, 0.01, 0.0, 0.0), _bank(9800.0, 200.0)),
        CampaignShotSample(2.0e-3, _plasma(2.5e6, 1200.0, 0.004, 0.001, 0.0, 0.0), _bank(9800.0, 200.0)),
        CampaignShotSample(3.0e-3, _plasma(2.5e6, 1500.0, 0.004, 0.001, 3.0e6, 0.0), _bank(9800.0, 200.0)),
        CampaignShotSample(4.0e-3, _plasma(0.0, 200.0, 0.02, 0.01, 0.0, 1500.0), _bank(9800.0, 200.0)),
        CampaignShotSample(5.0e-3, _plasma(0.0, 120.0, 0.02, 0.01, 0.0, 0.0), _bank(2000.0, 20.0)),
        CampaignShotSample(6.0e-3, _plasma(0.0, 40.0, 0.02, 0.01, 0.0, 0.0), _bank(9700.0, 180.0)),
        CampaignShotSample(7.0e-3, _plasma(100.0, 15.0, 0.02, 0.01, 0.0, 0.0), _bank(9800.0, 200.0)),
    )
    return CampaignShotPlan(
        shot_id=shot_id,
        samples=samples,
        initial_bank_voltage_V=5000.0,
        pulsed_mpc_admission_digest=_admission_digest("a"),
    )


def _orchestrator() -> MultiShotCampaignOrchestrator:
    return MultiShotCampaignOrchestrator("campaign-a", _scheduler_spec(), _bank_spec())


def test_campaign_runs_two_complete_shots_with_replay_extensions() -> None:
    report = _orchestrator().run([_complete_shot("shot-001"), _complete_shot("shot-002")])

    assert report["schema_version"] == MULTI_SHOT_CAMPAIGN_SCHEMA_VERSION
    assert report["shot_count"] == 2
    assert report["passed_count"] == 2
    assert report["failed_count"] == 0
    first = report["shots"][0]
    assert report["pulsed_mpc_evidence_schema_version"] == PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION
    assert report["pulsed_mpc_admission_digest_count"] == 2
    assert first["terminal_state"] == "idle"
    assert first["transition_states"] == report["expected_transition_states"]
    assert first["trigger_timestamp_ns"] == 2_000_000
    assert first["energy_recovered_J"] == pytest.approx(180.0)
    assert first["pulsed_mpc_evidence_schema_version"] == PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION
    assert first["pulsed_mpc_admission_digest"] == _admission_digest("a")
    assert len(first["pulse_id"]) == 36
    assert len(report["payload_sha256"]) == 64


def test_shot_result_replay_extension_carries_pulsed_mpc_digest() -> None:
    result = CampaignShotResult(
        shot_id="shot-001",
        pulse_id="123e4567-e89b-12d3-a456-426614174000",
        passed=True,
        failure_reason=None,
        terminal_state="idle",
        transition_states=("idle",),
        command_log=(),
        shot_phase_log=({"t": 0.0, "state": "burn", "reason": "admitted"},),
        capacitor_state_initial_J=200.0,
        capacitor_state_final_J=180.0,
        energy_recovered_J=20.0,
        trigger_timestamp_ns=0,
        pulsed_mpc_admission_digest=_admission_digest("b"),
    )

    extension = result.replay_v1_1_extension()

    assert extension["pulsed_mpc_evidence_schema_version"] == PULSED_MPC_DECISION_EVIDENCE_SCHEMA_VERSION
    assert extension["pulsed_mpc_admission_digest"] == _admission_digest("b")


def test_campaign_rejects_duplicate_shot_ids() -> None:
    shot = _complete_shot("shot-001")

    with pytest.raises(ValueError, match="duplicate shot_id"):
        _orchestrator().run([shot, shot])


def test_campaign_rejects_malformed_pulsed_mpc_digest() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        CampaignShotPlan(
            shot_id="bad-digest",
            samples=_complete_shot("reference").samples,
            initial_bank_voltage_V=5000.0,
            pulsed_mpc_admission_digest="A" * 64,
        )


def test_pyo3_table_bridge_preserves_pulsed_mpc_digest() -> None:
    np = pytest.importorskip("numpy")
    scpn_control_rs = pytest.importorskip("scpn_control_rs")
    if not hasattr(scpn_control_rs, "PyMultiShotCampaignOrchestrator"):
        pytest.skip("scpn_control_rs extension was not rebuilt with multi-shot campaign bindings")

    shot_ids = ["shot-001", "shot-002"]
    shots = [_complete_shot(shot_id) for shot_id in shot_ids]
    sample_shot_index: list[int] = []
    sample_t_s: list[float] = []
    plasma_rows: list[list[float]] = []
    bank_rows: list[list[float]] = []
    for shot_index, shot in enumerate(shots):
        for sample in shot.samples:
            assert sample.bank is not None
            sample_shot_index.append(shot_index)
            sample_t_s.append(sample.t_s)
            plasma_rows.append(
                [
                    sample.plasma.coil_current_A,
                    sample.plasma.temperature_eV,
                    sample.plasma.phase_lock_error_rad,
                    sample.plasma.reference_error_m,
                    sample.plasma.fusion_power_W,
                    sample.plasma.radial_velocity_m_s,
                ]
            )
            bank_rows.append([sample.bank.voltage_V, sample.bank.voltage_max_V, sample.bank.energy_J])

    orchestrator = scpn_control_rs.PyMultiShotCampaignOrchestrator(
        "campaign-a",
        scpn_control_rs.PyPulsedScenarioSpec(
            100.0,
            2.0e6,
            0.01,
            0.002,
            1.0e3,
            2.0e6,
            1.0e3,
            40.0,
            0.95,
            20.0,
            1.0e3,
            0.0,
        ),
        scpn_control_rs.PyCapacitorBankSpec(100e-6, 100e-6, 0.5, 10_000.0, 20.0),
        True,
    )
    report = orchestrator.run_table(
        shot_ids,
        np.asarray(sample_shot_index, dtype=np.uintp),
        np.asarray(sample_t_s, dtype=np.float64),
        np.asarray(plasma_rows, dtype=np.float64),
        np.asarray(bank_rows, dtype=np.float64),
        np.asarray([5000.0, 5000.0], dtype=np.float64),
        [shot.pulsed_mpc_admission_digest for shot in shots],
    )

    assert report["pulsed_mpc_admission_digest_count"] == 2
    assert report["shots"][0]["pulsed_mpc_admission_digest"] == _admission_digest("a")


def test_incomplete_lifecycle_fails_closed_without_aborting_campaign() -> None:
    incomplete = CampaignShotPlan(
        shot_id="incomplete",
        samples=(
            CampaignShotSample(0.0, _plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0), _bank(9800.0, 200.0)),
            CampaignShotSample(1.0e-3, _plasma(2.5e6, 10.0, 0.02, 0.01, 0.0, 0.0), _bank(9800.0, 200.0)),
        ),
        initial_bank_voltage_V=5000.0,
    )

    report = _orchestrator().run([incomplete, _complete_shot("shot-002")])

    assert report["passed_count"] == 1
    assert report["failed_count"] == 1
    assert report["shots"][0]["passed"] is False
    assert report["shots"][0]["failure_reason"] == "shot did not traverse the complete pulsed lifecycle"
    assert report["shots"][1]["passed"] is True


def test_non_monotone_sample_marks_only_that_shot_failed() -> None:
    bad = CampaignShotPlan(
        shot_id="bad-time",
        samples=(
            CampaignShotSample(1.0e-3, _plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0), _bank(9800.0, 200.0)),
            CampaignShotSample(0.0, _plasma(2.5e6, 10.0, 0.02, 0.01, 0.0, 0.0), _bank(9800.0, 200.0)),
        ),
        initial_bank_voltage_V=5000.0,
    )

    report = _orchestrator().run([bad])

    assert report["failed_count"] == 1
    assert report["shots"][0]["failure_reason"] == "t_s must be monotone"


def test_sample_rejects_non_telemetry_plasma() -> None:
    not_plasma: Any = object()
    with pytest.raises(TypeError, match="plasma must be PulsedPlasmaTelemetry"):
        CampaignShotSample(0.0, not_plasma)


def test_sample_rejects_non_telemetry_bank() -> None:
    not_bank: Any = object()
    with pytest.raises(TypeError, match="bank must be CapacitorBankTelemetry or None"):
        CampaignShotSample(0.0, _plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0), not_bank)


def test_sample_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="t_s must be non-negative"):
        CampaignShotSample(-1.0, _plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0))


def test_shot_plan_rejects_blank_shot_id() -> None:
    sample = CampaignShotSample(0.0, _plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0))
    with pytest.raises(ValueError, match="shot_id must not be empty"):
        CampaignShotPlan(shot_id="   ", samples=(sample,), initial_bank_voltage_V=5000.0)


def test_shot_plan_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="samples must not be empty"):
        CampaignShotPlan(shot_id="s", samples=(), initial_bank_voltage_V=5000.0)


def test_shot_plan_rejects_non_finite_initial_current() -> None:
    sample = CampaignShotSample(0.0, _plasma(0.0, 10.0, 0.02, 0.01, 0.0, 0.0))
    with pytest.raises(ValueError, match="initial_bank_current_A must be finite"):
        CampaignShotPlan(
            shot_id="s",
            samples=(sample,),
            initial_bank_voltage_V=5000.0,
            initial_bank_current_A=float("inf"),
        )


def test_orchestrator_rejects_blank_campaign_id() -> None:
    with pytest.raises(ValueError, match="campaign_id must not be empty"):
        MultiShotCampaignOrchestrator("  ", _scheduler_spec(), _bank_spec())


def test_orchestrator_rejects_wrong_scheduler_spec() -> None:
    not_spec: Any = object()
    with pytest.raises(TypeError, match="scheduler_spec must be PulsedScenarioSpec"):
        MultiShotCampaignOrchestrator("c", not_spec, _bank_spec())


def test_orchestrator_rejects_wrong_bank_spec() -> None:
    not_spec: Any = object()
    with pytest.raises(TypeError, match="bank_spec must be CapacitorBankSpec"):
        MultiShotCampaignOrchestrator("c", _scheduler_spec(), not_spec)


def test_run_rejects_empty_shots() -> None:
    with pytest.raises(ValueError, match="shots must not be empty"):
        _orchestrator().run(())


def test_run_rejects_non_shot_plan_entries() -> None:
    not_shot: Any = object()
    with pytest.raises(TypeError, match="shots must contain CampaignShotPlan entries"):
        _orchestrator().run((not_shot,))


def test_admission_failure_reason_flags_non_idle_terminal_state() -> None:
    orchestrator = MultiShotCampaignOrchestrator("c", _scheduler_spec(), _bank_spec(), require_complete_lifecycle=False)
    assert orchestrator._admission_failure_reason("burn", ()) == "shot did not return to idle"
