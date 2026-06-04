# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Multi-shot campaign orchestrator tests.
from __future__ import annotations

import pytest

from scpn_control.control.capacitor_bank_state import CapacitorBankSpec
from scpn_control.control.multi_shot_campaign import (
    MULTI_SHOT_CAMPAIGN_SCHEMA_VERSION,
    CampaignShotPlan,
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
    return CampaignShotPlan(shot_id=shot_id, samples=samples, initial_bank_voltage_V=5000.0)


def _orchestrator() -> MultiShotCampaignOrchestrator:
    return MultiShotCampaignOrchestrator("campaign-a", _scheduler_spec(), _bank_spec())


def test_campaign_runs_two_complete_shots_with_replay_extensions() -> None:
    report = _orchestrator().run([_complete_shot("shot-001"), _complete_shot("shot-002")])

    assert report["schema_version"] == MULTI_SHOT_CAMPAIGN_SCHEMA_VERSION
    assert report["shot_count"] == 2
    assert report["passed_count"] == 2
    assert report["failed_count"] == 0
    first = report["shots"][0]
    assert first["terminal_state"] == "idle"
    assert first["transition_states"] == report["expected_transition_states"]
    assert first["trigger_timestamp_ns"] == 2_000_000
    assert first["energy_recovered_J"] == pytest.approx(180.0)
    assert len(first["pulse_id"]) == 36
    assert len(report["payload_sha256"]) == 64


def test_campaign_rejects_duplicate_shot_ids() -> None:
    shot = _complete_shot("shot-001")

    with pytest.raises(ValueError, match="duplicate shot_id"):
        _orchestrator().run([shot, shot])


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
