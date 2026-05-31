# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Geometry-neutral contract tests

from __future__ import annotations

import pytest

from scpn_control.scpn.contracts import FeatureAxisSpec, ControlScales, ControlTargets, extract_features
from scpn_control.scpn.geometry_neutral_contracts import (
    ActuatorChannel,
    ActuatorSet,
    ControlObjective,
    DiagnosticChannel,
    DiagnosticFrame,
    MagneticConfiguration,
    ReplayScenario,
)


def test_magnetic_configuration_serialises_without_tokamak_axis_keys() -> None:
    cfg = MagneticConfiguration(
        name="public_w7x_like",
        device_class="stellarator",
        field_periods=5,
        coordinate_system="boozer_vmec_like",
        reference="public synthetic fixture",
    )

    payload = cfg.to_dict()

    assert payload["device_class"] == "stellarator"
    assert payload["field_periods"] == 5
    assert "R_axis_m" not in payload
    assert "Z_axis_m" not in payload


def test_actuator_channel_enforces_bounds_slew_and_latency() -> None:
    actuator = ActuatorChannel(
        name="helical_trim_A",
        unit="A",
        min_value=-100.0,
        max_value=100.0,
        slew_rate_per_s=50.0,
        latency_steps=2,
    )

    assert actuator.clamp(150.0) == 100.0
    assert actuator.apply_slew(previous=0.0, requested=100.0, dt_s=0.1) == 5.0
    assert actuator.latency_steps == 2


def test_actuator_set_rejects_duplicate_channels() -> None:
    channel = ActuatorChannel(
        name="trim",
        unit="A",
        min_value=-1.0,
        max_value=1.0,
        slew_rate_per_s=1.0,
    )

    with pytest.raises(ValueError, match="unique"):
        ActuatorSet(channels=(channel, channel))


def test_diagnostic_frame_is_mapping_and_rejects_duplicate_channels() -> None:
    frame = DiagnosticFrame(
        step=3,
        time_s=0.003,
        channels=(
            DiagnosticChannel(
                name="fieldline_spread",
                value=0.021,
                unit="rad",
                sigma=0.002,
                provenance="public_synthetic",
            ),
        ),
    )

    assert frame.as_mapping() == {"fieldline_spread": 0.021}

    duplicate = frame.channels[0]
    with pytest.raises(ValueError, match="unique"):
        DiagnosticFrame(step=0, time_s=0.0, channels=(duplicate, duplicate))


def test_replay_scenario_validates_fault_channels_and_serialises_contract() -> None:
    actuator = ActuatorChannel(
        name="helical_trim_A",
        unit="A",
        min_value=-1200.0,
        max_value=1200.0,
        slew_rate_per_s=4.0e5,
        latency_steps=1,
        failure_mode="stuck_supported",
    )
    scenario = ReplayScenario(
        name="stellarator_replay",
        seed=42,
        steps=8,
        dt_s=0.001,
        magnetic_configuration=MagneticConfiguration(
            name="public_w7x_like",
            device_class="stellarator",
            field_periods=5,
            coordinate_system="boozer_vmec_like",
            reference="public synthetic fixture",
        ),
        actuator_set=ActuatorSet(channels=(actuator,)),
        objective=ControlObjective(
            target_metrics={"fieldline_spread": 0.015},
            weights={"fieldline_spread": 1.0},
            constraints={"max_abs_current_A": 1200.0},
        ),
        initial_frame=DiagnosticFrame(
            step=0,
            time_s=0.0,
            channels=(
                DiagnosticChannel(
                    name="fieldline_spread",
                    value=0.04,
                    unit="rad",
                    sigma=0.002,
                    provenance="public_synthetic",
                ),
            ),
        ),
        fault_schedule={3: {"helical_trim_A": "stuck"}},
    )

    payload = scenario.to_dict()

    assert payload["magnetic_configuration"]["device_class"] == "stellarator"
    assert payload["fault_schedule"]["3"]["helical_trim_A"] == "stuck"

    with pytest.raises(KeyError, match="unknown actuator"):
        ReplayScenario(
            name="bad",
            seed=1,
            steps=4,
            dt_s=0.001,
            magnetic_configuration=scenario.magnetic_configuration,
            actuator_set=scenario.actuator_set,
            objective=scenario.objective,
            initial_frame=scenario.initial_frame,
            fault_schedule={1: {"missing": "stuck"}},
        )


def test_replay_scenario_rejects_non_integer_and_empty_fault_entries() -> None:
    actuator = ActuatorChannel(
        name="helical_trim_A",
        unit="A",
        min_value=-1200.0,
        max_value=1200.0,
        slew_rate_per_s=4.0e5,
        failure_mode="stuck_supported",
    )
    base_kwargs = {
        "name": "stellarator_replay",
        "seed": 42,
        "steps": 8,
        "dt_s": 0.001,
        "magnetic_configuration": MagneticConfiguration(
            name="public_w7x_like",
            device_class="stellarator",
            field_periods=5,
            coordinate_system="boozer_vmec_like",
            reference="public synthetic fixture",
        ),
        "actuator_set": ActuatorSet(channels=(actuator,)),
        "objective": ControlObjective(
            target_metrics={"fieldline_spread": 0.015},
            weights={"fieldline_spread": 1.0},
            constraints={"max_abs_current_A": 1200.0},
        ),
        "initial_frame": DiagnosticFrame(
            step=0,
            time_s=0.0,
            channels=(
                DiagnosticChannel(
                    name="fieldline_spread",
                    value=0.04,
                    unit="rad",
                    sigma=0.002,
                    provenance="public_synthetic",
                ),
            ),
        ),
    }

    with pytest.raises(TypeError, match="seed"):
        ReplayScenario(**(base_kwargs | {"seed": True, "fault_schedule": {}}))

    with pytest.raises(TypeError, match="steps"):
        ReplayScenario(**(base_kwargs | {"steps": 8.0, "fault_schedule": {}}))

    with pytest.raises(TypeError, match="fault schedule step"):
        ReplayScenario(**(base_kwargs | {"fault_schedule": {1.5: {"helical_trim_A": "stuck"}}}))

    with pytest.raises(ValueError, match="must not be empty"):
        ReplayScenario(**(base_kwargs | {"fault_schedule": {1: {}}}))

    with pytest.raises(ValueError, match="fault mode"):
        ReplayScenario(**(base_kwargs | {"fault_schedule": {1: {"helical_trim_A": "   "}}}))


def test_replay_scenario_rejects_inconsistent_replay_admission_contracts() -> None:
    unsupported_actuator = ActuatorChannel(
        name="helical_trim_A",
        unit="A",
        min_value=-1200.0,
        max_value=1200.0,
        slew_rate_per_s=4.0e5,
    )
    supported_actuator = ActuatorChannel(
        name="helical_trim_A",
        unit="A",
        min_value=-1200.0,
        max_value=1200.0,
        slew_rate_per_s=4.0e5,
        failure_mode="stuck_supported",
    )
    magnetic_configuration = MagneticConfiguration(
        name="public_w7x_like",
        device_class="stellarator",
        field_periods=5,
        coordinate_system="boozer_vmec_like",
        reference="public synthetic fixture",
    )
    initial_frame = DiagnosticFrame(
        step=0,
        time_s=0.0,
        channels=(
            DiagnosticChannel(
                name="fieldline_spread",
                value=0.04,
                unit="rad",
                sigma=0.002,
                provenance="public_synthetic",
            ),
        ),
    )
    objective = ControlObjective(
        target_metrics={"fieldline_spread": 0.015},
        weights={"fieldline_spread": 1.0},
        constraints={"max_abs_current_A": 1200.0},
    )
    base_kwargs = {
        "name": "stellarator_replay",
        "seed": 42,
        "steps": 8,
        "dt_s": 0.001,
        "magnetic_configuration": magnetic_configuration,
        "actuator_set": ActuatorSet(channels=(supported_actuator,)),
        "objective": objective,
        "initial_frame": initial_frame,
        "fault_schedule": {3: {"helical_trim_A": "stuck"}},
    }

    with pytest.raises(ValueError, match="initial_frame step"):
        ReplayScenario(
            **(
                base_kwargs
                | {
                    "initial_frame": DiagnosticFrame(
                        step=1,
                        time_s=0.0,
                        channels=initial_frame.channels,
                    )
                }
            )
        )

    with pytest.raises(ValueError, match="target_metrics missing"):
        ReplayScenario(
            **(
                base_kwargs
                | {
                    "objective": ControlObjective(
                        target_metrics={"missing_metric": 0.015},
                        weights={"missing_metric": 1.0},
                        constraints={"max_abs_current_A": 1200.0},
                    )
                }
            )
        )

    with pytest.raises(ValueError, match="actuator envelope"):
        ReplayScenario(
            **(
                base_kwargs
                | {
                    "objective": ControlObjective(
                        target_metrics={"fieldline_spread": 0.015},
                        weights={"fieldline_spread": 1.0},
                        constraints={"max_abs_current_A": 1200.1},
                    )
                }
            )
        )

    with pytest.raises(ValueError, match="failure_mode"):
        ReplayScenario(**(base_kwargs | {"actuator_set": ActuatorSet(channels=(unsupported_actuator,))}))

    with pytest.raises(ValueError, match="weights"):
        ControlObjective(
            target_metrics={"fieldline_spread": 0.015},
            weights={"fieldline_spread": 0.0},
            constraints={"max_abs_current_A": 1200.0},
        )


def test_contracts_reject_bool_integer_fields() -> None:
    with pytest.raises(TypeError, match="field_periods"):
        MagneticConfiguration(
            name="public_w7x_like",
            device_class="stellarator",
            field_periods=True,
            coordinate_system="boozer_vmec_like",
            reference="public synthetic fixture",
        )

    with pytest.raises(TypeError, match="latency_steps"):
        ActuatorChannel(
            name="helical_trim_A",
            unit="A",
            min_value=-1.0,
            max_value=1.0,
            slew_rate_per_s=1.0,
            latency_steps=False,
        )

    channel = DiagnosticChannel(
        name="fieldline_spread",
        value=0.04,
        unit="rad",
        sigma=0.002,
        provenance="public_synthetic",
    )
    with pytest.raises(TypeError, match="step"):
        DiagnosticFrame(step=True, time_s=0.0, channels=(channel,))


def test_existing_feature_axis_spec_handles_stellarator_metrics() -> None:
    features = extract_features(
        {"fieldline_spread": 0.021, "effective_ripple": 0.004},
        targets=ControlTargets(),
        scales=ControlScales(),
        feature_axes=[
            FeatureAxisSpec(
                obs_key="fieldline_spread",
                target=0.015,
                scale=0.012,
                pos_key="spread_too_low",
                neg_key="spread_too_high",
            )
        ],
        passthrough_keys=["effective_ripple"],
    )

    assert features["spread_too_high"] == pytest.approx(0.5)
    assert features["spread_too_low"] == 0.0
    assert features["effective_ripple"] == pytest.approx(0.004)
