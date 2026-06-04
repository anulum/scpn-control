# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: SCPN Control
# Description: Tests for geometry-neutral replay v1.1 pulsed-shot metadata.
from __future__ import annotations

from copy import deepcopy

import pytest

from scpn_control.scpn.geometry_neutral_replay import (
    SCHEMA_VERSION_V1_1,
    generate_report,
    load_geometry_neutral_replay_report,
    load_replay_schema,
    register_v1_1_schema,
    save_geometry_neutral_replay_report,
    validate_report,
)


def _v1_1_report() -> dict[str, object]:
    report = generate_report(steps=8, seed=20260604)
    bench = report["geometry_neutral_replay"]
    bench["schema_version"] = SCHEMA_VERSION_V1_1
    bench["pulse_id"] = "123e4567-e89b-12d3-a456-426614174000"
    bench["capacitor_state_initial_J"] = 18250.0
    bench["trigger_timestamp_ns"] = 500000
    bench["energy_recovered_J"] = 7350.25
    bench["shot_phase_log"] = [
        {"t": 0.0, "state": "precharge", "reason": "bank voltage admitted"},
        {"t": 0.0005, "state": "trigger", "reason": "ignitron trigger issued"},
        {"t": 0.0010, "state": "burn", "reason": "FRC burn window entered"},
    ]
    bench["frc_diagnostics"] = {
        "s_parameter_at_burn": 1.42,
        "mrti_peak_amplitude_m": 0.0032,
        "tilt_growth_rate_s_inv": -8.5,
    }
    return report


def test_v1_1_schema_registers_and_round_trips(tmp_path) -> None:
    report = _v1_1_report()

    schema = register_v1_1_schema()
    assert schema["$id"] == SCHEMA_VERSION_V1_1
    assert load_replay_schema(SCHEMA_VERSION_V1_1) == schema

    validate_report(report)
    output = save_geometry_neutral_replay_report(report, tmp_path / "replay.json")

    assert load_geometry_neutral_replay_report(output) == report


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("pulse_id", "not-a-uuid", "pulse_id"),
        ("capacitor_state_initial_J", -0.01, "capacitor_state_initial_J"),
    ],
)
def test_v1_1_rejects_invalid_scalar_extensions(field: str, value: object, match: str) -> None:
    report = _v1_1_report()
    bench = report["geometry_neutral_replay"]
    bench[field] = value

    with pytest.raises(ValueError, match=match):
        validate_report(report)


def test_v1_1_rejects_unsorted_shot_phase_log() -> None:
    report = _v1_1_report()
    bench = report["geometry_neutral_replay"]
    bench["shot_phase_log"] = [
        {"t": 0.002, "state": "burn", "reason": "late row first"},
        {"t": 0.001, "state": "trigger", "reason": "earlier row second"},
    ]

    with pytest.raises(ValueError, match="sorted"):
        validate_report(report)


def test_v1_1_validation_is_not_order_sensitive_for_original_report() -> None:
    report = _v1_1_report()
    shuffled = deepcopy(report)
    bench = shuffled["geometry_neutral_replay"]
    bench["shot_phase_log"] = list(reversed(bench["shot_phase_log"]))

    with pytest.raises(ValueError, match="sorted"):
        validate_report(shuffled)
