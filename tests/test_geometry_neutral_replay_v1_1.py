# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Tests for geometry-neutral replay v1.1 pulsed-shot metadata.
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from scpn_control.scpn.geometry_neutral_replay import (
    AER_ADMISSION_SCHEMA_VERSION,
    SCHEMA_VERSION_V1_1,
    attach_aer_admission_metadata,
    build_aer_admission_metadata,
    generate_report,
    load_geometry_neutral_replay_report,
    load_replay_schema,
    register_v1_1_schema,
    save_geometry_neutral_replay_report,
    validate_report,
)
from scpn_control.scpn.observation import AERControlObservation, SpikeBuffer, SpikeEvent

Report = dict[str, Any]


def _v1_1_report() -> Report:
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
    return cast(Report, report)


def _aer_admission_metadata() -> Report:
    buffer = SpikeBuffer(capacity=8)
    buffer.extend(
        [
            SpikeEvent(neuron_id=0, timestamp_ns=10),
            SpikeEvent(neuron_id=1, timestamp_ns=30),
            SpikeEvent(neuron_id=1, timestamp_ns=50),
        ]
    )
    observation = AERControlObservation(
        timestamp_ns=100,
        spike_stream=buffer,
        decode_window_ns=100,
        decode_strategy="rate",
        n_features=4,
        require_monotonic=True,
    )
    admission_report = observation.admission_report()
    features = observation.to_features()
    return cast(
        Report,
        build_aer_admission_metadata(
            admission_report=admission_report,
            decode_strategy="rate",
            decode_window_ns=100,
            n_features=4,
            feature_normalisation="unit",
            require_monotonic=True,
            feature_vector=features,
        ),
    )


def test_v1_1_schema_registers_and_round_trips(tmp_path: Path) -> None:
    report = _v1_1_report()

    schema = register_v1_1_schema()
    assert schema["$id"] == SCHEMA_VERSION_V1_1
    assert load_replay_schema(SCHEMA_VERSION_V1_1) == schema

    validate_report(report)
    output = save_geometry_neutral_replay_report(report, tmp_path / "replay.json")

    assert load_geometry_neutral_replay_report(output) == report


def test_v1_1_replay_save_is_byte_stable(tmp_path: Path) -> None:
    report = _v1_1_report()
    first = save_geometry_neutral_replay_report(report, tmp_path / "first.json")
    second = save_geometry_neutral_replay_report(
        json.loads(first.read_text(encoding="utf-8")),
        tmp_path / "second.json",
    )

    assert first.read_bytes() == second.read_bytes()


def test_v1_1_replay_attaches_digest_bound_aer_admission_metadata(tmp_path: Path) -> None:
    report = _v1_1_report()
    metadata = _aer_admission_metadata()

    attached = attach_aer_admission_metadata(report, metadata)
    bench = attached["geometry_neutral_replay"]
    validate_report(attached)
    assert bench["schema_version"] == SCHEMA_VERSION_V1_1
    assert bench["aer_admission"]["schema_version"] == AER_ADMISSION_SCHEMA_VERSION
    assert bench["aer_admission"]["monotonic_input"] is True
    assert bench["manifest"]["aer_admission_digest"]

    output = save_geometry_neutral_replay_report(attached, tmp_path / "aer_replay.json")
    assert load_geometry_neutral_replay_report(output) == attached


def test_v1_1_replay_rejects_aer_admission_digest_tampering() -> None:
    attached = attach_aer_admission_metadata(_v1_1_report(), _aer_admission_metadata())
    bench = cast(dict[str, Any], attached["geometry_neutral_replay"])
    bench["aer_admission"]["decode_window_ns"] = 101

    with pytest.raises(ValueError, match="aer admission digest"):
        validate_report(attached)


def test_v1_1_replay_rejects_aer_admission_without_manifest_digest() -> None:
    report = _v1_1_report()
    bench = report["geometry_neutral_replay"]
    bench["aer_admission"] = _aer_admission_metadata()

    with pytest.raises(ValueError, match="aer admission digest"):
        validate_report(report)


def test_v1_1_replay_rejects_inconsistent_aer_admission() -> None:
    metadata = _aer_admission_metadata()
    metadata["monotonic_input"] = False
    metadata["out_of_order_event_count"] = 1

    with pytest.raises(ValueError, match="strict monotonic"):
        attach_aer_admission_metadata(_v1_1_report(), metadata)


def test_v1_1_replay_rejects_impossible_aer_retention_count() -> None:
    metadata = _aer_admission_metadata()
    metadata["retained_events"] = int(metadata["capacity"]) + 1

    with pytest.raises(ValueError, match="retained_events"):
        attach_aer_admission_metadata(_v1_1_report(), metadata)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("pulse_id", "not-a-uuid", "pulse_id"),
        ("capacitor_state_initial_J", -0.01, "capacitor_state_initial_J"),
    ],
)
def test_v1_1_rejects_invalid_scalar_extensions(field: str, value: object, match: str) -> None:
    report = _v1_1_report()
    bench = cast(dict[str, Any], report["geometry_neutral_replay"])
    bench[field] = value

    with pytest.raises(ValueError, match=match):
        validate_report(report)


def test_v1_1_rejects_unsorted_shot_phase_log() -> None:
    report = _v1_1_report()
    bench = cast(dict[str, Any], report["geometry_neutral_replay"])
    bench["shot_phase_log"] = [
        {"t": 0.002, "state": "burn", "reason": "late row first"},
        {"t": 0.001, "state": "trigger", "reason": "earlier row second"},
    ]

    with pytest.raises(ValueError, match="sorted"):
        validate_report(report)


def test_v1_1_validation_is_not_order_sensitive_for_original_report() -> None:
    report = _v1_1_report()
    shuffled = deepcopy(report)
    bench = cast(dict[str, Any], shuffled["geometry_neutral_replay"])
    bench["shot_phase_log"] = list(reversed(bench["shot_phase_log"]))

    with pytest.raises(ValueError, match="sorted"):
        validate_report(shuffled)
