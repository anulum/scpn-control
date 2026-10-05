# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — Digital twin grid, actuator and metric domain tests

"""Check original declaration identity and presence-only provenance at the public reader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from test_digital_twin_reference_validation import _valid_digital_twin_reference_artifact

from validation.validate_digital_twin_reference import validate_digital_twin_reference

INVALID_GRID_INTEGERS: tuple[object, ...] = tuple([None, [], "4", True, -1, 4.0])
INVALID_STATE_NAMES: tuple[object, ...] = tuple([None, {}, "temperature", [], [" "], [False]])
INVALID_IDS_FLAGS: tuple[object, ...] = tuple([None, [], "true", 0, 1])


ERROR_FIELDS = (
    "final_avg_temp_relative_error",
    "q_profile_rmse",
    "actuator_lag_abs_error",
    "ids_roundtrip_abs_error",
    "island_mask_f1_error",
)


def declaration() -> dict[str, Any]:
    """Supply original engineering declarations and zero error bounds without physical twin evidence."""
    payload = cast(dict[str, Any], _valid_digital_twin_reference_artifact())
    payload["metrics"] = dict.fromkeys(ERROR_FIELDS, 0.0)
    payload["tolerances"] = dict.fromkeys(ERROR_FIELDS, 1.0)
    return payload


@pytest.mark.parametrize(
    ("field", "value", "accepted"),
    [
        *[(field, value, False) for field in ["grid_size", "time_steps", "seed"] for value in INVALID_GRID_INTEGERS],
        ("grid_size", 0, False),
        ("grid_size", 3, False),
        ("grid_size", 4, True),
        ("grid_size", 10**400, True),
        ("time_steps", 0, False),
        ("time_steps", 1, True),
        ("time_steps", 10**400, True),
        ("seed", 0, True),
        ("seed", 10**400, True),
        *[("state_variables", value, False) for value in INVALID_STATE_NAMES],
        ("state_variables", ["unknown", "unknown"], True),
        *[("has_ids_export", value, False) for value in INVALID_IDS_FLAGS],
        ("has_ids_export", True, True),
        ("has_ids_export", False, True),
    ],
)
def test_original_grid_domains(tmp_path: Path, field: str, value: object, accepted: bool) -> None:
    """Original topology integers remain uncapped, state names admit unknown duplicates and either boolean IDS flag passes."""
    payload = declaration()
    payload["grid_metadata"][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_digital_twin_reference(path)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "grid_metadata" for error in report["errors"])


@pytest.mark.parametrize(
    ("value", "accepted"),
    [(None, False), ([], False), ("0", False), (True, False), (-1, False), (0.0, False), (0, True), (10**400, True)],
)
def test_uncapped_integer_actuator_lag(tmp_path: Path, value: object, accepted: bool) -> None:
    """Actuator lag remains a nonboolean nonnegative integer without a cap."""
    payload = declaration()
    payload["actuator_metadata"]["actuator_tau_steps"] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_digital_twin_reference(path)["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize("field", ["actuator_rate_limit", "actuator_bias", "sensor_dropout_prob", "sensor_noise_std"])
@pytest.mark.parametrize("value", [None, [], "1", True, float("nan"), float("inf"), 10**400])
def test_actuator_numeric_refusals(tmp_path: Path, field: str, value: object) -> None:
    """Every numeric actuator/sensor field refuses nonnumbers, booleans, nonfinite conversion and overflowing integers."""
    payload = declaration()
    payload["actuator_metadata"][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_digital_twin_reference(path)
    assert report["status"] == "fail" and report["errors"][0]["field"] == "actuator_metadata"


@pytest.mark.parametrize(
    ("field", "value", "accepted"),
    [
        ("actuator_bias", -1e308, True),
        ("actuator_bias", 0, True),
        ("actuator_bias", 1e308, True),
        *[
            (field, value, accepted)
            for field in ["actuator_rate_limit", "sensor_noise_std"]
            for value, accepted in [(-1, False), (0, True), (5e-324, True), (1e308, True)]
        ],
        ("sensor_dropout_prob", -5e-324, False),
        ("sensor_dropout_prob", 0, True),
        ("sensor_dropout_prob", 1, True),
        ("sensor_dropout_prob", 1.001, False),
    ],
)
def test_signed_bias_and_actuator_boundaries(tmp_path: Path, field: str, value: float, accepted: bool) -> None:
    """Signed bias and uncapped nonnegative rate/noise retain admission; dropout includes zero and one."""
    payload = declaration()
    payload["actuator_metadata"][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_digital_twin_reference(path)["status"] == ("pass" if accepted else "fail")


@pytest.mark.parametrize(
    ("block", "field"),
    [("grid_metadata", field) for field in ["grid_size", "time_steps", "seed", "state_variables", "has_ids_export"]]
    + [
        ("actuator_metadata", field)
        for field in [
            "actuator_tau_steps",
            "actuator_rate_limit",
            "actuator_bias",
            "sensor_dropout_prob",
            "sensor_noise_std",
        ]
    ],
)
def test_all_required_metadata_fields(tmp_path: Path, block: str, field: str) -> None:
    """Each original required grid or actuator key remains mandatory through the persisted public reader."""
    payload = declaration()
    payload[block].pop(field)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    assert validate_digital_twin_reference(path)["errors"][0]["field"] == block


@pytest.mark.parametrize("field", ERROR_FIELDS)
@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, [], "1", True, -1, float("nan"), float("inf"), 10**400, 0, 1.001])
def test_error_and_bound_domains(tmp_path: Path, field: str, block: str, value: object) -> None:
    """Five declared errors remain finite nonnegative with positive finite inclusive bounds, refusing overflow."""
    payload = declaration()
    payload[block][field] = value
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    accepted = (block == "metrics" and value == 0) or (block == "tolerances" and value == 1.001)
    report = validate_digital_twin_reference(path)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in report["errors"])


@pytest.mark.parametrize(
    ("count", "accepted"),
    [
        (None, False),
        ([], False),
        ("1", False),
        (True, False),
        (0, False),
        (-1, False),
        (1.0, False),
        (1, True),
        (10**400, True),
    ],
)
def test_positive_uncapped_case_count(tmp_path: Path, count: object, accepted: bool) -> None:
    """Reference case count remains positive nonboolean integer without an artificial cap."""
    payload = declaration()
    payload["reference_case_count"] = count
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_digital_twin_reference(path)
    assert report["status"] == ("pass" if accepted else "fail")
    if accepted:
        assert report["entries"][0]["reference_case_count"] == count


def test_equal_bounds_and_metadata_only(tmp_path: Path) -> None:
    """Inclusive errors and independent uncalibrated metadata pass without new physical/export/reference authenticity rules."""
    payload = declaration()
    payload["metrics"] = dict(payload["tolerances"])
    payload["grid_metadata"].update(grid_size=10**400, state_variables=["unknown", "unknown"], has_ids_export=False)
    payload["actuator_metadata"].update(
        actuator_tau_steps=10**400,
        actuator_bias=-1e308,
        actuator_rate_limit=0,
        sensor_dropout_prob=1,
        sensor_noise_std=0,
    )
    payload["reference_artifact_sha256"] = "B" * 64
    payload["reference_doi"] = "../unresolved\x00presence"
    payload["units"]["extra"] = "unknown"
    payload["payload_sha256"] = "ignored-extra"
    path = tmp_path / "input.data"
    path.write_text(json.dumps(payload))
    report = validate_digital_twin_reference(path, require_reference_artifacts=True)
    assert report["status"] == "pass" and report["errors"] == []
    assert report["entries"] == [
        {
            "path": str(path),
            "source": "documented_public_reference",
            "model_id": payload["model_id"],
            "model_version": payload["model_version"],
            "reference_dataset_id": payload["reference_dataset_id"],
            "reference_case_count": 5,
        }
    ]
