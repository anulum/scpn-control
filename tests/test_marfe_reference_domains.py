# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — MARFE declared numeric contract tests

"""Exercise persisted MARFE numeric domains without computing radiation, fronts or density limits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from test_marfe_reference_validation import _valid_marfe_reference_artifact

from validation.validate_marfe_reference import canonical_artifact_sha256, validate_marfe_reference


def _inspect(tmp_path: Path, field: str, value: object) -> dict[str, object]:
    """Reseal a changed declaration and read it through the actual public validator."""
    payload = _valid_marfe_reference_artifact()
    payload[field] = value
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    return validate_marfe_reference(path, require_reference_artifacts=True)


@pytest.mark.parametrize("field", ["temperature_scan_eV", "density_scan_m3"])
@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [1],
        [1, True],
        [1, "2"],
        [1, 10**400],
        [1, float("nan")],
        [0, 1],
        [-1, 2],
        [1, 1],
        [2, 1],
        [1, 2],
        [1, 1e20],
    ],
)
def test_positive_scan_domains(tmp_path: Path, field: str, value: object) -> None:
    """Both scans require positive strictly increasing finite values, admitting values above one."""
    report = _inspect(tmp_path, field, value)
    accepted = value == [1, 2] or value == [1, 1e20]
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [0.1],
        [True, 0.2],
        [0.1, 10**400],
        [0, 0.2],
        [-1, 0.2],
        [0.2, 0.1],
        [0.1, 0.1],
        [0.1, 1],
        [0.1, 1.01],
        [1, 1],
    ],
)
def test_impurity_fraction_domain(tmp_path: Path, value: object) -> None:
    """Impurity endpoints preserve lower equality and inclusive upper one while rejecting zero or overflow."""
    report = _inspect(tmp_path, "impurity_fraction_range", value)
    accepted = value in [[0.1, 0.1], [0.1, 1], [1, 1]]
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(
            error["field"] == "impurity_fraction_range" for error in cast(list[dict[str, object]], report["errors"])
        )


@pytest.mark.parametrize("field", ["R0_m", "a_m", "q95", "connection_length_m"])
@pytest.mark.parametrize("value", [None, True, 0, -1, 10**400, float("inf"), float("nan")])
def test_geometry_scalars(tmp_path: Path, field: str, value: object) -> None:
    """Every geometry scalar requires representable positive numbers and refuses malformed declarations."""
    payload = _valid_marfe_reference_artifact()
    geometry = cast(dict[str, object], payload["geometry"])
    geometry[field] = value
    report = _inspect(tmp_path, "geometry", geometry)
    assert report["status"] == "fail"
    assert any(error["field"] == "geometry" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"R0_m": 1, "a_m": 1, "q95": 3, "connection_length_m": 10},
        {"R0_m": 1, "a_m": 2, "q95": 3, "connection_length_m": 10},
    ],
)
def test_geometry_structure_and_ordering(tmp_path: Path, value: object) -> None:
    """Geometry requires a dictionary with all fields and minor radius strictly below major."""
    report = _inspect(tmp_path, "geometry", value)
    assert report["status"] == "fail"
    assert any(error["field"] == "geometry" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize("field", ["P_SOL_W", "q_perp_W_m2"])
@pytest.mark.parametrize("value", [None, True, -1, 0, 10**400, float("inf"), float("nan"), 1])
def test_power_balance_domains(tmp_path: Path, field: str, value: object) -> None:
    """P_SOL is strictly positive while q_perp allows zero, with the same finite representation requirement."""
    payload = _valid_marfe_reference_artifact()
    power = cast(dict[str, object], payload["power_balance"])
    power[field] = value
    report = _inspect(tmp_path, "power_balance", power)
    accepted = (value == 1 and not isinstance(value, bool)) or (field == "q_perp_W_m2" and value == 0)
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == "power_balance" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize("value", [None, []])
def test_power_balance_structure(tmp_path: Path, value: object) -> None:
    """Malformed power block becomes a named finding without interpreter exceptions."""
    report = _inspect(tmp_path, "power_balance", value)
    assert report["status"] == "fail"
    assert any(error["field"] == "power_balance" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, True, "1", -1, 10**400, float("inf"), float("nan")])
def test_declared_metric_numbers(tmp_path: Path, block: str, value: object) -> None:
    """Error/bound numbers refuse wrong types, negativity, nonfinite and huge conversion safely."""
    payload = _valid_marfe_reference_artifact()
    mapping = cast(dict[str, object], payload[block])
    mapping["onset_temperature_relative_error"] = value
    report = _inspect(tmp_path, block, mapping)
    assert report["status"] == "fail"
    assert any(
        error["field"] == "onset_temperature_relative_error"
        for error in cast(list[dict[str, object]], report["errors"])
    )


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
def test_metric_blocks_must_be_objects(tmp_path: Path, block: str) -> None:
    """Both declared error and bound blocks must be objects."""
    report = _inspect(tmp_path, block, [])
    assert report["status"] == "fail"
    assert any(error["field"] == block for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize(
    "field",
    [
        "onset_temperature_relative_error",
        "density_limit_relative_error",
        "greenwald_fraction_error",
        "front_temperature_min_relative_error",
        "radiation_growth_rate_relative_error",
    ],
)
@pytest.mark.parametrize("mode", ["equal", "zero-error", "zero-bound", "excess"])
def test_each_declared_error_boundary(tmp_path: Path, field: str, mode: str) -> None:
    """Each of five declared error limits admits equality and zero errors, refusing zero bounds and excess."""
    payload = _valid_marfe_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    limits = cast(dict[str, object], payload["tolerances"])
    metrics[field] = limits[field]
    if mode == "zero-error":
        metrics[field] = 0
    elif mode == "zero-bound":
        limits[field] = 0
    elif mode == "excess":
        metrics[field] = 1
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_marfe_reference(path)
    assert report["status"] == ("pass" if mode in {"equal", "zero-error"} else "fail")
    if report["status"] == "fail":
        assert any(error["field"] == field for error in report["errors"])
