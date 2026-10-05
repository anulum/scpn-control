# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Control — EPED declared geometry and metric tests

"""Exercise declared EPED numeric domains through real persisted public inputs, without a pedestal solve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from test_eped_reference_validation import _valid_eped_reference_artifact

from validation.validate_eped_reference import canonical_artifact_sha256, validate_eped_reference


def _persist_and_validate(tmp_path: Path, field: str, value: object) -> dict[str, object]:
    """Reseal the original carrier after changing one declared block and inspect the actual file."""
    payload = _valid_eped_reference_artifact()
    payload[field] = value
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    return validate_eped_reference(path, require_reference_artifacts=True)


@pytest.mark.parametrize(
    "value",
    [None, [], [0], [0, True], [0, "1"], [0, 10**400], [0, float("nan")], [-0.1, 1], [0, 1.1], [0, 0], [1, 0], [0, 1]],
)
def test_grid_domains(tmp_path: Path, value: object) -> None:
    """Grid length, numeric representation, monotonicity and inclusive endpoints retain original admission."""
    report = _persist_and_validate(tmp_path, "rho_grid", value)
    assert report["status"] == (
        "pass" if value == [0, 1] and not isinstance(cast(list[object], value)[1], bool) else "fail"
    )
    if report["status"] == "fail":
        assert any(error["field"] == "rho_grid" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize("field", ["pedestal_width_range_psi_n", "beta_limit_range"])
@pytest.mark.parametrize(
    "value",
    [None, [], [0.1], [True, 0.2], [0.1, 10**400], [0, 0.2], [-1, 0.2], [0.2, 0.1], [0.1, 0.1], [0.1, 1], [0.1, 0.2]],
)
def test_width_and_beta_endpoints(tmp_path: Path, field: str, value: object) -> None:
    """Width equal endpoints remain allowed, beta equality fails, width excludes one while beta admits it."""
    report = _persist_and_validate(tmp_path, field, value)
    accepted = (
        value == [0.1, 0.2]
        or (field == "pedestal_width_range_psi_n" and value == [0.1, 0.1])
        or (field == "beta_limit_range" and value == [0.1, 1])
    )
    assert report["status"] == ("pass" if accepted else "fail")
    if not accepted:
        assert any(error["field"] == field for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kappa", True),
        ("kappa", 0),
        ("kappa", 10**400),
        ("delta", None),
        ("delta", 10**400),
        ("delta", -1),
        ("delta", 1),
        ("R0_m", 0),
        ("R0_m", 10**400),
        ("a_m", None),
        ("a_m", 10**400),
        ("a_m", 2),
        ("delta", -0.5),
    ],
)
def test_shaping_numbers(tmp_path: Path, field: str, value: object) -> None:
    """Every shaping scalar rejects invalid or overflowing values, preserving negative admitted triangularity."""
    payload = _valid_eped_reference_artifact()
    shaping = cast(dict[str, object], payload["shaping"])
    shaping[field] = value
    report = _persist_and_validate(tmp_path, "shaping", shaping)
    assert report["status"] == ("pass" if field == "delta" and value == -0.5 else "fail")
    if report["status"] == "fail":
        assert any(error["field"] == "shaping" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize("value", [None, {}, {"kappa": 1}])
def test_shaping_structure(tmp_path: Path, value: object) -> None:
    """Missing shaping dictionary or named fields fail before numeric use."""
    report = _persist_and_validate(tmp_path, "shaping", value)
    assert report["status"] == "fail"
    assert any(error["field"] == "shaping" for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
@pytest.mark.parametrize("value", [None, True, "1", -1, 10**400, float("inf"), float("nan")])
def test_declared_error_number_domains(tmp_path: Path, block: str, value: object) -> None:
    """Declared errors and bounds reject wrong types, negative, nonfinite and unrepresentable values."""
    payload = _valid_eped_reference_artifact()
    mapping = cast(dict[str, object], payload[block])
    mapping["pedestal_width_relative_error"] = value
    report = _persist_and_validate(tmp_path, block, mapping)
    assert report["status"] == "fail"
    assert any(
        error["field"] == "pedestal_width_relative_error" for error in cast(list[dict[str, object]], report["errors"])
    )


@pytest.mark.parametrize("block", ["metrics", "tolerances"])
def test_metric_blocks_must_be_objects(tmp_path: Path, block: str) -> None:
    """Malformed declared metric/bound blocks produce named findings."""
    report = _persist_and_validate(tmp_path, block, [])
    assert report["status"] == "fail"
    assert any(error["field"] == block for error in cast(list[dict[str, object]], report["errors"]))


@pytest.mark.parametrize(
    "field",
    [
        "pedestal_width_relative_error",
        "pedestal_height_relative_error",
        "pressure_limit_relative_error",
        "bootstrap_current_relative_error",
        "collisionality_width_order_error",
    ],
)
@pytest.mark.parametrize("mode", ["equal", "zero-error", "zero-bound", "excess"])
def test_each_declared_metric_boundary(tmp_path: Path, field: str, mode: str) -> None:
    """Each of five errors honors its own positive bound; equality and zero errors pass, excess and zero bounds fail."""
    payload = _valid_eped_reference_artifact()
    metrics = cast(dict[str, object], payload["metrics"])
    bounds = cast(dict[str, object], payload["tolerances"])
    metrics[field] = bounds[field]
    if mode == "zero-error":
        metrics[field] = 0
    elif mode == "zero-bound":
        bounds[field] = 0
    elif mode == "excess":
        metrics[field] = 1
    payload["payload_sha256"] = canonical_artifact_sha256(payload)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload))
    report = validate_eped_reference(path)
    assert report["status"] == ("pass" if mode in {"equal", "zero-error"} else "fail")
    if report["status"] == "fail":
        assert any(error["field"] == field for error in report["errors"])
